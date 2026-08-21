"""Headless sensor: watch the webcam, tell Unity which jutsu was cast.

This is the recogniser with the demo's face taken off. It draws nothing and speaks
nothing -- Unity owns the screen and the speakers now. What leaves this process is four
kinds of small JSON datagram (see :mod:`handsign.bridge`), and what comes back is
permission to keep recognising.

    python 08_bridge.py --weights runs/handsign/yolo11m_disjoint/weights/best.pt

Develop the game without a camera or a GPU:

    python 08_bridge.py --fake-cast

then type a jutsu number (or part of its name) and press Enter to cast it. Every other
part of the pipeline behaves identically, so the whole battle system can be built and
debugged this way.

Relationship to 04_demo.py: that script is still the standalone showcase, with the
cel-shaded effects and the voice lines. This one shares the same recognition core
(`HandSignDetector`, `SignSmoother`, `SequenceTracker`) and adds nothing to it.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from handsign import (
    DEFAULT_ACCEPT_CONF,
    DEFAULT_CONF,
    SequenceTracker,
    SignSmoother,
    load_jutsu,
)
from handsign.bridge import (
    DEFAULT_CAST_REPEAT,
    DEFAULT_DOWNLINK_PORT,
    DEFAULT_UPLINK_PORT,
    RecognitionGate,
    UplinkEncoder,
    decode_control,
)
from handsign.transport import UdpDownlink, UdpUplink
from handsign.videolink import DEFAULT_LOCAL_PORT, FrameEncoder


class BridgeError(RuntimeError):
    """Raised on unusable configuration."""


# --------------------------------------------------------------------------- helpers


def open_source(source: str):
    """Open a webcam index or a video file path. Mirrors 04_demo.py."""
    import cv2

    if source.isdigit():
        # CAP_DSHOW avoids a multi-second open delay on Windows' default backend.
        cap = cv2.VideoCapture(
            int(source), cv2.CAP_DSHOW if sys.platform == "win32" else 0
        )
    else:
        if not Path(source).exists():
            raise BridgeError(f"source {source!r} does not exist")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise BridgeError(f"could not open source {source!r}")
    return cap


def stdin_reader(queue: "Queue[str]") -> None:
    """Feed typed lines to the main loop without blocking it."""
    for line in sys.stdin:
        queue.put(line.strip())


def resolve_jutsu(jutsu_list, token: str):
    """Find a jutsu by 1-based index or case-insensitive substring."""
    token = token.strip()
    if not token:
        return None

    if token.isdigit():
        index = int(token) - 1
        return jutsu_list[index] if 0 <= index < len(jutsu_list) else None

    lowered = token.lower()
    for candidate in jutsu_list:
        if candidate.name.lower() == lowered:
            return candidate
    for candidate in jutsu_list:
        if lowered in candidate.name.lower():
            return candidate
    return None


def draw_preview(frame, detection, accept_conf: float, signs, status: str, fps: float):
    """Debug overlay: what the model sees, and what the smoother has committed to.

    Deliberately plain -- this is a diagnostic, not the game. Unity draws the duel.
    The distinction that matters here is *seen* versus *committed*: the model names a
    sign every frame, but only a sign that wins the voting window becomes real. Showing
    both is what tells you whether a miss was the camera or the smoother.
    """
    import cv2

    canvas = frame.copy()
    height, width = canvas.shape[:2]

    if detection is not None:
        x1, y1, x2, y2 = (int(v) for v in detection.box)
        committed = detection.confidence >= accept_conf
        colour = (80, 220, 80) if committed else (60, 160, 220)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        label = f"{detection.name} {detection.confidence:.2f}"
        if not committed:
            label += f"  (below {accept_conf:.2f})"
        cv2.putText(canvas, label, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "no hand sign detected", (14, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 70, 220), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width, 34), (24, 24, 24), -1)
    cv2.putText(canvas, f"{status}   {fps:4.1f} fps", (14, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, height - 42), (width, height), (24, 24, 24), -1)
    sequence = " > ".join(signs[-6:]) if signs else "(no signs committed yet)"
    cv2.putText(canvas, sequence, (14, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 235, 235), 2, cv2.LINE_AA)

    return canvas


def show_preview(canvas) -> bool:
    """Draw one debug frame. Returns False when the user asks to quit."""
    import cv2

    cv2.imshow("08_bridge -- camera debug (Unity draws the game)", canvas)
    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)


def print_menu(jutsu_list) -> None:
    print("\n  fake-cast mode -- type a number or part of a name, then Enter")
    for i, jutsu in enumerate(jutsu_list, 1):
        signs = " ".join(jutsu.signs)
        print(f"   {i:>2}. {jutsu.name:<26} {signs}")
    print("    q. quit\n")


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/handsign/yolo11m_disjoint/weights/best.pt"),
        help="ignored with --fake-cast",
    )
    parser.add_argument("--jutsu", type=Path, default=Path("demo-short.csv"),
                        help="sequence table; must use the same names as Unity's battle_stats.csv")
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--accept-conf", type=float, default=DEFAULT_ACCEPT_CONF,
                        help="confidence below which a frame does not vote")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--window", type=int, default=25, help="voting window in frames")
    parser.add_argument("--min-votes", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="seconds of idle hands before a partial sequence is dropped")

    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--uplink-port", type=int, default=DEFAULT_UPLINK_PORT)
    parser.add_argument("--downlink-port", type=int, default=DEFAULT_DOWNLINK_PORT)
    parser.add_argument("--cast-repeat", type=int, default=DEFAULT_CAST_REPEAT,
                        help="copies of each cast datagram; UDP may drop one")
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)

    parser.add_argument("--fake-cast", action="store_true",
                        help="no camera, no model: cast from the keyboard")
    parser.add_argument("--ignore-lock", action="store_true",
                        help="keep recognising even when Unity says to stop (debugging only)")
    parser.add_argument("--preview", action="store_true",
                        help="debug window showing what the model sees. Unity draws the "
                             "real game; this is only for checking the camera path")
    parser.add_argument("--no-video", action="store_true",
                        help="do not send the face cam to Unity")
    parser.add_argument("--video-port", type=int, default=DEFAULT_LOCAL_PORT)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-quality", type=int, default=55, metavar="1-100")
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not flip the frame. Must match 04_demo.py, which mirrors "
                             "before inference")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    jutsu_list = load_jutsu(args.jutsu)
    if not jutsu_list:
        raise BridgeError(f"no jutsu loaded from {args.jutsu}")

    encoder = UplinkEncoder(cast_repeat=args.cast_repeat)
    gate = RecognitionGate()

    uplink = UdpUplink(args.unity_host, args.uplink_port)
    downlink = UdpDownlink(args.downlink_port)

    print(f"[bridge] uplink   -> {args.unity_host}:{args.uplink_port}")
    print(f"[bridge] downlink <- 0.0.0.0:{args.downlink_port}")
    print(f"[bridge] {len(jutsu_list)} jutsu from {args.jutsu}")

    try:
        if args.fake_cast:
            return run_fake_cast(args, jutsu_list, encoder, gate, uplink, downlink)
        return run_live(args, jutsu_list, encoder, gate, uplink, downlink)
    finally:
        uplink.close()
        downlink.close()


def pump_downlink(downlink, gate: RecognitionGate, smoother, tracker, quiet: bool) -> None:
    """Apply Unity's control messages; reset recognition state when the gate closes.

    This is the half of the recognition lock that lives on this side. Without the reset,
    a sequence half-entered before the lock would survive it and complete the instant
    recognition resumed.
    """
    now = time.time()
    for payload in downlink.poll():
        message = decode_control(payload)
        if message is None:
            continue

        if message.type == "rejected" and not quiet:
            print(f"[bridge] rejected: {message.jutsu} ({message.reason})")

        closed = gate.apply(message, now)
        if not closed:
            continue

        if smoother is not None:
            smoother.reset()
        if tracker is not None:
            tracker.buffer.clear()

        if not quiet:
            print(f"[bridge] {gate.describe(now)}")


def send_cast(encoder, uplink, jutsu, quiet: bool) -> None:
    uplink.send_all(encoder.cast(jutsu.name, jutsu.signs))
    if not quiet:
        print(f"[bridge] cast #{encoder.last_seq}: {jutsu.name}")


# --------------------------------------------------------------------------- fake mode


def run_fake_cast(args, jutsu_list, encoder, gate, uplink, downlink) -> int:
    """Keyboard-driven casting. No camera, no model, no GPU."""
    print_menu(jutsu_list)

    typed: "Queue[str]" = Queue()
    threading.Thread(target=stdin_reader, args=(typed,), daemon=True).start()

    last_heartbeat = 0.0
    interval = 1.0 / max(0.1, args.heartbeat_hz)

    while True:
        pump_downlink(downlink, gate, None, None, args.quiet)

        now = time.time()
        if now - last_heartbeat >= interval:
            last_heartbeat = now
            uplink.send(encoder.heartbeat(0.0))

        try:
            line = typed.get(timeout=0.05)
        except Empty:
            continue

        if line.lower() in {"q", "quit", "exit"}:
            return 0
        if not line:
            continue
        if line == "?":
            print_menu(jutsu_list)
            continue

        jutsu = resolve_jutsu(jutsu_list, line)
        if jutsu is None:
            print(f"[bridge] no jutsu matches {line!r} -- type ? for the list")
            continue

        if not gate.enabled and not args.ignore_lock:
            print(f"[bridge] {gate.describe(now)} -- Unity is not accepting casts")
            continue

        # Mirror what live mode sends, so Unity cannot tell the difference.
        for sign in jutsu.signs:
            uplink.send(encoder.sign(sign, 1.0))
        send_cast(encoder, uplink, jutsu, args.quiet)


# --------------------------------------------------------------------------- live mode


def run_live(args, jutsu_list, encoder, gate, uplink, downlink) -> int:
    """The real thing: webcam -> model -> smoother -> sequence -> Unity."""
    from handsign import HandSignDetector

    detector = HandSignDetector(
        args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device
    )
    smoother = SignSmoother(window=args.window, min_votes=args.min_votes)
    tracker = SequenceTracker(jutsu_list, timeout_s=args.timeout)

    cap = open_source(args.source)
    print("[bridge] recognising -- Ctrl+C to stop")
    if args.preview:
        print("[bridge] preview window open (debug only) -- press q to close the window")

    video = None
    video_link = None
    if not args.no_video:
        video = FrameEncoder(
            width=args.video_width, quality=args.video_quality, fps=args.video_fps
        )
        video_link = UdpUplink(args.unity_host, args.video_port)
        print(f"[bridge] face cam -> {args.unity_host}:{args.video_port} "
              f"({args.video_width}px q{args.video_quality} @{args.video_fps:g}fps)")

    committed: list[str] = []
    frames = 0
    fps = 0.0
    fps_mark = time.time()
    last_heartbeat = 0.0
    interval = 1.0 / max(0.1, args.heartbeat_hz)

    try:
        while True:
            pump_downlink(downlink, gate, smoother, tracker, args.quiet)
            now = time.time()

            if now - last_heartbeat >= interval:
                last_heartbeat = now
                uplink.send(encoder.heartbeat(fps))

            ok, frame = cap.read()
            if not ok:
                print("[bridge] source ended")
                return 0

            # Mirror before inference, exactly as 04_demo.py does. The model must see
            # the same orientation it was validated against, so this is not a display
            # choice -- flipping only the preview would silently change accuracy.
            if not args.no_mirror:
                import cv2

                frame = cv2.flip(frame, 1)

            # ── The face cam is sent BEFORE the lock check, deliberately. ──
            # The lock exists to stop *recognition*, not to stop the camera. Skipping
            # this while locked would freeze the player's own face for the six seconds
            # their attack is in flight, which reads as a crash.
            if video is not None and video.due(now):
                packet = video.encode(frame, now)
                if packet is not None:
                    video_link.send(packet)
                elif video.dropped_oversize == 1:
                    video.dropped_oversize += 1        # warn once, then stay quiet
                    print("[bridge] a video frame was too large to send; "
                          "lower --video-quality or --video-width")

            # Skipping inference is the point of the lock, not just an optimisation:
            # it frees the GPU for the 6 seconds an attack is in flight.
            if not gate.enabled and not args.ignore_lock:
                if args.preview:
                    committed.clear()
                    if not show_preview(
                        draw_preview(frame, None, args.accept_conf, committed,
                                     gate.describe(now), fps)
                    ):
                        return 0
                else:
                    time.sleep(0.01)
                continue

            detection = detector.detect(frame)
            voting = (
                detection.name
                if detection and detection.confidence >= args.accept_conf
                else None
            )

            sign = smoother.update(voting)
            if sign:
                committed.append(sign)
                if not args.quiet:
                    conf = detection.confidence if detection else 0.0
                    print(f"[bridge] sign: {sign} ({conf:.2f})   {' > '.join(committed)}")

                uplink.send(
                    encoder.sign(sign, detection.confidence if detection else 0.0)
                )

                jutsu = tracker.add(sign, now)
                if jutsu:
                    send_cast(encoder, uplink, jutsu, args.quiet)
                    committed.clear()
                else:
                    for candidate, depth in tracker.progress():
                        uplink.send(
                            encoder.progress(candidate.name, depth, len(candidate.signs))
                        )
                        break

            before = len(tracker.buffer)
            tracker.tick(now, holding=smoother.held is not None)
            if len(tracker.buffer) < before:
                committed.clear()          # the sequence timed out; mirror that on screen

            if args.preview:
                if not show_preview(
                    draw_preview(frame, detection, args.accept_conf, committed,
                                 gate.describe(now), fps)
                ):
                    return 0

            frames += 1
            if frames % 30 == 0:
                elapsed = now - fps_mark
                if elapsed > 0:
                    fps = 30.0 / elapsed
                fps_mark = now

    except KeyboardInterrupt:
        print("\n[bridge] stopped")
        return 0
    finally:
        cap.release()
        if video_link is not None:
            video_link.close()
        if args.preview:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
