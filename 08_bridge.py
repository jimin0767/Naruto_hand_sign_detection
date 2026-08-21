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

            # Skipping inference is the point of the lock, not just an optimisation:
            # it frees the GPU for the 6 seconds an attack is in flight.
            if not gate.enabled and not args.ignore_lock:
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
                uplink.send(
                    encoder.sign(sign, detection.confidence if detection else 0.0)
                )

                jutsu = tracker.add(sign, now)
                if jutsu:
                    send_cast(encoder, uplink, jutsu, args.quiet)
                else:
                    for candidate, depth in tracker.progress():
                        uplink.send(
                            encoder.progress(candidate.name, depth, len(candidate.signs))
                        )
                        break

            tracker.tick(now, holding=smoother.held is not None)

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


if __name__ == "__main__":
    sys.exit(main())
