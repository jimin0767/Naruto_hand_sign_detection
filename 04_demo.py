"""Live webcam demo: detect hand signs, smooth them over time, match jutsu sequences.

Temporal smoothing matters more here than raw per-frame accuracy. Measured top-1 on an
unseen subject is 92%, but the errors are not uniformly distributed -- `hare` is confused
with `monkey` in a third of cases, because both are "one hand stacked over the other" and
differ only in finger detail that motion blur erases. Per-frame that reads as flicker; a
majority vote over a short window removes almost all of it.

The design keeps three things apart so each can be tested without a camera:
  SignSmoother     frame predictions -> confirmed signs (debounce + hysteresis)
  SequenceTracker  confirmed signs   -> matched jutsu (with idle timeout)
  draw_overlay     state             -> pixels

Usage:
    python 04_demo.py                          # default webcam
    python 04_demo.py --source 1               # second camera
    python 04_demo.py --source clip.mp4        # a video file
    python 04_demo.py --record out.mp4         # save what you demo
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

from handsign import (
    CANONICAL,
    DEFAULT_ACCEPT_CONF,
    DEFAULT_CONF,
    HandSignDetector,
    HandSignError,
    Jutsu,
    SequenceTracker,
    SignSmoother,
    load_jutsu,
)

DemoError = HandSignError


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, box, label, confidence, smoother, tracker, now, fps, voting=True):
    """Draw detection, smoothing state, sign buffer, and any matched jutsu.

    `voting` distinguishes a detection confident enough to count toward a sign from one
    the model is merely guessing at -- drawn grey, so idle-hand false positives are
    visible as rejected rather than invisible.
    """
    height, width = frame.shape[:2]
    accent = (0, 235, 60) if voting else (130, 130, 130)

    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 2)
        tag = f"{label} {confidence:.2f}" + ("" if voting else "  (ignored)")
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), accent, -1)
        cv2.putText(frame, tag, (x1 + 4, max(th, y1 - 5)), FONT, 0.6, (0, 0, 0), 2)

    # Held sign + stability bar: shows *why* a sign has or hasn't registered yet.
    if smoother.held:
        cv2.putText(frame, smoother.held.upper(), (14, 40), FONT, 1.1, (0, 235, 60), 3)
        filled = int(160 * smoother.stability)
        cv2.rectangle(frame, (14, 52), (174, 62), (70, 70, 70), -1)
        cv2.rectangle(frame, (14, 52), (14 + filled, 62), (0, 235, 60), -1)

    cv2.putText(frame, f"{fps:4.1f} FPS", (width - 110, 28), FONT, 0.6, (200, 200, 200), 2)

    # Sign buffer along the bottom.
    sequence = " > ".join(tracker.buffer) if tracker.buffer else "-"
    cv2.rectangle(frame, (0, height - 42), (width, height), (25, 25, 28), -1)
    cv2.putText(frame, sequence[:70], (14, height - 15), FONT, 0.65, (235, 235, 235), 2)

    partial = tracker.progress()
    if partial and not tracker.banner(now):
        best, depth = partial[0]
        cv2.putText(frame, f"{best.name}  ({depth}/{len(best.signs)})",
                    (14, height - 52), FONT, 0.55, (150, 190, 255), 2)

    hit = tracker.banner(now)
    if hit:
        # Low and translucent. Centred and opaque covers the hands -- the one thing the
        # audience is actually looking at when the jutsu fires.
        text = hit.name
        scale = min(1.3, (width - 60) / max(len(text) * 22, 1))
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 3)
        x = max(10, (width - tw) // 2)
        baseline = int(height * 0.78)
        top, bottom = baseline - th - 14, baseline + 26

        panel = frame[max(0, top):min(height, bottom), :].copy()
        frame[max(0, top):min(height, bottom), :] = cv2.addWeighted(
            panel, 0.25, cv2.GaussianBlur(panel, (0, 0), 8), 0.75, -30
        )
        cv2.putText(frame, text, (x, baseline), FONT, scale, (60, 220, 255), 3)
        if hit.english:
            (ew, _), _ = cv2.getTextSize(hit.english, FONT, 0.55, 2)
            cv2.putText(frame, hit.english, (max(10, (width - ew) // 2), baseline + 20),
                        FONT, 0.55, (210, 210, 210), 2)
    return frame


# --------------------------------------------------------------------------------------

def open_source(source: str) -> cv2.VideoCapture:
    """Open a webcam index or a video file path."""
    if source.isdigit():
        # CAP_DSHOW avoids a multi-second open delay on Windows' default backend.
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    else:
        if not Path(source).exists():
            raise DemoError(f"source {source!r} does not exist")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise DemoError(f"could not open source {source!r}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("runs/handsign/yolo11m_disjoint/weights/best.pt"))
    parser.add_argument("--jutsu", type=Path, default=Path("jutsu.yaml"))
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--window", type=int, default=25,
                        help="smoothing window in frames; at ~78 FPS, 25 is about 0.3s")
    parser.add_argument("--min-votes", type=int, default=18)
    parser.add_argument(
        "--accept-conf", type=float, default=DEFAULT_ACCEPT_CONF,
        help="a frame only votes toward a sign above this confidence, separate from "
             "--conf which controls what is drawn. Every training image contains exactly "
             "one sign, so the model has no 'not a sign' output and will name one for "
             "idle hands -- but typically with lower confidence than a real sign.",
    )
    parser.add_argument("--timeout", type=float, default=4.0,
                        help="seconds of inactivity before a partial sequence is cleared")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--headless", action="store_true",
                        help="skip the preview window; for CI, or recording a clip on a "
                             "machine with no display")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"error: {args.weights} not found; run 03_train.py first", file=sys.stderr)
        return 1

    jutsu = load_jutsu(args.jutsu)
    print(f"loaded {len(jutsu)} jutsu from {args.jutsu}")

    detector = HandSignDetector(args.weights, conf=args.conf, imgsz=args.imgsz,
                                device=args.device)

    cap = open_source(args.source)
    smoother = SignSmoother(args.window, args.min_votes)
    tracker = SequenceTracker(jutsu, timeout_s=args.timeout)

    writer = None
    fps_window: deque[float] = deque(maxlen=30)
    print("running -- q or Esc to quit, r to reset the sequence")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)   # selfie view

            started = time.perf_counter()
            hit = detector.detect(frame)
            box = hit.box if hit else None
            label = hit.name if hit else None
            confidence = hit.confidence if hit else 0.0

            now = time.time()
            # Two thresholds. The box is drawn above --conf so you can see what the model
            # is thinking, but only a confident detection votes toward committing a sign.
            voting_label = label if confidence >= args.accept_conf else None
            confirmed = smoother.update(voting_label)
            if confirmed:
                hit = tracker.add(confirmed, now)
                print(f"  {confirmed:8s} -> [{' > '.join(tracker.buffer) or '(matched)'}]"
                      + (f"   *** {hit.name} ***" if hit else ""))
            tracker.tick(now)

            fps_window.append(time.perf_counter() - started)
            fps = len(fps_window) / max(sum(fps_window), 1e-6)
            frame = draw_overlay(frame, box, label, confidence, smoother, tracker, now,
                                 fps, voting=voting_label is not None)

            if args.record:
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            if not args.headless:
                cv2.imshow("NARUTO hand signs", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    smoother.reset()
                    tracker.buffer.clear()
                    print("  -- reset")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.record}")
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
