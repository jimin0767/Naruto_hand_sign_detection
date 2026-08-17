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
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

CANONICAL = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger",
]

# Measured on the held-out subject: 0.25 gives 92.1% top-1, while 0.5 drops to 85.2%
# because 12% of frames stop detecting anything at all. Raising this makes the demo look
# cleaner per-box while actually losing more signs.
DEFAULT_CONF = 0.25


class DemoError(RuntimeError):
    """Raised on unusable configuration or an unopenable video source."""


# --------------------------------------------------------------------------------------
# Temporal smoothing
# --------------------------------------------------------------------------------------

class SignSmoother:
    """Turn noisy per-frame predictions into stable, de-duplicated sign events.

    Two problems to solve at once. Flicker: a held sign occasionally misreads for a frame
    or two. Repetition: a sign held for a second produces 30 identical frames, which must
    register as *one* sign, not thirty.

    Both fall out of tracking a "held" sign. A new sign is emitted only when the window
    agrees on something different from what is currently held; the held sign clears when
    the window agrees the hands are down, which is what lets the same sign be performed
    twice in a row.
    """

    def __init__(self, window: int = 9, min_votes: int = 6, clear_votes: int = 6):
        if min_votes > window or clear_votes > window:
            raise DemoError(
                f"min_votes ({min_votes}) and clear_votes ({clear_votes}) cannot exceed "
                f"window ({window}); the threshold would be unreachable"
            )
        self.window: deque[str | None] = deque(maxlen=window)
        self.min_votes = min_votes
        self.clear_votes = clear_votes
        self.held: str | None = None

    def update(self, prediction: str | None) -> str | None:
        """Feed one frame's top-1 class (or None). Returns a newly confirmed sign, if any."""
        self.window.append(prediction)
        counts = Counter(p for p in self.window if p is not None)

        if counts:
            candidate, votes = counts.most_common(1)[0]
            if votes >= self.min_votes:
                if candidate != self.held:
                    self.held = candidate
                    return candidate
                return None

        if sum(1 for p in self.window if p is None) >= self.clear_votes:
            self.held = None
        return None

    @property
    def stability(self) -> float:
        """Fraction of the window agreeing with the held sign -- shown as a progress bar."""
        if not self.window or self.held is None:
            return 0.0
        return sum(1 for p in self.window if p == self.held) / self.window.maxlen

    def reset(self) -> None:
        self.window.clear()
        self.held = None


# --------------------------------------------------------------------------------------
# Sequence matching
# --------------------------------------------------------------------------------------

@dataclass
class Jutsu:
    name: str
    english: str
    signs: list[str]


@dataclass
class SequenceTracker:
    """Accumulate confirmed signs and report when the tail matches a known jutsu."""

    jutsu: list[Jutsu]
    timeout_s: float = 4.0
    max_length: int = 16
    buffer: list[str] = field(default_factory=list)
    last_sign_at: float = 0.0
    matched: Jutsu | None = None
    matched_at: float = 0.0

    def __post_init__(self) -> None:
        # Longest first, so a 6-sign jutsu wins over a 3-sign one sharing its ending.
        self.jutsu = sorted(self.jutsu, key=lambda j: len(j.signs), reverse=True)

    def add(self, sign: str, now: float) -> Jutsu | None:
        self.buffer.append(sign)
        del self.buffer[: max(0, len(self.buffer) - self.max_length)]
        self.last_sign_at = now

        for candidate in self.jutsu:
            n = len(candidate.signs)
            if n and self.buffer[-n:] == candidate.signs:
                self.matched = candidate
                self.matched_at = now
                self.buffer.clear()   # consume, so the next attempt starts clean
                return candidate
        return None

    def tick(self, now: float) -> None:
        """Expire a stale partial sequence. Call once per frame."""
        if self.buffer and now - self.last_sign_at > self.timeout_s:
            self.buffer.clear()

    def banner(self, now: float, hold_s: float = 3.0) -> Jutsu | None:
        """The jutsu to display right now, or None once its moment has passed."""
        if self.matched and now - self.matched_at <= hold_s:
            return self.matched
        return None

    def progress(self) -> list[tuple[Jutsu, int]]:
        """Jutsu whose opening signs match the buffer, with how far along each is."""
        out = []
        for candidate in self.jutsu:
            for depth in range(len(self.buffer), 0, -1):
                if candidate.signs[:depth] == self.buffer[-depth:]:
                    out.append((candidate, depth))
                    break
        return sorted(out, key=lambda pair: -pair[1])


def load_jutsu(path: Path) -> list[Jutsu]:
    """Read and validate jutsu.yaml, rejecting signs the model cannot produce."""
    if not path.exists():
        raise DemoError(f"{path} not found")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("jutsu") or []
    if not entries:
        raise DemoError(f"{path} defines no jutsu")

    out = []
    for entry in entries:
        signs = [str(s).strip().lower() for s in entry.get("signs", [])]
        if not signs:
            raise DemoError(f"jutsu {entry.get('name')!r} has no signs")
        unknown = sorted(set(signs) - set(CANONICAL))
        if unknown:
            # Gassho and Mizunoe land here: real signs the reference project detects but
            # this model was never trained on. Failing loudly beats a jutsu that silently
            # can never fire.
            raise DemoError(
                f"jutsu {entry.get('name')!r} uses sign(s) {unknown} that this model "
                f"cannot detect. Valid signs: {CANONICAL}"
            )
        out.append(Jutsu(str(entry.get("name", "?")), str(entry.get("english", "")), signs))
    return out


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, box, label, confidence, smoother, tracker, now, fps):
    """Draw detection, smoothing state, sign buffer, and any matched jutsu."""
    height, width = frame.shape[:2]

    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 235, 60), 2)
        tag = f"{label} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 235, 60), -1)
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
    parser.add_argument("--window", type=int, default=9, help="smoothing window in frames")
    parser.add_argument("--min-votes", type=int, default=6)
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

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

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
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   device=args.device, verbose=False)[0]

            box = label = None
            confidence = 0.0
            boxes = result.boxes
            if len(boxes):
                # Top-1: the demo commits to one sign per frame, which is how the 92%
                # figure was measured. Scoring every box tells a different, worse story.
                best = int(boxes.conf.argmax())
                box = boxes.xyxy[best].tolist()
                label = CANONICAL[int(boxes.cls[best])]
                confidence = float(boxes.conf[best])

            now = time.time()
            confirmed = smoother.update(label)
            if confirmed:
                hit = tracker.add(confirmed, now)
                print(f"  {confirmed:8s} -> [{' > '.join(tracker.buffer) or '(matched)'}]"
                      + (f"   *** {hit.name} ***" if hit else ""))
            tracker.tick(now)

            fps_window.append(time.perf_counter() - started)
            fps = len(fps_window) / max(sum(fps_window), 1e-6)
            frame = draw_overlay(frame, box, label, confidence, smoother, tracker, now, fps)

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
