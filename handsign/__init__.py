"""NARUTO hand sign detection -- reusable inference and sequence logic.

Minimal integration:

    from handsign import HandSignDetector, SignSmoother, SequenceTracker, load_jutsu

    detector = HandSignDetector("best.pt")
    smoother = SignSmoother()                       # 25-frame window, 18 votes
    tracker  = SequenceTracker(load_jutsu("jutsu.yaml"))

    detection = detector.detect(frame)              # BGR numpy array
    voting = detection.name if detection and detection.confidence >= 0.60 else None
    if (sign := smoother.update(voting)):
        if (jutsu := tracker.add(sign, time.time())):
            print("cast", jutsu.name)
    tracker.tick(time.time())

`detector` requires ultralytics and torch; `smoothing` needs neither, so it can be driven
from recorded predictions or from a different inference backend entirely.
"""

from .classes import CANONICAL, CANONICAL_INDEX, ROMAJI
from .detector import DEFAULT_ACCEPT_CONF, DEFAULT_CONF, Detection, HandSignDetector
from .smoothing import (
    HandSignError,
    Jutsu,
    SequenceTracker,
    SignSmoother,
    load_jutsu,
)

__all__ = [
    "CANONICAL", "CANONICAL_INDEX", "ROMAJI",
    "HandSignDetector", "Detection", "DEFAULT_CONF", "DEFAULT_ACCEPT_CONF",
    "SignSmoother", "SequenceTracker", "Jutsu", "load_jutsu", "HandSignError",
]
