"""Inference wrapper: a frame in, at most one hand sign out.

The model is a detector, but every training image contains exactly one sign, so in
practice it behaves as a localiser plus a 12-way classifier. This wrapper commits to the
single highest-confidence box per frame -- which is how the 92% top-1 figure was measured,
and how a game should consume it. Scoring every returned box tells a different and much
worse story, because the model emits low-confidence duplicates alongside the right answer.

Accepts .pt or .onnx; Ultralytics dispatches on the extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .classes import CANONICAL

# Measured on the held-out subject: conf 0.25 gives 92.1% top-1, while 0.5 gives 85.2%
# because 12% of frames then detect nothing at all.
DEFAULT_CONF = 0.25

# Separate, stricter gate for *committing* to a sign. No training image lacks a sign, so
# the model has no "not a sign" output and will name one for idle hands -- usually with
# lower confidence than a real sign. See README "Known behaviour".
DEFAULT_ACCEPT_CONF = 0.60


@dataclass(frozen=True)
class Detection:
    """One hand sign in one frame."""

    class_id: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels

    @property
    def romaji(self) -> str:
        from .classes import ROMAJI
        return ROMAJI[self.name]


class HandSignDetector:
    """Load a model once, then call `detect` per frame."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = DEFAULT_CONF,
        imgsz: int = 640,
        device: str | int = 0,
    ):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        # A model whose class list disagrees with ours would produce plausible-looking
        # nonsense rather than an error, which is the exact failure this project exists
        # to prevent. Check it at load time.
        names = getattr(self.model, "names", None)
        if names:
            ordered = tuple(names[i] for i in sorted(names))
            if ordered != CANONICAL:
                raise ValueError(
                    f"weights declare class order {ordered}, but this package expects "
                    f"{CANONICAL}. Indices would be silently mismatched."
                )

    def detect(self, frame) -> Detection | None:
        """Return the highest-confidence sign in a BGR frame, or None."""
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        boxes = result.boxes
        if not len(boxes):
            return None
        best = int(boxes.conf.argmax())
        class_id = int(boxes.cls[best])
        return Detection(
            class_id=class_id,
            name=CANONICAL[class_id],
            confidence=float(boxes.conf[best]),
            box=tuple(float(v) for v in boxes.xyxy[best]),
        )
