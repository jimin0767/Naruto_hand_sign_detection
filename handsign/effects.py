"""Jutsu visual effects, composited onto the camera frame.

Effects are procedural rather than sprite-based: lightning is regenerated every frame from
a seeded RNG, which is what produces the restless crackle. Replaying a fixed animation
reads as a looping GIF pasted on the video; noise that never repeats reads as electricity.

Everything is drawn additively into a region of interest around the hand, so cost scales
with the effect's size rather than the frame's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def jagged_path(
    rng: np.random.Generator,
    start: tuple[float, float],
    end: tuple[float, float],
    displace: float,
    min_displace: float = 2.0,
) -> list[tuple[float, float]]:
    """Midpoint-displacement path between two points.

    Recursively splits the segment and jitters the midpoint by a halving amount, which is
    the standard way to get a self-similar bolt: sharp kinks at every scale rather than a
    smooth curve with noise on top.
    """
    if displace < min_displace:
        return [start, end]
    mid = (
        (start[0] + end[0]) / 2 + rng.uniform(-displace, displace),
        (start[1] + end[1]) / 2 + rng.uniform(-displace, displace),
    )
    left = jagged_path(rng, start, mid, displace / 2, min_displace)
    right = jagged_path(rng, mid, end, displace / 2, min_displace)
    return left[:-1] + right


def radial_bolts(
    rng: np.random.Generator,
    centre: tuple[float, float],
    radius: float,
    count: int,
    branch_chance: float = 0.55,
) -> list[list[tuple[float, float]]]:
    """Bolts radiating from a core, some of which fork partway along."""
    bolts = []
    for i in range(count):
        angle = rng.uniform(0, 2 * math.pi) if count < 4 else (
            2 * math.pi * i / count + rng.uniform(-0.35, 0.35)
        )
        reach = radius * rng.uniform(0.55, 1.25)
        tip = (centre[0] + math.cos(angle) * reach, centre[1] + math.sin(angle) * reach)
        path = jagged_path(rng, centre, tip, radius * 0.28)
        bolts.append(path)

        if rng.random() < branch_chance and len(path) > 3:
            k = rng.integers(1, len(path) - 1)
            fork_angle = angle + rng.uniform(-1.1, 1.1)
            fork_len = reach * rng.uniform(0.25, 0.55)
            fork_tip = (path[k][0] + math.cos(fork_angle) * fork_len,
                        path[k][1] + math.sin(fork_angle) * fork_len)
            bolts.append(jagged_path(rng, path[k], fork_tip, radius * 0.16))
    return bolts


@dataclass
class EffectSpec:
    """Look and timing for one jutsu's effect."""
    core: tuple[int, int, int] = (255, 255, 255)     # BGR, the hot centre of a bolt
    glow: tuple[int, int, int] = (255, 190, 90)      # BGR, the bloom around it
    duration: float = 2.6
    charge: float = 0.18                             # ramp-in, seconds
    fade: float = 0.45                               # ramp-out, seconds
    bolts: int = 9
    radius_scale: float = 0.95                       # relative to the detection box
    flash: float = 0.35                              # full-frame flash strength at t=0


CHIDORI = EffectSpec(
    core=(255, 250, 235),       # very slightly cool white
    glow=(255, 165, 45),        # saturated electric blue in BGR
    duration=5.0,
    bolts=15,                   # denser reads as electricity; sparse reads as cracks
    radius_scale=1.0,
    flash=0.34,
)

# Effects are matched by jutsu name. Unlisted jutsu simply show the banner.
EFFECTS: dict[str, EffectSpec] = {
    "Chidori": CHIDORI,
}


def effect_for(jutsu_name: str | None) -> EffectSpec | None:
    return EFFECTS.get(jutsu_name) if jutsu_name else None


class LightningEffect:
    """Draws a crackling electric ball anchored to a point, for a fixed duration."""

    def __init__(self, spec: EffectSpec, seed: int | None = None):
        self.spec = spec
        self.rng = np.random.default_rng(seed)

    def intensity(self, age: float) -> float:
        """Envelope in [0, 1]: ramp up, hold, ramp down."""
        s = self.spec
        if age < 0 or age > s.duration:
            return 0.0
        if age < s.charge:
            return age / s.charge
        if age > s.duration - s.fade:
            return max(0.0, (s.duration - age) / s.fade)
        return 1.0

    def draw(self, frame: np.ndarray, centre: tuple[float, float],
             radius: float, age: float) -> np.ndarray:
        """Composite the effect onto `frame` in place and return it."""
        level = self.intensity(age)
        if level <= 0.01:
            return frame

        s = self.spec
        H, W = frame.shape[:2]
        r = max(24.0, radius * s.radius_scale)
        pad = int(r * 1.9)
        cx, cy = int(centre[0]), int(centre[1])
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1, y1 = min(W, cx + pad), min(H, cy + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame

        roi = frame[y0:y1, x0:x1]
        layer = np.zeros_like(roi)
        local = (centre[0] - x0, centre[1] - y0)

        # Bolt count and reach breathe with the envelope so the ball ignites and dies
        # rather than popping in at full strength.
        count = max(3, int(s.bolts * (0.55 + 0.45 * level)))
        bolts = radial_bolts(self.rng, local, r * (0.7 + 0.3 * level), count)
        polys = [np.asarray(b, dtype=np.int32) for b in bolts if len(b) > 1]

        # Wide dim pass, blurred, becomes the bloom.
        glow = tuple(int(c * level) for c in s.glow)
        cv2.polylines(layer, polys, False, glow, max(2, int(r * 0.075)), cv2.LINE_AA)
        cv2.circle(layer, (int(local[0]), int(local[1])), int(r * 0.26), glow, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), max(3.0, r * 0.16))

        # Thin bright pass on top is the visible arc itself.
        core = tuple(int(c * level) for c in s.core)
        cv2.polylines(layer, polys, False, core, max(1, int(r * 0.024)), cv2.LINE_AA)
        cv2.circle(layer, (int(local[0]), int(local[1])),
                   max(2, int(r * 0.10)), core, -1, cv2.LINE_AA)

        # Additive: light adds to the scene, it does not replace it. Alpha blending here
        # would grey the bolts out against a bright background.
        frame[y0:y1, x0:x1] = cv2.add(roi, layer)

        if age < s.charge and s.flash:
            strength = s.flash * (1.0 - age / max(s.charge, 1e-6))
            tint = np.full_like(frame, np.array(s.glow, np.uint8))
            cv2.addWeighted(frame, 1.0, tint, strength * 0.5, 0, frame)
        return frame


class AnchorTracker:
    """Smoothed hand position for anchoring an effect.

    Raw detection boxes are unusable as an anchor for three reasons: they jitter a few
    pixels every frame, they vanish entirely on frames where the model finds nothing, and
    they occasionally jump across the frame when the model latches onto something else.
    Attached directly, an effect vibrates, freezes, and teleports.

    So: exponential smoothing for the jitter, short velocity-based coasting for dropouts,
    and a snap threshold so a genuine large movement is followed immediately instead of
    being slurred across half a second.
    """

    def __init__(
        self,
        smoothing: float = 0.45,
        snap_distance: float = 160.0,
        coast_s: float = 0.6,
        damping: float = 0.85,
    ):
        self.smoothing = smoothing
        self.snap_distance = snap_distance
        self.coast_s = coast_s
        self.damping = damping
        self.centre: tuple[float, float] | None = None
        self.radius = 0.0
        self.velocity = (0.0, 0.0)
        self.last_seen: float | None = None
        self.last_update: float | None = None

    def update(
        self, box: tuple[float, float, float, float] | None, now: float
    ) -> tuple[tuple[float, float], float] | None:
        """Feed this frame's box (or None) and get the anchor to draw at."""
        dt = 0.0 if self.last_update is None else max(0.0, now - self.last_update)
        self.last_update = now

        if box is not None:
            x1, y1, x2, y2 = box
            target = ((x1 + x2) / 2, (y1 + y2) / 2)
            target_r = max(x2 - x1, y2 - y1) * 0.62

            if self.centre is None:
                self.centre, self.radius, self.velocity = target, target_r, (0.0, 0.0)
            else:
                dx, dy = target[0] - self.centre[0], target[1] - self.centre[1]
                if (dx * dx + dy * dy) ** 0.5 > self.snap_distance:
                    # A jump this large is the model relocating, not a hand moving.
                    # Smoothing it would drag the effect across the frame.
                    self.centre, self.velocity = target, (0.0, 0.0)
                    self.radius = target_r
                else:
                    # Adaptive smoothing. A fixed factor cannot win: heavy enough to
                    # kill a few pixels of jitter is far too heavy to follow a hand
                    # moving quickly, which shows up as the effect trailing behind.
                    # Scaling the factor with distance smooths small wobble hard while
                    # following real movement almost immediately.
                    distance = (dx * dx + dy * dy) ** 0.5
                    reach = min(1.0, distance / max(self.snap_distance, 1e-6))
                    a = self.smoothing + (0.95 - self.smoothing) * reach
                    moved = (self.centre[0] + dx * a, self.centre[1] + dy * a)
                    if dt > 1e-4:
                        self.velocity = ((moved[0] - self.centre[0]) / dt,
                                         (moved[1] - self.centre[1]) / dt)
                    self.centre = moved
                    self.radius += (target_r - self.radius) * a
            self.last_seen = now

        elif self.centre is not None and self.last_seen is not None:
            # Coast on the last known velocity, decaying, so a brief dropout glides
            # instead of freezing. Beyond coast_s the effect simply holds position --
            # guessing further would send it drifting off screen.
            if now - self.last_seen <= self.coast_s and dt > 0:
                self.centre = (self.centre[0] + self.velocity[0] * dt,
                               self.centre[1] + self.velocity[1] * dt)
                self.velocity = (self.velocity[0] * self.damping,
                                 self.velocity[1] * self.damping)
            else:
                self.velocity = (0.0, 0.0)

        if self.centre is None:
            return None
        return self.centre, self.radius

    def reset(self) -> None:
        self.centre = None
        self.radius = 0.0
        self.velocity = (0.0, 0.0)
        self.last_seen = self.last_update = None
