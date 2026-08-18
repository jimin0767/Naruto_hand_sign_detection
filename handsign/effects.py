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
from pathlib import Path

import cv2
import numpy as np


def blit_rgba(dst, tile, x, y, alpha=1.0):
    """Alpha-composite an RGBA tile onto a BGR image, clipped to bounds."""
    th, tw = tile.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + tw), min(H, y + th)
    if x0 >= x1 or y0 >= y1:
        return
    patch = tile[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (patch[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    dst[y0:y1, x0:x1] = (patch[:, :, :3] * a + dst[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)


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
# Values are either an EffectSpec (procedural) or a TransformSpec (sprite); the caller
# builds the matching effect class.
EFFECTS: dict[str, object] = {}


def effect_for(jutsu_name: str | None):
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
        # Smoothed box extent. `radius` suits a round effect; a sprite needs a real
        # height, and deriving one back out of `radius` is a units error waiting to
        # happen -- so the tracker keeps what it actually measured.
        self.size: tuple[float, float] = (0.0, 0.0)
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

            target_size = (x2 - x1, y2 - y1)
            if self.centre is None:
                self.centre, self.radius, self.velocity = target, target_r, (0.0, 0.0)
                self.size = target_size
            else:
                dx, dy = target[0] - self.centre[0], target[1] - self.centre[1]
                if (dx * dx + dy * dy) ** 0.5 > self.snap_distance:
                    # A jump this large is the model relocating, not a hand moving.
                    # Smoothing it would drag the effect across the frame.
                    self.centre, self.velocity = target, (0.0, 0.0)
                    self.radius, self.size = target_r, target_size
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
                    self.size = (self.size[0] + (target_size[0] - self.size[0]) * a,
                                 self.size[1] + (target_size[1] - self.size[1]) * a)
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
        self.size = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.last_seen = self.last_update = None


# --------------------------------------------------------------------------------------
# Sprite transformation (smoke puff, then a character standing where you are)
# --------------------------------------------------------------------------------------

def key_background(bgr: np.ndarray, tolerance: int = 8) -> np.ndarray:
    """Build an alpha channel by flood-filling the background inward from the border.

    Thresholding on "near white" is the obvious approach and the wrong one -- it punches
    holes through every pale part of the subject. Filling from the border only removes
    background actually connected to the edge, so interior light areas survive.

    A second pass keeps only components that touch the border, repairing leaks where an
    anti-aliased edge lets the fill seep into the subject.

    Tolerance is not "smaller is safer": a checkerboard transparency backdrop alternates
    between two values, so too tight a tolerance cannot swallow both and leaves speckle.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    work = bgr.copy()
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    seeds = [(x, y) for x in range(0, w, 2) for y in (0, h - 1)]
    seeds += [(x, y) for y in range(0, h, 2) for x in (0, w - 1)]
    for x, y in seeds:
        if mask[y + 1, x + 1] == 0:
            cv2.floodFill(work, mask, (x, y), 0,
                          (tolerance,) * 3, (tolerance,) * 3, flags)

    background = (mask[1:h + 1, 1:w + 1] > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(background)
    border_labels = {int(v) for row in
                     (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
                     for v in row} - {0}
    background = np.isin(labels, list(border_labels)).astype(np.uint8)
    return np.where(background > 0, 0, 255).astype(np.uint8)


def load_sprite(path: str | Path, tolerance: int = 8) -> np.ndarray:
    """Load an RGBA sprite, keying a flat background if the file has no usable alpha.

    Returns the image cropped to its opaque bounds, so placement can align to the subject
    rather than to whatever padding the source file happened to carry.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sprite not found: {path}")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not decode {path}")

    if raw.ndim == 3 and raw.shape[2] == 4 and (raw[:, :, 3] < 250).mean() > 0.02:
        rgba = raw
    else:
        bgr = raw[:, :, :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        rgba = np.dstack([bgr, key_background(bgr, tolerance)])

    # Pull the alpha in by a pixel. Edge pixels are a blend of subject and the old
    # background, so keeping them draws a bright halo around the whole sprite.
    alpha = cv2.erode(rgba[:, :, 3], np.ones((3, 3), np.uint8), iterations=1)
    rgba[:, :, 3] = cv2.GaussianBlur(alpha, (0, 0), 0.8)

    ys, xs = np.where(rgba[:, :, 3] > 16)
    if len(xs) == 0:
        raise ValueError(f"{path} is fully transparent after keying")
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@dataclass
class TransformSpec:
    """Timing and placement for a sprite transformation."""
    duration: float = 6.0
    smoke_s: float = 0.55          # puff covers the swap for this long
    fade: float = 0.5
    scale: float = 1.3             # sprite height relative to the person box height
    smoke_colour: tuple[int, int, int] = (232, 232, 236)
    puffs: int = 16


class TransformEffect:
    """A puff of smoke, then a character sprite standing where the person is.

    The sprite covers the person rather than replacing their pixels, which is why no
    background inpainting is needed -- and why the smoke matters: it hides the instant
    of the swap, exactly as the source material does.
    """

    def __init__(self, spec: TransformSpec, sprite: np.ndarray, seed: int | None = None):
        self.spec = spec
        self.sprite = sprite
        self.rng = np.random.default_rng(seed)

    def sprite_alpha(self, age: float) -> float:
        s = self.spec
        if age < s.smoke_s * 0.55:
            return 0.0                                  # still hidden by the puff
        if age > s.duration - s.fade:
            return max(0.0, (s.duration - age) / s.fade)
        ramp = (age - s.smoke_s * 0.55) / max(s.smoke_s * 0.45, 1e-6)
        return min(1.0, ramp)

    def smoke_alpha(self, age: float) -> float:
        s = self.spec
        if age < 0 or age > s.smoke_s:
            return 0.0
        return float(np.sin(np.pi * min(1.0, age / s.smoke_s)) ** 0.6)

    def _draw_smoke(self, frame, centre, radius, age) -> None:
        level = self.smoke_alpha(age)
        if level <= 0.01:
            return
        H, W = frame.shape[:2]
        grow = 0.55 + 1.15 * (age / max(self.spec.smoke_s, 1e-6))
        layer = np.zeros_like(frame)
        for _ in range(self.spec.puffs):
            angle = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(0, radius * 0.9) * grow
            px = int(centre[0] + math.cos(angle) * dist)
            py = int(centre[1] + math.sin(angle) * dist * 1.15)
            pr = int(self.rng.uniform(0.30, 0.62) * radius * grow)
            cv2.circle(layer, (px, py), max(4, pr), self.spec.smoke_colour, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), max(6.0, radius * 0.16))
        cv2.addWeighted(frame, 1.0, layer, level, 0, frame)

    def _draw_sprite(self, frame, centre, radius, age) -> None:
        """`radius` is half the person box HEIGHT, not the lightning-style radius."""
        level = self.sprite_alpha(age)
        if level <= 0.01:
            return
        H, W = frame.shape[:2]
        sh, sw = self.sprite.shape[:2]
        target_h = max(16, int(radius * 2 * self.spec.scale))
        target_w = max(8, int(sw * target_h / sh))
        resized = cv2.resize(self.sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Top-aligned to the person box, not centred: on a torso-up webcam shot the box
        # ends at the chest, so centring would sink her head into the middle of the body.
        x = int(centre[0] - target_w / 2)
        y = int(centre[1] - radius)
        blit_rgba(frame, resized, x, y, level)

    def draw(self, frame: np.ndarray, centre: tuple[float, float],
             radius: float, age: float) -> np.ndarray:
        """`radius` is half the person box height; see `_draw_sprite`."""
        if age < 0 or age > self.spec.duration:
            return frame
        self._draw_sprite(frame, centre, radius, age)
        self._draw_smoke(frame, centre, radius, age)     # smoke sits in front
        return frame



# --------------------------------------------------------------------------------------
# Clone jutsu (copies of the caster, receding into the distance)
# --------------------------------------------------------------------------------------

@dataclass
class CloneSpec:
    """Arrangement and timing for duplicated copies of the caster."""
    duration: float = 6.0
    count: int = 6                 # mirrored pairs, so even numbers arrange symmetrically
    stagger: float = 0.13          # delay between each clone popping in
    fade: float = 0.6
    # Spacing is a fraction of the FRAME width, not the cutout width. A caster with
    # arms outstretched produces a near-frame-wide cutout, so cutout-relative spacing
    # throws the far pairs off screen while the near pair buries the caster.
    spread: float = 0.155          # horizontal gap per depth step, as a fraction of frame
    shrink: float = 0.17           # scale lost per depth step
    rise: float = 0.10             # upward shift per depth step, as a fraction of height
    smoke_s: float = 0.45
    smoke_colour: tuple[int, int, int] = (232, 232, 236)


def cutout_from_mask(frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Build a tight RGBA cutout of the masked subject, plus its top-left in the frame."""
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() < 64:
        return None
    # Keep only the largest blob: the segmenter occasionally picks up furniture or a
    # reflection as a second "person", and a floating scrap of laundry breaks the illusion.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)

    ys, xs = np.where(binary > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = frame[y0:y1, x0:x1]
    alpha = (binary[y0:y1, x0:x1] * 255).astype(np.uint8)
    # Soften the cut edge slightly; a hard mask boundary reads as a paper cutout.
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)

    if y1 >= frame.shape[0]:
        # The subject runs off the bottom of the frame, so the silhouette ends in a
        # straight horizontal cut. Invisible on the caster -- it sits on the frame edge --
        # but a clone is shrunk and lifted, which drags that line into open picture where
        # it reads as a pasted rectangle. Dissolve it into a gradient instead.
        height = alpha.shape[0]
        band = max(4, int(height * 0.18))
        ramp = np.linspace(1.0, 0.0, band, dtype=np.float32)[:, None]
        alpha[-band:] = (alpha[-band:] * ramp).astype(np.uint8)

    return np.dstack([patch, alpha]), (x0, y0)


class CloneEffect:
    """Duplicate the caster into a receding formation.

    The cutout is refreshed every frame, so the clones mirror the caster's movement --
    that live sync is what sells them as clones rather than as pasted stills.

    Drawn far-to-near, with the caster's own cutout composited last so the real person
    always ends up in front of their copies.
    """

    def __init__(self, spec: CloneSpec, seed: int | None = None):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.cutout: np.ndarray | None = None
        self.origin: tuple[int, int] = (0, 0)

    def set_cutout(self, cutout: np.ndarray | None, origin: tuple[int, int]) -> None:
        """Called once per frame with a fresh segmentation, if one is available."""
        if cutout is not None:
            self.cutout, self.origin = cutout, origin

    def placements(self) -> list[tuple[int, float, float, float]]:
        """(depth, side, scale, index) for each clone, far ones first."""
        out = []
        for i in range(self.spec.count):
            depth = i // 2 + 1
            side = -1.0 if i % 2 == 0 else 1.0
            out.append((depth, side, max(0.25, 1.0 - self.spec.shrink * depth), i))
        out.sort(key=lambda p: -p[0])          # far to near
        return out

    def alpha_for(self, index: int, age: float) -> float:
        s = self.spec
        start = index * s.stagger
        if age < start:
            return 0.0
        if age > s.duration - s.fade:
            return max(0.0, (s.duration - age) / s.fade)
        return min(1.0, (age - start) / max(s.smoke_s * 0.5, 1e-6))

    def _puff(self, frame, cx, cy, radius, strength) -> None:
        if strength <= 0.02:
            return
        layer = np.zeros_like(frame)
        for _ in range(7):
            angle = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(0, radius * 0.7)
            cv2.circle(layer,
                       (int(cx + math.cos(angle) * dist), int(cy + math.sin(angle) * dist)),
                       max(4, int(radius * self.rng.uniform(0.3, 0.6))),
                       self.spec.smoke_colour, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), max(5.0, radius * 0.18))
        cv2.addWeighted(frame, 1.0, layer, strength, 0, frame)

    def draw(self, frame: np.ndarray, centre: tuple[float, float],
             radius: float, age: float) -> np.ndarray:
        if self.cutout is None or age < 0 or age > self.spec.duration:
            return frame
        s = self.spec
        ch, cw = self.cutout.shape[:2]
        ox, oy = self.origin

        for depth, side, scale, index in self.placements():
            level = self.alpha_for(index, age)
            if level <= 0.01:
                continue
            tw, th = max(8, int(cw * scale)), max(8, int(ch * scale))
            resized = cv2.resize(self.cutout, (tw, th), interpolation=cv2.INTER_AREA)
            # Flank the caster, and lift each step back so the formation recedes rather
            # than standing in a flat row. Overlap is fine -- it reads as a crowd.
            x = int(ox + cw / 2 + side * frame.shape[1] * s.spread * depth - tw / 2)
            y = int(oy + ch * (1 - scale) - ch * s.rise * depth)

            appearing = age - index * s.stagger
            if appearing < s.smoke_s:
                self._puff(frame, x + tw / 2, y + th / 2, max(20.0, tw * 0.45),
                           float(np.sin(np.pi * max(0.0, appearing) / s.smoke_s) ** 0.7))
            blit_rgba(frame, resized, x, y, level)

        # The caster goes back on top: clones drawn near them would otherwise cover the
        # real person, which looks like they vanished rather than multiplied.
        blit_rgba(frame, self.cutout, ox, oy, 1.0)
        return frame

def estimate_yaw(keypoints) -> float | None:
    """Head yaw in [-1, 1] from COCO pose keypoints: -1 full left, 0 facing camera, +1 right.

    Uses where the nose sits between the eyes. Facing the camera the nose is centred
    between them; as the head turns it slides toward the near eye and the eyes crowd
    together. Ear visibility disambiguates a strong turn, where one ear disappears.

    `keypoints` is the COCO 17x3 array: 0 nose, 1 left eye, 2 right eye, 3 left ear,
    4 right ear.
    """
    nose, left_eye, right_eye = keypoints[0], keypoints[1], keypoints[2]
    if nose[2] < 0.3 or left_eye[2] < 0.3 or right_eye[2] < 0.3:
        return None

    centre_x = (left_eye[0] + right_eye[0]) / 2
    span = abs(left_eye[0] - right_eye[0])
    if span < 3:
        return None
    # Normalised by half the eye span, so it is scale invariant.
    yaw = float((nose[0] - centre_x) / (span / 2))

    left_ear, right_ear = keypoints[3], keypoints[4]
    if left_ear[2] < 0.3 <= right_ear[2]:
        yaw = max(yaw, 0.35)
    elif right_ear[2] < 0.3 <= left_ear[2]:
        yaw = min(yaw, -0.35)
    return max(-1.0, min(1.0, yaw))


# --------------------------------------------------------------------------------------
# Flame breath (a jet of fire from the mouth)
# --------------------------------------------------------------------------------------

@dataclass
class FlameSpec:
    """Shape, behaviour and timing for the flame jet."""
    duration: float = 5.0
    ramp: float = 0.30             # seconds for the jet to reach full pressure
    fade: float = 0.7
    reach: float = 0.55            # jet length as a fraction of frame width
    spread: float = 0.26           # cone half-angle, radians
    yaw_swing: float = 1.15        # radians of aim at full head yaw
    rate: int = 620                # particles per SECOND; must stay under
                                   # max_particles/life or the cap truncates the
                                   # oldest particles, which are the jet's far end
    life: float = 0.85             # particle lifetime, seconds
    buoyancy: float = 0.55         # upward drift as a fraction of frame height per second
    turbulence: float = 0.75       # random walk strength
    size: float = 0.034            # particle radius at death, fraction of frame height
    max_particles: int = 700


def fire_lut() -> np.ndarray:
    """256-entry BGR ramp, shaped (256, 1, 3) as cv2.applyColorMap requires.

    Black to dark red to orange to yellow to white.

    Flame colour tracks temperature, so the renderer accumulates a scalar heat field and
    maps it through this once. Colouring each particle individually cannot produce the
    continuous gradient that makes fire look like fire; overlapping coloured blobs just
    read as overlapping blobs.
    """
    stops = [
        (0.00, (0, 0, 0)),
        (0.18, (0, 0, 60)),
        (0.34, (0, 24, 160)),
        (0.52, (10, 90, 235)),
        (0.70, (40, 175, 255)),
        (0.86, (150, 240, 255)),
        (1.00, (255, 255, 255)),
    ]
    lut = np.zeros((256, 1, 3), np.uint8)
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / max(t1 - t0, 1e-6)
                lut[i, 0] = [int(c0[j] + (c1[j] - c0[j]) * k) for j in range(3)]
                break
    return lut


_FIRE_LUT = fire_lut()


class FlameBreathEffect:
    """A jet of fire breathed from the mouth, aimed by the caster head yaw.

    Particles rather than geometry, because fire has no silhouette to draw. Each particle
    is a puff of hot gas launched from the mouth that slows, rises as it cools, and swells
    as it dissipates. They accumulate into a scalar heat field which is blurred once and
    mapped through a temperature ramp, and that is what produces continuous flame instead
    of a pile of overlapping orange circles.
    """

    def __init__(self, spec: FlameSpec, seed: int | None = None):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.yaw = 0.0
        self.last_time: float | None = None
        # Columns: x, y, vx, vy, age, heat
        self.particles = np.zeros((0, 6), np.float32)

    def set_yaw(self, yaw: float | None) -> None:
        """Aim the jet where the caster is looking; None keeps the previous value."""
        if yaw is not None:
            self.yaw += (yaw - self.yaw) * 0.3

    def intensity(self, age: float) -> float:
        s = self.spec
        if age < 0 or age > s.duration:
            return 0.0
        if age < s.ramp:
            return age / s.ramp
        if age > s.duration - s.fade:
            return max(0.0, (s.duration - age) / s.fade)
        return 1.0

    def aim(self) -> float:
        """Jet direction in radians. 0 points right, pi points left."""
        turn = max(-1.0, min(1.0, self.yaw))
        if turn >= 0:
            return turn * self.spec.yaw_swing * 0.5
        return math.pi - abs(turn) * self.spec.yaw_swing * 0.5

    def _spawn(self, mouth, frame_size, level, count):
        s = self.spec
        width, height = frame_size
        speed = s.reach * width / max(s.life, 1e-6)
        angle = self.aim()
        new = np.zeros((count, 6), np.float32)
        jitter = self.rng.normal(0.0, s.spread * 0.5, count)
        # Speed varies per particle. A uniform jet reads as a solid cone; the spread of
        # velocities is what gives flame its ragged, licking edge.
        speeds = speed * self.rng.uniform(0.45, 1.15, count) * (0.55 + 0.45 * level)
        new[:, 0] = mouth[0] + self.rng.normal(0.0, height * 0.008, count)
        new[:, 1] = mouth[1] + self.rng.normal(0.0, height * 0.008, count)
        new[:, 2] = np.cos(angle + jitter) * speeds
        new[:, 3] = np.sin(angle + jitter) * speeds
        new[:, 5] = self.rng.uniform(0.7, 1.0, count)
        self.particles = np.vstack([self.particles, new])[-s.max_particles:]

    def _advance(self, dt, frame_size):
        s = self.spec
        if not len(self.particles):
            return
        p = self.particles
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt
        # Drag first, then buoyancy: hot gas slows quickly and then rises as it cools,
        # which curls the tail of the jet upward instead of leaving it a straight line.
        decay = max(0.0, 1.0 - 1.1 * dt)
        p[:, 2] *= decay
        p[:, 3] *= decay
        p[:, 3] -= s.buoyancy * frame_size[1] * dt
        if s.turbulence:
            wobble = s.turbulence * frame_size[1] * dt
            p[:, 2] += self.rng.normal(0.0, wobble, len(p))
            p[:, 3] += self.rng.normal(0.0, wobble, len(p))
        p[:, 4] += dt
        self.particles = p[p[:, 4] < s.life]

    def draw(self, frame: np.ndarray, centre: tuple[float, float],
             radius: float, age: float) -> np.ndarray:
        """`centre` is the mouth. `radius` is unused; size comes from the frame."""
        s = self.spec
        level = self.intensity(age)
        height, width = frame.shape[:2]

        dt = 1 / 30 if self.last_time is None else min(0.1, max(1e-3, age - self.last_time))
        self.last_time = age
        self._advance(dt, (width, height))
        if level > 0.01:
            self._spawn(centre, (width, height), level,
                        max(1, int(s.rate * dt * level)))
        if not len(self.particles):
            return frame

        # Accumulate at half resolution: the field is blurred anyway, so the detail is
        # thrown away regardless, and rasterising 700 circles costs a quarter as much.
        scale = 0.5
        hh, hw = int(height * scale), int(width * scale)
        heat = np.zeros((hh, hw), np.float32)
        max_radius = s.size * height * scale
        for x, y, _, _, born, hot in self.particles:
            t = float(born) / s.life
            # Cool and swell with age: a small bright core leaving the mouth becomes a
            # large faint smudge by the time it dies.
            radius_px = max(1, int(max_radius * (0.18 + 0.82 * t)))
            # float(): OpenCV rejects numpy scalars for the colour argument.
            # Each particle contributes only a sliver. Full-strength blobs saturate the
            # field to white on the first overlap, which is a glowing lozenge, not fire --
            # the gradient has to be built up by accumulation.
            value = float(float(hot) * 0.46 * (1.0 - t) ** 1.5 * level)
            cv2.circle(heat, (int(x * scale), int(y * scale)), radius_px, value, -1)

        # The ignition point itself: the jet is spread thin over its length, so without
        # this the mouth never reaches white and the whole plume stays dull red.
        cv2.circle(heat, (int(centre[0] * scale), int(centre[1] * scale)),
                   max(2, int(max_radius * 0.85)), float(level), -1)

        heat = cv2.GaussianBlur(heat, (0, 0), max(1.5, max_radius * 0.30))
        heat = np.clip(heat, 0.0, 1.0)
        # Colour at half resolution too, then upscale: the alternative is three
        # full-resolution float passes for detail the blur already discarded.
        coloured = cv2.applyColorMap((heat * 255).astype(np.uint8), _FIRE_LUT)
        coloured = (coloured * heat[:, :, None]).astype(np.uint8)
        coloured = cv2.resize(coloured, (width, height), interpolation=cv2.INTER_LINEAR)
        # Additive, so cool outer wisps stay translucent while the core burns to white.
        frame[:] = cv2.add(frame, coloured)
        return frame


# --------------------------------------------------------------------------------------
# Registry -- last in the file so every spec class above is already defined.
# --------------------------------------------------------------------------------------

TRANSFORMATION = TransformSpec(duration=6.0, smoke_s=0.55, scale=1.3)
CLONE = CloneSpec(duration=6.0, count=6)
DRAGON_FLAME = FlameSpec()

EFFECTS.update({
    "Chidori": CHIDORI,
    "Transformation Jutsu": TRANSFORMATION,
    "Clone Jutsu": CLONE,
    "Dragon Flame Jutsu": DRAGON_FLAME,
})
