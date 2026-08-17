"""Demo presentation layer: camera panel, sign card strip, countdown, jutsu reveal.

Kept apart from the capture loop and from the recognition logic so the look can change
without touching either.

Kanji cannot be drawn by OpenCV -- `cv2.putText` has no CJK glyphs. PIL can, but calling
into PIL every frame is wasteful when there are only twelve possible characters, so every
glyph and label is rasterised once at construction and blitted thereafter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .classes import CANONICAL, KANJI, ROMAJI
from .effects import blit_rgba  # noqa: F401  re-exported for existing importers

# Dark, high-contrast palette. Orange accent reads as Naruto without being garish, and
# stays legible against the skin tones and room lighting that dominate a webcam frame.
INK = (18, 16, 14)            # panel background (BGR)
CARD = (34, 31, 28)           # card fill
CARD_EDGE = (58, 54, 50)
ACCENT = (24, 122, 255)       # orange (BGR)
ACCENT_DIM = (40, 70, 110)
TEXT = (238, 238, 240)
MUTED = (150, 146, 142)
GREEN = (80, 220, 90)
GREY = (130, 130, 130)

FONT = cv2.FONT_HERSHEY_SIMPLEX

CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/YuGothB.ttc",     # Yu Gothic Bold
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]

STRIP_H = 172
CARD_W, CARD_H = 104, 92
LABEL_H = 34
CARD_GAP = 12
GROW_S = 0.22                 # card entry animation, seconds


def find_cjk_font() -> str | None:
    for path in CJK_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def rounded_rect(img, p1, p2, colour, radius=10, thickness=-1):
    """Filled or outlined rounded rectangle. cv2 has no primitive for this."""
    x1, y1 = p1
    x2, y2 = p2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, colour, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), colour, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), colour, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), colour, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), colour, thickness)
        for cx, cy, a0, a1 in ((x1 + radius, y1 + radius, 180, 270),
                               (x2 - radius, y1 + radius, 270, 360),
                               (x1 + radius, y2 - radius, 90, 180),
                               (x2 - radius, y2 - radius, 0, 90)):
            cv2.ellipse(img, (cx, cy), (radius, radius), 0, a0, a1, colour, thickness)


def centre_text(img, text, cx, y, scale, colour, thickness=2):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(img, text, (int(cx - w / 2), y), FONT, scale, colour, thickness, cv2.LINE_AA)
    return w


@dataclass
class Card:
    """One recognised sign in the strip."""
    sign: str
    born: float


@dataclass
class UIState:
    """Everything the renderer needs, gathered by the demo loop."""
    box: tuple[float, float, float, float] | None = None
    label: str | None = None
    confidence: float = 0.0
    voting: bool = True
    held: str | None = None
    stability: float = 0.0
    cards: list[Card] = field(default_factory=list)
    time_left: float | None = None
    timeout_s: float = 3.0
    jutsu_name: str | None = None
    jutsu_element: str = ""
    jutsu_age: float = 0.0
    fps: float = 0.0
    # Jutsu effect, anchored to the hand. Owned by the demo loop; drawn here so it sits
    # above the video but below the HUD, keeping the readouts legible through it.
    effect: object | None = None
    effect_centre: tuple[float, float] | None = None
    effect_radius: float = 0.0
    effect_age: float = 0.0


class DemoUI:
    """Composites the camera panel and the card strip into one canvas."""

    def __init__(self, width: int, height: int, font_path: str | None = None):
        self.width = width
        self.height = height
        self.font_path = font_path or find_cjk_font()
        self.kanji_tiles: dict[str, np.ndarray] = {}
        self.big_kanji: dict[str, np.ndarray] = {}
        if self.font_path:
            self._prerender()
        else:
            print("  note: no CJK font found; showing romaji instead of kanji",
                  file=sys.stderr)

    # -- asset preparation ------------------------------------------------------------

    def _text_tile(self, text: str, size: int, colour) -> np.ndarray:
        """Rasterise one string to a tight RGBA tile using PIL."""
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(self.font_path, size)
        box = font.getbbox(text)
        w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        img = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
        # `colour` is BGR and the tile is composited straight into a BGR canvas, so the
        # channels are passed through unswapped -- PIL's "R" slot ends up in OpenCV's
        # blue channel, which is exactly what we want. Converting here would double-swap.
        ImageDraw.Draw(img).text((-box[0] + 4, -box[1] + 4), text,
                                 font=font, fill=(colour[0], colour[1], colour[2], 255))
        return np.array(img)

    def _prerender(self) -> None:
        # Only twelve glyphs exist, so this is cheap and removes PIL from the frame loop.
        for name in CANONICAL:
            self.kanji_tiles[name] = self._text_tile(KANJI[name], 58, TEXT)
            self.big_kanji[name] = self._text_tile(KANJI[name], 54, ACCENT)

    # -- pieces -----------------------------------------------------------------------

    def _draw_detection(self, frame, st: UIState) -> None:
        if st.box is None:
            return
        x1, y1, x2, y2 = (int(v) for v in st.box)
        colour = GREEN if st.voting else GREY
        # Corner brackets rather than a full rectangle: less of the hands is covered.
        L = max(18, min(40, (x2 - x1) // 4))
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), colour, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), colour, 3, cv2.LINE_AA)
        tag = f"{st.label} {st.confidence:.2f}" + ("" if st.voting else "  ignored")
        cv2.putText(frame, tag, (x1, max(18, y1 - 10)), FONT, 0.55, colour, 2, cv2.LINE_AA)

    def _draw_held(self, frame, st: UIState) -> None:
        if not st.held:
            return
        pad, pw, ph = 14, 252, 86
        x0, y0 = pad, pad
        rounded_rect(frame, (x0, y0), (x0 + pw, y0 + ph), INK, 12, -1)
        rounded_rect(frame, (x0, y0), (x0 + pw, y0 + ph), CARD_EDGE, 12, 1)
        tx = x0 + 18
        if self.font_path:
            tile = self.big_kanji[st.held]
            th, tw = tile.shape[:2]
            blit_rgba(frame, tile, x0 + 16, y0 + (ph - th) // 2 - 4)
            tx = x0 + 16 + tw + 14
        cv2.putText(frame, st.held.upper(), (tx, y0 + 40), FONT, 0.78, TEXT, 2, cv2.LINE_AA)
        cv2.putText(frame, ROMAJI[st.held], (tx, y0 + 62), FONT, 0.48, MUTED, 1, cv2.LINE_AA)
        # Stability bar: shows why a sign has or has not committed yet.
        bar_y = y0 + ph - 12
        cv2.rectangle(frame, (x0 + 14, bar_y), (x0 + pw - 14, bar_y + 4), CARD_EDGE, -1)
        bw = int((pw - 28) * min(1.0, st.stability))
        if bw:
            cv2.rectangle(frame, (x0 + 14, bar_y), (x0 + 14 + bw, bar_y + 4), ACCENT, -1)

    def _draw_timer(self, frame, st: UIState) -> None:
        """Countdown ring, top right. Only present while the clock is actually running."""
        if st.time_left is None:
            return
        W = frame.shape[1]
        r, cx, cy = 34, W - 74, 74
        rounded_rect(frame, (W - 142, 14), (W - 14, 134), INK, 14, -1)
        rounded_rect(frame, (W - 142, 14), (W - 14, 134), CARD_EDGE, 14, 1)
        frac = max(0.0, min(1.0, st.time_left / max(st.timeout_s, 1e-6)))
        cv2.circle(frame, (cx, cy), r, ACCENT_DIM, 4, cv2.LINE_AA)
        if frac > 0:
            # Runs anticlockwise from 12 o'clock, so the arc visibly drains.
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360 * frac,
                        ACCENT if frac > 0.33 else (60, 60, 235), 4, cv2.LINE_AA)
        centre_text(frame, f"{st.time_left:.1f}", cx, cy + 8, 0.72, TEXT, 2)
        centre_text(frame, "RESET IN", cx, 124, 0.38, MUTED, 1)

    def _draw_strip(self, canvas, st: UIState, now: float) -> None:
        """The card row below the camera: kanji above, English below."""
        top = self.height
        cv2.rectangle(canvas, (0, top), (self.width, top + STRIP_H), INK, -1)
        cv2.line(canvas, (0, top), (self.width, top), CARD_EDGE, 1)

        if st.jutsu_name:
            self._draw_jutsu(canvas, st, top)
            return

        if not st.cards:
            centre_text(canvas, "PERFORM A HAND SIGN", self.width // 2,
                        top + STRIP_H // 2 + 6, 0.6, (70, 66, 62), 2)
            return

        step = CARD_W + CARD_GAP
        visible = st.cards[-max(1, (self.width - 40) // step):]
        total = len(visible) * step - CARD_GAP
        x = (self.width - total) // 2
        y = top + 14

        for i, card in enumerate(visible):
            age = now - card.born
            grow = min(1.0, age / GROW_S)
            ease = 1 - (1 - grow) ** 3            # ease-out cubic
            # Newest card is accented; the arrow between cards shows sequence order.
            newest = i == len(visible) - 1
            cw = int(CARD_W * (0.85 + 0.15 * ease))
            ch = int(CARD_H * (0.85 + 0.15 * ease))
            cx = x + i * step + (CARD_W - cw) // 2
            cy = y + (CARD_H - ch) // 2

            rounded_rect(canvas, (cx, cy), (cx + cw, cy + ch), CARD, 12, -1)
            rounded_rect(canvas, (cx, cy), (cx + cw, cy + ch),
                         ACCENT if newest else CARD_EDGE, 12, 2 if newest else 1)

            if self.font_path and grow > 0.15:
                tile = self.kanji_tiles[card.sign]
                th, tw = tile.shape[:2]
                blit_rgba(canvas, tile, cx + (cw - tw) // 2, cy + (ch - th) // 2, ease)
            else:
                centre_text(canvas, ROMAJI[card.sign], cx + cw // 2, cy + ch // 2 + 8,
                            0.7, TEXT, 2)

            ly = y + CARD_H + 8
            rounded_rect(canvas, (x + i * step, ly),
                         (x + i * step + CARD_W, ly + LABEL_H), CARD, 8, -1)
            centre_text(canvas, card.sign.upper(), x + i * step + CARD_W // 2,
                        ly + 23, 0.52, TEXT if newest else MUTED, 2 if newest else 1)

            if i:
                centre_text(canvas, ">", x + i * step - CARD_GAP // 2,
                            y + CARD_H // 2 + 6, 0.5, CARD_EDGE, 2)

    def _draw_jutsu(self, canvas, st: UIState, top: int) -> None:
        """Replaces the card strip when a jutsu completes."""
        # Brief flash on arrival, then settle.
        flash = max(0.0, 1.0 - st.jutsu_age / 0.35)
        if flash:
            overlay = canvas[top:top + STRIP_H].copy()
            overlay[:] = ACCENT
            cv2.addWeighted(overlay, 0.25 * flash, canvas[top:top + STRIP_H],
                            1 - 0.25 * flash, 0, canvas[top:top + STRIP_H])

        cv2.line(canvas, (0, top), (self.width, top), ACCENT, 3)
        if st.jutsu_element:
            centre_text(canvas, st.jutsu_element.upper(), self.width // 2,
                        top + 40, 0.62, ACCENT, 2)
        scale = min(1.6, (self.width - 80) / max(len(st.jutsu_name) * 24, 1))
        centre_text(canvas, st.jutsu_name, self.width // 2, top + 96, scale, TEXT, 3)

    # -- entry point ------------------------------------------------------------------

    def render(self, frame, st: UIState, now: float) -> np.ndarray:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        canvas = np.full((self.height + STRIP_H, self.width, 3), INK, np.uint8)
        canvas[:self.height] = frame

        if st.effect is not None and st.effect_centre is not None:
            view = canvas[:self.height]
            st.effect.draw(view, st.effect_centre, st.effect_radius, st.effect_age)

        self._draw_detection(canvas, st)
        self._draw_held(canvas, st)
        self._draw_timer(canvas, st)
        self._draw_strip(canvas, st, now)
        cv2.putText(canvas, f"{st.fps:.0f} FPS", (self.width - 92, self.height - 14),
                    FONT, 0.5, MUTED, 1, cv2.LINE_AA)
        return canvas
