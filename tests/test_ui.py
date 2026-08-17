"""Tests for the demo's presentation layer.

Rendering correctness is mostly a matter of taste, but a few things are objectively
checkable: the canvas geometry, the alpha blit (which reaches into raw pixel buffers and
will happily read out of bounds), the BGR channel order, and that every state the demo can
reach actually draws instead of throwing mid-presentation.
"""

from __future__ import annotations

import numpy as np
import pytest

from handsign.classes import CANONICAL, KANJI
from handsign.ui import (
    ACCENT,
    STRIP_H,
    Card,
    DemoUI,
    UIState,
    blit_rgba,
    find_cjk_font,
    rounded_rect,
)


@pytest.fixture
def frame():
    return np.full((360, 640, 3), 60, np.uint8)


@pytest.fixture(scope="module")
def ui():
    return DemoUI(640, 360)


class TestBlitRgba:
    def test_opaque_tile_replaces_pixels(self):
        dst = np.zeros((10, 10, 3), np.uint8)
        tile = np.full((4, 4, 4), 255, np.uint8)
        blit_rgba(dst, tile, 2, 2)
        assert dst[3, 3].tolist() == [255, 255, 255]
        assert dst[0, 0].tolist() == [0, 0, 0]

    def test_transparent_tile_changes_nothing(self):
        dst = np.full((8, 8, 3), 50, np.uint8)
        tile = np.zeros((4, 4, 4), np.uint8)          # alpha 0
        blit_rgba(dst, tile, 1, 1)
        assert (dst == 50).all()

    def test_half_alpha_blends(self):
        dst = np.zeros((6, 6, 3), np.uint8)
        tile = np.zeros((2, 2, 4), np.uint8)
        tile[:, :, :3] = 200
        tile[:, :, 3] = 128
        blit_rgba(dst, tile, 0, 0)
        assert 95 <= int(dst[0, 0, 0]) <= 105

    def test_alpha_argument_scales_further(self):
        dst = np.zeros((6, 6, 3), np.uint8)
        tile = np.full((2, 2, 4), 255, np.uint8)
        blit_rgba(dst, tile, 0, 0, alpha=0.5)
        assert 120 <= int(dst[0, 0, 0]) <= 135

    @pytest.mark.parametrize("x,y", [(-3, -3), (8, 8), (-50, 2), (2, 50), (9, 9)])
    def test_out_of_bounds_never_raises(self, x, y):
        """Cards animate in from partly off-canvas; a crash here kills the demo."""
        dst = np.zeros((10, 10, 3), np.uint8)
        blit_rgba(dst, np.full((4, 4, 4), 255, np.uint8), x, y)

    def test_partial_overlap_clips_correctly(self):
        dst = np.zeros((10, 10, 3), np.uint8)
        blit_rgba(dst, np.full((4, 4, 4), 255, np.uint8), -2, -2)
        assert dst[1, 1].tolist() == [255, 255, 255]
        assert dst[2, 2].tolist() == [0, 0, 0]


class TestRoundedRect:
    def test_filled_covers_the_centre(self, frame):
        rounded_rect(frame, (10, 10), (100, 80), (7, 8, 9), 10, -1)
        assert frame[45, 55].tolist() == [7, 8, 9]

    def test_outline_leaves_the_centre_alone(self, frame):
        before = frame[45, 55].copy()
        rounded_rect(frame, (10, 10), (100, 80), (7, 8, 9), 10, 2)
        assert frame[45, 55].tolist() == before.tolist()

    def test_radius_clamped_on_a_tiny_rect(self, frame):
        rounded_rect(frame, (5, 5), (9, 9), (1, 2, 3), 50, -1)   # must not raise

    def test_zero_size_rect_is_safe(self, frame):
        rounded_rect(frame, (20, 20), (20, 20), (1, 2, 3), 6, -1)


class TestGlyphs:
    def test_a_cjk_font_is_available(self):
        assert find_cjk_font() is not None, "no CJK font; kanji would fall back to romaji"

    def test_all_twelve_glyphs_prerendered(self, ui):
        assert set(ui.kanji_tiles) == set(CANONICAL)
        assert set(ui.big_kanji) == set(CANONICAL)

    def test_tiles_are_rgba_and_non_empty(self, ui):
        for name, tile in ui.kanji_tiles.items():
            assert tile.ndim == 3 and tile.shape[2] == 4, name
            assert tile[:, :, 3].max() > 0, f"{name} rendered blank"

    def test_accent_glyph_keeps_bgr_channel_order(self, ui):
        """A BGR/RGB double-swap here renders the orange accent as blue."""
        tile = ui.big_kanji["tiger"]
        lit = tile[tile[:, :, 3] > 200][:, :3]
        assert len(lit) > 0
        mean = lit.mean(axis=0)
        assert mean[2] > mean[0], f"expected red-dominant BGR, got {mean}"
        assert abs(mean[0] - ACCENT[0]) < 40 and abs(mean[2] - ACCENT[2]) < 40

    def test_kanji_are_distinct(self, ui):
        shapes = {name: t[:, :, 3].sum() for name, t in ui.kanji_tiles.items()}
        assert len(set(shapes.values())) > 8, "glyphs look identical -- font may lack CJK"

    def test_every_class_has_a_kanji(self):
        assert set(KANJI) == set(CANONICAL)


class TestRender:
    def test_canvas_is_frame_plus_strip(self, ui, frame):
        out = ui.render(frame, UIState(), 0.0)
        assert out.shape == (360 + STRIP_H, 640, 3)

    def test_mismatched_frame_is_resized(self, ui):
        out = ui.render(np.zeros((200, 300, 3), np.uint8), UIState(), 0.0)
        assert out.shape == (360 + STRIP_H, 640, 3)

    def test_empty_state_draws_a_prompt(self, ui, frame):
        out = ui.render(frame, UIState(), 0.0)
        assert out[360:].std() > 0, "strip should not be flat when empty"

    def test_cards_render(self, ui, frame):
        st = UIState(cards=[Card("tiger", 0.0), Card("snake", 0.1)])
        out = ui.render(frame, st, 1.0)
        assert out[360:].std() > 0

    @pytest.mark.parametrize("n", [1, 3, 6, 12, 30])
    def test_long_sequences_do_not_overflow(self, ui, frame, n):
        """More cards than fit must clip to the newest, not run off the canvas."""
        st = UIState(cards=[Card(CANONICAL[i % 12], 0.0) for i in range(n)])
        out = ui.render(frame, st, 5.0)
        assert out.shape == (360 + STRIP_H, 640, 3)

    def test_timer_absent_when_clock_not_running(self, ui, frame):
        a = ui.render(frame.copy(), UIState(), 0.0)
        b = ui.render(frame.copy(), UIState(time_left=2.0, timeout_s=3.0), 0.0)
        assert not np.array_equal(a[:150, 480:], b[:150, 480:])

    @pytest.mark.parametrize("left", [3.0, 1.5, 0.2, 0.0])
    def test_timer_renders_at_every_fraction(self, ui, frame, left):
        ui.render(frame.copy(), UIState(time_left=left, timeout_s=3.0), 0.0)

    def test_jutsu_reveal_replaces_the_strip(self, ui, frame):
        cards = [Card("ox", 0.0)]
        a = ui.render(frame.copy(), UIState(cards=cards), 1.0)
        b = ui.render(frame.copy(), UIState(cards=cards, jutsu_name="Chidori",
                                            jutsu_element="Lightning Style"), 1.0)
        assert not np.array_equal(a[360:], b[360:])

    def test_very_long_jutsu_name_fits(self, ui, frame):
        ui.render(frame, UIState(jutsu_name="Summoning: Fanged Pursuit Jutsu",
                                 jutsu_element="Earth Style"), 1.0)

    def test_detection_and_held_draw(self, ui, frame):
        st = UIState(box=(100.0, 80.0, 300.0, 260.0), label="tiger", confidence=0.91,
                     voting=True, held="tiger", stability=0.8, fps=60.0)
        ui.render(frame, st, 1.0)

    def test_rejected_detection_draws(self, ui, frame):
        st = UIState(box=(10.0, 10.0, 60.0, 60.0), label="hare",
                     confidence=0.3, voting=False)
        ui.render(frame, st, 1.0)

    def test_box_touching_the_edges_is_safe(self, ui, frame):
        st = UIState(box=(0.0, 0.0, 640.0, 360.0), label="ox", confidence=0.7)
        ui.render(frame, st, 1.0)

    def test_render_does_not_mutate_the_source_frame(self, ui):
        f = np.full((360, 640, 3), 77, np.uint8)
        ui.render(f, UIState(held="tiger", stability=1.0), 1.0)
        assert (f == 77).all(), "the caller's frame was drawn on"


class TestRomajiFallback:
    def test_renders_without_a_cjk_font(self, frame):
        """A machine with no Japanese font must still show a usable demo."""
        ui = DemoUI(640, 360, font_path=None)
        ui.font_path = None
        ui.kanji_tiles, ui.big_kanji = {}, {}
        out = ui.render(frame, UIState(cards=[Card("tiger", 0.0)], held="tiger"), 1.0)
        assert out.shape == (360 + STRIP_H, 640, 3)
