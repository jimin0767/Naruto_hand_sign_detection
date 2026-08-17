"""Tests for the jutsu effect layer.

An effect that throws mid-cast ruins the demo, and the geometry reaches into raw pixel
buffers near frame edges, so the bounds cases matter more than the aesthetics.
"""

from __future__ import annotations

import numpy as np
import pytest

from handsign.effects import (
    CHIDORI,
    EFFECTS,
    EffectSpec,
    LightningEffect,
    effect_for,
    jagged_path,
    radial_bolts,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def frame():
    return np.full((360, 640, 3), 30, np.uint8)


class TestJaggedPath:
    def test_starts_and_ends_where_asked(self, rng):
        path = jagged_path(rng, (0.0, 0.0), (100.0, 0.0), 20.0)
        assert path[0] == (0.0, 0.0)
        assert path[-1] == (100.0, 0.0)

    def test_subdivides(self, rng):
        assert len(jagged_path(rng, (0.0, 0.0), (100.0, 0.0), 40.0)) > 4

    def test_tiny_displacement_is_a_straight_segment(self, rng):
        assert jagged_path(rng, (0.0, 0.0), (10.0, 0.0), 0.5) == [(0.0, 0.0), (10.0, 0.0)]

    def test_actually_deviates_from_the_straight_line(self, rng):
        path = jagged_path(rng, (0.0, 0.0), (100.0, 0.0), 30.0)
        assert max(abs(y) for _, y in path) > 1.0

    def test_deterministic_for_a_given_seed(self):
        a = jagged_path(np.random.default_rng(7), (0.0, 0.0), (50.0, 50.0), 10.0)
        b = jagged_path(np.random.default_rng(7), (0.0, 0.0), (50.0, 50.0), 10.0)
        assert a == b

    def test_reseeding_changes_the_shape(self):
        """Bolts must not repeat frame to frame, or it reads as a looping GIF."""
        a = jagged_path(np.random.default_rng(1), (0.0, 0.0), (50.0, 50.0), 10.0)
        b = jagged_path(np.random.default_rng(2), (0.0, 0.0), (50.0, 50.0), 10.0)
        assert a != b


class TestRadialBolts:
    def test_produces_at_least_the_requested_count(self, rng):
        assert len(radial_bolts(rng, (50.0, 50.0), 30.0, 6, branch_chance=0.0)) == 6

    def test_branches_add_extra_bolts(self, rng):
        assert len(radial_bolts(rng, (50.0, 50.0), 30.0, 6, branch_chance=1.0)) > 6

    def test_bolts_originate_at_the_centre(self, rng):
        for bolt in radial_bolts(rng, (50.0, 50.0), 30.0, 5, branch_chance=0.0):
            assert bolt[0] == (50.0, 50.0)

    def test_reach_is_bounded_by_radius(self, rng):
        centre, radius = (0.0, 0.0), 40.0
        pts = [p for b in radial_bolts(rng, centre, radius, 8) for p in b]
        assert max(np.hypot(x, y) for x, y in pts) < radius * 2.6

    def test_every_bolt_has_at_least_two_points(self, rng):
        assert all(len(b) >= 2 for b in radial_bolts(rng, (0.0, 0.0), 25.0, 8))


class TestIntensityEnvelope:
    @pytest.fixture
    def fx(self):
        return LightningEffect(EffectSpec(duration=2.0, charge=0.2, fade=0.5), seed=0)

    def test_silent_before_and_after(self, fx):
        assert fx.intensity(-0.1) == 0.0
        assert fx.intensity(2.5) == 0.0

    def test_ramps_up_during_charge(self, fx):
        assert fx.intensity(0.0) == pytest.approx(0.0)
        assert fx.intensity(0.1) == pytest.approx(0.5)
        assert fx.intensity(0.2) == pytest.approx(1.0)

    def test_holds_at_full(self, fx):
        assert fx.intensity(1.0) == 1.0

    def test_fades_out(self, fx):
        assert fx.intensity(1.75) == pytest.approx(0.5)
        assert fx.intensity(2.0) == pytest.approx(0.0)

    def test_never_leaves_zero_one(self, fx):
        for t in np.linspace(-0.5, 3.0, 200):
            assert 0.0 <= fx.intensity(float(t)) <= 1.0


class TestDraw:
    def test_brightens_the_frame(self, frame):
        fx = LightningEffect(CHIDORI, seed=1)
        before = frame.mean()
        fx.draw(frame, (320.0, 180.0), 60.0, 1.0)
        assert frame.mean() > before

    def test_nothing_drawn_when_expired(self, frame):
        fx = LightningEffect(CHIDORI, seed=1)
        before = frame.copy()
        fx.draw(frame, (320.0, 180.0), 60.0, 99.0)
        assert np.array_equal(frame, before)

    def test_additive_never_wraps_around(self, frame):
        """Additive blending must saturate, not overflow to black."""
        frame[:] = 250
        LightningEffect(CHIDORI, seed=2).draw(frame, (320.0, 180.0), 60.0, 1.0)
        assert frame.min() >= 250

    @pytest.mark.parametrize("centre", [
        (0.0, 0.0), (639.0, 359.0), (-40.0, 180.0), (700.0, 180.0),
        (320.0, -30.0), (320.0, 400.0),
    ])
    def test_offscreen_anchors_never_raise(self, frame, centre):
        """The hand can sit at or past the frame edge mid-cast."""
        LightningEffect(CHIDORI, seed=3).draw(frame, centre, 60.0, 1.0)

    @pytest.mark.parametrize("radius", [0.0, 1.0, 12.0, 400.0])
    def test_extreme_radii_are_safe(self, frame, radius):
        LightningEffect(CHIDORI, seed=4).draw(frame, (320.0, 180.0), radius, 1.0)

    def test_consecutive_frames_differ(self, frame):
        """Regenerating the bolts each frame is what produces the crackle."""
        fx = LightningEffect(CHIDORI, seed=5)
        a = frame.copy(); fx.draw(a, (320.0, 180.0), 60.0, 1.0)
        b = frame.copy(); fx.draw(b, (320.0, 180.0), 60.0, 1.0)
        assert not np.array_equal(a, b)

    def test_effect_is_blue_dominant(self, frame):
        """Chidori reads as electric blue; a BGR mix-up would make it orange."""
        fx = LightningEffect(CHIDORI, seed=6)
        fx.draw(frame, (320.0, 180.0), 70.0, 1.0)
        lit = frame[frame.max(axis=2) > 120]
        assert len(lit) > 0
        assert lit[:, 0].mean() > lit[:, 2].mean()

    def test_returns_the_same_array(self, frame):
        fx = LightningEffect(CHIDORI, seed=7)
        assert fx.draw(frame, (320.0, 180.0), 50.0, 1.0) is frame

    def test_fainter_near_the_end(self, frame):
        fx = LightningEffect(CHIDORI, seed=8)
        mid = frame.copy(); fx.draw(mid, (320.0, 180.0), 60.0, 1.2)
        late = frame.copy(); fx.draw(late, (320.0, 180.0), 60.0, CHIDORI.duration - 0.05)
        assert late.mean() < mid.mean()


class TestRegistry:
    def test_chidori_registered(self):
        assert effect_for("Chidori") is CHIDORI

    def test_unknown_jutsu_has_no_effect(self):
        assert effect_for("Fireball Jutsu") is None

    def test_none_is_safe(self):
        assert effect_for(None) is None

    def test_every_registered_name_exists_in_the_jutsu_table(self):
        """An effect keyed to a misspelled jutsu would silently never fire."""
        from pathlib import Path

        from handsign import load_jutsu
        names = {j.name for j in load_jutsu(Path(__file__).resolve().parents[1] / "jutsu.csv")}
        assert set(EFFECTS) <= names, f"no such jutsu: {set(EFFECTS) - names}"
