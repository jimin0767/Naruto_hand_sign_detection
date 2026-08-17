"""Tests for the jutsu effect layer.

An effect that throws mid-cast ruins the demo, and the geometry reaches into raw pixel
buffers near frame edges, so the bounds cases matter more than the aesthetics.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from handsign.effects import (
    CHIDORI,
    EFFECTS,
    AnchorTracker,
    CloneEffect,
    CloneSpec,
    EffectSpec,
    TransformEffect,
    TransformSpec,
    cutout_from_mask,
    key_background,
    load_sprite,
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


class TestAnchorTracker:
    """The anchor decides whether the effect looks stuck to the hand or floating near it."""

    def _run(self, tracker, boxes, dt=0.033):
        return [tracker.update(b, i * dt) for i, b in enumerate(boxes)]

    def test_first_box_is_adopted_immediately(self):
        t = AnchorTracker()
        (centre, radius) = t.update((100.0, 100.0, 200.0, 200.0), 0.0)
        assert centre == (150.0, 150.0)
        assert radius == pytest.approx(62.0)

    def test_returns_none_before_any_detection(self):
        assert AnchorTracker().update(None, 0.0) is None

    def test_small_jitter_is_damped(self):
        """A few pixels of wobble must not make the effect vibrate."""
        t = AnchorTracker()
        t.update((100.0, 100.0, 200.0, 200.0), 0.0)
        deviations = []
        for i, d in enumerate([3, -3, 3, -3, 2, -2], 1):
            (cx, _), _ = t.update((100.0 + d, 100.0, 200.0 + d, 200.0), i * 0.033)
            deviations.append(abs(cx - 150.0))
        assert max(deviations) < 2.0

    def test_fast_motion_is_followed_closely(self):
        """Adaptive smoothing: a fixed factor heavy enough to kill jitter lags badly here."""
        t = AnchorTracker()
        t.update((100.0, 100.0, 200.0, 200.0), 0.0)
        for i in range(1, 8):
            x = 100.0 + 40 * i
            (cx, _), _ = t.update((x, 100.0, x + 100.0, 200.0), i * 0.033)
        assert abs((150.0 + 40 * 7) - cx) < 30.0

    def test_adaptive_beats_fixed_smoothing_on_fast_motion(self):
        """Pins the reason the adaptive term exists."""
        lags = {}
        for name, snap in (("adaptive", 160.0), ("effectively_fixed", 1e9)):
            t = AnchorTracker(snap_distance=snap)
            t.update((100.0, 100.0, 200.0, 200.0), 0.0)
            for i in range(1, 8):
                x = 100.0 + 40 * i
                (cx, _), _ = t.update((x, 100.0, x + 100.0, 200.0), i * 0.033)
            lags[name] = abs((150.0 + 40 * 7) - cx)
        assert lags["adaptive"] < lags["effectively_fixed"] * 0.6

    def test_large_jump_snaps_instead_of_sliding(self):
        """A jump this big is the model relocating, not a hand moving."""
        t = AnchorTracker(snap_distance=100.0)
        t.update((0.0, 0.0, 100.0, 100.0), 0.0)
        (centre, _) = t.update((500.0, 0.0, 600.0, 100.0), 0.033)
        assert centre == (550.0, 50.0)

    def test_dropout_coasts_then_holds(self):
        t = AnchorTracker(coast_s=0.1)
        t.update((0.0, 0.0, 100.0, 100.0), 0.0)
        for i in range(1, 5):
            t.update((40.0 * i, 0.0, 100.0 + 40.0 * i, 100.0), i * 0.033)
        moving = t.centre[0]
        (coasted, _), _ = t.update(None, 0.20)
        assert coasted > moving                      # kept gliding
        held = t.centre[0]
        for k in range(6):
            t.update(None, 0.5 + k * 0.033)
        assert t.centre[0] == pytest.approx(held, abs=1e-6)   # then parked

    def test_position_survives_a_long_dropout(self):
        """Hands leaving frame must not send the effect to the origin."""
        t = AnchorTracker()
        t.update((300.0, 200.0, 400.0, 300.0), 0.0)
        for i in range(1, 40):
            result = t.update(None, i * 0.033)
        assert result is not None
        assert result[0][0] == pytest.approx(350.0, abs=30.0)

    def test_radius_is_smoothed_too(self):
        t = AnchorTracker()
        t.update((0.0, 0.0, 100.0, 100.0), 0.0)
        before = t.radius
        (_, radius) = t.update((0.0, 0.0, 200.0, 200.0), 0.033)
        assert before < radius < 200.0 * 0.62        # moved toward, not jumped to

    def test_reset_clears_state(self):
        t = AnchorTracker()
        t.update((0.0, 0.0, 100.0, 100.0), 0.0)
        t.reset()
        assert t.centre is None and t.update(None, 1.0) is None

    def test_zero_dt_does_not_divide_by_zero(self):
        t = AnchorTracker()
        t.update((0.0, 0.0, 100.0, 100.0), 5.0)
        t.update((10.0, 0.0, 110.0, 100.0), 5.0)     # identical timestamp

    def test_converges_on_a_stationary_hand(self):
        t = AnchorTracker()
        for i in range(40):
            (centre, _) = t.update((100.0, 100.0, 200.0, 200.0), i * 0.033)
        assert centre == pytest.approx((150.0, 150.0), abs=0.5)


class TestDurationOverride:
    def test_spec_duration_is_replaceable(self):
        from dataclasses import replace
        longer = replace(CHIDORI, duration=9.0)
        assert LightningEffect(longer).intensity(7.0) == 1.0
        assert LightningEffect(CHIDORI).intensity(7.0) == 0.0

    def test_default_is_the_longer_duration(self):
        assert CHIDORI.duration >= 5.0


class TestKeyBackground:
    def _checkerboard(self, w=80, h=120):
        """Mimics a 'transparent' PNG flattened to RGB: two near-white values."""
        bg = np.full((h, w, 3), 255, np.uint8)
        bg[::8, ::8] = 254
        return bg

    def test_flat_background_is_fully_keyed(self):
        alpha = key_background(self._checkerboard())
        assert (alpha == 0).mean() > 0.95

    def test_subject_is_kept(self):
        img = self._checkerboard()
        cv2.rectangle(img, (20, 30), (60, 90), (40, 20, 160), -1)
        alpha = key_background(img)
        assert alpha[60, 40] == 255
        assert alpha[2, 2] == 0

    def test_pale_subject_interior_survives(self):
        """A white shape inside the subject must not be keyed out with the background."""
        img = self._checkerboard()
        cv2.rectangle(img, (20, 30), (60, 90), (40, 20, 160), -1)
        cv2.rectangle(img, (32, 45), (48, 70), (255, 255, 255), -1)   # enclosed white
        alpha = key_background(img)
        assert alpha[55, 40] == 255, "enclosed white was keyed out"

    def test_returns_uint8_mask_of_matching_shape(self):
        img = self._checkerboard(40, 50)
        alpha = key_background(img)
        assert alpha.shape == (50, 40) and alpha.dtype == np.uint8


class TestLoadSprite:
    def _write(self, tmp_path, name="s.png"):
        # Realistic sprite dimensions: the de-fringe erode+blur is a fixed pixel width,
        # so on a 30px-wide test shape it eats a visible fraction of the interior while
        # on a real 300px sprite it is negligible.
        # A circle, not a rectangle: cropping a solid rectangle to its opaque bounds
        # leaves no transparent pixels at all, which makes "background was removed"
        # untestable.
        img = np.full((600, 400, 3), 255, np.uint8)
        cv2.circle(img, (200, 300), 150, (40, 20, 160), -1)
        p = tmp_path / name
        cv2.imwrite(str(p), img)
        return p

    def test_loads_rgb_and_keys_it(self, tmp_path):
        sprite = load_sprite(self._write(tmp_path))
        assert sprite.shape[2] == 4
        assert (sprite[:, :, 3] == 0).any(), "background not keyed"
        assert (sprite[:, :, 3] > 200).any(), "subject not kept"

    def test_cropped_to_opaque_bounds(self, tmp_path):
        """Placement aligns to the subject, not to whatever padding the file carried."""
        sprite = load_sprite(self._write(tmp_path))
        assert sprite.shape[0] < 600 and sprite.shape[1] < 400

    def test_existing_alpha_is_respected(self, tmp_path):
        rgba = np.zeros((600, 400, 4), np.uint8)
        rgba[100:500, 50:350] = (10, 200, 10, 255)
        p = tmp_path / "a.png"
        cv2.imwrite(str(p), rgba)
        sprite = load_sprite(p)
        assert sprite.shape[2] == 4 and (sprite[:, :, 3] > 200).mean() > 0.9

    def test_defringe_keeps_the_interior_fully_opaque(self, tmp_path):
        """Edge softening must not translate into a semi-transparent subject."""
        sprite = load_sprite(self._write(tmp_path))
        assert sprite[:, :, 3].max() == 255
        opaque = (sprite[:, :, 3] > 200).mean()
        assert opaque > 0.7, f"interior went translucent ({opaque:.2f})"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sprite(tmp_path / "nope.png")

    def test_fully_flat_image_raises(self, tmp_path):
        p = tmp_path / "blank.png"
        cv2.imwrite(str(p), np.full((40, 40, 3), 255, np.uint8))
        with pytest.raises(ValueError, match="fully transparent"):
            load_sprite(p)

    def test_the_shipped_sprite_loads_if_present(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "assets" / "sakura.png"
        if not path.exists():
            pytest.skip("assets/sakura.png is gitignored; supply your own")
        sprite = load_sprite(path)
        assert sprite.shape[2] == 4
        assert 0.2 < (sprite[:, :, 3] > 16).mean() < 0.95


class TestTransformEffect:
    @pytest.fixture
    def sprite(self):
        rgba = np.zeros((200, 100, 4), np.uint8)
        rgba[:, :, :3] = (30, 80, 220)
        rgba[:, :, 3] = 255
        return rgba

    @pytest.fixture
    def fx(self, sprite):
        return TransformEffect(TransformSpec(duration=4.0, smoke_s=0.5), sprite, seed=0)

    def test_sprite_hidden_behind_the_initial_puff(self, fx):
        assert fx.sprite_alpha(0.0) == 0.0
        assert fx.sprite_alpha(0.1) == 0.0

    def test_sprite_appears_after_the_puff_peaks(self, fx):
        assert fx.sprite_alpha(0.5) == pytest.approx(1.0)

    def test_sprite_fades_at_the_end(self, fx):
        assert 0.0 < fx.sprite_alpha(3.8) < 1.0
        assert fx.sprite_alpha(4.0) == pytest.approx(0.0)

    def test_smoke_rises_and_clears(self, fx):
        assert fx.smoke_alpha(0.0) == pytest.approx(0.0)
        assert fx.smoke_alpha(0.25) > 0.8
        assert fx.smoke_alpha(0.6) == 0.0

    def test_draw_changes_the_frame(self, fx, frame):
        before = frame.copy()
        fx.draw(frame, (320.0, 180.0), 150.0, 1.0)
        assert not np.array_equal(frame, before)

    def test_nothing_drawn_when_expired(self, fx, frame):
        before = frame.copy()
        fx.draw(frame, (320.0, 180.0), 150.0, 99.0)
        assert np.array_equal(frame, before)

    def test_sprite_is_top_aligned(self, fx, frame):
        """A torso-up person box ends at the chest; centring sinks her head into it."""
        fx.draw(frame, (320.0, 200.0), 150.0, 1.0)
        top_band = frame[55:70, 300:340]
        assert top_band.std() > 0, "nothing drawn near the top of the person box"

    def test_scale_controls_sprite_height(self, sprite, frame):
        small = TransformEffect(TransformSpec(scale=0.5, smoke_s=0.01), sprite, seed=1)
        big = TransformEffect(TransformSpec(scale=2.0, smoke_s=0.01), sprite, seed=1)
        a, b = frame.copy(), frame.copy()
        small.draw(a, (320.0, 180.0), 60.0, 1.0)
        big.draw(b, (320.0, 180.0), 60.0, 1.0)
        assert (b != 30).sum() > (a != 30).sum()

    @pytest.mark.parametrize("centre", [(0.0, 0.0), (639.0, 359.0), (-100.0, 180.0)])
    def test_offscreen_anchor_is_safe(self, fx, frame, centre):
        fx.draw(frame, centre, 150.0, 1.0)

    def test_tiny_radius_is_safe(self, fx, frame):
        fx.draw(frame, (320.0, 180.0), 0.0, 1.0)


class TestEffectRegistryTypes:
    def test_transformation_is_registered_as_a_sprite_effect(self):
        assert isinstance(effect_for("Transformation Jutsu"), TransformSpec)

    def test_chidori_is_still_procedural(self):
        assert isinstance(effect_for("Chidori"), EffectSpec)

    def test_anchor_tracker_exposes_box_size(self):
        """TransformEffect needs a real height; deriving one from `radius` is a units bug."""
        t = AnchorTracker()
        t.update((100.0, 50.0, 200.0, 350.0), 0.0)
        assert t.size == pytest.approx((100.0, 300.0))


class TestCutoutFromMask:
    def _scene(self, h=200, w=300):
        return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)

    def test_returns_rgba_and_origin(self):
        frame = self._scene()
        mask = np.zeros((200, 300), np.float32)
        mask[50:150, 80:200] = 1.0
        rgba, origin = cutout_from_mask(frame, mask)
        assert rgba.shape == (100, 120, 4)
        assert origin == (80, 50)

    def test_background_is_transparent(self):
        frame = self._scene()
        mask = np.zeros((200, 300), np.float32)
        mask[50:150, 80:200] = 1.0
        rgba, _ = cutout_from_mask(frame, mask)
        assert rgba[50, 60, 3] == 255

    def test_empty_mask_returns_none(self):
        assert cutout_from_mask(self._scene(), np.zeros((200, 300), np.float32)) is None

    def test_tiny_mask_returns_none(self):
        mask = np.zeros((200, 300), np.float32)
        mask[0:4, 0:4] = 1.0
        assert cutout_from_mask(self._scene(), mask) is None

    def test_keeps_only_the_largest_blob(self):
        """The segmenter picks up furniture as a second 'person'; a floating scrap of
        laundry cloned six times destroys the illusion."""
        frame = self._scene()
        mask = np.zeros((200, 300), np.float32)
        mask[50:150, 80:200] = 1.0        # the person
        mask[10:20, 10:25] = 1.0          # a stray blob
        rgba, origin = cutout_from_mask(frame, mask)
        assert origin == (80, 50), "cutout expanded to include the stray blob"

    def test_bottom_cut_is_dissolved_when_it_touches_the_frame_edge(self):
        """A shrunk, raised clone drags that straight edge into open picture."""
        frame = self._scene()
        mask = np.zeros((200, 300), np.float32)
        mask[50:200, 80:200] = 1.0        # runs off the bottom
        rgba, _ = cutout_from_mask(frame, mask)
        alpha = rgba[:, :, 3]
        assert alpha[-1, 60] < 40, "hard bottom edge survived"
        assert alpha[len(alpha) // 2, 60] > 200, "fade ate the body"

    def test_no_fade_when_the_subject_ends_inside_the_frame(self):
        frame = self._scene()
        mask = np.zeros((200, 300), np.float32)
        mask[50:150, 80:200] = 1.0        # clear of the bottom
        rgba, _ = cutout_from_mask(frame, mask)
        assert rgba[-1, 60, 3] > 200


class TestCloneEffect:
    @pytest.fixture
    def cutout(self):
        rgba = np.zeros((240, 120, 4), np.uint8)
        rgba[:, :, :3] = (60, 90, 200)
        rgba[:, :, 3] = 255
        return rgba

    @pytest.fixture
    def fx(self, cutout):
        e = CloneEffect(CloneSpec(duration=5.0, count=6, stagger=0.1), seed=0)
        e.set_cutout(cutout, (200, 60))
        return e

    def test_placement_count_matches_spec(self, fx):
        assert len(fx.placements()) == 6

    def test_placements_are_mirrored_pairs(self, fx):
        sides = sorted(side for _, side, _, _ in fx.placements())
        assert sides.count(-1.0) == sides.count(1.0) == 3

    def test_drawn_far_to_near(self, fx):
        depths = [d for d, _, _, _ in fx.placements()]
        assert depths == sorted(depths, reverse=True)

    def test_deeper_clones_are_smaller(self, fx):
        by_depth = {d: s for d, _, s, _ in fx.placements()}
        assert by_depth[1] > by_depth[2] > by_depth[3]

    def test_scale_never_collapses(self):
        fx = CloneEffect(CloneSpec(count=20, shrink=0.4))
        assert all(scale >= 0.25 for _, _, scale, _ in fx.placements())

    def test_clones_appear_staggered(self, fx):
        assert fx.alpha_for(0, 0.02) < fx.alpha_for(0, 0.4)
        assert fx.alpha_for(5, 0.05) == 0.0        # last clone has not started

    def test_all_clones_visible_mid_effect(self, fx):
        assert all(fx.alpha_for(i, 2.0) == 1.0 for i in range(6))

    def test_all_fade_together_at_the_end(self, fx):
        assert all(0.0 <= fx.alpha_for(i, 4.9) < 0.5 for i in range(6))

    def test_draw_without_a_cutout_is_a_no_op(self, frame):
        fx = CloneEffect(CloneSpec())
        before = frame.copy()
        fx.draw(frame, (320.0, 180.0), 100.0, 1.0)
        assert np.array_equal(frame, before)

    def test_draw_changes_the_frame(self, fx, frame):
        before = frame.copy()
        fx.draw(frame, (320.0, 180.0), 100.0, 2.0)
        assert not np.array_equal(frame, before)

    def test_caster_is_composited_last(self, fx, frame):
        """Clones drawn near the caster would otherwise cover them, which reads as
        vanishing rather than multiplying."""
        fx.draw(frame, (320.0, 180.0), 100.0, 2.0)
        ox, oy = fx.origin
        assert frame[oy + 100, ox + 60].tolist() == [60, 90, 200]

    def test_nothing_drawn_when_expired(self, fx, frame):
        before = frame.copy()
        fx.draw(frame, (320.0, 180.0), 100.0, 99.0)
        assert np.array_equal(frame, before)

    def test_cutout_near_the_frame_edge_is_safe(self, cutout, frame):
        fx = CloneEffect(CloneSpec(), seed=1)
        fx.set_cutout(cutout, (620, 340))
        fx.draw(frame, (320.0, 180.0), 100.0, 2.0)

    def test_set_cutout_ignores_none(self, fx, cutout):
        fx.set_cutout(None, (0, 0))
        assert fx.cutout is cutout and fx.origin == (200, 60)

    def test_clone_count_is_configurable(self, cutout):
        fx = CloneEffect(CloneSpec(count=4))
        assert len(fx.placements()) == 4


class TestCloneRegistry:
    def test_clone_jutsu_registered(self):
        assert isinstance(effect_for("Clone Jutsu"), CloneSpec)

    def test_all_three_effects_are_distinct_types(self):
        kinds = {type(effect_for(n)).__name__
                 for n in ("Chidori", "Transformation Jutsu", "Clone Jutsu")}
        assert kinds == {"EffectSpec", "TransformSpec", "CloneSpec"}
