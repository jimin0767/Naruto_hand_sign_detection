"""Tests for the smoothing and sequence logic behind the demo.

These live in the `handsign` package rather than the demo script so they can run without
a camera -- and so a game can reuse them. The behaviours tested here are the ones that
decide whether the demo feels solid or twitchy on stage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


import handsign as demo   # noqa: E402  the logic now lives in the package


def feed(smoother, predictions):
    """Push a list of frame predictions, returning every sign that got confirmed."""
    return [s for s in (smoother.update(p) for p in predictions) if s]


class TestSignSmoother:
    def test_confirms_after_enough_agreement(self):
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, ["tiger"] * 6) == ["tiger"]

    def test_does_not_confirm_below_threshold(self):
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, ["tiger"] * 5) == []

    def test_held_sign_emits_only_once(self):
        """A sign held for a second is 30 frames but must register as one sign."""
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, ["tiger"] * 40) == ["tiger"]

    def test_single_frame_flicker_is_ignored(self):
        """hare/monkey misreads for one frame; the window must absorb it."""
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, ["hare"] * 6 + ["monkey"] + ["hare"] * 6) == ["hare"]

    def test_genuine_change_is_confirmed(self):
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, ["hare"] * 9 + ["monkey"] * 9) == ["hare", "monkey"]

    def test_same_sign_twice_needs_a_gap(self):
        """Consecutive identical signs only register again after the hands drop."""
        s = demo.SignSmoother(window=9, min_votes=6, clear_votes=6)
        assert feed(s, ["tiger"] * 9 + [None] * 9 + ["tiger"] * 9) == ["tiger", "tiger"]

    def test_no_gap_means_no_second_registration(self):
        s = demo.SignSmoother(window=9, min_votes=6, clear_votes=6)
        assert feed(s, ["tiger"] * 9 + [None] * 2 + ["tiger"] * 9) == ["tiger"]

    def test_empty_frames_alone_confirm_nothing(self):
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, [None] * 30) == []

    def test_held_clears_when_hands_drop(self):
        s = demo.SignSmoother(window=9, min_votes=6, clear_votes=6)
        feed(s, ["tiger"] * 9)
        assert s.held == "tiger"
        feed(s, [None] * 9)
        assert s.held is None

    def test_stability_is_zero_before_any_hold(self):
        assert demo.SignSmoother().stability == 0.0

    def test_stability_rises_with_agreement(self):
        s = demo.SignSmoother(window=10, min_votes=6)
        feed(s, ["ox"] * 10)
        assert s.stability == pytest.approx(1.0)

    def test_reset_clears_everything(self):
        s = demo.SignSmoother(window=9, min_votes=6)
        feed(s, ["ox"] * 9)
        s.reset()
        assert s.held is None and s.stability == 0.0

    def test_unreachable_threshold_is_rejected(self):
        """min_votes > window would silently never confirm anything."""
        with pytest.raises(demo.HandSignError, match="cannot exceed"):
            demo.SignSmoother(window=5, min_votes=9)

    def test_contested_window_confirms_nothing(self):
        s = demo.SignSmoother(window=9, min_votes=6)
        assert feed(s, ["hare", "monkey"] * 6) == []


def make_tracker(**kwargs):
    jutsu = [
        demo.Jutsu("Short", "", ["tiger", "snake"]),
        demo.Jutsu("Long", "", ["ram", "monkey", "tiger", "snake"]),
        demo.Jutsu("Other", "", ["ox", "hare", "monkey"]),
    ]
    return demo.SequenceTracker(jutsu, **kwargs)


class TestSequenceTracker:
    def test_matches_a_complete_sequence(self):
        t = make_tracker()
        assert t.add("tiger", 0.0) is None
        assert t.add("snake", 0.1).name == "Short"

    def test_prefers_the_longest_match(self):
        """A 4-sign jutsu ending in tiger>snake must beat the 2-sign tiger>snake."""
        t = make_tracker()
        for i, sign in enumerate(["ram", "monkey", "tiger"]):
            t.add(sign, i * 0.1)
        assert t.add("snake", 0.4).name == "Long"

    def test_buffer_clears_after_a_match(self):
        t = make_tracker()
        t.add("tiger", 0.0)
        t.add("snake", 0.1)
        assert t.buffer == []

    def test_unrelated_prefix_does_not_match(self):
        t = make_tracker()
        assert t.add("ox", 0.0) is None
        assert t.add("hare", 0.1) is None

    def test_timeout_clears_a_partial_sequence(self):
        """The clock starts on the first idle tick, not when the sign was added."""
        t = make_tracker(timeout_s=4.0)
        t.add("tiger", 0.0)
        t.tick(1.0)              # hands released; countdown begins here
        t.tick(6.0)
        assert t.buffer == []

    def test_timeout_does_not_clear_a_fresh_sequence(self):
        t = make_tracker(timeout_s=4.0)
        t.add("tiger", 0.0)
        t.tick(2.0)
        assert t.buffer == ["tiger"]

    def test_sequence_broken_by_timeout_does_not_match(self):
        t = make_tracker(timeout_s=4.0)
        t.add("tiger", 0.0)
        t.tick(1.0)
        t.tick(9.0)
        assert t.add("snake", 10.0) is None

    def test_holding_a_sign_suspends_the_countdown(self):
        """Holding one sign for a minute must not wipe the sequence."""
        t = make_tracker(timeout_s=3.0)
        t.add("tiger", 0.0)
        for tick in range(1, 60):
            t.tick(float(tick), holding=True)
        assert t.buffer == ["tiger"]

    def test_countdown_restarts_after_a_hold(self):
        t = make_tracker(timeout_s=3.0)
        t.add("tiger", 0.0)
        t.tick(10.0, holding=True)      # still holding: no clock
        t.tick(11.0)                    # released: clock starts now
        t.tick(13.0)
        assert t.buffer == ["tiger"]    # only 2s idle
        t.tick(15.0)
        assert t.buffer == []

    def test_time_left_is_none_while_holding(self):
        t = make_tracker(timeout_s=3.0)
        t.add("tiger", 0.0)
        t.tick(1.0, holding=True)
        assert t.time_left(1.0) is None

    def test_time_left_is_none_with_an_empty_buffer(self):
        assert make_tracker().time_left(5.0) is None

    def test_time_left_counts_down(self):
        t = make_tracker(timeout_s=3.0)
        t.add("tiger", 0.0)
        t.tick(1.0)
        assert t.time_left(1.0) == pytest.approx(3.0)
        assert t.time_left(2.5) == pytest.approx(1.5)

    def test_time_left_floors_at_zero(self):
        t = make_tracker(timeout_s=3.0)
        t.add("tiger", 0.0)
        t.tick(1.0)
        assert t.time_left(99.0) == 0.0

    def test_adding_a_sign_cancels_the_countdown(self):
        t = make_tracker(timeout_s=3.0)
        t.add("tiger", 0.0)
        t.tick(1.0)
        t.add("ox", 2.0)
        assert t.idle_since is None and t.time_left(2.0) is None

    def test_buffer_is_bounded(self):
        t = make_tracker(max_length=4)
        for i in range(12):
            t.add("bird", float(i))
        assert len(t.buffer) == 4

    def test_match_survives_leading_junk(self):
        """Only the tail must match, so a fumbled opening does not block the jutsu."""
        t = make_tracker()
        for i, sign in enumerate(["bird", "bird", "dog", "tiger"]):
            t.add(sign, i * 0.1)
        assert t.add("snake", 0.5).name == "Short"

    def test_banner_shows_then_expires(self):
        t = make_tracker()
        t.add("tiger", 0.0)
        t.add("snake", 1.0)
        assert t.banner(1.5, hold_s=3.0).name == "Short"
        assert t.banner(9.0, hold_s=3.0) is None

    def test_banner_is_none_before_any_match(self):
        assert make_tracker().banner(1.0) is None

    def test_progress_reports_partial_depth(self):
        t = make_tracker()
        t.add("ox", 0.0)
        t.add("hare", 0.1)
        best, depth = t.progress()[0]
        assert best.name == "Other" and depth == 2

    def test_progress_is_empty_without_a_prefix_match(self):
        t = make_tracker()
        t.add("dragon", 0.0)
        assert t.progress() == []


class TestLoadJutsu:
    def test_loads_the_shipped_file(self):
        entries = demo.load_jutsu(Path(__file__).resolve().parents[1] / "jutsu.csv")
        assert entries
        for j in entries:
            assert set(j.signs) <= set(demo.CANONICAL)

    def test_rejects_a_sign_the_model_cannot_detect(self, tmp_path):
        """Gassho is real but untrained here; a jutsu using it could never fire."""
        p = tmp_path / "j.yaml"
        p.write_text("jutsu:\n  - name: X\n    signs: [tiger, gassho]\n")
        with pytest.raises(demo.HandSignError, match="gassho"):
            demo.load_jutsu(p)

    def test_rejects_an_empty_sign_list(self, tmp_path):
        p = tmp_path / "j.yaml"
        p.write_text("jutsu:\n  - name: X\n    signs: []\n")
        with pytest.raises(demo.HandSignError, match="no signs"):
            demo.load_jutsu(p)

    def test_rejects_a_file_with_no_jutsu(self, tmp_path):
        p = tmp_path / "j.yaml"
        p.write_text("jutsu: []\n")
        with pytest.raises(demo.HandSignError, match="no jutsu"):
            demo.load_jutsu(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(demo.HandSignError, match="not found"):
            demo.load_jutsu(tmp_path / "nope.yaml")

    def test_sign_names_are_normalised(self, tmp_path):
        p = tmp_path / "j.yaml"
        p.write_text("jutsu:\n  - name: X\n    signs: ['  TIGER ', Snake]\n")
        assert demo.load_jutsu(p)[0].signs == ["tiger", "snake"]


class TestIntegration:
    def test_noisy_frames_produce_a_clean_jutsu_match(self):
        """End to end: realistic flicker in, exactly one jutsu out."""
        smoother = demo.SignSmoother(window=9, min_votes=6, clear_votes=6)
        tracker = make_tracker()
        frames = (
            ["tiger"] * 7 + ["snake"]          # one-frame misread mid-hold
            + ["tiger"] * 3 + [None] * 9       # hands drop
            + ["snake"] * 7 + ["tiger"]        # another misread
            + ["snake"] * 3
        )
        matched = []
        for i, prediction in enumerate(frames):
            confirmed = smoother.update(prediction)
            if confirmed:
                hit = tracker.add(confirmed, i * 0.03)
                if hit:
                    matched.append(hit.name)
        assert matched == ["Short"]

    def test_conf_default_matches_the_measured_optimum(self):
        assert demo.DEFAULT_CONF == 0.25

    def test_canonical_matches_training(self):
        """Compared as sequences: the package exports a tuple, scripts wrap it in a list
        because yaml.safe_dump emits a Python-specific tag for tuples."""
        train = _load("train_for_demo_compare", "03_train.py")
        assert list(demo.CANONICAL) == list(train.CANONICAL)
