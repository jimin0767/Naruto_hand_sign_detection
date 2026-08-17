"""Contract tests for the public `handsign` package.

This is what teammates import, so these assertions are the integration contract. The class
*order* in particular is load-bearing: the model emits integer indices, and anything
decoding them with a different order gets plausible-looking wrong answers rather than an
error. That is the exact failure the whole pipeline was built to prevent.
"""

from __future__ import annotations

import importlib

import pytest

import handsign
from handsign.classes import CANONICAL, CANONICAL_INDEX, ROMAJI


class TestClassContract:
    def test_exactly_twelve_classes(self):
        assert len(CANONICAL) == 12

    def test_frozen_order(self):
        """Pinned literally. If this test fails, every exported model is invalidated."""
        assert list(CANONICAL) == [
            "bird", "boar", "dog", "dragon", "hare", "horse",
            "monkey", "ox", "ram", "rat", "snake", "tiger",
        ]

    def test_canonical_is_immutable(self):
        with pytest.raises((TypeError, AttributeError)):
            CANONICAL[0] = "nope"      # type: ignore[index]

    def test_index_lookup_is_the_inverse(self):
        for i, name in enumerate(CANONICAL):
            assert CANONICAL_INDEX[name] == i
            assert CANONICAL[CANONICAL_INDEX[name]] == name

    def test_romaji_covers_every_class(self):
        assert set(ROMAJI) == set(CANONICAL)

    def test_no_duplicates(self):
        assert len(set(CANONICAL)) == len(CANONICAL)


class TestPublicApi:
    @pytest.mark.parametrize("name", [
        "CANONICAL", "CANONICAL_INDEX", "ROMAJI",
        "HandSignDetector", "Detection", "DEFAULT_CONF", "DEFAULT_ACCEPT_CONF",
        "SignSmoother", "SequenceTracker", "Jutsu", "load_jutsu", "HandSignError",
    ])
    def test_exported(self, name):
        assert hasattr(handsign, name), f"{name} missing from handsign"
        assert name in handsign.__all__

    def test_smoothing_has_no_heavy_dependencies(self):
        """Importable without torch/ultralytics, so a game can drive it from any backend."""
        module = importlib.import_module("handsign.smoothing")
        source = module.__file__
        assert source and "smoothing" in source
        text = open(source, encoding="utf-8").read()
        for heavy in ("import torch", "import ultralytics", "from ultralytics"):
            assert heavy not in text, f"{heavy} leaked into handsign.smoothing"

    def test_thresholds_match_measurements(self):
        assert handsign.DEFAULT_CONF == 0.25
        assert handsign.DEFAULT_ACCEPT_CONF == 0.60


class TestDetectionType:
    def test_fields(self):
        d = handsign.Detection(11, "tiger", 0.93, (1.0, 2.0, 3.0, 4.0))
        assert (d.class_id, d.name, d.confidence, d.box) == (11, "tiger", 0.93, (1.0, 2.0, 3.0, 4.0))

    def test_romaji_property(self):
        assert handsign.Detection(11, "tiger", 0.9, (0, 0, 1, 1)).romaji == "Tora"

    def test_is_frozen(self):
        d = handsign.Detection(0, "bird", 0.5, (0, 0, 1, 1))
        with pytest.raises(Exception):
            d.name = "tiger"      # type: ignore[misc]


class TestSmootherDefaults:
    def test_defaults_are_internally_consistent(self):
        s = handsign.SignSmoother()
        assert s.min_votes <= s.window.maxlen
        assert s.clear_votes <= s.window.maxlen

    @pytest.mark.parametrize("window", [3, 5, 9, 15, 25, 60])
    def test_derived_clear_votes_never_exceeds_window(self, window):
        """A constant default would crash as soon as a caller shrinks the window."""
        s = handsign.SignSmoother(window=window, min_votes=max(1, window // 2))
        assert s.clear_votes <= window

    def test_default_window_is_sized_for_high_frame_rates(self):
        """9 frames is 0.115s at the ~78 FPS this runs at -- too short to debounce."""
        assert handsign.SignSmoother().window.maxlen >= 20


class TestJutsuCsv:
    """The shipped jutsu table, and the CSV loader that reads it."""

    @staticmethod
    def table():
        from pathlib import Path
        return handsign.load_jutsu(Path(__file__).resolve().parents[1] / "jutsu.csv")

    def test_loads(self):
        assert len(self.table()) >= 15

    def test_every_sign_is_detectable(self):
        for j in self.table():
            assert set(j.signs) <= set(CANONICAL), f"{j.name} uses an undetectable sign"

    def test_no_jutsu_is_unreachable(self):
        """A shorter jutsu completing mid-sequence would clear the buffer and block it."""
        assert handsign.find_unreachable(self.table()) == []

    def test_no_duplicate_sequences(self):
        seqs = [tuple(j.signs) for j in self.table()]
        assert len(seqs) == len(set(seqs))

    def test_sequences_are_long_enough_to_be_deliberate(self):
        """One- and two-sign jutsu fire by accident during other sequences."""
        for j in self.table():
            assert len(j.signs) >= 3, f"{j.name} is only {len(j.signs)} signs"

    def test_sequences_are_short_enough_to_demo(self):
        for j in self.table():
            assert len(j.signs) <= 7, f"{j.name} is {len(j.signs)} signs"

    def test_names_are_unique(self):
        names = [j.name for j in self.table()]
        assert len(names) == len(set(names))


class TestFindUnreachable:
    def test_detects_a_blocking_prefix(self):
        """[a,b] completes at index 2 of [x,a,b,c], clearing the buffer before it ends."""
        blocked = handsign.find_unreachable([
            handsign.Jutsu("long", "", ["x", "a", "b", "c"]),
            handsign.Jutsu("short", "", ["a", "b"]),
        ])
        assert blocked == [("long", "short", 3)]

    def test_suffix_overlap_is_not_blocking(self):
        """Tried longest-first, so a shorter tail does not shadow the longer jutsu."""
        assert handsign.find_unreachable([
            handsign.Jutsu("long", "", ["x", "a", "b"]),
            handsign.Jutsu("short", "", ["a", "b"]),
        ]) == []

    def test_disjoint_jutsu_are_fine(self):
        assert handsign.find_unreachable([
            handsign.Jutsu("a", "", ["rat", "ox"]),
            handsign.Jutsu("b", "", ["dog", "bird", "ram"]),
        ]) == []

    def test_equal_length_never_blocks(self):
        assert handsign.find_unreachable([
            handsign.Jutsu("a", "", ["rat", "ox"]),
            handsign.Jutsu("b", "", ["rat", "ox"]),
        ]) == []


class TestCsvLoader:
    def test_header_row_is_optional(self, tmp_path):
        a = tmp_path / "a.csv"; a.write_text("Fire Style,X,snake,ram,tiger\n")
        b = tmp_path / "b.csv"; b.write_text("element,jutsu,sign1,sign2,sign3\nFire Style,X,snake,ram,tiger\n")
        assert [j.signs for j in handsign.load_jutsu(a)] == [j.signs for j in handsign.load_jutsu(b)]

    def test_trailing_empty_columns_ignored(self, tmp_path):
        p = tmp_path / "j.csv"; p.write_text("Fire Style,X,snake,ram,tiger,,,,\n")
        assert handsign.load_jutsu(p)[0].signs == ["snake", "ram", "tiger"]

    def test_element_is_kept(self, tmp_path):
        p = tmp_path / "j.csv"; p.write_text("Water Style,X,dragon,tiger,hare\n")
        assert handsign.load_jutsu(p)[0].english == "Water Style"

    def test_case_and_space_normalised(self, tmp_path):
        p = tmp_path / "j.csv"; p.write_text("Fire Style,X, SNAKE , Ram ,tiger\n")
        assert handsign.load_jutsu(p)[0].signs == ["snake", "ram", "tiger"]

    def test_blank_rows_skipped(self, tmp_path):
        p = tmp_path / "j.csv"; p.write_text("Fire Style,X,snake,ram,tiger\n\n,,,,\n")
        assert len(handsign.load_jutsu(p)) == 1

    def test_undetectable_sign_rejected(self, tmp_path):
        p = tmp_path / "j.csv"; p.write_text("Fire Style,X,snake,gassho,tiger\n")
        with pytest.raises(handsign.HandSignError, match="gassho"):
            handsign.load_jutsu(p)

    def test_row_without_signs_rejected(self, tmp_path):
        p = tmp_path / "j.csv"; p.write_text("Fire Style,X\n")
        with pytest.raises(handsign.HandSignError, match="no signs"):
            handsign.load_jutsu(p)

    def test_utf8_bom_tolerated(self, tmp_path):
        """Excel writes a BOM, and users will edit this file in Excel."""
        p = tmp_path / "j.csv"
        p.write_bytes(b"\xef\xbb\xbfelement,jutsu,sign1,sign2,sign3\nFire Style,X,snake,ram,tiger\n")
        assert handsign.load_jutsu(p)[0].signs == ["snake", "ram", "tiger"]
