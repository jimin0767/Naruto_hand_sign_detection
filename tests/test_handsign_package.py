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
