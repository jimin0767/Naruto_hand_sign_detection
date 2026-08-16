"""Tests for 02_split.py.

The load-bearing property is that a subject never straddles two splits. Group assignment
is what enforces it, and `verify_no_leakage` is the independent check on that -- so both
are tested, including the case where the checker must actually fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "split_dataset", Path(__file__).resolve().parents[1] / "02_split.py"
)
split = importlib.util.module_from_spec(_spec)
sys.modules["split_dataset"] = split
_spec.loader.exec_module(split)


def hashes_from_patterns(patterns: list[list[int]]) -> np.ndarray:
    """Build a hash matrix from explicit bit rows, padded to 64 bits."""
    return np.array([(p + [0] * 64)[:64] for p in patterns], dtype=np.uint8)


# ------------------------------------------------------------------- component discovery

class TestConnectedComponents:
    def test_unlinked_sources_stay_separate(self):
        groups = split.connected_components(["a", "b", "c"], {})
        assert sorted(groups) == [["a"], ["b"], ["c"]]

    def test_linked_pair_merges(self):
        groups = split.connected_components(["a", "b", "c"], {("a", "b"): {0, 1}})
        assert sorted(groups) == [["a", "b"], ["c"]]

    def test_link_is_transitive(self):
        """cs<->wilsons and otani<->wilsons must pull all three into one group."""
        links = {("cs", "wilsons"): {0}, ("otani", "wilsons"): {1}}
        groups = split.connected_components(["cs", "otani", "wilsons", "vgu"], links)
        assert sorted(groups) == [["cs", "otani", "wilsons"], ["vgu"]]

    def test_groups_are_sorted_largest_first(self):
        links = {("a", "b"): {0}, ("b", "c"): {1}}
        groups = split.connected_components(["a", "b", "c", "d"], links)
        assert groups[0] == ["a", "b", "c"]

    def test_every_source_appears_exactly_once(self):
        names = ["a", "b", "c", "d", "e"]
        groups = split.connected_components(names, {("a", "e"): {0}, ("b", "c"): {1}})
        flat = [s for g in groups for s in g]
        assert sorted(flat) == names


class TestFindNearDuplicateLinks:
    def test_identical_frames_in_different_sources_link(self):
        hashes = hashes_from_patterns([[1, 0, 1, 0], [1, 0, 1, 0]])
        links = split.find_near_duplicate_links(hashes, np.array(["a", "b"]), 5)
        assert list(links) == [("a", "b")]

    def test_distant_frames_do_not_link(self):
        hashes = hashes_from_patterns([[1] * 40, [0] * 64])
        links = split.find_near_duplicate_links(hashes, np.array(["a", "b"]), 5)
        assert links == {}

    def test_same_source_duplicates_are_not_links(self):
        """Within-source duplication is expected and irrelevant to grouping."""
        hashes = hashes_from_patterns([[1, 0, 1], [1, 0, 1]])
        links = split.find_near_duplicate_links(hashes, np.array(["a", "a"]), 5)
        assert links == {}

    def test_threshold_is_respected(self):
        hashes = hashes_from_patterns([[0] * 64, [1] * 6 + [0] * 58])
        sources = np.array(["a", "b"])
        assert split.find_near_duplicate_links(hashes, sources, 5) == {}
        assert ("a", "b") in split.find_near_duplicate_links(hashes, sources, 6)

    def test_key_order_is_canonical(self):
        hashes = hashes_from_patterns([[1, 1], [1, 1]])
        links = split.find_near_duplicate_links(hashes, np.array(["zebra", "alpha"]), 5)
        assert list(links) == [("alpha", "zebra")]

    def test_spans_chunk_boundary(self):
        """Matches must be found even when the pair straddles the chunking window."""
        patterns = [[0] * 64 for _ in range(10)]
        hashes = hashes_from_patterns(patterns)
        sources = np.array(["a"] * 5 + ["b"] * 5)
        links = split.find_near_duplicate_links(hashes, sources, 0, chunk=3)
        assert ("a", "b") in links


# ------------------------------------------------------------------------- assignment

class TestAssignSplits:
    GROUPS = [["cs", "otani", "wilsons"], ["vgu"], ["yylunxie"], ["chayawat"], ["minsub"]]

    def test_named_sources_are_held_out(self):
        got = split.assign_splits(self.GROUPS, ["chayawat"], ["yylunxie"])
        assert got["chayawat"] == "val" and got["yylunxie"] == "test"

    def test_unnamed_sources_default_to_train(self):
        got = split.assign_splits(self.GROUPS, ["chayawat"], ["yylunxie"])
        assert got["vgu"] == "train" and got["minsub"] == "train"

    def test_naming_one_member_holds_out_the_whole_group(self):
        """cs cannot be held out alone -- wilsons carries 69% of the same content."""
        got = split.assign_splits(self.GROUPS, ["cs"], ["yylunxie"])
        assert got["cs"] == got["otani"] == got["wilsons"] == "val"

    def test_every_source_receives_a_split(self):
        got = split.assign_splits(self.GROUPS, ["chayawat"], ["yylunxie"])
        assert set(got) == {s for g in self.GROUPS for s in g}

    def test_unknown_source_raises(self):
        with pytest.raises(split.SplitError, match="unknown source"):
            split.assign_splits(self.GROUPS, ["nonexistent"], ["yylunxie"])

    def test_same_group_in_val_and_test_raises(self):
        """Holding out cs as val and wilsons as test measures nothing -- same person."""
        with pytest.raises(split.SplitError, match="same person"):
            split.assign_splits(self.GROUPS, ["cs"], ["wilsons"])

    def test_multiple_groups_per_split(self):
        got = split.assign_splits(self.GROUPS, ["chayawat", "minsub"], ["yylunxie"])
        assert got["chayawat"] == got["minsub"] == "val"


# ----------------------------------------------------------------------- leak detection

class TestVerifyNoLeakage:
    def test_clean_split_reports_no_offenders(self):
        hashes = hashes_from_patterns([[1] * 30, [0] * 64])
        assert split.verify_no_leakage(hashes, np.array(["train", "val"]), 5) == []

    def test_detects_a_leaking_split(self):
        """The guard must actually fire -- group assignment is not self-verifying."""
        hashes = hashes_from_patterns([[1, 0, 1, 1], [1, 0, 1, 1]])
        offenders = split.verify_no_leakage(hashes, np.array(["train", "val"]), 5)
        assert offenders == [(0, 1)]

    def test_duplicates_inside_one_split_are_not_offenders(self):
        hashes = hashes_from_patterns([[1, 1], [1, 1]])
        assert split.verify_no_leakage(hashes, np.array(["train", "train"]), 5) == []

    def test_detects_leak_across_chunk_boundary(self):
        hashes = hashes_from_patterns([[0] * 64] * 8)
        splits = np.array(["train"] * 4 + ["test"] * 4)
        assert len(split.verify_no_leakage(hashes, splits, 0, chunk=3)) == 16

    def test_near_but_not_identical_still_counts(self):
        hashes = hashes_from_patterns([[0] * 64, [1, 1, 1] + [0] * 61])
        assert split.verify_no_leakage(hashes, np.array(["train", "test"]), 5) == [(0, 1)]


class TestCoarseHashBits:
    def test_returns_64_bits(self, tmp_path):
        path = tmp_path / "x.png"
        Image.fromarray(np.random.default_rng(0).integers(
            0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        bits = split.coarse_hash_bits(path)
        assert bits.shape == (64,) and set(np.unique(bits)) <= {0, 1}

    def test_matches_between_a_crop_and_its_rescale(self, tmp_path):
        """otani's overlap with wilsons is a differently scaled crop of one session."""
        rng = np.random.default_rng(1)
        base = np.kron(rng.integers(0, 255, (8, 8), dtype=np.uint8), np.ones((32, 32), np.uint8))
        big, small = tmp_path / "big.png", tmp_path / "small.png"
        Image.fromarray(base).save(big)
        Image.fromarray(base).resize((128, 128), Image.Resampling.LANCZOS).save(small)
        assert np.array_equal(split.coarse_hash_bits(big), split.coarse_hash_bits(small))


class TestClassTable:
    def test_counts_per_split(self):
        rows = [{"classes": "tiger"}, {"classes": "tiger"}, {"classes": "bird"}]
        table = split.class_table(rows, np.array(["train", "val", "train"]))
        assert table["train"]["tiger"] == 1
        assert table["train"]["bird"] == 1
        assert table["val"]["tiger"] == 1

    def test_handles_multi_class_images(self):
        table = split.class_table([{"classes": "tiger|bird"}], np.array(["train"]))
        assert table["train"]["tiger"] == 1 and table["train"]["bird"] == 1

    def test_all_splits_present_even_when_empty(self):
        table = split.class_table([{"classes": "ox"}], np.array(["train"]))
        assert set(table) == {"train", "val", "test"}


class TestConfiguration:
    def test_canonical_matches_build_stage(self):
        """The two stages hardcode the class list separately; they must not drift."""
        name = "build_for_compare"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / "01_build_dataset.py"
        )
        build = importlib.util.module_from_spec(spec)
        # Register before exec: @dataclass resolves annotations via sys.modules.
        sys.modules[name] = build
        try:
            spec.loader.exec_module(build)
            assert split.CANONICAL == build.CANONICAL
        finally:
            del sys.modules[name]

    def test_defaults_do_not_overlap(self):
        assert not set(split.DEFAULT_VAL) & set(split.DEFAULT_TEST)
