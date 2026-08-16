"""Tests for the non-training logic in 03_train.py.

Training itself is not unit-tested. What is tested is everything that could silently
point the trainer at the wrong data or misreport the result -- the split selection and
the metric extraction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


train = _load("train_model", "03_train.py")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    for prefix in ("", "random_"):
        for name in ("train", "val", "test"):
            (tmp_path / f"{prefix}{name}.txt").write_text("./images/a.jpg\n")
    return tmp_path


class TestBuildSplitYaml:
    def test_disjoint_selects_unprefixed_lists(self, dataset, tmp_path):
        out = train.build_split_yaml(dataset, "disjoint", tmp_path / "out" / "d.yaml")
        config = yaml.safe_load(out.read_text())
        assert config["train"] == "train.txt"
        assert config["val"] == "val.txt"

    def test_random_selects_prefixed_lists(self, dataset, tmp_path):
        out = train.build_split_yaml(dataset, "random", tmp_path / "out" / "r.yaml")
        config = yaml.safe_load(out.read_text())
        assert config["train"] == "random_train.txt"
        assert config["test"] == "random_test.txt"

    def test_creates_missing_parent_directory(self, dataset, tmp_path):
        out = train.build_split_yaml(dataset, "disjoint", tmp_path / "a" / "b" / "c.yaml")
        assert out.exists()

    def test_class_names_and_count_are_consistent(self, dataset, tmp_path):
        config = yaml.safe_load(
            train.build_split_yaml(dataset, "disjoint", tmp_path / "d.yaml").read_text()
        )
        assert config["nc"] == len(config["names"]) == 12
        assert config["names"] == train.CANONICAL

    def test_path_is_absolute_and_posix(self, dataset, tmp_path):
        config = yaml.safe_load(
            train.build_split_yaml(dataset, "disjoint", tmp_path / "d.yaml").read_text()
        )
        assert "\\" not in config["path"]
        assert Path(config["path"]).is_absolute()

    def test_missing_list_file_raises(self, tmp_path):
        (tmp_path / "train.txt").write_text("./images/a.jpg\n")
        with pytest.raises(FileNotFoundError, match="run 02_split.py"):
            train.build_split_yaml(tmp_path, "disjoint", tmp_path / "d.yaml")


class FakeBox:
    """Stands in for ultralytics' results.box."""

    def __init__(self, ap_class_index, ap50, ap, p, r, map50, mean_ap):
        self.ap_class_index = ap_class_index
        self.ap50, self.ap, self.p, self.r = ap50, ap, p, r
        self.map50, self.map = map50, mean_ap
        self.mp, self.mr = sum(p) / len(p), sum(r) / len(r)


class FakeMetrics:
    def __init__(self, box):
        self.box = box


class TestSummarize:
    def test_extracts_overall_metrics(self):
        box = FakeBox([0, 1], [0.9, 0.8], [0.6, 0.5], [0.85, 0.75], [0.88, 0.7], 0.85, 0.55)
        summary = train.summarize(FakeMetrics(box), train.CANONICAL)
        assert summary["mAP50"] == 0.85
        assert summary["mAP50_95"] == 0.55

    def test_maps_ap_onto_the_right_class_names(self):
        """ap arrays are indexed by position, not by class id -- an easy off-by-one."""
        box = FakeBox([11, 3], [0.9, 0.4], [0.6, 0.2], [0.8, 0.3], [0.7, 0.2], 0.65, 0.4)
        summary = train.summarize(FakeMetrics(box), train.CANONICAL)
        assert summary["per_class"]["tiger"]["AP50"] == 0.9
        assert summary["per_class"]["dragon"]["AP50"] == 0.4

    def test_absent_classes_are_omitted_not_zeroed(self):
        """A class missing from a split has no AP; recording 0.0 would misreport it."""
        box = FakeBox([0], [0.9], [0.6], [0.8], [0.7], 0.9, 0.6)
        summary = train.summarize(FakeMetrics(box), train.CANONICAL)
        assert set(summary["per_class"]) == {"bird"}
        assert "tiger" not in summary["per_class"]

    def test_all_twelve_classes_present(self):
        n = 12
        box = FakeBox(list(range(n)), [0.9] * n, [0.6] * n, [0.8] * n, [0.7] * n, 0.9, 0.6)
        summary = train.summarize(FakeMetrics(box), train.CANONICAL)
        assert set(summary["per_class"]) == set(train.CANONICAL)

    def test_values_are_rounded(self):
        box = FakeBox([0], [0.123456789], [0.987654321], [0.5], [0.5], 0.1, 0.9)
        row = train.summarize(FakeMetrics(box), train.CANONICAL)["per_class"]["bird"]
        assert row["AP50"] == 0.1235


class TestAugmentationPolicy:
    def test_no_vertical_flip(self):
        """jannat was dropped for exactly this; do not reintroduce it."""
        assert train.AUGMENTATION["flipud"] == 0.0

    def test_no_mixup(self):
        """Blending two signs into one frame produces a label that means nothing."""
        assert train.AUGMENTATION["mixup"] == 0.0

    def test_rotation_stays_small(self):
        assert 0 < train.AUGMENTATION["degrees"] <= 15

    def test_colour_jitter_is_raised(self):
        """Skin tone and lighting are the scarcest axes; defaults are 0.7/0.4."""
        assert train.AUGMENTATION["hsv_s"] >= 0.7
        assert train.AUGMENTATION["hsv_v"] >= 0.4

    def test_horizontal_flip_on_by_default(self):
        assert train.AUGMENTATION["fliplr"] == 0.5

    def test_canonical_matches_other_stages(self):
        split = _load("split_for_train_compare", "02_split.py")
        assert train.CANONICAL == split.CANONICAL
