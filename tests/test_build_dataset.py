"""Tests for the pure logic in 01_build_dataset.py.

Weighted toward the class remap, because that is the one defect in this pipeline that
produces no error and no crash -- only a mediocre mAP that looks like a modelling problem.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "build_dataset", Path(__file__).resolve().parents[1] / "01_build_dataset.py"
)
build = importlib.util.module_from_spec(_spec)
sys.modules["build_dataset"] = build
_spec.loader.exec_module(build)


# ----------------------------------------------------------------------------- remapping

class TestNormalizeClassName:
    @pytest.mark.parametrize("raw,expected", [
        ("bird", "bird"), ("Bird", "bird"), ("  TIGER  ", "tiger"),
    ])
    def test_english_names(self, raw, expected):
        assert build.normalize_class_name(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("tori", "bird"), ("i", "boar"), ("inu", "dog"), ("tatsu", "dragon"),
        ("u", "hare"), ("uma", "horse"), ("saru", "monkey"), ("ushi", "ox"),
        ("hitsuji", "ram"), ("ne", "rat"), ("mi", "snake"), ("tora", "tiger"),
    ])
    def test_romaji_names(self, raw, expected):
        """otani's full vocabulary. Single letters `i` and `u` are real classes."""
        assert build.normalize_class_name(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Hitsuji -ram-", "ram"), ("Tora -tiger-", "tiger"), ("Ne -rat-", "rat"),
    ])
    def test_chayawat_decorated_form(self, raw, expected):
        assert build.normalize_class_name(raw) == expected

    @pytest.mark.parametrize("raw,expected", [("Bird-", "bird"), ("Snake-", "snake")])
    def test_trailing_dash_form(self, raw, expected):
        assert build.normalize_class_name(raw) == expected

    def test_serpent_is_snake(self):
        """kasidit and yylunxie say 'serpent' where everyone else says 'snake'."""
        assert build.normalize_class_name("Serpent") == "snake"

    def test_unknown_name_raises(self):
        with pytest.raises(build.BuildError, match="unrecognized class name"):
            build.normalize_class_name("gassho")

    def test_alias_covers_canonical_vocabulary(self):
        assert set(build.ALIAS.values()) == set(build.CANONICAL)


class TestIndexSpaces:
    """The actual class-name lists shipped by the sources, verbatim from their data.yaml."""

    ENGLISH = ["bird", "boar", "dog", "dragon", "hare", "horse",
               "monkey", "ox", "ram", "rat", "snake", "tiger"]
    ROMAJI = ["hitsuji", "i", "inu", "mi", "ne", "saru",
              "tatsu", "tora", "tori", "u", "uma", "ushi"]
    DECORATED = ["Hitsuji -ram-", "I -boar-", "Inu -dog-", "Mi -snake-", "Ne -rat-",
                 "Saru -monkey-", "Tatsu -dragon-", "Tora -tiger-", "Tori -bird-",
                 "U -hare-", "Uma -horse-", "Ushi -ox-"]

    def test_english_index_zero_is_bird(self):
        assert build.normalize_class_name(self.ENGLISH[0]) == "bird"

    def test_romaji_index_zero_is_ram_not_bird(self):
        """The defect this pipeline exists to prevent: otani's 0 is ram, not bird."""
        assert build.normalize_class_name(self.ROMAJI[0]) == "ram"

    def test_romaji_index_ten_is_horse_not_snake(self):
        assert build.normalize_class_name(self.ROMAJI[10]) == "horse"

    @pytest.mark.parametrize("names", [ENGLISH, ROMAJI, DECORATED])
    def test_every_ordering_covers_all_twelve_classes(self, names):
        mapped = [build.normalize_class_name(n) for n in names]
        assert sorted(mapped) == sorted(build.CANONICAL)

    def test_romaji_and_decorated_agree_elementwise(self):
        """chayawat and otani use the same order, so they must remap identically."""
        assert ([build.normalize_class_name(n) for n in self.ROMAJI]
                == [build.normalize_class_name(n) for n in self.DECORATED])

    def test_orderings_genuinely_differ(self):
        """Guards the premise: if these matched, no remap would be needed."""
        assert ([build.normalize_class_name(n) for n in self.ENGLISH]
                != [build.normalize_class_name(n) for n in self.ROMAJI])


# ---------------------------------------------------------------------------- letterbox

def _letterboxed(width=64, height=64, bar=10, fill=200) -> Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[bar:height - bar, :, :] = fill
    return Image.fromarray(canvas)


class TestFindContentBox:
    def test_detects_horizontal_bars(self):
        assert build.find_content_box(_letterboxed(bar=10)) == (0, 10, 64, 54)

    def test_detects_vertical_bars(self):
        canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        canvas[:, 8:56, :] = 180
        assert build.find_content_box(Image.fromarray(canvas)) == (8, 0, 56, 64)

    def test_no_bars_returns_full_frame(self):
        image = Image.fromarray(np.full((40, 60, 3), 128, dtype=np.uint8))
        assert build.find_content_box(image) == (0, 0, 60, 40)

    def test_all_black_returns_full_frame(self):
        """Must not produce a zero-size crop."""
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        assert build.find_content_box(image) == (0, 0, 32, 32)

    def test_tolerates_stray_bright_pixel_in_bar(self):
        """JPEG ringing at the bar edge must not defeat the scan."""
        canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        canvas[10:54, :, :] = 200
        canvas[3, 30, :] = 255
        assert build.find_content_box(Image.fromarray(canvas)) == (0, 10, 64, 54)

    def test_dark_content_is_not_mistaken_for_a_bar(self):
        image = Image.fromarray(np.full((40, 40, 3), 40, dtype=np.uint8))
        assert build.find_content_box(image) == (0, 0, 40, 40)


class TestRecropBox:
    def test_centre_box_stays_centred(self):
        out = build.recrop_box((0.5, 0.5, 0.2, 0.2), (64, 64), (0, 10, 64, 54))
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(0.5)

    def test_height_scales_by_crop_ratio(self):
        """Cropping 64 -> 44 rows makes a box occupy proportionally more of the frame."""
        _, _, _, h = build.recrop_box((0.5, 0.5, 0.2, 0.2), (64, 64), (0, 10, 64, 54))
        assert h == pytest.approx(0.2 * 64 / 44)

    def test_width_untouched_by_horizontal_bars(self):
        _, _, w, _ = build.recrop_box((0.5, 0.5, 0.2, 0.2), (64, 64), (0, 10, 64, 54))
        assert w == pytest.approx(0.2)

    def test_offcentre_box_moves_up(self):
        _, y, _, _ = build.recrop_box((0.5, 0.25, 0.2, 0.2), (64, 64), (0, 10, 64, 54))
        assert y == pytest.approx((0.25 * 64 - 10) / 44)

    def test_roundtrip_against_pixel_arithmetic(self):
        size, content = (100, 100), (0, 20, 100, 80)
        xc, yc, w, h = build.recrop_box((0.5, 0.5, 0.4, 0.4), size, content)
        assert (yc - h / 2) * 60 == pytest.approx(0.5 * 100 - 0.4 * 100 / 2 - 20)


class TestClipBox:
    def test_inside_box_untouched(self):
        box, changed = build.clip_box((0.5, 0.5, 0.2, 0.2))
        assert not changed and box == pytest.approx((0.5, 0.5, 0.2, 0.2))

    def test_overhanging_box_is_truncated_not_translated(self):
        """marcs has 43 boxes hanging off the frame; the visible part must be preserved."""
        box, changed = build.clip_box((0.1, 0.5, 0.4, 0.2))
        assert changed
        assert box[0] == pytest.approx(0.15)   # centre of [0.0, 0.3]
        assert box[2] == pytest.approx(0.3)    # width truncated from 0.4

    def test_clips_all_four_edges(self):
        box, changed = build.clip_box((0.5, 0.5, 2.0, 2.0))
        assert changed and box == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_fully_outside_box_collapses_to_zero_area(self):
        box, _ = build.clip_box((1.5, 0.5, 0.2, 0.2))
        assert box[2] == pytest.approx(0.0)

    def test_result_is_always_within_unit_frame(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            raw = tuple(rng.uniform(-0.5, 1.5, 4))
            (xc, yc, w, h), _ = build.clip_box(raw)
            assert -1e-9 <= xc - w / 2 and xc + w / 2 <= 1 + 1e-9
            assert -1e-9 <= yc - h / 2 and yc + h / 2 <= 1 + 1e-9


# --------------------------------------------------------------------------------- misc

class TestPerceptualHash:
    def test_bit_width(self):
        image = Image.fromarray(np.random.default_rng(0).integers(
            0, 255, (40, 40, 3), dtype=np.uint8))
        assert len(build.perceptual_hash(image, 8)) == 8      # 64 bits
        assert len(build.perceptual_hash(image, 16)) == 32    # 256 bits

    def test_is_deterministic(self):
        image = Image.fromarray(np.random.default_rng(1).integers(
            0, 255, (50, 50, 3), dtype=np.uint8))
        assert build.perceptual_hash(image, 16) == build.perceptual_hash(image, 16)

    def test_survives_rescaling(self):
        """kasidit duplicates chayawat at half resolution; both must hash alike."""
        rng = np.random.default_rng(2)
        big = Image.fromarray(np.kron(
            rng.integers(0, 255, (16, 16), dtype=np.uint8), np.ones((16, 16), np.uint8)))
        small = big.resize((128, 128), Image.Resampling.LANCZOS)
        assert build.perceptual_hash(big, 8) == build.perceptual_hash(small, 8)

    def test_distinguishes_different_images(self):
        rng = np.random.default_rng(3)
        a = Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8))
        b = Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8))
        assert build.perceptual_hash(a, 16) != build.perceptual_hash(b, 16)


class TestParseLabelLine:
    def test_parses_a_normal_line(self):
        assert build.parse_label_line("3 0.5 0.5 0.2 0.4") == (3, (0.5, 0.5, 0.2, 0.4))

    @pytest.mark.parametrize("line", ["", "   ", "\t"])
    def test_blank_lines_return_none(self, line):
        assert build.parse_label_line(line) is None

    def test_float_class_index_is_accepted(self):
        """Some exports write '3.0' rather than '3'."""
        assert build.parse_label_line("3.0 0.5 0.5 0.2 0.4")[0] == 3

    def test_extra_columns_ignored(self):
        """Segmentation-style exports carry trailing polygon points."""
        assert build.parse_label_line("1 0.5 0.5 0.2 0.4 0.9 0.9")[0] == 1

    def test_truncated_line_raises(self):
        with pytest.raises(build.BuildError, match="malformed"):
            build.parse_label_line("3 0.5 0.5")


class TestConfiguration:
    def test_twelve_canonical_classes(self):
        assert len(build.CANONICAL) == 12 == len(set(build.CANONICAL))

    def test_canonical_order_is_alphabetical(self):
        """Matches the majority of sources, so their labels pass through unchanged."""
        assert build.CANONICAL == sorted(build.CANONICAL)

    def test_index_lookup_agrees_with_list(self):
        for i, name in enumerate(build.CANONICAL):
            assert build.CANONICAL_INDEX[name] == i

    def test_every_exclusion_carries_a_reason(self):
        assert set(build.EXCLUDED) == {"sworkspace", "jannat", "kasidit", "handsigns"}
        assert all(len(reason) > 20 for reason in build.EXCLUDED.values())

    def test_label_path_derivation(self):
        got = build.label_path_for(Path("00_data/vgu/train/images/a_jpg.rf.abc.jpg"))
        assert got == Path("00_data/vgu/train/labels/a_jpg.rf.abc.txt")

    def test_label_path_only_rewrites_the_images_component(self):
        """A source directory literally named 'images' must not be rewritten."""
        got = build.label_path_for(Path("00_data/images/train/images/x.jpg"))
        assert got == Path("00_data/images/train/labels/x.txt")
