"""Merge the Roboflow Universe hand-sign exports into one canonical YOLO dataset.

Twelve sources were audited (see docs/superpowers/specs/); four are excluded because
their labels are unusable rather than merely low-value. The remaining eight disagree on
class *indices*, so every label is remapped through a canonical table before merging.

Outputs a flat image/label pool plus a manifest. Splitting happens in 02_split.py, which
reads the manifest -- keeping "what data exists" separate from "how it is partitioned".

Usage:
    python 01_build_dataset.py
    python 01_build_dataset.py --dedup-bits 64      # aggressive, matches original spec
    python 01_build_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# --------------------------------------------------------------------------------------
# Canonical schema
# --------------------------------------------------------------------------------------

# English alphabetical, matching the ordering used by the majority of sources.
# The single source of truth for class order lives in handsign/classes.py.
# Four copies of this list is how the index-mismatch bug comes back.
from handsign.classes import CANONICAL as _C, CANONICAL_INDEX

CANONICAL = list(_C)  # noqa: E402

# Every spelling observed across the twelve sources, mapped to canonical English.
# Romaji entries matter most: `otani` and `chayawat` order their classes by the romaji
# name, so their index 0 is `hitsuji` (ram), not `bird`.
ALIAS: dict[str, str] = {
    "tori": "bird", "bird": "bird",
    "i": "boar", "boar": "boar",
    "inu": "dog", "dog": "dog",
    "tatsu": "dragon", "dragon": "dragon",
    "u": "hare", "hare": "hare", "rabbit": "hare",
    "uma": "horse", "horse": "horse",
    "saru": "monkey", "monkey": "monkey",
    "ushi": "ox", "ox": "ox",
    "hitsuji": "ram", "ram": "ram", "sheep": "ram",
    "ne": "rat", "rat": "rat", "mouse": "rat",
    "mi": "snake", "snake": "snake", "serpent": "snake",
    "tora": "tiger", "tiger": "tiger",
}

# Sources excluded from the merge, with the audit finding that disqualified each.
EXCLUDED: dict[str, str] = {
    "sworkspace": "box geometry systematically wrong (w/h 1.5-3.3 vs 0.53-0.86 elsewhere); "
                  "520 of 1863 images show a sign but carry an empty label",
    "jannat": "90-degree and vertical-flip augmentation invalid for hand signs; "
              "299 images duplicate wilsons un-augmented",
    "kasidit": "500 of 1050 images are chayawat frames downscaled 512->256",
    "handsigns": "covers 2 of 12 classes across ~12 distinct moments",
}

# Sources exported with "Fit (black edges)" -- letterbox bars must be cropped and the
# box coordinates renormalized against the content region.
LETTERBOXED: set[str] = {"otani"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "valid", "test")


class BuildError(RuntimeError):
    """Raised when input data violates an assumption the merge depends on."""


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_build_dataset.py)
# --------------------------------------------------------------------------------------

def normalize_class_name(raw: str) -> str:
    """Map a source's class name onto the canonical vocabulary.

    Handles the decorated forms seen in the wild: ``'Hitsuji -ram-'`` (chayawat) and
    ``'Bird-'`` (jannat). Raises rather than guessing -- a silently mismapped class is
    precisely the failure this whole stage exists to prevent, and it would surface only
    as a mediocre mAP rather than an error.
    """
    text = raw.strip().lower()

    # 'hitsuji -ram-' carries the English name inside the dashes; prefer it.
    if " -" in text and text.endswith("-"):
        inner = text.split(" -")[-1].rstrip("-").strip()
        if inner in ALIAS:
            return ALIAS[inner]

    stripped = text.rstrip("-").strip()
    if stripped in ALIAS:
        return ALIAS[stripped]

    raise BuildError(
        f"unrecognized class name {raw!r}. Add it to ALIAS in 01_build_dataset.py "
        f"-- refusing to guess, since a wrong mapping is silent."
    )


def perceptual_hash(image: Image.Image, side: int) -> bytes:
    """Difference hash: compare each pixel to its right neighbour on a `side`x`side` grid.

    `side=8` yields 64 bits (coarse -- collides across consecutive video frames),
    `side=16` yields 256 bits (tight -- collides only on true duplicates).
    """
    grid = image.convert("L").resize((side + 1, side), Image.Resampling.LANCZOS)
    cells = np.asarray(grid, dtype=np.int16)
    return np.packbits((cells[:, 1:] > cells[:, :-1]).flatten()).tobytes()


def find_content_box(
    image: Image.Image, luma_threshold: int = 16
) -> tuple[int, int, int, int]:
    """Locate the non-letterbox region as ``(left, top, right, bottom)``.

    A row or column counts as a bar when at least 98% of its pixels are near-black. The
    fraction test (rather than a max or a percentile) is what tolerates the handful of
    bright pixels JPEG ringing leaves along the bar edge, without being thrown off by a
    single outlier the way a percentile on a narrow image would be.
    """
    luma = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = luma.shape

    dark = luma < luma_threshold
    row_dark = dark.mean(axis=1) >= 0.98
    col_dark = dark.mean(axis=0) >= 0.98

    def first_content(is_dark: np.ndarray) -> int:
        idx = int(np.argmin(is_dark))
        return idx if not is_dark[idx] else len(is_dark)

    top = first_content(row_dark)
    bottom = len(row_dark) - first_content(row_dark[::-1])
    left = first_content(col_dark)
    right = len(col_dark) - first_content(col_dark[::-1])

    # An all-black image would produce an empty crop; keep it whole and let the caller
    # decide, rather than emitting a zero-size image.
    if top >= bottom or left >= right:
        return 0, 0, width, height
    return left, top, right, bottom


def recrop_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    content: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Renormalize a YOLO ``(xc, yc, w, h)`` box from the full frame onto a crop."""
    xc, yc, bw, bh = box
    width, height = image_size
    left, top, right, bottom = content
    crop_w, crop_h = right - left, bottom - top
    return (
        (xc * width - left) / crop_w,
        (yc * height - top) / crop_h,
        bw * width / crop_w,
        bh * height / crop_h,
    )


def clip_box(
    box: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float, float], bool]:
    """Clamp a box to the unit frame, returning ``(box, was_modified)``.

    Clips the corners and rebuilds the centre/extent, so a box hanging off the edge is
    truncated at the border rather than being translated inward. A box lying entirely
    outside collapses to zero area rather than inverting to a negative extent.
    """
    def clamp(value: float) -> float:
        return min(1.0, max(0.0, value))

    xc, yc, bw, bh = box
    # Clamping both corners into [0,1] keeps them ordered, so the extents below are
    # never negative and the rebuilt centre never lands outside the frame.
    x1, x2 = clamp(xc - bw / 2), clamp(xc + bw / 2)
    y1, y2 = clamp(yc - bh / 2), clamp(yc + bh / 2)
    clipped = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    changed = any(abs(a - b) > 1e-9 for a, b in zip(box, clipped))
    return clipped, changed


def parse_label_line(line: str) -> tuple[int, tuple[float, float, float, float]] | None:
    """Parse one YOLO label line, or return None if it is blank."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) < 5:
        raise BuildError(f"malformed label line: {line!r}")
    values = [float(p) for p in parts[1:5]]
    return int(float(parts[0])), (values[0], values[1], values[2], values[3])


# --------------------------------------------------------------------------------------
# Source reading
# --------------------------------------------------------------------------------------

@dataclass
class Sample:
    source: str
    origin: Path
    label_path: Path
    boxes: list[tuple[int, tuple[float, float, float, float]]]
    width: int
    height: int
    fine_hash: bytes = b""
    coarse_hash: bytes = b""


@dataclass
class Stats:
    scanned: int = 0
    unlabeled: int = 0
    empty_label: int = 0
    clipped_boxes: int = 0
    letterbox_cropped: int = 0
    per_source: Counter = field(default_factory=Counter)


def load_class_map(source_dir: Path) -> dict[int, str]:
    """Read a source's data.yaml and build ``{source_index -> canonical_name}``."""
    config = yaml.safe_load((source_dir / "data.yaml").read_text())
    names = config["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    return {i: normalize_class_name(name) for i, name in enumerate(names)}


def label_path_for(image_path: Path) -> Path:
    """Roboflow mirrors ``.../images/x.jpg`` as ``.../labels/x.txt``."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_source(source_dir: Path, stats: Stats) -> list[Sample]:
    """Load every labelled image from one source, with classes already remapped."""
    source = source_dir.name
    class_map = load_class_map(source_dir)
    samples: list[Sample] = []

    for split in SPLIT_DIRS:
        image_dir = source_dir / split / "images"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stats.scanned += 1

            label_path = label_path_for(image_path)
            if not label_path.exists():
                stats.unlabeled += 1
                continue

            boxes = []
            for line in label_path.read_text().splitlines():
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                index, box = parsed
                if index not in class_map:
                    raise BuildError(
                        f"{label_path}: class index {index} outside the "
                        f"{len(class_map)} classes declared in {source}/data.yaml"
                    )
                boxes.append((CANONICAL_INDEX[class_map[index]], box))

            # Roboflow emits an empty label file for negatives. We drop them here because
            # the excluded `sworkspace` proved these are unreliable -- 520 of its "empty"
            # images plainly show a sign. Genuine hard negatives would need verifying.
            if not boxes:
                stats.empty_label += 1
                continue

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except OSError as exc:
                raise BuildError(f"cannot read {image_path}: {exc}") from exc

            samples.append(Sample(source, image_path, label_path, boxes, width, height))

    stats.per_source[source] = len(samples)
    return samples


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------

def compute_hashes(samples: list[Sample], coarse_side: int, verbose: bool) -> None:
    """Attach a tight (256-bit) and a coarse hash to every sample."""
    for i, sample in enumerate(samples):
        with Image.open(sample.origin) as image:
            sample.fine_hash = perceptual_hash(image, 16)
            sample.coarse_hash = perceptual_hash(image, coarse_side)
        if verbose and i and i % 2000 == 0:
            print(f"    hashed {i}/{len(samples)}", flush=True)


def deduplicate(
    samples: list[Sample], coarse_side: int
) -> tuple[list[Sample], dict[str, int]]:
    """Drop duplicates, preferring the highest-resolution copy of each.

    Cross-source duplicates are removed unconditionally: `cs` and `wilsons` share frames,
    and were they ever split apart, those shared frames would leak train into test.

    Within a source, duplicates cannot leak (the split is source-disjoint), so removal is
    only about not over-weighting a source. `coarse_side=16` therefore drops just true
    duplicates; pass 8 for the aggressive variant that also collapses near-identical
    consecutive video frames.
    """
    counts = {"cross_source": 0, "within_source": 0}

    # Pass 1 -- cross-source, on the tight hash so only genuine shared frames collide.
    by_fine: dict[bytes, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_fine[sample.fine_hash].append(sample)

    survivors: list[Sample] = []
    for group in by_fine.values():
        if len({s.source for s in group}) > 1:
            best = max(group, key=lambda s: (s.width * s.height, s.source))
            counts["cross_source"] += len(group) - 1
            # Keep only same-source siblings of the winner; pass 2 thins those.
            survivors.extend([s for s in group if s.source == best.source])
        else:
            survivors.extend(group)

    # Pass 2 -- within-source, on the coarse hash.
    by_source_coarse: dict[tuple[str, bytes], list[Sample]] = defaultdict(list)
    for sample in survivors:
        by_source_coarse[(sample.source, sample.coarse_hash)].append(sample)

    kept: list[Sample] = []
    for group in by_source_coarse.values():
        best = max(group, key=lambda s: (s.width * s.height, str(s.origin)))
        counts["within_source"] += len(group) - 1
        kept.append(best)

    kept.sort(key=lambda s: (s.source, str(s.origin)))
    return kept, counts


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------

def write_sample(sample: Sample, out_dir: Path, index: int, stats: Stats) -> dict:
    """Emit one image/label pair; returns its manifest row."""
    stem = f"{sample.source}_{index:05d}"
    suffix = sample.origin.suffix.lower()
    boxes = [(cls, box) for cls, box in sample.boxes]
    width, height = sample.width, sample.height

    if sample.source in LETTERBOXED:
        with Image.open(sample.origin) as image:
            content = find_content_box(image)
            if content != (0, 0, image.width, image.height):
                cropped = image.crop(content)
                boxes = [
                    (cls, recrop_box(box, (image.width, image.height), content))
                    for cls, box in boxes
                ]
                width, height = cropped.size
                cropped.convert("RGB").save(out_dir / "images" / f"{stem}.jpg", quality=95)
                suffix = ".jpg"
                stats.letterbox_cropped += 1
            else:
                shutil.copy2(sample.origin, out_dir / "images" / f"{stem}{suffix}")
    else:
        shutil.copy2(sample.origin, out_dir / "images" / f"{stem}{suffix}")

    lines = []
    for cls, box in boxes:
        box, changed = clip_box(box)
        if changed:
            stats.clipped_boxes += 1
        if box[2] <= 0 or box[3] <= 0:
            continue  # fully outside the frame after cropping
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
    (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    return {
        "id": stem,
        "source": sample.source,
        "origin": str(sample.origin.relative_to(sample.origin.parents[3])),
        "image": f"images/{stem}{suffix}",
        "width": width,
        "height": height,
        "n_boxes": len(lines),
        "classes": "|".join(CANONICAL[cls] for cls, _ in boxes),
        "fine_hash": sample.fine_hash.hex(),
    }


def write_data_yaml(out_dir: Path) -> None:
    """Write data.yaml. Split lists are created later by 02_split.py.

    The lists live at the dataset root, not in a subdirectory: Ultralytics resolves a
    ``./`` path inside a list file against *that file's own parent*, so a list in
    ``splits/`` would send it looking for ``splits/images/``.
    """
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()).replace("\\", "/"),
                "train": "train.txt",
                "val": "val.txt",
                "test": "test.txt",
                "nc": len(CANONICAL),
                "names": CANONICAL,
            },
            sort_keys=False,
        )
    )


# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("00_data"))
    parser.add_argument("--out", type=Path, default=Path("01_dataset"))
    parser.add_argument(
        "--dedup-bits", type=int, choices=(64, 256), default=256,
        help="within-source duplicate sensitivity; 64 also collapses near-identical "
             "consecutive video frames (default: 256)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"error: {args.raw} not found", file=sys.stderr)
        return 1

    coarse_side = 8 if args.dedup_bits == 64 else 16
    sources = sorted(
        d.name for d in args.raw.iterdir() if d.is_dir() and (d / "data.yaml").exists()
    )
    keepers = [s for s in sources if s not in EXCLUDED]

    print(f"sources found: {len(sources)}   using: {len(keepers)}")
    for name in sources:
        if name in EXCLUDED:
            print(f"  - EXCLUDE {name:12s} {EXCLUDED[name]}")

    stats = Stats()
    samples: list[Sample] = []
    print("\nreading sources ...")
    for name in keepers:
        found = read_source(args.raw / name, stats)
        samples.extend(found)
        print(f"  {name:12s} {len(found):6d} labelled images")

    print(f"\ntotal labelled: {len(samples)}"
          f"   (skipped {stats.unlabeled} unlabeled, {stats.empty_label} empty-label)")

    print(f"\nhashing (within-source dedup at {args.dedup_bits}-bit) ...")
    compute_hashes(samples, coarse_side, verbose=True)
    kept, dropped = deduplicate(samples, coarse_side)
    print(f"  cross-source duplicates removed: {dropped['cross_source']}")
    print(f"  within-source duplicates removed: {dropped['within_source']}")
    print(f"  remaining: {len(kept)}")

    class_counts = Counter(CANONICAL[cls] for s in kept for cls, _ in s.boxes)
    source_counts = Counter(s.source for s in kept)

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        print(f"\nwriting to {args.out} ...")
        for sub in ("images", "labels", "splits"):
            (args.out / sub).mkdir(parents=True, exist_ok=True)
        rows = [write_sample(s, args.out, i, stats) for i, s in enumerate(kept)]
        with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_data_yaml(args.out)
        print(f"  {len(rows)} images, manifest.csv, data.yaml")
        print(f"  letterbox-cropped: {stats.letterbox_cropped}"
              f"   boxes clipped to frame: {stats.clipped_boxes}")

    print("\nper-source:")
    for name, count in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {count:6d}")
    print("\nper-class boxes:")
    total = sum(class_counts.values())
    for name in CANONICAL:
        count = class_counts[name]
        print(f"  {name:8s} {count:5d}  {count / total * 100:5.2f}%")
    print(f"  {'TOTAL':8s} {total:5d}   imbalance max/min = "
          f"{max(class_counts.values()) / min(class_counts.values()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
