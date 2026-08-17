"""Partition the merged dataset into subject-disjoint train / val / test splits.

Splitting by *source* is not sufficient. `otani` and `wilsons` turned out to contain the
same recording session at different crops -- 44% of otani has a near-duplicate in wilsons
-- so holding out otani while training on wilsons would leak. That overlap was invisible
in the raw exports because otani shipped letterboxed, and only surfaced once 01_build
cropped the bars.

This script therefore *derives* subject groups from image content rather than trusting
directory names: sources are linked when they share near-duplicate frames, and each
connected component is assigned to exactly one split. It then re-verifies the result and
refuses to write a split that leaks.

Usage:
    python 02_split.py
    python 02_split.py --val chayawat minsub --test yylunxie
    python 02_split.py --threshold 8        # stricter linkage
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# The single source of truth for class order lives in handsign/classes.py.
# Four copies of this list is how the index-mismatch bug comes back.
from handsign.classes import CANONICAL as _C

CANONICAL = list(_C)  # noqa: E402

# Sources whose *group* is held out. Named per source; the group each belongs to is
# discovered at runtime, so naming any member holds out the whole component.
DEFAULT_VAL = ["chayawat", "minsub"]
DEFAULT_TEST = ["yylunxie"]

# 64-bit dHash Hamming distance below which two frames count as the same visual moment.
# Deliberately loose: over-linking merely makes the held-out estimate more conservative,
# whereas under-linking silently inflates it.
DEFAULT_THRESHOLD = 5


class SplitError(RuntimeError):
    """Raised when a split cannot be produced without leaking."""


# --------------------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------------------

def coarse_hash_bits(path: Path, side: int = 8) -> np.ndarray:
    """64 unpacked dHash bits -- unpacked so Hamming distance is a plain XOR-sum."""
    with Image.open(path) as image:
        grid = image.convert("L").resize((side + 1, side), Image.Resampling.LANCZOS)
        cells = np.asarray(grid, dtype=np.int16)
    return (cells[:, 1:] > cells[:, :-1]).flatten().astype(np.uint8)


def load_or_build_hashes(dataset: Path, rows: list[dict], verbose: bool) -> np.ndarray:
    """Hash every image, caching to disk -- this is the slow step and it rarely changes."""
    cache = dataset / "hashes64.npy"
    if cache.exists():
        cached = np.load(cache)
        if len(cached) == len(rows):
            if verbose:
                print(f"  loaded cached hashes from {cache.name}")
            return cached
        print(f"  cache size {len(cached)} != {len(rows)} rows; rehashing")

    if verbose:
        print(f"  hashing {len(rows)} images ...")
    hashes = np.zeros((len(rows), 64), dtype=np.uint8)
    for i, row in enumerate(rows):
        hashes[i] = coarse_hash_bits(dataset / row["image"])
        if verbose and i and i % 2500 == 0:
            print(f"    {i}/{len(rows)}", flush=True)
    np.save(cache, hashes)
    return hashes


def find_near_duplicate_links(
    hashes: np.ndarray, sources: np.ndarray, threshold: int, chunk: int = 1200
) -> dict[tuple[str, str], set[int]]:
    """Return ``{(source_a, source_b): {image indices involved}}`` for near-duplicates."""
    links: dict[tuple[str, str], set[int]] = defaultdict(set)
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        distance = (block[:, None, :] != hashes[None, :, :]).sum(2)
        for a, b in zip(*np.where(distance <= threshold)):
            i, j = start + int(a), int(b)
            if i >= j or sources[i] == sources[j]:
                continue
            key = (sources[i], sources[j]) if sources[i] < sources[j] else (sources[j], sources[i])
            links[key].update((i, j))
    return links


def connected_components(
    all_sources: list[str], links: dict[tuple[str, str], set[int]]
) -> list[list[str]]:
    """Merge sources that share near-duplicate frames into subject groups."""
    parent = {s: s for s in all_sources}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[str, list[str]] = defaultdict(list)
    for source in all_sources:
        groups[find(source)].append(source)
    return sorted((sorted(v) for v in groups.values()), key=len, reverse=True)


# --------------------------------------------------------------------------------------
# Assignment and verification
# --------------------------------------------------------------------------------------

def assign_splits(
    groups: list[list[str]], val_names: list[str], test_names: list[str]
) -> dict[str, str]:
    """Map each source to a split, holding out whole groups.

    Naming any member of a group holds out the entire group, since its members are not
    separable without leaking.
    """
    group_of = {s: i for i, g in enumerate(groups) for s in g}
    for name in val_names + test_names:
        if name not in group_of:
            raise SplitError(f"unknown source {name!r}; available: {sorted(group_of)}")

    val_groups = {group_of[n] for n in val_names}
    test_groups = {group_of[n] for n in test_names}
    if val_groups & test_groups:
        shared = [groups[i] for i in val_groups & test_groups]
        raise SplitError(
            f"val and test would share subject group(s) {shared} -- they contain the same "
            f"person, so holding both out separately does not measure generalization."
        )

    assignment = {}
    for i, group in enumerate(groups):
        split = "val" if i in val_groups else "test" if i in test_groups else "train"
        for source in group:
            assignment[source] = split
    return assignment


def verify_no_leakage(
    hashes: np.ndarray, splits: np.ndarray, threshold: int, chunk: int = 1200
) -> list[tuple[int, int]]:
    """Re-check the finished split directly, independent of how groups were derived."""
    offenders: list[tuple[int, int]] = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        distance = (block[:, None, :] != hashes[None, :, :]).sum(2)
        for a, b in zip(*np.where(distance <= threshold)):
            i, j = start + int(a), int(b)
            if i < j and splits[i] != splits[j]:
                offenders.append((i, j))
    return offenders


def class_table(rows: list[dict], splits: np.ndarray) -> dict[str, Counter]:
    table: dict[str, Counter] = {name: Counter() for name in ("train", "val", "test")}
    for row, split in zip(rows, splits):
        for name in row["classes"].split("|"):
            table[split][name] += 1
    return table


# --------------------------------------------------------------------------------------

def resolve_listed_path(list_file: Path, line: str) -> Path:
    """Resolve a line from a YOLO list file exactly as Ultralytics does.

    Ultralytics rewrites a leading ``./`` to the list file's own parent directory. Mirrored
    here so `verify_listing` fails on a path Ultralytics would fail on, rather than on one
    Python's own relative-path rules happen to dislike.
    """
    if line.startswith("./"):
        return list_file.parent / line[2:]
    return Path(line)


def write_split_files(
    dataset: Path, rows: list[dict], splits: np.ndarray, prefix: str = ""
) -> list[Path]:
    """Write YOLO list files at the dataset root and return their paths.

    Root, not a subdirectory: see the note in 01_build_dataset.write_data_yaml.
    """
    written = []
    for name in ("train", "val", "test"):
        paths = [f"./{r['image']}" for r, s in zip(rows, splits) if s == name]
        target = dataset / f"{prefix}{name}.txt"
        target.write_text("\n".join(paths) + "\n")
        written.append(target)
    return written


def verify_listing(list_file: Path) -> None:
    """Confirm every listed image, and its label, exists where the trainer will look."""
    lines = [x for x in list_file.read_text().splitlines() if x.strip()]
    if not lines:
        raise SplitError(f"{list_file.name} is empty")
    for line in lines:
        image = resolve_listed_path(list_file, line)
        if not image.exists():
            raise SplitError(f"{list_file.name}: {line} resolves to {image}, which does not exist")
        label = Path(str(image).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        label = label.with_suffix(".txt")
        if not label.exists():
            raise SplitError(f"{list_file.name}: no label for {image.name} (expected {label})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("01_dataset"))
    parser.add_argument("--val", nargs="+", default=DEFAULT_VAL)
    parser.add_argument("--test", nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--allow-leakage", action="store_true",
        help="write the split even if verification finds cross-split near-duplicates",
    )
    args = parser.parse_args()

    manifest = args.dataset / "manifest.csv"
    if not manifest.exists():
        print(f"error: {manifest} not found; run 01_build_dataset.py first", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    sources = np.array([r["source"] for r in rows])
    print(f"{len(rows)} images from {len(set(sources))} sources")

    hashes = load_or_build_hashes(args.dataset, rows, verbose=True)

    print(f"\nlinking sources that share frames (Hamming <= {args.threshold}) ...")
    links = find_near_duplicate_links(hashes, sources, args.threshold)
    totals = Counter(sources)
    for (a, b), involved in sorted(links.items(), key=lambda kv: -len(kv[1])):
        na = sum(1 for i in involved if sources[i] == a)
        nb = sum(1 for i in involved if sources[i] == b)
        print(f"  {a} <-> {b}: {na}/{totals[a]} ({na / totals[a] * 100:.1f}%) "
              f"and {nb}/{totals[b]} ({nb / totals[b] * 100:.1f}%)")
    if not links:
        print("  none -- every source is independent")

    groups = connected_components(sorted(set(sources.tolist())), links)
    print("\nsubject groups:")
    for i, group in enumerate(groups, 1):
        print(f"  {i}. {group}  ({sum(totals[s] for s in group)} images)")

    assignment = assign_splits(groups, args.val, args.test)
    splits = np.array([assignment[s] for s in sources])

    print("\nverifying the finished split ...")
    offenders = verify_no_leakage(hashes, splits, args.threshold)
    if offenders:
        pairs = Counter(
            tuple(sorted((f"{sources[i]}/{splits[i]}", f"{sources[j]}/{splits[j]}")))
            for i, j in offenders
        )
        print(f"  FAIL: {len(offenders)} cross-split near-duplicate pairs")
        for pair, count in pairs.most_common(10):
            print(f"    {count:6d}  {pair[0]} <-> {pair[1]}")
        if not args.allow_leakage:
            raise SplitError(
                "refusing to write a leaking split; rerun with different --val/--test, "
                "or --allow-leakage to override"
            )
    else:
        print("  OK: no cross-split near-duplicates")

    written = write_split_files(args.dataset, rows, splits)
    for list_file in written:
        verify_listing(list_file)
    print(f"  OK: all listed images and labels resolve")

    # A random split, for contrast only. Reporting both makes the cost of the naive
    # approach visible -- it is typically 10-15 mAP points of self-deception.
    rng = random.Random(0)
    shuffled = list(range(len(rows)))
    rng.shuffle(shuffled)
    n_val, n_test = int(len(rows) * 0.1), int(len(rows) * 0.1)
    random_splits = np.array(["train"] * len(rows), dtype=object)
    for i in shuffled[:n_val]:
        random_splits[i] = "val"
    for i in shuffled[n_val:n_val + n_test]:
        random_splits[i] = "test"
    for list_file in write_split_files(args.dataset, rows, random_splits, prefix="random_"):
        verify_listing(list_file)

    counts = Counter(splits.tolist())
    table = class_table(rows, splits)
    print("\n" + "=" * 74)
    print(f"{'split':6s}{'imgs':>7s}{'share':>8s}   sources")
    for name in ("train", "val", "test"):
        members = sorted({s for s, sp in assignment.items() if sp == name})
        print(f"{name:6s}{counts[name]:7d}{counts[name] / len(rows) * 100:7.1f}%   "
              f"{', '.join(members)}")

    print(f"\n{'class':9s}{'train':>8s}{'val':>7s}{'test':>7s}")
    for name in CANONICAL:
        print(f"{name:9s}{table['train'][name]:8d}{table['val'][name]:7d}{table['test'][name]:7d}")
    missing = {
        split: [c for c in CANONICAL if not table[split][c]] for split in ("train", "val", "test")
    }
    for split, names in missing.items():
        if names:
            print(f"  WARNING: {split} is missing {names}")

    meta = {
        "threshold": args.threshold,
        "groups": groups,
        "assignment": assignment,
        "counts": {k: int(v) for k, v in counts.items()},
    }
    (args.dataset / "grouping.json").write_text(json.dumps(meta, indent=2))
    print("\nwrote {train,val,test}.txt, random_*.txt, grouping.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
