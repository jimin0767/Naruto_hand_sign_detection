"""Train YOLO11m on the merged hand-sign dataset and report held-out metrics.

Model sizing is deliberately moderate. With ~2,500 distinct visual moments across 8
recording setups, capacity is not the bottleneck -- a larger backbone overfits those
8 backgrounds harder while barely fitting in 8 GB. See the spec for the measured
VRAM/latency table behind the choice.

Augmentation is tuned for this dataset's actual scarcity: colour and lighting diversity,
not image count. Hence raised HSV ranges and moderate geometry.

Usage:
    python 03_train.py                              # subject-disjoint split
    python 03_train.py --split random               # for contrast; not the honest number
    python 03_train.py --no-fliplr                  # the planned ablation
    python 03_train.py --epochs 5 --name smoke      # quick sanity run
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sys
from pathlib import Path

import yaml

CANONICAL = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger",
]

# Augmentation rationale, per the spec:
#   hsv_s/hsv_v raised  -- skin tone and lighting are the scarcest diversity axes
#   scale 0.5           -- the user sits at an unknown distance from the webcam
#   degrees 10          -- signs are upright; large rotations are what made jannat unusable
#   mosaic + close_mosaic -- strongest regularizer available, but disabled for the final
#                            epochs so training ends on realistic single-subject framing
#   erasing 0.4         -- occlusion robustness
AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.5,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.0,          # a hand sign is never upside down
    "fliplr": 0.5,          # overridden by --no-fliplr; see module docstring of the spec
    "mosaic": 1.0,
    "mixup": 0.0,           # blends two signs into one frame; meaningless here
    "erasing": 0.4,
}


def resolve_cache_mode(requested: str, workers: int, start_method: str) -> tuple[str, str]:
    """Pick a safe image-cache mode, returning ``(mode, explanation)``.

    ``cache='ram'`` is only safe where workers are *forked*. Under ``spawn`` (Windows, and
    macOS by default) the dataset object is pickled into every worker, so an N-GB RAM cache
    becomes N GB *per worker*. With 8 workers and a 9.4 GB cache that is ~75 GB of demand:
    in practice the machine swaps until the workers die with MemoryError, leaving the parent
    alive and the GPU idle at 0%.

    Disk caching sidesteps it -- workers read shared .npy files instead of carrying copies.
    """
    if requested != "ram" or workers == 0 or start_method == "fork":
        return requested, ""
    return "disk", (
        f"cache='ram' is unsafe with workers={workers} under the '{start_method}' start "
        f"method: the cache is pickled into each worker rather than shared. "
        f"Falling back to cache='disk'. Use --workers 0 to force RAM caching."
    )


def build_split_yaml(dataset: Path, split: str, out: Path) -> Path:
    """Write a data.yaml pointing at either the disjoint or the random list files."""
    prefix = "" if split == "disjoint" else "random_"
    for name in ("train", "val", "test"):
        listing = dataset / f"{prefix}{name}.txt"
        if not listing.exists():
            raise FileNotFoundError(f"{listing} missing; run 02_split.py first")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset.resolve()).replace("\\", "/"),
                "train": f"{prefix}train.txt",
                "val": f"{prefix}val.txt",
                "test": f"{prefix}test.txt",
                "nc": len(CANONICAL),
                "names": CANONICAL,
            },
            sort_keys=False,
        )
    )
    return out


def summarize(metrics, class_names: list[str]) -> dict:
    """Pull per-class AP out of an Ultralytics results object."""
    per_class = {}
    # `ap_class_index` lists which classes actually appeared; a class absent from the
    # split has no AP rather than an AP of zero, and conflating those would mislead.
    for position, class_index in enumerate(metrics.box.ap_class_index):
        per_class[class_names[int(class_index)]] = {
            "AP50": round(float(metrics.box.ap50[position]), 4),
            "AP50_95": round(float(metrics.box.ap[position]), 4),
            "precision": round(float(metrics.box.p[position]), 4),
            "recall": round(float(metrics.box.r[position]), 4),
        }
    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class": per_class,
    }


def print_report(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"  mAP@0.5      {summary['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 {summary['mAP50_95']:.4f}   "
          f"(capped by the 0.75 annotator IoU floor -- see spec 1.2)")
    print(f"  precision    {summary['precision']:.4f}    recall {summary['recall']:.4f}")
    print(f"\n  {'class':9s}{'AP50':>8s}{'AP50-95':>9s}{'prec':>8s}{'recall':>8s}")
    for name in CANONICAL:
        row = summary["per_class"].get(name)
        if row is None:
            print(f"  {name:9s}{'absent':>8s}")
            continue
        print(f"  {name:9s}{row['AP50']:8.3f}{row['AP50_95']:9.3f}"
              f"{row['precision']:8.3f}{row['recall']:8.3f}")
    weakest = sorted(summary["per_class"].items(), key=lambda kv: kv[1]["AP50"])[:3]
    if weakest:
        print("\n  weakest classes: "
              + ", ".join(f"{n} ({v['AP50']:.3f})" for n, v in weakest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("01_dataset"))
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("disjoint", "random"), default="disjoint")
    parser.add_argument("--no-fliplr", action="store_true",
                        help="the planned ablation: disable horizontal flip")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache", choices=("ram", "disk", "none"), default="disk",
        help="cache decoded images; decoding dominates epoch time (248s vs ~90s). "
             "Defaults to disk because 'ram' is unsafe with spawned workers -- see "
             "resolve_cache_mode()",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if not (args.dataset / "manifest.csv").exists():
        print(f"error: {args.dataset} not built; run 01_build_dataset.py", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    run_name = args.name or f"{Path(args.model).stem}_{args.split}" + (
        "_noflip" if args.no_fliplr else ""
    )
    # Absolute: Ultralytics resolves a *relative* project against its own runs_dir
    # setting, which silently nests output at runs/detect/runs/handsign/.
    project = (Path.cwd() / "runs" / "handsign").resolve()
    data_yaml = build_split_yaml(
        args.dataset, args.split, project / f"data_{args.split}.yaml"
    )

    augmentation = dict(AUGMENTATION)
    if args.no_fliplr:
        augmentation["fliplr"] = 0.0

    cache_mode, cache_note = resolve_cache_mode(
        args.cache, args.workers, multiprocessing.get_start_method()
    )
    if cache_note:
        print(f"NOTE: {cache_note}")

    print(f"model {args.model}   split {args.split}   imgsz {args.imgsz}   "
          f"batch {args.batch}   epochs {args.epochs}")
    print(f"fliplr {augmentation['fliplr']}   cache {cache_mode}   "
          f"workers {args.workers}   run -> {project / run_name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=str(project),
        name=run_name,
        exist_ok=True,
        close_mosaic=min(15, max(1, args.epochs // 6)),
        cache=False if cache_mode == "none" else cache_mode,
        amp=True,
        plots=True,
        **augmentation,
    )

    run_dir = project / run_name
    best = run_dir / "weights" / "best.pt"
    print(f"\nbest weights: {best}")

    # Validation was used for checkpoint selection, so it is no longer an unbiased
    # estimate. Test is the number to quote.
    report = {}
    for split_name in ("val", "test"):
        metrics = YOLO(str(best)).val(
            data=str(data_yaml), split=split_name, imgsz=args.imgsz,
            batch=args.batch, device=args.device, project=str(project),
            name=f"{run_name}_{split_name}", exist_ok=True, plots=True, verbose=False,
        )
        report[split_name] = summarize(metrics, CANONICAL)
        print_report(f"=== {split_name.upper()} ({args.split} split)", report[split_name])

    report["config"] = {
        "model": args.model, "split": args.split, "imgsz": args.imgsz,
        "batch": args.batch, "epochs": args.epochs, "fliplr": augmentation["fliplr"],
        "seed": args.seed,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {run_dir / 'report.json'}")
    print(f"confusion matrix: {run_dir}_test/confusion_matrix_normalized.png")

    if args.split == "disjoint" and report["test"]["mAP50"] > 0.97:
        print("\nWARNING: test mAP@0.5 above 0.97 on a subject-disjoint split is "
              "implausible here. Suspect a leak before celebrating (spec 4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
