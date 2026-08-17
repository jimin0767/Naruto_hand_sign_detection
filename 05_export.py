"""Export trained weights to ONNX for game engines, and verify the export agrees.

An export that silently disagrees with the source model is worse than no export, so this
does not just call `.export()` -- it re-runs both models over held-out images and compares
top-1 predictions. A mismatch here would show up in-game as a model that "sometimes"
misreads signs, which is nearly impossible to debug from the engine side.

ONNX targets Unity Sentis, Unreal NNE, onnxruntime (C#/C++/Python), and the web. For
TensorRT, CoreML, or TFLite, pass --format; the same verification runs either way.

Usage:
    python 05_export.py
    python 05_export.py --format engine        # TensorRT, needs the same GPU at runtime
    python 05_export.py --no-verify
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from handsign.classes import CANONICAL

DEFAULT_WEIGHTS = Path("runs/handsign/yolo11m_disjoint/weights/best.pt")


def verify_parity(pt_path: Path, exported: Path, dataset: Path, n: int, device: str) -> bool:
    """Compare top-1 predictions between the source and exported models."""
    from ultralytics import YOLO

    manifest = dataset / "manifest.csv"
    if not manifest.exists():
        print(f"  skipping verification: {manifest} not found")
        return True

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    # Held-out subject only -- verifying on training images would hide a real regression
    # behind memorised examples.
    held_out = [r for r in rows if r["source"] == "yylunxie"] or rows
    random.seed(0)
    sample = [str(dataset / r["image"]) for r in random.sample(held_out, min(n, len(held_out)))]

    source_model, exported_model = YOLO(str(pt_path)), YOLO(str(exported))
    agree = 0
    worst = 0.0
    for path in sample:
        a = source_model.predict(path, conf=0.25, imgsz=640, device=device, verbose=False)[0].boxes
        b = exported_model.predict(path, conf=0.25, imgsz=640, verbose=False)[0].boxes
        ca = int(a.cls[int(a.conf.argmax())]) if len(a) else -1
        cb = int(b.cls[int(b.conf.argmax())]) if len(b) else -1
        agree += ca == cb
        if len(a) and len(b):
            worst = max(worst, abs(float(a.conf.max()) - float(b.conf.max())))

    print(f"  top-1 agreement: {agree}/{len(sample)}   max confidence delta: {worst:.4f}")
    return agree == len(sample)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--format", default="onnx",
                        help="onnx (default), engine, coreml, tflite, torchscript ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dataset", type=Path, default=Path("01_dataset"))
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"error: {args.weights} not found; run 03_train.py first", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    names = getattr(model, "names", None)
    if names:
        ordered = tuple(names[i] for i in sorted(names))
        if ordered != CANONICAL:
            print(f"error: weights declare {ordered}, expected {CANONICAL}", file=sys.stderr)
            return 1
        print(f"class order verified: {list(CANONICAL)}")

    print(f"exporting {args.weights.name} -> {args.format} at {args.imgsz}px ...")
    kwargs = dict(format=args.format, imgsz=args.imgsz, dynamic=False)
    if args.format == "onnx":
        kwargs.update(opset=args.opset, simplify=True)
    exported = Path(model.export(**kwargs))
    print(f"  wrote {exported}  ({exported.stat().st_size / 2**20:.1f} MB)")

    if args.no_verify:
        print("  --no-verify: skipping parity check")
        return 0

    print(f"\nverifying against {args.weights.name} on held-out images ...")
    if not verify_parity(args.weights, exported, args.dataset, args.samples, args.device):
        print("\nFAIL: exported model disagrees with the source weights.", file=sys.stderr)
        print("Do not ship this export -- see INTEGRATION.md section 2.", file=sys.stderr)
        return 1
    print("\nOK: export matches the source model.")
    print("Integration contract: INTEGRATION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
