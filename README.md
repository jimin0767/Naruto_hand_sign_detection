# NARUTO Hand Sign Detection

Real-time detection of the 12 Naruto zodiac hand signs from a webcam, with temporal
smoothing and jutsu sequence matching. A rebuild of
[Kazuhito00/NARUTO-HandSignDetection](https://github.com/Kazuhito00/NARUTO-HandSignDetection)
using only public datasets.

**YOLO11m @ 640** · **92.1% top-1** on an unseen person · **~80 FPS** on an RTX 4070 Laptop

> **Integrating this into a game or another app? Read [INTEGRATION.md](INTEGRATION.md).**
> It covers the ONNX contract, the class-index order, and why you must smooth over time.

---

## Quick start

```bash
pip install -r requirements.txt
python 04_demo.py
```

`q` or `Esc` quits, `r` resets the sign sequence. Useful flags:

```bash
python 04_demo.py --source 1              # a different camera
python 04_demo.py --source clip.mp4       # a video file
python 04_demo.py --accept-conf 0.75      # stricter; fewer false positives
python 04_demo.py --record out.mp4        # save the session
```

Weights are **not** in this repo — download `best.pt` (and `best.onnx`) from the
[Releases](../../releases) page and place them at
`runs/handsign/yolo11m_disjoint/weights/`.

## Using it from Python

```python
import time
from handsign import HandSignDetector, SignSmoother, SequenceTracker, load_jutsu

detector = HandSignDetector("best.pt")
smoother = SignSmoother()                       # 25-frame window, 18 votes
tracker  = SequenceTracker(load_jutsu("jutsu.yaml"))

detection = detector.detect(frame)              # BGR numpy array
voting = detection.name if detection and detection.confidence >= 0.60 else None
if (sign := smoother.update(voting)):
    if (jutsu := tracker.add(sign, time.time())):
        print("cast", jutsu.name)
tracker.tick(time.time())
```

`handsign.smoothing` has no torch or ultralytics dependency, so it can be driven from any
inference backend — or ported to another language.

## The 12 classes

```
0 bird   1 boar   2 dog    3 dragon   4 hare    5 horse
6 monkey 7 ox     8 ram    9 rat     10 snake  11 tiger
```

This order is load-bearing. It is defined once in `handsign/classes.py` and imported
everywhere; see [INTEGRATION.md §1](INTEGRATION.md) for why.

The reference project also detects **Gassho** (hand clap) and **Mizunoe** — no public
dataset covers them, so jutsu requiring those signs cannot be triggered here.

## Results

| | val (2 unseen people) | test (1 unseen person) |
|---|---|---|
| **top-1 @ conf 0.25** | **92.1%** | **98.1%** |
| mAP@0.5 | 0.9634 | 0.9872 |
| mAP@0.5:0.95 | 0.5428 | 0.6085 |

Evaluated on **subject-disjoint** splits — no person appears in both training and
evaluation. A random split scores considerably higher and means nothing.

Weakest class is `hare` (67% on val), most often confused with `monkey`.
mAP@0.5:0.95 is capped by label noise: two annotators labelling the *same photograph*
agree only to 0.75 IoU.

## Rebuilding from scratch

Needs the 12 Roboflow exports in `00_data/<source>/`, ~13 GB of disk, and a CUDA GPU.

```bash
python 01_build_dataset.py     # merge 8 sources, remap classes, dedup  -> 11,202 images
python 02_split.py             # content-derived subject-disjoint splits
python 03_train.py             # YOLO11m, 640, batch 12                 -> ~3.5 h
python 05_export.py            # ONNX for game engines
```

Each stage prints what it did and refuses to proceed on bad input. `01` aborts on an
unrecognised class name rather than guessing; `02` refuses to write a split that leaks.

```bash
python -m pytest tests/ -q     # 186 tests
```

## How the dataset was built

12 Roboflow sources, 16,850 images → **11,202** usable. Four sources were dropped:

| dropped | why |
|---|---|
| `sworkspace` | box geometry systematically wrong; 520 images show a sign but carry an empty label |
| `jannat` | 90°/vertical-flip augmentation, invalid for hand signs |
| `kasidit` | 500 of 1,050 images are `chayawat` frames downscaled 512→256 |
| `handsigns` | 2 of 12 classes, ~12 distinct moments |

Two findings shaped everything downstream:

- **The sources disagree on class indices** — three incompatible orderings, one keyed on
  Japanese romaji. A naive merge mislabels ~18% of boxes, silently.
- **Splitting by source is not enough.** `otani` and `wilsons` turned out to be the *same
  recording session* at different crops — 44% of `otani` has a near-duplicate in
  `wilsons`. Splits are therefore derived from image content, not directory names.

Full analysis: [`docs/superpowers/specs/`](docs/superpowers/specs/).

## Known limitations

- **It always predicts something.** Every training image contains exactly one sign, so the
  model has no "not a sign" output and will name one for idle hands. Mitigated by the 0.60
  commit threshold and debouncing; properly fixed by retraining with background images.
- **12 classes, not 14** — no Gassho or Mizunoe.
- **~15 people, 8 recording setups.** Narrow coverage of lighting and skin tone.
- **Jutsu sequences in `jutsu.yaml` are unverified.** Manga, anime, and databooks disagree.
  Treat the file as a starting point you own.

## Licence and attribution

Trained on datasets that are **CC BY 4.0** (7 of 8), which requires attribution when
redistributing the weights. See [ATTRIBUTION.md](ATTRIBUTION.md) — keep it with any copy
of the model.

Ultralytics YOLO11 is **AGPL-3.0**; check your obligations before shipping commercially.
NARUTO is a trademark of Masashi Kishimoto / Shueisha; this project is unaffiliated and
educational.
