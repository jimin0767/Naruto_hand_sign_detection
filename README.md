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
python 04_demo.py --source 1                    # a different camera
python 04_demo.py --source clip.mp4             # a video file
python 04_demo.py --jutsu demo-short.csv        # 7 short jutsu, easier to present
python 04_demo.py --accept-conf 0.75            # stricter; fewer false positives
python 04_demo.py --record out.mp4              # save the session
```

Weights are **not** in this repo — download `best.pt` (and `best.onnx`) from the
[Releases](../../releases) page and place them at
`runs/handsign/yolo11m_disjoint/weights/`.

### Reading the screen

| element | meaning |
|---|---|
| **Green corner brackets** | a detection that counts toward a sign |
| **Grey brackets, "ignored"** | detected but below `--accept-conf`, so it does not vote |
| **Panel, top left** | the sign currently being held, with a bar showing how close it is to committing |
| **Ring, top right** | countdown to sequence reset. Appears **only after you release a sign** — holding one steady never expires it |
| **Card strip, bottom** | signs recognised so far: kanji above, English below. The same sign twice in a row does not stack |
| **Orange banner** | a completed jutsu; it replaces the card strip for a few seconds |

**If a sign will not commit,** watch the brackets and the stability bar. Grey brackets mean
the model sees the sign but is not confident enough to count it — the bar stalls partway
and nothing registers. Lower the gate for that sign rather than assuming the model is
blind to it:

```bash
python 04_demo.py --accept-conf 0.50
```

Confidence varies by class and by person. During testing `ox` swung between 0.31 and 0.82
on the same held pose, which was enough to stall it at the default 0.60.

## Using it from Python

```python
import time
from handsign import HandSignDetector, SignSmoother, SequenceTracker, load_jutsu

detector = HandSignDetector("best.pt")
smoother = SignSmoother()                       # 25-frame window, 18 votes
tracker  = SequenceTracker(load_jutsu("jutsu.csv"))

detection = detector.detect(frame)              # BGR numpy array
voting = detection.name if detection and detection.confidence >= 0.60 else None
if (sign := smoother.update(voting)):
    if (jutsu := tracker.add(sign, time.time())):
        print("cast", jutsu.name)
tracker.tick(time.time(), holding=smoother.held is not None)
```

Pass `holding` to `tick` so the reset countdown pauses while a sign is being held —
otherwise a long hold silently wipes the sequence mid-cast.

`handsign.smoothing` has no torch or ultralytics dependency, so it can be driven from any
inference backend — or ported to another language. `handsign.ui` is the demo's rendering
layer and is not needed to consume the model.

## The 12 classes

```
0 bird   1 boar   2 dog    3 dragon   4 hare    5 horse
6 monkey 7 ox     8 ram    9 rat     10 snake  11 tiger
```

This order is load-bearing. It is defined once in `handsign/classes.py` and imported
everywhere; see [INTEGRATION.md §1](INTEGRATION.md) for why.

The reference project also detects **Gassho** (hand clap) and **Mizunoe** — no public
dataset covers them, so jutsu requiring those signs cannot be triggered here. The same
applies to the **Clone Seal** (crossed fingers), which is not one of the twelve: Shadow
Clone and Water Shark Bullet are therefore not representable.

## Jutsu

Two tables ship. `jutsu.csv` holds all 19 sequences; `demo-short.csv` is a curated seven
— one per element plus the two general techniques, none longer than four signs — which is
easier to perform live. Pick one with `--jutsu`.

Both use the reference project's column layout, in English:

```
element,jutsu,sign1,sign2,...
Fire Style,Fireball Jutsu,snake,ram,monkey,boar,horse,tiger
```

Sequences come from [Narutopedia](https://naruto.fandom.com) infoboxes. Entries are
restricted to 3–7 signs: one- and two-sign jutsu fire by accident during other sequences,
and the longest canonical jutsu (Water Dragon, 44 signs) is impractical to perform.

`load_jutsu` rejects any sign the model cannot detect, and warns if a jutsu is
**unreachable** — a shorter jutsu completing partway through a longer one clears the
buffer and blocks it. Edit freely; the loader checks your work.

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
python -m pytest tests/ -q     # 253 tests
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
- **Jutsu sequences may differ from your preferred source.** `jutsu.csv` follows
  [Narutopedia](https://naruto.fandom.com) infobox data, which disagrees with the
  reference project on two entries (see below). Manga, anime, and databooks are not
  consistent; treat the file as a starting point you own.

## Licence and attribution

Trained on datasets that are **CC BY 4.0** (7 of 8), which requires attribution when
redistributing the weights. See [ATTRIBUTION.md](ATTRIBUTION.md) — keep it with any copy
of the model.

Ultralytics YOLO11 is **AGPL-3.0**; check your obligations before shipping commercially.
NARUTO is a trademark of Masashi Kishimoto / Shueisha; this project is unaffiliated and
educational.
