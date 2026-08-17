# Integrating the hand-sign model into a game

Everything an engine-side implementation needs. Read §1 even if you skim the rest — a
wrong class order produces confident, plausible, wrong answers rather than an error.

## 1. Class order — do not reorder

The model emits integer class indices. This mapping is the contract:

```
0 bird     1 boar     2 dog      3 dragon
4 hare     5 horse    6 monkey   7 ox
8 ram      9 rat     10 snake   11 tiger
```

English alphabetical. The twelve original datasets used **three mutually incompatible
index orders** — one keyed on Japanese romaji, where index 0 is *ram*, not *bird* — and
merging them naively mislabels ~18% of boxes. That failure is silent: it degrades accuracy
without raising anything. Hardcode the list above verbatim, or read it from the ONNX
metadata key `names`.

Romaji display names, if you want them in-game: Tori, I, Inu, Tatsu, U, Uma, Saru, Ushi,
Hitsuji, Ne, Mi, Tora.

## 2. ONNX contract

`best.onnx`, exported at opset 17, static batch.

| | |
|---|---|
| input | `images`, `float32[1, 3, 640, 640]` |
| output | `output0`, `float32[1, 16, 8400]` |

**Preprocessing** — letterbox, do not stretch:
1. Scale the frame so its longest side is 640, preserving aspect ratio.
2. Paste onto a 640×640 canvas filled with **114** grey, centred.
3. BGR → **RGB**, transpose HWC → CHW, divide by 255.0. No mean/std normalisation.

**Output decoding** — `16 = 4 bbox + 12 classes`, and `8400 = 80² + 40² + 20²` anchor
points. There is **no objectness row** (YOLOv8-style head).

- Rows `0..3` — `cx, cy, w, h` in **pixels of the 640×640 letterboxed frame**, not
  normalised, not relative to your original frame.
- Rows `4..15` — one score per class, each an **independent sigmoid** in `[0,1]`. They do
  **not** sum to 1; there is no softmax. Confidence for an anchor is the max over rows
  4..15, and the predicted class is its argmax.

Then: filter by confidence, run NMS (IoU 0.7), and undo the letterbox —
`x_orig = (x_640 - pad_left) / scale`.

Verified against the PyTorch weights on 60 held-out images: **60/60 identical top-1
classes**, max confidence delta 0.0002.

## 3. Thresholds

| setting | value | why |
|---|---|---|
| detection confidence | **0.25** | measured 92.1% top-1 vs 85.2% at 0.5, where 12% of frames detect nothing |
| commit confidence | **0.60** | separate, stricter gate before acting on a sign — see §5 |
| NMS IoU | 0.7 | Ultralytics default |
| input size | 640 | as trained; other sizes degrade accuracy |

## 4. Use the top-1 box, not all boxes

Every training image contains exactly one sign, so treat this as a localiser plus a 12-way
classifier: **take the single highest-confidence box per frame and ignore the rest.** The
model emits low-confidence duplicates alongside the correct answer, and scoring all of
them produces a far worse — and misleading — accuracy picture.

## 5. Smooth over time, or it will flicker

Per-frame top-1 is 92% on an unseen person, so roughly **one frame in twelve names the
wrong sign**. Acting on raw per-frame output will visibly misfire. The errors are
structured, not random: `hare` reads as `monkey` about 20% of the time, because both are
"one hand stacked over the other" and differ only in finger detail that motion blur
erases.

Two rules, both necessary:

- **Debounce.** Only commit a sign after it holds the majority of a rolling window.
  Size the window against your frame rate — 25 frames is ~0.3 s at 78 FPS, but only
  0.42 s at 60 FPS and 0.83 s at 30 FPS. Tune in *seconds*, not frames.
- **De-duplicate.** A sign held for one second is ~30 frames and must register **once**.
  Track a "currently held" sign; emit only when the window agrees on something
  *different*, and clear the hold when the window agrees the hands are down. Clearing on
  hands-down is what lets the same sign be performed twice in a row.

Reference implementation: `handsign/smoothing.py` (`SignSmoother`, `SequenceTracker`).
It is pure Python with no torch or ultralytics dependency, so you can port it directly or
drive it from your own backend.

## 6. Known behaviour: it always predicts something

**All 11,202 training images contain exactly one hand sign.** No public source provides
negatives, so the model has no "not a sign" output and will name one of twelve for idle
hands, a wave, or a coffee cup.

Mitigations, in increasing order of effectiveness:
1. The 0.60 commit threshold (§3) — idle-hand guesses are usually less confident.
2. Debouncing (§5) — false positives are unstable, real signs are not.
3. **Retrain with background images.** The real fix: a few hundred frames with empty label
   files teaches the missing concept. Not yet done.

Design around this. Gate recognition behind an explicit "start casting" input rather than
running it open-loop, and the problem largely disappears.

## 7. Performance

12.5 ms inference at 640 on an RTX 4070 Laptop (~80 FPS end-to-end including capture and
overlay). Budget accordingly — you likely do **not** need to run this every frame. Every
second or third frame is usually plenty, since the debounce window spans several frames
anyway, and it frees GPU time for rendering.

CPU inference via onnxruntime works but is much slower; measure before relying on it.

## 8. Accuracy, honestly

| | held-out subject A | held-out subject B |
|---|---|---|
| top-1 @ conf 0.25 | 92.1% | 98.1% |
| mAP@0.5 | 0.963 | 0.987 |

Two unseen people, and they differ by 6 points of top-1 — expect variance across players.
Weakest class is `hare` at 67% on subject A. `mAP` figures are included because they are
conventional, but **top-1 is the number that predicts in-game behaviour**; mAP integrates
over confidence thresholds you will never actually use.

Trained on ~15 people across 8 recording setups, so lighting and skin tone coverage is
narrow. If in-game accuracy disappoints for a particular player, the fix is more training
subjects, not more tuning.
