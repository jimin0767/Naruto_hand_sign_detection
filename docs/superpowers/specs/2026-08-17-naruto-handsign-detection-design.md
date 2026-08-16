# NARUTO Hand Sign Detection — Design

**Date:** 2026-08-17
**Status:** Approved, ready for implementation planning
**Reference project:** [Kazuhito00/NARUTO-HandSignDetection](https://github.com/Kazuhito00/NARUTO-HandSignDetection) (YOLOX-Nano, 416px, 14 classes)

## Goal

Rebuild the reference project using only public Roboflow Universe datasets, targeting a
live-webcam demo on an RTX 4070 Laptop (8 GB VRAM). The demo detects one of 12 Naruto
zodiac hand signs per frame and matches sign sequences against a jutsu table.

## Scope boundary

This spec covers dataset construction, training, evaluation, and the webcam demo. It does
**not** cover collecting new imagery (see Known Limitations).

---

## 1. Dataset audit findings

Twelve Roboflow Universe exports were audited in `00_data/`. Raw totals differ slightly
from the figures we started with: the directory holds **16,850 images and 16,326 boxes**
(the 16,326 figure is boxes, not images; the gap is 524 images carrying empty labels).

### 1.1 Class index incompatibility (blocking)

Sources use three mutually incompatible index spaces:

| Ordering | Sources | index 0 | index 10 |
|---|---|---|---|
| English alphabetical | vgu, wilsons, cs, minsub, sworkspace, yylunxie, kasidit, jannat | `bird` | `snake`/`serpent` |
| Romaji alphabetical | otani, chayawat | `hitsuji` (ram) | `uma` (horse) |
| Ad-hoc | handsigns (nc=2), marcs (nc=1) | `horse` / `tiger` | — |

Naive concatenation mislabels ~18% of boxes. A canonical remap table is required and is
specified in §2.1.

**Validation of the remap:** 813 image groups are byte-identical across sources. After
applying the remap, class labels agree on **100%** of them (0 conflicts). Had the romaji
mapping been wrong, `otani`/`chayawat` overlaps would have surfaced as conflicts.

### 1.2 Annotator noise floor

On those same cross-source duplicates — the *same photograph* annotated independently by
two different people — mean box IoU is **0.75** (median 0.76, <0.5 in ~1% of cases).

Consequence: mAP@0.5 is a meaningful metric here; mAP@0.75 and above measures annotator
disagreement more than model skill. We do not optimize for it.

### 1.3 Per-source assessment

| source | imgs | boxes | classes | verdict | rationale |
|---|---|---|---|---|---|
| vgu | 6,142 | 6,138 | 12 | **keep** | best source; 3–5 subjects, tight boxes, webcam domain; video frames → 2,978 after dedup |
| yylunxie | 1,200 | 1,200 | 12 | **keep** | clean, tight, perfectly balanced (100/class) |
| minsub | 240 | 240 | 12 | **keep** | highest per-image diversity (228 clusters / 240 images) |
| wilsons | 2,056 | 2,056 | 12 | **keep** | clean; 299 images shared with jannat |
| otani | 684 | 684 | 12 | **keep** | tight boxes; letterboxed with black bars, needs un-padding |
| chayawat | 1,200 | 1,200 | 12 | **keep** | correct labels, near-zero diversity (one subject/shirt/background) |
| cs | 376 | 376 | 12 | **keep** | close-up domain; 49% of boxes cover >50% of frame |
| marcs | 208 | 208 | 1 | **keep** | tiger only, loose boxes, 43 out-of-bounds |
| sworkspace | 1,863 | 1,343 | 12 | **DROP** | box geometry systematically wrong; 520 unlabeled positives |
| jannat | 1,716 | 1,716 | 12 | **DROP** | 90°/vertical-flip augmentation invalid for hand signs; 299 dupes of wilsons |
| kasidit | 1,050 | 1,050 | 12 | **DROP** | 500 of 1,050 are chayawat frames downscaled 512→256 |
| handsigns | 115 | 115 | 2 | **DROP** | 2 of 12 classes, ~12 distinct moments |

### 1.4 Evidence for dropping `sworkspace`

Box aspect and position, compared against healthy sources:

| source | w mean | h mean | w/h | y center | centers in top 40% |
|---|---|---|---|---|---|
| vgu | 0.353 | 0.447 | 0.86 | 0.574 | 9% |
| wilsons | 0.293 | 0.510 | 0.60 | 0.514 | 10% |
| yylunxie | 0.204 | 0.415 | 0.53 | 0.485 | 27% |
| **sworkspace** | 0.536 | 0.343 | **1.5–3.3** | **0.336** | **59%** |

Hand-sign boxes are taller than wide and sit mid-frame; sworkspace's are wide, flat, and
land on the subject's face. Three coordinate-transform hypotheses were tested
(as-is, letterbox un-pad, aspect rescale by W/H) — none recovers the hands. The error is
present in train, valid, *and* test, so it is baked into the annotations rather than being
an augmentation artifact. Additionally 520 of 1,863 images show a clear sign with an empty
label, which would train the model to suppress correct detections.

### 1.5 Pre-existing split contamination

The shipped Roboflow train/valid/test splits already leak. Count of byte-identical image
groups straddling splits *within* a single source:

kasidit 122, chayawat 97, wilsons 95, jannat 31, otani 17, sworkspace 11, handsigns 5,
yylunxie 4, vgu 3.

The shipped splits are therefore discarded and rebuilt (§2.2).

### 1.6 What remains

```
16,326 boxes claimed
12,102  after dropping the 4 bad sources
11,202  after duplicate removal (256-bit dHash)   ← working dataset
 6,757  under the aggressive 64-bit variant (--dedup-bits 64)
~2,500  distinct visual moments (perceptual clustering)
   ~15  distinct people across 8 recording setups
```

**Dedup sensitivity — revised during implementation.** The original plan deduplicated on a
64-bit dHash, yielding 6,757 images. Measurement showed a 256-bit hash drops only 900
images against 64-bit's 5,345; the 4,445-image gap is not duplicates but consecutive video
frames that collide at 8×8 resolution while genuinely differing at 16×16.

The justification for aggressive dedup also weakened once the split became source-disjoint:
duplicates *within* a source can no longer leak across splits, because the whole source
lands on one side. What remains is source over-weighting, which is better addressed with
sampling weights — tunable and reversible — than by permanently deleting 44% of the data.

Revised policy, implemented in `01_build_dataset.py`:
- **Cross-source** duplicates are removed unconditionally on the 256-bit hash. `cs` and
  `wilsons` share 56 frames; were they ever split apart, those frames would leak.
- **Within-source** duplicates are removed on a configurable hash, default 256-bit.
  `--dedup-bits 64` reproduces the original 6,757-image dataset.

A safety check confirmed dedup does not destroy signal: of 1,620 collision groups under the
64-bit hash, only 3 mix classes (1 under 256-bit).

**The binding constraint is background diversity (8 setups), not image count.** This drives
the split strategy (§2.2), the augmentation choices (§2.3), and the model sizing (§3).

---

## 2. Pipeline

### 2.1 Stage 0 — Build (`01_build_dataset.py`)

Canonical class order (English alphabetical, matching the majority of sources):

```
0 bird   1 boar   2 dog    3 dragon  4 hare   5 horse
6 monkey 7 ox     8 ram    9 rat    10 snake 11 tiger
```

Alias table maps every observed spelling to canonical: romaji (`tori`→bird, `i`→boar,
`inu`→dog, `tatsu`→dragon, `u`→hare, `uma`→horse, `saru`→monkey, `ushi`→ox,
`hitsuji`→ram, `ne`→rat, `mi`→snake, `tora`→tiger), `serpent`→snake, and the
`Hitsuji -ram-` / `Bird-` decorated forms.

Steps:
1. Read the 8 keeper sources; remap class indices through the alias table.
2. Drop duplicates per §1.6, keeping the highest-resolution copy.
3. Strip `otani`'s letterbox bars and re-normalize its box coordinates.
4. Clip out-of-bounds boxes to frame.
5. Write a flat image/label pool plus `data.yaml`.
6. Emit `manifest.csv`: one row per output image with source, original path, hash,
   dimensions, and classes — so provenance of any image is answerable later.

Splitting is deliberately *not* done here. Stage 0 answers "what data exists"; stage 1
answers "how is it partitioned". Keeping them apart means the split can be changed without
rebuilding 11,202 images.

**Verification performed.** The remap was confirmed from two independent directions:
chayawat's original `class_N` filenames form a clean 1:1 bijection with the canonical
classes, and chayawat (romaji ordering) agrees with kasidit (English ordering) on all 500
images the two share. Letterbox cropping and box renormalization were confirmed visually
across a sample of otani outputs.

**Fail loudly.** If an unrecognized class name appears, abort rather than silently
assigning a default. A silent miscategorization here is the exact failure mode this whole
stage exists to prevent.

### 2.2 Stage 1 — Split (`02_split.py`)

**Splitting by source is not sufficient — revised during implementation.** Once `01_build`
cropped otani's letterbox bars, otani was found to share a recording session with wilsons:
**44% of otani images have a near-duplicate in wilsons, and 69% of wilsons in otani** —
the same person, shirt, room, and desk, at a different crop. The overlap was invisible in
the raw exports because the black bars dominated otani's perceptual hash. Visual inspection
confirmed it is the same footage, not merely the same pose.

Sources are therefore linked by *content*, not by directory name: two sources are merged
when they share frames within Hamming distance 5 of a 64-bit dHash, and each connected
component is assigned to exactly one split. The threshold is deliberately loose —
over-linking makes the held-out estimate conservative, under-linking silently inflates it.

Discovered subject groups:

| group | sources | images |
|---|---|---|
| 1 | cs, otani, wilsons | 2,681 |
| 2 | vgu | 5,676 |
| 3 | yylunxie | 1,200 |
| 4 | chayawat | 1,197 |
| 5 | minsub | 240 |
| 6 | marcs | 208 |

Resulting split:

| split | sources | imgs | share | per-class |
|---|---|---|---|---|
| train | vgu, wilsons, otani, cs, marcs | 8,565 | 76.5% | 476–1,237 |
| val | chayawat, minsub | 1,437 | 12.8% | 119–120 |
| test | yylunxie | 1,200 | 10.7% | 100 |

Val and test are near-perfectly class-balanced, so per-class metrics are directly
comparable without reweighting. `marcs` is kept in train because a tiger-only source would
distort per-class metrics in a held-out split.

The script re-verifies the finished split independently of how groups were derived and
**refuses to write a leaking split** unless `--allow-leakage` is passed. Naming any member
of a group in `--val`/`--test` holds out the whole group, since its members are not
separable.

A random split is emitted alongside — **not for training**, but to quantify and present the
gap between the two, which demonstrates why the disjoint split is the honest number.

### 2.3 Stage 2 — Train (`03_train.py`)

YOLO11m, 640px, batch 12, bf16, COCO-pretrained.

| param | value | rationale |
|---|---|---|
| `hsv_s` / `hsv_v` | 0.7 / 0.5 | raised; skin tone and lighting are the scarcest diversity axes |
| `scale` | 0.5 | user sits at unknown distance |
| `degrees` | 10 | signs are upright; large rotations are what made jannat unusable |
| `mosaic` | 1.0 | strongest available regularizer for a diversity-starved set |
| `close_mosaic` | 15 | finish training on realistic single-subject framing |
| `erasing` | 0.4 | occlusion robustness |
| `fliplr` | 0.5 | see below |

**`fliplr` rationale.** Mirroring swaps which hand is on top — for `dog` and `monkey` that
is a genuine left-handed variation, not a different class, and no two of the 12 signs are
mirror images of each other. It also matches inference conditions, since webcam previews
are conventionally mirrored. Defaulted **on**, and designated the first ablation: if the
confusion matrix shows a specific mirror pair collapsing, it comes back off.

### 2.4 Stage 3 — Demo (`04_demo.py`)

Webcam capture → YOLO11m inference → temporal smoothing over a rolling window → sequence
matching against a jutsu table, restricted to jutsu expressible with the 12 available signs.

Temporal smoothing matters more than raw per-frame accuracy for perceived demo quality: a
model at 0.85 mAP that flickers between classes looks worse than the same model with a
5-frame majority vote.

---

## 3. Model selection

Measured on the target GPU (RTX 4070 Laptop, 8,188 MiB), 640px, bf16 autocast, fused fp16
eager inference at batch 1:

| model | params | bs8 | bs16 | FPS |
|---|---|---|---|---|
| yolo11s | 9.4M | 1.8G | 3.5G | 124 |
| **yolo11m** | **20.1M** | **3.6G** | **7.1G** | **101** |
| yolo11l | 25.3M | 4.7G | 9.1G ✗ | 56 |
| yolo11x | 56.9M | 7.0G | 13.6G ✗ | 63 |
| yolo12m | 20.1M | 5.1G | 9.9G ✗ | 78 |
| yolo26m | 21.8M | 4.1G | 8.0G | 94 |
| yolo26l | 26.2M | 5.1G | 10.0G ✗ | 64 |
| rtdetr-l | 32.8M | — | — | 48 |

✗ exceeds VRAM. Note Windows WDDM spills past 8 GB into system RAM rather than raising OOM,
so exceeding it degrades throughput ~10× instead of failing fast.

**Decision: YOLO11m @ 640, batch 12** (~5.3 GB by interpolation between the bs8 and bs16
rows, leaving headroom for the dataloader, EMA weights, and desktop compositing).

Rejected alternatives:
- **yolo11x / yolo11l** — capacity is not the bottleneck. With ~2,500 distinct visual
  moments across 8 backgrounds, a larger backbone overfits those backgrounds harder while
  barely fitting in memory.
- **rtdetr-l** — DETR-family models need substantially more data and longer schedules to
  converge; a poor fit at this data scale.
- **yolo26m** — a legitimate alternative (NMS-free, consistent latency, comparable VRAM),
  rejected only for being less battle-tested than YOLO11.
- **Two-stage detector→crop→classifier** — genuinely stronger against the background
  shortcut, but roughly double the work. Held as an upgrade path if per-class confusion
  (rather than localization) proves limiting.

---

## 4. Evaluation

- **Primary:** mAP@0.5 on the subject-disjoint test set.
- **Secondary:** 12×12 confusion matrix, to distinguish class confusion from localization failure.
- **Reported alongside:** the same metrics under a random split, to show the gap.
- **Not optimized:** mAP@0.75+, which is capped by the 0.75 annotator IoU floor (§1.2).

**Expected range: mAP@0.5 of 0.80–0.92.** A result above 0.97 should be treated as evidence
of a leak, not success — given the contamination already found in the shipped splits (§1.5),
this is a live risk.

---

## 5. Known limitations

1. **12 classes, not 14.** The reference project detects Mizunoe and Gassho (hand claps);
   no public source here covers them. Several jutsu sequences terminate in Gassho, so the
   demo's jutsu table must be restricted. Recording these two signs is roughly 30 minutes of
   webcam capture and is the single highest-value data addition available — deliberately out
   of scope for this spec.
2. **`chayawat` contributes 614 training images of one subject in one shirt.** Retained for
   now; if the confusion matrix shows it dominating, cap its contribution.
3. **`cs` is a close-up domain** (49% of boxes >50% of frame) that does not match webcam
   framing. Retained as hard-case variety, flagged as a candidate for removal.
4. **`marcs` is tiger-only**, contributing to tiger's over-representation (13.0% of boxes
   vs 6.2% for rat — 2.09× imbalance). Mild enough to leave uncorrected initially.
5. **Test set is a single subject** (`yylunxie`). Subject-disjoint and therefore honest, but
   narrow; a strong test score reflects generalization to *one* unseen person. Val covers
   two subjects (`chayawat`, `minsub`), so it is the more robust of the two for model
   selection despite being the smaller sample per subject.
6. **Subject grouping rests on a perceptual-hash threshold.** Two sources sharing a subject
   in visually *dissimilar* footage — different room, different lighting — would not link
   and would be split apart. The otani/wilsons case was caught only because the framing was
   near-identical. Treat held-out scores as an upper bound on true generalization.

---

## 6. Deliverables

```
01_build_dataset.py   merge + remap + dedup + manifest
02_split.py           subject-disjoint split (+ random split for comparison)
03_train.py           YOLO11m training
04_demo.py            webcam demo with temporal smoothing + jutsu matching
docs/                 this spec, audit outputs, contact sheets
```
