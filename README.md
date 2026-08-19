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
python 04_demo.py --no-voice                    # do not speak the jutsu name
```

Jutsu can shout their own name in the character's voice — supply the clips with
[`06_voice.py`](#voice).

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
| **Lightning** | Chidori's effect, tracking your hand for 5 s (`--effect-seconds` to change) |
| **Smoke, then a character** | Transformation Jutsu, tracking your whole body for 6 s |
| **A crowd of you** | Clone Jutsu, six copies mirroring your movement for 6 s |
| **Fire from your mouth** | Dragon Flame Jutsu, a jet of flame for 5 s, aimed by your head |
| **Water spheres** | Water Bullet Jutsu, a volley of four pressurised rounds |
| **An earthen dragon head** | Earth Dragon Bullet, lunging out along your aim |
| **Crescent wind blades** | Vacuum Wave, six expanding pressure waves |

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

### Effects

Chidori casts procedural lightning on your hand. Bolts are regenerated every frame from
midpoint displacement rather than replaying a sprite animation — noise that never repeats
reads as electricity, a looping clip reads as a sticker.

The effect follows your hand via `AnchorTracker`, which cannot use the raw detection box:
boxes jitter a few pixels every frame, vanish on frames the model misses, and occasionally
jump across the frame. Attached directly, an effect vibrates, freezes, and teleports.

So the smoothing factor **scales with how far the box moved** — small wobble is damped
hard, real movement is followed almost immediately. A single fixed factor cannot do both:
heavy enough to kill jitter lags roughly 47 px behind a fast hand, which is what "not
following well" looks like. Adaptive smoothing halves that to ~22 px. Brief dropouts coast
on the last velocity rather than freezing, and a jump beyond `snap_distance` is taken
immediately since that is the model relocating, not a hand moving.

This works at all *because* of the model's main weakness: it always emits a box whether or
not you are making a valid sign — useless for recognition, serviceable as a hand tracker.

**Transformation Jutsu** does a puff of smoke and then draws a character sprite standing
where you are, tracked with `yolo11n` person detection. The sprite *covers* you rather than
replacing your pixels, which is why no background inpainting is needed — and why the smoke
matters: it hides the instant of the swap, exactly as the source material does.

The image is **not in this repo** — character art is copyrighted, so supply your own:

```bash
# any forward-facing cutout, roughly 600-1000px tall
cp your_character.png assets/sakura.png
python 04_demo.py                       # or --transform-image path/to/other.png
```

A transparent PNG is ideal, but a flat white or checkerboard background is keyed
automatically by flood-filling inward from the border. Thresholding on "near white" is the
obvious approach and the wrong one — it punches holes through pale parts of the subject;
filling from the border only removes background actually connected to the edge. If the
sprite looks too large or small for you, adjust `--transform-scale` (default 1.3, meaning
sprite height = 1.3x your detected person box).

`assets/` is gitignored. Teammates supply their own copy.

**Clone Jutsu** segments you with `yolo11n-seg` and composites six copies in a receding
formation, each popping in with its own puff of smoke. The cutout is refreshed **every
frame**, so the clones mirror your movement — that live sync is what sells them as clones
rather than pasted stills. `--clones N` changes the count.

Two details that are not obvious. Clone spacing is a fraction of the **frame** width, not
of your cutout: with your arms outstretched the cutout is nearly frame-wide, so
cutout-relative spacing throws the far pairs off screen. And when you run off the bottom
of the frame your silhouette ends in a straight horizontal cut — invisible on you, since
it sits on the frame edge, but a shrunk and lifted clone drags that line into open picture
where it reads as a pasted rectangle, so the bottom of every cutout is dissolved into a
gradient.

**Dragon Flame Jutsu** breathes a jet of fire from your mouth and **aims it where you are
looking** — face the camera and it goes straight out, turn your head and it follows.

The mouth comes from the COCO pose model's nose keypoint, offset downward by your eye
separation so it holds at any distance. Yaw comes from where the nose sits *between* the
eyes: centred means facing the camera, and as you turn it slides toward the near eye while
the eyes crowd together. Ear visibility disambiguates a strong turn, where one ear
disappears entirely.

Fire is a particle system, not geometry — flame has no silhouette to draw. Each particle is
a puff of hot gas launched from the mouth that slows, rises as it cools, and swells as it
dissipates. They accumulate into a scalar **heat field** which is blurred once and mapped
through a temperature ramp, and that is what produces continuous flame instead of a pile of
overlapping orange circles. The field is built and coloured at half resolution, since the
blur discards that detail anyway; doing it full-size cost 3x the time for no visible gain.

The shape is a **narrow neck at the lips flaring into a billowing mass**, which is what
real fire-breathing looks like. That flare comes from a lateral `bloom` force that ramps
with particle age, not from spraying wide at the mouth — spraying wide just gives a uniform
cone. A scrolling multi-octave noise field carves the billows; without it the plume is the
right shape but reads as a glow rather than fire.

Every force scales by frame **width**, matching `reach`. Mixing width and height makes the
physics depend on the camera's aspect ratio — a 4:3 panel multiplies the scattering forces
by 1.33x against an unchanged jet length, and the plume blows apart into loose blobs.

Tune it to your framing:

```bash
python 04_demo.py --flame-reach 0.7      # longer jet
python 04_demo.py --flame-spread 0.4     # wider, fatter cone
python 04_demo.py --flame-size 0.09      # heavier, more opaque fire
python 04_demo.py --flame-bloom 2.2      # flares harder into a fireball
```

Additive light on a bright background washes out, so it reads far more strongly against a
darker wall. If it looks weak on camera, raise `--flame-size` before anything else.

If you raise `rate` or `life` in `FlameSpec`, keep `rate * life` under `max_particles`.
The cap discards the *oldest* particles, which are the far end of the jet, so exceeding it
silently shortens the flame no matter what `reach` says.

**Water Bullet Jutsu** fires a volley of four pressurised spheres, each shedding a spray
tail. The stagger between shots is what makes it read as rounds rather than one thrown
ball. Same heat-field technique as the flame, through a blue-to-white ramp.

**Earth Dragon Bullet** lunges a dragon head of packed earth out along your aim — the head
geometry is shared with the old fire dragon, but composited **opaquely** rather than
additively. That is the whole difference between earth and fire: additive blending makes
anything look like it emits light, so rock rendered that way reads as a glowing ghost.
Earth has to occlude.

**Vacuum Wave** launches six crescent blades of compressed air. Wind is invisible, so the
effect is its edge: thin bright arcs that thin and dim as they expand, with a faint wash
behind. Drawing it as a solid body would read as water or gas rather than pressure.

Add more in `handsign/effects.py` — `EFFECTS` maps a jutsu name to an `EffectSpec`
(procedural), `TransformSpec` (sprite), `CloneSpec` (segmentation), `FlameSpec`,
`WaterBulletSpec`, `EarthDragonSpec`, or `VacuumWaveSpec`. A name that does not match the
jutsu table is caught by tests.

**Testing effects:** composite onto a *fresh* frame each step, as the demo does. Reusing
one canvas accumulates every frame of an additive effect, which makes a modest plume look
like a blown-out beam — an artifact that cost real tuning time here.

`load_jutsu` rejects any sign the model cannot detect, and warns if a jutsu is
**unreachable** — a shorter jutsu completing partway through a longer one clears the
buffer and blocks it. Edit freely; the loader checks your work.

### Voice

A jutsu can shout its own name as it fires, in the character's actual voice. Speech
synthesis is the obvious alternative and it is the wrong one — a system voice reading
"Chidori" flatly undoes the moment the effect just built. So the demo plays a **clip you
cut out of the show**, and `06_voice.py` does the cutting:

```bash
# 1. find the line: prints the non-silent ranges, so no video editor is needed
python 06_voice.py --video Kakashi_chidori.mp4 --scan

   1.     1.42 ->    3.18 s  ( 1.76s)   --start 1.42 --end 3.18

# 2. cut it, using the range from step 1
python 06_voice.py --video Kakashi_chidori.mp4 --jutsu Chidori --start 1.42 --end 3.18

# 3. check what has a voice and what does not
python 06_voice.py --list
```

That writes `assets/voice/chidori.wav`, which `04_demo.py` picks up with no further
configuration — cast the sequence and Kakashi says the line. Add `--play` to hear the
result immediately, and `--gain 3` if it still sits too low against the room.

The clip is trimmed, downmixed to mono 44.1 kHz 16-bit PCM, and **peak-normalised** to
−1 dBFS. Levelling is not polish: anime is mixed with a lot of headroom, so a raw extract
plays back noticeably quieter than the room expects and the line disappears under
ambient noise. Normalising every clip also means they all land at the same level, which
matters as soon as there is more than one. Short fades top and tail the cut — a hard cut
mid-waveform pops.

Clips are matched to jutsu **by name**: `assets/voice/fireball_jutsu.wav` fires for
"Fireball Jutsu". `Chidori.WAV` and `Fireball Jutsu.wav` work too, since both sides are
slugified before comparison.

Playback needs an audio backend, and the demo takes the first one that exists:

| backend | notes |
|---|---|
| `sounddevice` | cross-platform, lowest latency, cleanest cut-off. `pip install sounddevice` |
| `simpleaudio` | cross-platform, WAV only |
| `winsound` | Windows stdlib — **no install**, which is why sound works out of the box |
| `ffplay` / `afplay` / `aplay` | last resort, spawns a process per cast |

Pin one with `--audio-backend sounddevice`, or mute everything with `--no-voice`.

Two things are deliberate. Clips are decoded **at startup**, not on the cast — reading a
file inside the frame loop costs a visible hitch on exactly the frame that matters. And
nothing about voice can raise into that loop: a missing clip, an unreadable file, or a
machine with no audio device at all leaves the demo running silently rather than dying
mid-presentation.

`assets/` is gitignored, and anime audio is copyrighted — the clips stay on your machine,
exactly like the transform sprite. `06_voice.py` needs ffmpeg on PATH, or
`pip install imageio-ffmpeg`, which ships its own copy.

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

`06_voice.py` is not part of that pipeline — it cuts jutsu voice clips out of a video and
can be run at any point.

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
