# Attribution

This model is a derivative work of eight public datasets from
[Roboflow Universe](https://universe.roboflow.com). **Seven are licensed CC BY 4.0**, which
requires attribution when the work or any derivative — including trained model weights — is
redistributed. Keep this file with any copy of the weights.

## Datasets used in training

| Source | License | Link |
|---|---|---|
| `vgu` — naruto hand sign v3 | CC BY 4.0 | https://universe.roboflow.com/vgu-aeaes/naruto-hand-sign/dataset/3 |
| `wilsons` — Naruto Hand Detection v1 | CC BY 4.0 | https://universe.roboflow.com/wilsons-workspace-fa8q1/naruto-hand-detection/dataset/1 |
| `yylunxie` — Naruto hand sign v2 | CC BY 4.0 | https://universe.roboflow.com/yylunxie/naruto-hand-sign-p8toe/dataset/2 |
| `chayawat` — Naruto v2 | CC BY 4.0 | https://universe.roboflow.com/chayawats-workspace-z7lzz/naruto-lkanh/dataset/2 |
| `otani` — NARUTO IN v5 | CC BY 4.0 | https://universe.roboflow.com/otani-sbz1y/naruto-in/dataset/5 |
| `cs` — naruto hand sign detection v2 | CC BY 4.0 | https://universe.roboflow.com/cs-pw2ff/naruto-hand-sign-detection-z6qoa/dataset/2 |
| `marcs` — naruto-hand-seals v1 | CC BY 4.0 | https://universe.roboflow.com/marcs-workspace-gkr8r/naruto-hand-seals-g34t6/dataset/1 |
| `minsub` — naruto v8 | MIT | https://universe.roboflow.com/minsub-song-wt0yo/naruto-pthxg/dataset/8 |

Four further datasets were evaluated and **excluded** for label-quality reasons
(see the design spec, §1.3): `sworkspace`, `jannat`, `kasidit`, `handsigns`.

## Software

- **Ultralytics YOLO11** — **AGPL-3.0**. Note this is a copyleft licence: distributing a
  network-accessible service built on Ultralytics can trigger source-disclosure
  obligations. For closed-source or commercial use, Ultralytics sells a separate
  Enterprise licence. Verify your situation before shipping anything beyond an internal
  or educational project.
- PyTorch (BSD-3), OpenCV (Apache-2.0), NumPy (BSD-3), Pillow (HPND).

## Voice clips and character art

Neither is in this repository, and neither should be added to it. The jutsu clips built by
`06_voice.py` and the sprite used by Transformation Jutsu are extracted from the anime,
which is copyrighted by its rights holders; `assets/` is gitignored so they stay on the
machine that made them. Supply your own from media you own, for demonstration and
educational use.

Note that a clip cut with no `--start`/`--end`, or across the `all` range `--scan`
reports, carries whatever else is in that stretch of the scene — sound effects and
background score along with the line. That is the point of it, and it does not change the
above: the clip stays local either way.

## Trademark

NARUTO is a trademark of Masashi Kishimoto / Shueisha. This project is unaffiliated,
non-commercial, and educational. The name and the jutsu list refer to the source work;
they are not licensed from the rights holders.
