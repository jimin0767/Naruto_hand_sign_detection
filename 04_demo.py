"""Live webcam demo: detect hand signs, smooth them, match jutsu sequences.

Layout: camera panel on top, a strip of recognised sign cards below it (kanji large,
English beneath), a countdown ring top-right, and a jutsu reveal that replaces the strip
when a sequence completes.

The countdown deliberately does not run while a sign is being held -- it measures "have
you stopped casting", not "how long since the last sign". Holding one sign steady for
longer than the timeout must not wipe the sequence.

Usage:
    python 04_demo.py                          # default webcam
    python 04_demo.py --source 1               # second camera
    python 04_demo.py --source clip.mp4        # a video file
    python 04_demo.py --record out.mp4         # save what you demo
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2

from handsign import (
    DEFAULT_ACCEPT_CONF,
    DEFAULT_CONF,
    HandSignDetector,
    HandSignError,
    SequenceTracker,
    SignSmoother,
    load_jutsu,
)
from handsign.effects import (
    AnchorTracker,
    LightningEffect,
    TransformEffect,
    TransformSpec,
    effect_for,
    load_sprite,
)
from handsign.ui import Card, DemoUI, UIState

DemoError = HandSignError

JUTSU_HOLD_S = 3.5          # how long the jutsu name replaces the card strip


def open_source(source: str) -> cv2.VideoCapture:
    """Open a webcam index or a video file path."""
    if source.isdigit():
        # CAP_DSHOW avoids a multi-second open delay on Windows' default backend.
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    else:
        if not Path(source).exists():
            raise DemoError(f"source {source!r} does not exist")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise DemoError(f"could not open source {source!r}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("runs/handsign/yolo11m_disjoint/weights/best.pt"))
    parser.add_argument("--jutsu", type=Path, default=Path("jutsu.csv"))
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--window", type=int, default=25,
                        help="smoothing window in frames; at ~78 FPS, 25 is about 0.3s")
    parser.add_argument("--min-votes", type=int, default=18)
    parser.add_argument(
        "--accept-conf", type=float, default=DEFAULT_ACCEPT_CONF,
        help="a frame only votes toward a sign above this confidence, separate from "
             "--conf which controls what is drawn. Every training image contains exactly "
             "one sign, so the model has no 'not a sign' output and will name one for "
             "idle hands -- but typically with lower confidence than a real sign.",
    )
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="seconds after releasing a sign before the sequence resets")
    parser.add_argument("--width", type=int, default=960, help="camera panel width")
    parser.add_argument("--effect-seconds", type=float, default=None,
                        help="override how long a jutsu effect lasts")
    parser.add_argument("--transform-image", type=Path, default=Path("assets/sakura.png"),
                        help="sprite for Transformation Jutsu; a flat background is keyed "
                             "automatically if the file has no alpha channel")
    parser.add_argument("--transform-scale", type=float, default=None,
                        help="sprite height relative to the detected person box; raise if "
                             "the sprite looks too small for you")
    parser.add_argument("--person-model", default="yolo11n.pt",
                        help="COCO model used to locate you during a sprite transform")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--headless", action="store_true",
                        help="skip the preview window; for CI, or recording on a machine "
                             "with no display")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"error: {args.weights} not found; run 03_train.py first", file=sys.stderr)
        return 1

    jutsu = load_jutsu(args.jutsu)
    print(f"loaded {len(jutsu)} jutsu from {args.jutsu}")

    detector = HandSignDetector(args.weights, conf=args.conf, imgsz=args.imgsz,
                                device=args.device)
    cap = open_source(args.source)
    smoother = SignSmoother(args.window, args.min_votes)
    tracker = SequenceTracker(jutsu, timeout_s=args.timeout)

    ok, probe = cap.read()
    if not ok:
        print("error: could not read a frame from the source", file=sys.stderr)
        return 1
    ph, pw = probe.shape[:2]
    panel_w = args.width
    panel_h = int(round(ph * panel_w / pw))
    ui = DemoUI(panel_w, panel_h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cards: list[Card] = []
    jutsu_name: str | None = None
    jutsu_element = ""
    jutsu_at = 0.0
    effect: LightningEffect | None = None
    effect_at = 0.0
    # Runs every frame, not just during an effect, so the anchor is already warm and
    # settled the instant a jutsu fires.
    anchor = AnchorTracker()
    # A second tracker: sprite transforms need the whole person, not the hands.
    person_anchor = AnchorTracker(snap_distance=260.0)
    person_model = None

    sprite = None
    if args.transform_image.exists():
        sprite = load_sprite(args.transform_image)
        print(f"loaded transform sprite {args.transform_image} {sprite.shape[1]}x{sprite.shape[0]}")
    else:
        print(f"note: {args.transform_image} not found -- Transformation Jutsu will show "
              f"the banner only")
    writer = None
    frame_times: deque[float] = deque(maxlen=30)
    print("running -- q or Esc to quit, r to reset the sequence")

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)          # selfie view

            started = time.perf_counter()
            hit = detector.detect(frame)
            now = time.time()

            voting = hit.name if hit and hit.confidence >= args.accept_conf else None
            confirmed = smoother.update(voting)

            if confirmed:
                # Consecutive duplicates never stack: no jutsu in the table repeats a sign
                # back-to-back, so a repeat is far more likely to be a re-detection than
                # a deliberate second cast.
                if not cards or cards[-1].sign != confirmed:
                    cards.append(Card(confirmed, now))
                    matched = tracker.add(confirmed, now)
                    if matched:
                        jutsu_name, jutsu_element, jutsu_at = (
                            matched.name, matched.english, now)
                        cards = []
                        spec = effect_for(matched.name)
                        if isinstance(spec, TransformSpec):
                            if sprite is None:
                                spec = None          # no asset, no effect
                            else:
                                if args.effect_seconds:
                                    spec = replace(spec, duration=args.effect_seconds)
                                if args.transform_scale:
                                    spec = replace(spec, scale=args.transform_scale)
                                if person_model is None:
                                    from ultralytics import YOLO
                                    person_model = YOLO(args.person_model)
                                person_anchor.reset()
                                effect, effect_at = TransformEffect(spec, sprite), now
                        elif spec:
                            if args.effect_seconds:
                                spec = replace(spec, duration=args.effect_seconds)
                            effect, effect_at = LightningEffect(spec), now
                        print(f"  *** {matched.name} ***"
                              + ("  [effect]" if spec else ""))
                    else:
                        print(f"  {confirmed:8s} [{' > '.join(tracker.buffer)}]")

            tracker.tick(now, holding=smoother.held is not None)
            if not tracker.buffer and cards:
                cards = []                          # the countdown expired
            if jutsu_name and now - jutsu_at > JUTSU_HOLD_S:
                jutsu_name = None

            frame_times.append(time.perf_counter() - started)
            state = UIState(
                box=hit.box if hit else None,
                label=hit.name if hit else None,
                confidence=hit.confidence if hit else 0.0,
                voting=voting is not None,
                held=smoother.held,
                stability=smoother.stability,
                cards=cards,
                time_left=tracker.time_left(now),
                timeout_s=args.timeout,
                jutsu_name=jutsu_name,
                jutsu_element=jutsu_element if jutsu_name else "",
                jutsu_age=now - jutsu_at,
                fps=len(frame_times) / max(sum(frame_times), 1e-6),
            )
            scale = panel_w / frame.shape[1]
            if state.box:
                state.box = tuple(v * scale for v in state.box)
            tracked = anchor.update(state.box, now)

            if effect is not None:
                if now - effect_at > effect.spec.duration:
                    effect = None
                else:
                    if isinstance(effect, TransformEffect):
                        # Person detection only runs while a sprite transform is on
                        # screen, so it costs nothing the rest of the time.
                        pr = person_model.predict(frame, conf=0.35, classes=[0],
                                                  verbose=False, device=args.device)[0]
                        pbox = None
                        if len(pr.boxes):
                            best = int(pr.boxes.conf.argmax())
                            pbox = tuple(float(v) * scale
                                         for v in pr.boxes.xyxy[best].tolist())
                        anchored = person_anchor.update(pbox, now)
                        if anchored is not None:
                            # Half the box HEIGHT: the sprite is scaled and top-aligned
                            # against the person's vertical extent.
                            anchored = (anchored[0], person_anchor.size[1] / 2)
                    else:
                        anchored = tracked
                    if anchored is not None:
                        state.effect = effect
                        state.effect_centre, state.effect_radius = anchored
                        state.effect_age = now - effect_at
            canvas = ui.render(frame, state, now)

            if args.record:
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)

            if not args.headless:
                cv2.imshow("NARUTO hand signs", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    smoother.reset()
                    tracker.buffer.clear()
                    cards, jutsu_name, effect = [], None, None
                    anchor.reset()
                    person_anchor.reset()
                    print("  -- reset")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.record}")
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
