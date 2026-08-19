"""Cut a jutsu's sound out of a video and install it as a voice clip.

The demo plays a jutsu's sound the moment it fires (see `handsign/voice.py`). This builds
the clips it plays: point it at an episode clip, name the jutsu, and it extracts the
audio, trims it, levels it, and writes `assets/voice/<jutsu>.wav` -- the exact name and
format the demo looks for.

**The whole scene, not just the line.** A shout on its own is the thinner half of what the
show plays: Kakashi says "Chidori" *over* the crackle of it charging, and the crackle is
most of why it lands. So the default is to take everything in the range you give -- voice,
effect, and score together -- and cutting a bare voice line is the special case, not this.
Give no `--start`/`--end` at all and an already-trimmed clip is used end to end.

Two settings exist for that case in particular. `--stereo` keeps the source's stereo image
instead of folding wide-panned effects into mono, where they partly cancel. And `--level
rms` levels by average energy rather than by the single loudest sample, so a three-second
scene and a one-second shout end up at the same loudness rather than merely the same
ceiling -- with `--limit` on top if the effect towers far enough over the line that the
ceiling alone cannot be got past.

Levelling at all is not optional polish. Anime is mixed with a lot of headroom, so a raw
extract plays back noticeably quieter than the room expects and the line gets lost under
the demo's own noise; every clip is levelled so they all land in the same place.

Typical run -- an already-trimmed clip of the moment, taken whole:

    python 06_voice.py --video Kakashi_chidori.mp4 --jutsu Chidori --stereo --play

From something longer, find the scene first. Watch for it coming back as several rows:
one row is the voice on its own, and the `all` row is the scene.

    python 06_voice.py --video episode.mp4 --scan
    python 06_voice.py --video episode.mp4 --jutsu Chidori --start 1.42 --end 3.18 --stereo
    python 06_voice.py --list

Needs ffmpeg. Either put it on PATH (`winget install Gyan.FFmpeg`, `brew install
ffmpeg`, `apt install ffmpeg`) or `pip install imageio-ffmpeg`, which ships a private
copy this script will find on its own.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

from handsign import load_jutsu
from handsign.voice import DEFAULT_VOICE_DIR, EXTENSIONS, VoicePlayer, find_clip, slugify

# 16-bit PCM at 44.1 kHz. Not a preference: winsound and simpleaudio read plain PCM WAV
# and nothing else, and those are the two backends that work without an install.
SAMPLE_RATE = 44100
PEAK_DBFS = -1.0            # leave a hair of headroom so nothing clips on playback
RMS_DBFS = -20.0            # target average energy under `--level rms`
FADE_IN_S = 0.015           # a hard cut into a waveform mid-cycle pops audibly
FADE_OUT_S = 0.060

# Peak this far above the average means one loud moment towers over the rest of the clip
# -- an effect hit, usually. Speech alone sits around 13 dB and a mixed scene 15-18, so
# this is set just past where normal content stops reaching.
WIDE_CREST_DB = 18.0


class VoiceBuildError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    """ffmpeg from PATH, else the copy imageio-ffmpeg bundles."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise VoiceBuildError(
            "ffmpeg not found. Install it (winget install Gyan.FFmpeg / brew install "
            "ffmpeg / apt install ffmpeg) or run: pip install imageio-ffmpeg"
        )


def run_ffmpeg(ffmpeg: str, args: list[str]) -> str:
    """Run ffmpeg and return its stderr, which is where it reports everything."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", *args],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        raise VoiceBuildError(f"ffmpeg failed:\n{tail}")
    return result.stderr


def parse_time(value: str) -> float:
    """Seconds from "12.5", "1:03", or "0:01:03.25"."""
    parts = str(value).split(":")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad timestamp {value!r}; try 12.5 or 1:03.5")
    return seconds


def trim_args(start: float | None, end: float | None) -> list[str]:
    """Input-side seek arguments.

    `-ss` goes *before* `-i` so ffmpeg seeks instead of decoding the skipped audio, and
    the length is given as `-t` rather than `-to`: with an input-side seek, `-to` is
    measured from the seek point, which quietly produces a clip of the wrong length.
    """
    args = []
    if start:
        args += ["-ss", f"{start:.3f}"]
    if end is not None:
        span = end - (start or 0.0)
        if span <= 0:
            raise VoiceBuildError(f"--end {end} is not after --start {start or 0}")
        args += ["-t", f"{span:.3f}"]
    return args


def measure_levels(ffmpeg: str, source: Path, start: float | None, end: float | None,
                   channels: int = 1) -> tuple[float, float]:
    """(peak, mean) level of the selected range in dBFS, where 0 is full scale.

    Two numbers rather than one, because they disagree in exactly the case this tool
    exists for. A cut holding nothing but the shout has its peak a predictable ~13 dB
    above its average, so levelling against either lands in the same place. A cut that
    keeps the scene -- the Chidori crackling under the line -- peaks on the effect, and
    matching *that* to full scale is what leaves the voice sounding buried.

    Measured through the same channel layout the output will be written with, which is not
    a detail: a wide-panned effect loses level when it is folded to mono, sometimes all of
    it. Measuring the stereo source and then applying that gain to a mono downmix lands the
    clip as far below target as the fold cost it -- silently, since every number printed
    still looks right.

    The fold has to happen *in the filter chain*, via aformat, rather than as the `-ac` on
    the output: `-ac` is applied after the graph, so volumedetect placed behind it would go
    on measuring the untouched source and the bug would survive its own fix.
    """
    layout = "mono" if channels == 1 else "stereo"
    stderr = run_ffmpeg(ffmpeg, [
        *trim_args(start, end), "-i", str(source), "-vn",
        "-af", f"aformat=channel_layouts={layout},volumedetect", "-f", "null", "-",
    ])
    peak = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", stderr)
    mean = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", stderr)
    if not peak:
        raise VoiceBuildError(
            f"no audio track found in {source.name} -- check the clip actually has sound"
        )
    # volumedetect always reports both; treating a missing mean as the peak simply makes
    # rms levelling behave like peak levelling rather than crashing on a surprise.
    return float(peak.group(1)), float(mean.group(1)) if mean else float(peak.group(1))


def measure_peak(ffmpeg: str, source: Path, start: float | None, end: float | None,
                 channels: int = 1) -> float:
    """Loudest sample in the selected range, in dBFS."""
    return measure_levels(ffmpeg, source, start, end, channels)[0]


def level_gain(peak: float, mean: float, mode: str, peak_target: float,
               rms_target: float, limit: bool = False) -> float:
    """dB to apply so the clip lands where `mode` says it should.

    `peak` puts the loudest sample at `peak_target` and changes nothing else, so the mix
    survives exactly as the show made it -- the voice sits against the effect at the ratio
    it was dubbed at. That is the right default, and for a clip that is only a voice it is
    also the whole story.

    `rms` aims the *average* at `rms_target` instead. Across a folder of clips that is what
    makes them sound equally loud, since a three-second scene and a one-second shout
    normalised to the same peak are not the same loudness at all. The ceiling still wins:
    without `limit` the gain is clamped so the loudest sample stays under `peak_target`,
    which on a clip whose effect towers over the line leaves rms doing exactly what peak
    did. `limit` is the way out of that -- see `limiter_filter`.
    """
    if mode == "none":
        return 0.0
    if mode == "peak":
        return peak_target - peak
    wanted = rms_target - mean
    return wanted if limit else min(wanted, peak_target - peak)


def limiter_filter(ceiling_dbfs: float) -> str:
    """A lookahead limiter holding the output under `ceiling_dbfs`.

    What it buys: with the ceiling enforced here rather than by shrinking the gain, the
    quiet nine tenths of a clip can come up to a usable level while the one loud hit is
    held down instead of dragging everything else with it. What it costs: that hit is no
    longer at the ratio the show mixed it at. Hence opt-in.

    `level=0` matters -- alimiter auto-levels its output by default, which would undo the
    gain that was just carefully chosen.
    """
    return (f"alimiter=limit={10 ** (ceiling_dbfs / 20):.4f}"
            f":attack=5:release=50:level=0")


def scan_segments(ffmpeg: str, source: Path, noise_db: float = -32.0,
                  min_silence: float = 0.25) -> list[tuple[float, float]]:
    """Ranges that are not silence, i.e. candidate lines.

    Cutting by ear means scrubbing a video editor; this gets you to the shout in one
    command by detecting the silence around it and returning the gaps between.
    """
    stderr = run_ffmpeg(ffmpeg, [
        "-i", str(source), "-vn",
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ])

    duration = None
    if (match := re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)):
        h, m, s = match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    starts = [float(v) for v in re.findall(r"silence_start:\s*(-?\d+(?:\.\d+)?)", stderr)]
    ends = [float(v) for v in re.findall(r"silence_end:\s*(-?\d+(?:\.\d+)?)", stderr)]

    # Invert the silences. The file starts loud unless a silence starts at ~0, and ends
    # loud unless the last event was a silence_start.
    edges = sorted([(s, "start") for s in starts] + [(e, "end") for e in ends])
    segments: list[tuple[float, float]] = []
    open_at = 0.0 if not edges or edges[0][1] == "start" else None
    for when, kind in edges:
        if kind == "start" and open_at is not None:
            if when - open_at > 0.05:
                segments.append((open_at, when))
            open_at = None
        elif kind == "end":
            open_at = when
    if open_at is not None and duration and duration - open_at > 0.05:
        segments.append((open_at, duration))
    return segments


def report_crest(peak: float, mean: float, mode: str, peak_target: float,
                 rms_target: float, limiting: bool) -> None:
    """Say when the levelling chosen is not the levelling this clip wants.

    A clip that is only a voice sits about 13 dB peak-to-average and every mode treats it
    the same, so there is nothing to say. A clip carrying the scene's effects can sit at
    twice that, and there the mode matters -- but so does the ceiling, which quietly turns
    `rms` back into `peak` on the very clips `rms` was reached for. Both are worth a line,
    because neither shows up as anything but a clip that plays too quietly on the day.
    """
    crest = peak - mean
    if crest <= WIDE_CREST_DB or limiting:
        return
    if mode == "peak":
        print(f"  note: {crest:.0f} dB between this clip's loudest moment and its average "
              f"-- an\n        effect, most likely, well above the voice. Peak levelling "
              f"keeps that\n        balance exactly as mixed; --level rms matches this "
              f"clip's loudness to\n        your others instead.", file=sys.stderr)
    elif crest > peak_target - rms_target:
        held = crest - (peak_target - rms_target)
        print(f"  note: the {peak_target:g} dBFS ceiling held rms levelling {held:.0f} dB "
              f"short of its\n        {rms_target:g} dBFS target -- this clip's peak is "
              f"{crest:.0f} dB above its average, so it\n        cannot get there "
              f"cleanly. --limit trades that peak for the {held:.0f} dB.",
              file=sys.stderr)


def scene_span(segments: list[tuple[float, float]]) -> tuple[float, float] | None:
    """First sound to last sound across every segment -- the scene, not one piece of it.

    `scan_segments` breaks at every dip below the threshold, which inside a single scene
    means the gaps between the shout, the effect, and the score. Rejoining them is nearly
    always the range wanted; picking one row is how a clip ends up as a bare voice line.
    """
    return (segments[0][0], segments[-1][1]) if segments else None


def extract(ffmpeg: str, source: Path, destination: Path, start: float | None,
            end: float | None, gain_db: float, fade: bool, channels: int = 1,
            limit_dbfs: float | None = None) -> None:
    """Write the trimmed, levelled clip.

    `channels` is 1 by default because a shout is centred anyway and mono is half the
    file. Pass 2 for a clip that keeps the scene's effects: those are mixed wide, and
    folding them down sums two out-of-phase sides into a thinner effect than the show has.

    Filter order is load-bearing: gain, then the limiter that catches what the gain pushed
    over, then the fades. Fading last means the fades themselves are never limited, so the
    cut stays clean however hard the middle of the clip is being held down.
    """
    filters = [f"volume={gain_db:.2f}dB"] if abs(gain_db) > 0.01 else []
    if limit_dbfs is not None:
        filters.append(limiter_filter(limit_dbfs))
    if fade:
        filters.append(f"afade=t=in:st=0:d={FADE_IN_S}")
        # An out-fade needs to know where the end is, which only holds for a bounded cut.
        if end is not None:
            span = end - (start or 0.0)
            if span > FADE_OUT_S * 2:
                filters.append(
                    f"afade=t=out:st={span - FADE_OUT_S:.3f}:d={FADE_OUT_S}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(ffmpeg, [
        *trim_args(start, end), "-i", str(source),
        "-vn", "-ac", str(channels), "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        *(["-af", ",".join(filters)] if filters else []),
        "-y", str(destination),
    ])
    if not destination.exists() or destination.stat().st_size <= 44:
        raise VoiceBuildError(
            f"{destination} came out empty -- is the --start/--end range inside the clip?"
        )


def pad_range(start: float | None, end: float | None,
              pad: float) -> tuple[float | None, float | None]:
    """Widen a cut by `pad` seconds each way, without running off the front of the file.

    Silence detection ends a segment where the sound drops below a threshold, which for an
    effect means partway down its own tail. Padding puts the tail back. A one-sided range
    stays one-sided: there is nothing to add to an end that is already the end of file.
    """
    if pad <= 0 or (start is None and end is None):
        return start, end
    return (max(0.0, (start or 0.0) - pad), None if end is None else end + pad)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def wav_summary(path: Path) -> str:
    """"(3.02s, stereo)" -- enough to tell a bare voice line from a whole scene at a glance."""
    if path.suffix.lower() != ".wav":
        return ""
    try:
        with wave.open(str(path), "rb") as handle:
            length = handle.getnframes() / float(handle.getframerate())
            channels = handle.getnchannels()
    except Exception as exc:
        return f"  (unreadable: {exc})"
    return f"  ({length:.2f}s, {'stereo' if channels > 1 else 'mono'})"


def list_clips(voice_dir: Path, jutsu_csv: Path) -> int:
    """Show which jutsu have a voice and which are still silent."""
    if not jutsu_csv.exists():
        print(f"error: {jutsu_csv} not found", file=sys.stderr)
        return 1
    jutsu = load_jutsu(jutsu_csv)
    print(f"voice clips in {voice_dir}:\n")
    have = 0
    for entry in sorted(jutsu, key=lambda j: j.name):
        path = find_clip(voice_dir, entry.name)
        if path is None:
            print(f"  [ ] {entry.name:36s} -- {slugify(entry.name)}.wav")
            continue
        have += 1
        print(f"  [x] {entry.name:36s} {path.name}{wav_summary(path)}")
    print(f"\n{have}/{len(jutsu)} jutsu have a voice")

    known = {slugify(entry.name) for entry in jutsu}
    if voice_dir.is_dir():
        strays = [p.name for p in sorted(voice_dir.iterdir())
                  if p.suffix.lower() in EXTENSIONS and slugify(p.stem) not in known]
        if strays:
            print(f"\nnot matched to any jutsu in {jutsu_csv.name}: {', '.join(strays)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", "-i", type=Path,
                        help="video or audio file holding the line (mp4, mkv, wav, mp3 ...)")
    parser.add_argument("--jutsu", "-j",
                        help="jutsu the clip belongs to, e.g. Chidori; sets the output name")
    parser.add_argument("--out", "-o", type=Path,
                        help="explicit output path, instead of <voice-dir>/<jutsu>.wav")
    parser.add_argument("--voice-dir", type=Path, default=DEFAULT_VOICE_DIR)
    parser.add_argument("--jutsu-csv", type=Path, default=Path("jutsu.csv"))
    parser.add_argument("--start", type=parse_time, default=None,
                        help="cut in at this point: seconds, or mm:ss.s. Omit --start and "
                             "--end both to take the whole file, effects and all")
    parser.add_argument("--end", type=parse_time, default=None, help="cut out at this point")
    parser.add_argument("--pad", type=float, default=0.0,
                        help="widen the cut by this many seconds each way, to keep an "
                             "effect tail that --scan trimmed off")
    parser.add_argument("--scan", action="store_true",
                        help="print the non-silent ranges in the file and exit -- use it "
                             "to find the line without opening an editor")
    parser.add_argument("--noise", type=float, default=-32.0,
                        help="--scan silence threshold in dB (default -32; raise toward 0 "
                             "if background music hides the gaps)")
    parser.add_argument("--level", choices=("peak", "rms", "none"), default="peak",
                        help="peak: loudest sample hits --peak (default; leaves the mix "
                             "untouched otherwise). rms: average energy hits --rms, so "
                             "clips of different kinds sit at the same loudness next to "
                             "each other. none: source level")
    parser.add_argument("--peak", type=float, default=PEAK_DBFS,
                        help=f"dBFS ceiling for the loudest sample (default {PEAK_DBFS}); "
                             f"also caps --level rms so nothing clips")
    parser.add_argument("--rms", type=float, default=RMS_DBFS,
                        help=f"target average dBFS for --level rms (default {RMS_DBFS})")
    parser.add_argument("--limit", action="store_true",
                        help="with --level rms, hold the peak down with a limiter instead "
                             "of holding the gain down -- brings the voice up under an "
                             "effect that towers over it, at the cost of squashing the "
                             "effect's loudest moment")
    parser.add_argument("--gain", type=float, default=0.0,
                        help="extra dB on top of levelling")
    parser.add_argument("--no-normalize", action="store_true",
                        help="same as --level none")
    parser.add_argument("--stereo", action="store_true",
                        help="keep both channels instead of downmixing to mono -- worth it "
                             "for a clip carrying the scene's sound effects")
    parser.add_argument("--no-fade", action="store_true",
                        help="skip the short in/out fades that hide the cut")
    parser.add_argument("--play", action="store_true", help="play the result once, to check it")
    parser.add_argument("--list", action="store_true",
                        help="show which jutsu have a clip and which do not")
    args = parser.parse_args()

    if args.list:
        return list_clips(args.voice_dir, args.jutsu_csv)

    if not args.video:
        parser.error("--video is required (or use --list)")
    if not args.video.exists():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 1

    try:
        ffmpeg = find_ffmpeg()

        if args.scan:
            segments = scan_segments(ffmpeg, args.video, args.noise)
            print(f"non-silent ranges in {args.video.name} (threshold {args.noise:g} dB):\n")
            if not segments:
                print("  none -- the whole file is below the threshold; try --noise -45")
            for i, (begin, finish) in enumerate(segments, 1):
                print(f"  {i:2d}.  {begin:7.2f} -> {finish:7.2f} s  "
                      f"({finish - begin:5.2f}s)   --start {begin:.2f} --end {finish:.2f}")

            # Silence detection splits a scene wherever it dips quiet, so the shout and the
            # effect around it usually land in different rows above. Taking one row is how
            # you end up with a bare voice line; the span across all of them is the scene.
            if len(segments) > 1:
                first, last = scene_span(segments)
                print(f"\n  all.  {first:7.2f} -> {last:7.2f} s  ({last - first:5.2f}s)   "
                      f"--start {first:.2f} --end {last:.2f}")
                print("\nOne row is the voice on its own. `all` keeps the effects between "
                      "them,\nwhich is what the show plays -- as does giving no range at "
                      "all on an\nalready-trimmed clip. Re-run with --jutsu and your pick.")
            else:
                print("\nRe-run with --jutsu and that range -- or with no range at all, "
                      "which\nkeeps the file end to end.")
            return 0

        if not args.jutsu and not args.out:
            parser.error("--jutsu (or --out) is required when building a clip")

        destination = args.out or args.voice_dir / f"{slugify(args.jutsu)}.wav"
        if destination.suffix.lower() != ".wav":
            print(f"warning: {destination.suffix} clips only play if you have installed "
                  f"sounddevice; .wav plays everywhere", file=sys.stderr)

        # A typo in --jutsu writes a file the demo will never look for, and the only
        # symptom is silence. Check it against the table now.
        if args.jutsu and args.jutsu_csv.exists():
            names = [entry.name for entry in load_jutsu(args.jutsu_csv)]
            match = [n for n in names if slugify(n) == slugify(args.jutsu)]
            if match:
                args.jutsu = match[0]           # adopt the table's spelling
            else:
                near = [n for n in names if slugify(args.jutsu) in slugify(n)]
                print(f"warning: no jutsu named {args.jutsu!r} in {args.jutsu_csv.name}"
                      + (f" -- did you mean {near[0]!r}?" if near else ""), file=sys.stderr)

        args.start, args.end = pad_range(args.start, args.end, args.pad)

        channels = 2 if args.stereo else 1
        mode = "none" if args.no_normalize else args.level
        limiting = args.limit and mode == "rms"
        if args.limit and not limiting:
            # Without this the flag is simply ignored, and the clip that comes out is one
            # the user believes they asked for something else.
            print(f"warning: --limit only applies to --level rms, not --level {mode} "
                  f"-- ignoring it", file=sys.stderr)

        gain = args.gain
        if mode != "none":
            peak, mean = measure_levels(ffmpeg, args.video, args.start, args.end, channels)
            gain += level_gain(peak, mean, mode, args.peak, args.rms, limiting)
            print(f"peak {peak:+.1f} dBFS, average {mean:+.1f} dBFS "
                  f"-> {mode} levelling{', limited' if limiting else ''}, "
                  f"applying {gain:+.1f} dB")
            report_crest(peak, mean, mode, args.peak, args.rms, limiting)

        if args.start is None and args.end is None:
            span = "whole file"
        elif args.end is None:
            span = f"{args.start:.2f}s -> end"
        else:
            span = f"{args.start or 0:.2f} -> {args.end:.2f}s"
        print(f"extracting {args.video.name} [{span}, "
              f"{'stereo' if args.stereo else 'mono'}] -> {destination}")
        extract(ffmpeg, args.video, destination, args.start, args.end, gain,
                fade=not args.no_fade, channels=channels,
                limit_dbfs=args.peak if limiting else None)
        print(f"  wrote {destination}  "
              f"({wav_duration(destination):.2f}s, {destination.stat().st_size / 1024:.0f} KB)")

        if args.play:
            player = VoicePlayer(destination.parent, [args.jutsu or destination.stem])
            if player.available:
                print(f"  playing via {player.backend.name} ...")
                player.play(args.jutsu or destination.stem)
                # Nothing else to do, but exiting kills the sound on every backend.
                time.sleep(wav_duration(destination) + 0.3)
                player.stop()
            else:
                print(f"  {player.describe()}", file=sys.stderr)

        print(f"\nrun the demo -- {args.jutsu or destination.stem} now speaks:")
        print("    python 04_demo.py")
    except VoiceBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
