"""Cut a character's voice line out of a video and install it as a jutsu voice clip.

The demo speaks a jutsu the moment it fires (see `handsign/voice.py`). This builds the
clips it plays: point it at an episode clip, name the jutsu, and it extracts the audio,
trims it, levels it, and writes `assets/voice/<jutsu>.wav` -- the exact name and format
the demo looks for.

Levelling is not optional polish. Anime audio is mixed with a lot of headroom, so a raw
extract plays back noticeably quieter than the room expects and the line gets lost under
the demo's own noise; every clip is peak-normalised so they all land at the same level.

Typical run -- find the shout, then cut it:

    python 06_voice.py --video Kakashi_chidori.mp4 --scan
    python 06_voice.py --video Kakashi_chidori.mp4 --jutsu Chidori --start 1.4 --end 3.2
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

# 16-bit PCM mono at 44.1 kHz. Not a preference: winsound and simpleaudio read plain PCM
# WAV and nothing else, and those are the two backends that work without an install.
SAMPLE_RATE = 44100
PEAK_DBFS = -1.0            # leave a hair of headroom so nothing clips on playback
FADE_IN_S = 0.015           # a hard cut into a waveform mid-cycle pops audibly
FADE_OUT_S = 0.060


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


def measure_peak(ffmpeg: str, source: Path, start: float | None, end: float | None) -> float:
    """Loudest sample in the selected range, in dBFS (0 is full scale)."""
    stderr = run_ffmpeg(ffmpeg, [
        *trim_args(start, end), "-i", str(source),
        "-vn", "-af", "volumedetect", "-f", "null", "-",
    ])
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", stderr)
    if not match:
        raise VoiceBuildError(
            f"no audio track found in {source.name} -- check the clip actually has sound"
        )
    return float(match.group(1))


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


def extract(ffmpeg: str, source: Path, destination: Path, start: float | None,
            end: float | None, gain_db: float, fade: bool) -> None:
    """Write the trimmed, levelled clip."""
    filters = [f"volume={gain_db:.2f}dB"] if abs(gain_db) > 0.01 else []
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
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        *(["-af", ",".join(filters)] if filters else []),
        "-y", str(destination),
    ])
    if not destination.exists() or destination.stat().st_size <= 44:
        raise VoiceBuildError(
            f"{destination} came out empty -- is the --start/--end range inside the clip?"
        )


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


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
        length = f"  ({wav_duration(path):.2f}s)" if path.suffix.lower() == ".wav" else ""
        print(f"  [x] {entry.name:36s} {path.name}{length}")
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
                        help="cut in at this point: seconds, or mm:ss.s")
    parser.add_argument("--end", type=parse_time, default=None, help="cut out at this point")
    parser.add_argument("--scan", action="store_true",
                        help="print the non-silent ranges in the file and exit -- use it "
                             "to find the line without opening an editor")
    parser.add_argument("--noise", type=float, default=-32.0,
                        help="--scan silence threshold in dB (default -32; raise toward 0 "
                             "if background music hides the gaps)")
    parser.add_argument("--peak", type=float, default=PEAK_DBFS,
                        help=f"normalise the loudest sample to this dBFS (default {PEAK_DBFS})")
    parser.add_argument("--gain", type=float, default=0.0,
                        help="extra dB on top of normalisation")
    parser.add_argument("--no-normalize", action="store_true",
                        help="keep the source level; only --gain is applied")
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
            print("\nPick the one holding the line, then re-run with --jutsu and that range.")
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

        gain = args.gain
        if not args.no_normalize:
            peak = measure_peak(ffmpeg, args.video, args.start, args.end)
            gain += args.peak - peak
            print(f"peak {peak:+.1f} dBFS -> applying {gain:+.1f} dB")

        if args.start is None and args.end is None:
            span = "whole file"
        elif args.end is None:
            span = f"{args.start:.2f}s -> end"
        else:
            span = f"{args.start or 0:.2f} -> {args.end:.2f}s"
        print(f"extracting {args.video.name} [{span}] -> {destination}")
        extract(ffmpeg, args.video, destination, args.start, args.end, gain,
                fade=not args.no_fade)
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
