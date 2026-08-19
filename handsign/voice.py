"""Jutsu audio clips, cut from the show.

The banner announces a jutsu visually; this plays it the way the show does. Speech
synthesis is the obvious alternative and it sounds like a train station announcement:
"Chidori" read flatly by a system voice undoes the moment the effect just built. The clip
has to be the real one -- and a clip carrying the whole scene, the shout over the crackle
of the jutsu charging, beats the shout alone by roughly the same distance again. Whatever
06_voice.py was pointed at gets played back: voice, effects, score, mono or stereo.

Clips live under `assets/voice/`, named after the jutsu they belong to:

    assets/voice/chidori.wav                  ->  "Chidori"
    assets/voice/fireball_jutsu.wav           ->  "Fireball Jutsu"

Build them from a video with `06_voice.py`; the naming is handled for you.

A clip runs to its own end. Nothing here trims one to the length of the banner or of the
visual effect, so a three-second scene plays all three seconds; only casting the next
jutsu, or resetting, cuts it off.

Playback must never stall the capture loop -- 25 ms of blocking is two dropped frames --
so every backend here starts the clip and returns immediately, and clips are decoded once
at startup rather than read from disk mid-cast. The backends are tried in order and the
first one that imports wins:

    sounddevice           cross-platform, clean stop, best latency -- install it if you can
    simpleaudio           cross-platform, WAV only
    winsound              Windows stdlib: no install, which is why the demo has sound out
                          of the box on the machine it is usually presented from
    ffplay/afplay/aplay   last resort, spawns a process per cast

Only the first two are worth installing; the fallbacks exist so a fresh clone makes noise
without anyone reading a setup section first.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

DEFAULT_VOICE_DIR = Path("assets/voice")

# WAV first, and by a distance: three of the four backends read nothing else. The
# compressed formats only work if sounddevice is installed alongside soundfile, so
# 06_voice.py always writes WAV.
EXTENSIONS = (".wav", ".ogg", ".flac", ".mp3", ".m4a")


class VoiceError(RuntimeError):
    """Raised on an unusable clip or an unusable audio backend."""


def slugify(name: str) -> str:
    """Jutsu name -> file stem.

    "Summoning: Fanged Pursuit Jutsu" -> "summoning_fanged_pursuit_jutsu". Punctuation and
    case vary between the CSV, the wiki, and whatever the user typed when saving the file,
    so both sides go through here rather than being compared directly.
    """
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in name)
    return "_".join(cleaned.split())


def find_clip(voice_dir: Path | str, jutsu_name: str) -> Path | None:
    """Locate the clip for one jutsu, or None."""
    voice_dir = Path(voice_dir)
    if not voice_dir.is_dir():
        return None
    slug = slugify(jutsu_name)

    for ext in EXTENSIONS:
        exact = voice_dir / f"{slug}{ext}"
        if exact.exists():
            return exact

    # Fall back to a slug comparison over the directory. Files arrive named "Chidori.wav"
    # or "chidori - kakashi.wav" at least as often as they arrive named correctly, and a
    # silent demo is a rotten way to find that out.
    for path in sorted(voice_dir.iterdir()):
        if path.suffix.lower() in EXTENSIONS and slugify(path.stem) == slug:
            return path
    return None


def read_wav(path: Path | str):
    """Decode a PCM WAV to an int16 array plus its sample rate.

    Stdlib `wave` rather than soundfile: the only format that matters here is the one
    06_voice.py writes, and this keeps the module import-light.
    """
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise VoiceError(
                f"{path}: {handle.getsampwidth() * 8}-bit WAV; rebuild it with 06_voice.py "
                f"(16-bit PCM)"
            )
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    # Copied, not a view on `raw`: frombuffer hands back a read-only array, and playback
    # libraries that take a buffer pointer are entitled to object to that.
    data = np.frombuffer(raw, dtype="<i2").copy()
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, rate


# -- backends -------------------------------------------------------------------------
#
# Each one loads a clip to an opaque handle and plays that handle without blocking. The
# split matters: `load` may be slow (decode, device open), `play` may not.


class SoundDeviceBackend:
    name = "sounddevice"

    def __init__(self):
        import sounddevice

        # Installed but with no usable output device -- a headless box, a VM, a machine
        # whose audio service is not running -- would otherwise win the backend race and
        # only fail on the first cast. Fail here instead, so the next backend gets a turn.
        sounddevice.check_output_settings()
        self.sd = sounddevice

    def load(self, path: Path):
        return read_wav(path)

    def play(self, clip) -> None:
        data, rate = clip
        self.sd.stop()                  # a second cast cuts the first line off
        self.sd.play(data, rate)

    def stop(self) -> None:
        self.sd.stop()


class SimpleAudioBackend:
    name = "simpleaudio"

    def __init__(self):
        import simpleaudio

        self.sa = simpleaudio
        self.handle = None

    def load(self, path: Path):
        return self.sa.WaveObject.from_wave_file(str(path))

    def play(self, clip) -> None:
        self.stop()
        self.handle = clip.play()

    def stop(self) -> None:
        if self.handle is not None:
            self.handle.stop()
            self.handle = None


class WinsoundBackend:
    name = "winsound"

    def __init__(self):
        import winsound                 # ImportError anywhere but Windows

        self.winsound = winsound

    def load(self, path: Path):
        # PlaySound takes a path, so there is nothing to decode -- but it also fails
        # *silently* on a file it cannot parse, so the file is opened once here to turn
        # that into an error at startup instead of a mystery during a demo.
        with wave.open(str(path), "rb"):
            pass
        return str(path)

    def play(self, clip) -> None:
        self.winsound.PlaySound(
            clip, self.winsound.SND_FILENAME | self.winsound.SND_ASYNC
        )

    def stop(self) -> None:
        self.winsound.PlaySound(None, self.winsound.SND_PURGE)


class CommandBackend:
    """Hand the file to whatever command-line player exists."""

    name = "command"
    CANDIDATES = (
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "error"]),
        ("afplay", []),                 # macOS
        ("aplay", ["-q"]),              # ALSA
        ("paplay", []),                 # PulseAudio
    )

    def __init__(self, exe: str | None = None):
        self.process = None
        for candidate, flags in self.CANDIDATES:
            if exe and candidate != exe:
                continue
            found = shutil.which(candidate)
            if found:
                self.name = candidate
                self.command = [found, *flags]
                return
        raise VoiceError(f"{exe or 'no command-line audio player'} not found on PATH")

    def load(self, path: Path):
        return str(path)

    def play(self, clip) -> None:
        self.stop()
        self.process = subprocess.Popen(
            [*self.command, clip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None


BACKENDS = (SoundDeviceBackend, SimpleAudioBackend, WinsoundBackend, CommandBackend)


def open_backend(prefer: str | None = None):
    """First backend that works on this machine, or None if none do.

    `prefer` names one and only that one is tried, so a machine where the default choice
    sounds wrong can be pinned -- and a bad name raises rather than silently falling
    through to something else.
    """
    if not prefer:
        for backend in BACKENDS:
            try:
                return backend()
            except Exception:
                continue
        return None

    key = prefer.strip().lower()
    players = [exe for exe, _ in CommandBackend.CANDIDATES]
    if key in players:
        return CommandBackend(key)
    for backend in BACKENDS:
        if key == backend.name:
            return backend()
    raise VoiceError(
        f"unknown audio backend {prefer!r}; try one of "
        f"{', '.join([b.name for b in BACKENDS[:-1]] + players)}"
    )


# -- player ---------------------------------------------------------------------------


class VoicePlayer:
    """Plays the voice clip for a jutsu, if one exists.

    Constructed with the jutsu names up front so every clip is decoded before the camera
    starts. A missing clip, an unreadable file, or a machine with no audio device at all
    are all non-events: the demo runs silently rather than failing. Nothing in here may
    raise into the frame loop -- a sound effect is not worth dropping a presentation over.
    """

    def __init__(
        self,
        voice_dir: Path | str = DEFAULT_VOICE_DIR,
        jutsu_names=(),
        backend=None,
        prefer: str | None = None,
    ):
        self.voice_dir = Path(voice_dir)
        self.backend = backend if backend is not None else open_backend(prefer)
        # Keyed by slug, so a caller holding the name in a different spelling than the
        # CSV -- a game, a REPL, an edited table -- still gets its clip.
        self.clips: dict[str, object] = {}
        self.paths: dict[str, Path] = {}        # by jutsu name, for reporting
        self.missing: list[str] = []
        self.broken: list[tuple[str, str]] = []
        self.playing: str | None = None
        if self.backend is not None:
            self._preload(jutsu_names)

    def _preload(self, jutsu_names) -> None:
        for name in jutsu_names:
            path = find_clip(self.voice_dir, name)
            if path is None:
                self.missing.append(name)
                continue
            try:
                self.clips[slugify(name)] = self.backend.load(path)
                self.paths[name] = path
            except Exception as exc:
                self.broken.append((name, f"{path.name}: {exc}"))

    @property
    def available(self) -> bool:
        return bool(self.clips)

    def play(self, jutsu_name: str) -> bool:
        """Start the clip for `jutsu_name`. Returns whether anything was played."""
        clip = self.clips.get(slugify(jutsu_name))
        if clip is None:
            return False
        try:
            self.backend.play(clip)
        except Exception as exc:
            # One bad playback disables the voice rather than repeating the failure on
            # every cast for the rest of the session.
            print(f"  voice disabled: {exc}", file=sys.stderr)
            self.clips.clear()
            return False
        self.playing = jutsu_name
        return True

    def stop(self) -> None:
        if self.backend is None:
            return
        try:
            self.backend.stop()
        except Exception:
            pass
        self.playing = None

    def describe(self) -> str:
        """One line for the demo's startup output."""
        if self.backend is None:
            return (
                "voice: no audio backend (pip install sounddevice, or put ffplay on PATH)"
            )
        if not self.clips:
            return (
                f"voice: no clips in {self.voice_dir} -- build one with "
                f"06_voice.py --jutsu Chidori --video clip.mp4"
            )
        have = ", ".join(sorted(self.paths))
        return f"voice: {len(self.clips)} clip(s) via {self.backend.name} -- {have}"
