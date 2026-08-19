"""Tests for the jutsu voice clips.

Sound is the one part of the demo nobody can check in CI by listening, so what is tested
here is everything around the speaker: that a clip is found under the names people
actually save files as, that playback never blocks or throws into the frame loop, and
that a missing clip, a broken clip, or a machine with no audio device leaves a working
demo rather than a stack trace mid-presentation.
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import pytest

from handsign.voice import (
    CommandBackend,
    VoiceError,
    VoicePlayer,
    find_clip,
    open_backend,
    read_wav,
    slugify,
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


voice_script = _load("voice_script", "06_voice.py")


def write_wav(path: Path, seconds: float = 0.05, rate: int = 44100, channels: int = 1,
              width: int = 2) -> Path:
    """A silent but structurally valid PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\0" * int(rate * seconds) * channels * width)
    return path


class FakeBackend:
    """Records what it was asked to play instead of making noise."""

    name = "fake"

    def __init__(self, fail_on_play: bool = False):
        self.loaded: list[str] = []
        self.played: list[str] = []
        self.stops = 0
        self.fail_on_play = fail_on_play

    def load(self, path):
        self.loaded.append(Path(path).name)
        return str(path)

    def play(self, clip):
        if self.fail_on_play:
            raise RuntimeError("no audio device")
        self.played.append(Path(clip).name)

    def stop(self):
        self.stops += 1


class TestSlugify:
    def test_lowercases_and_underscores(self):
        assert slugify("Fireball Jutsu") == "fireball_jutsu"

    def test_drops_punctuation(self):
        assert slugify("Summoning: Fanged Pursuit Jutsu") == "summoning_fanged_pursuit_jutsu"

    def test_collapses_runs_of_separators(self):
        assert slugify("  Water  --  Trumpet ") == "water_trumpet"

    def test_is_idempotent(self):
        """The demo slugifies jutsu names and 06_voice.py slugifies filenames; both sides
        have to land on the same string or nothing ever matches."""
        assert slugify(slugify("Earth Dragon Bullet")) == slugify("Earth Dragon Bullet")


class TestFindClip:
    def test_finds_exact_slug(self, tmp_path):
        write_wav(tmp_path / "chidori.wav")
        assert find_clip(tmp_path, "Chidori").name == "chidori.wav"

    def test_finds_multiword_jutsu(self, tmp_path):
        write_wav(tmp_path / "fireball_jutsu.wav")
        assert find_clip(tmp_path, "Fireball Jutsu") is not None

    def test_tolerates_the_name_people_actually_save(self, tmp_path):
        """"Chidori.WAV" and "Fireball Jutsu.wav" are what comes out of an editor."""
        write_wav(tmp_path / "Chidori.WAV")
        write_wav(tmp_path / "Fireball Jutsu.wav")
        assert find_clip(tmp_path, "Chidori") is not None
        assert find_clip(tmp_path, "Fireball Jutsu") is not None

    def test_missing_clip_is_none(self, tmp_path):
        assert find_clip(tmp_path, "Chidori") is None

    def test_missing_directory_is_none(self, tmp_path):
        assert find_clip(tmp_path / "nope", "Chidori") is None

    def test_wav_wins_over_compressed(self, tmp_path):
        """Only one backend reads mp3; preferring it would mute the other three."""
        write_wav(tmp_path / "chidori.wav")
        (tmp_path / "chidori.mp3").write_bytes(b"\0")
        assert find_clip(tmp_path, "Chidori").suffix == ".wav"

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("chidori")
        assert find_clip(tmp_path, "Chidori") is None


class TestReadWav:
    def test_returns_samples_and_rate(self, tmp_path):
        data, rate = read_wav(write_wav(tmp_path / "a.wav", seconds=0.1, rate=8000))
        assert rate == 8000
        assert len(data) == 800

    def test_stereo_is_shaped_per_channel(self, tmp_path):
        data, _ = read_wav(write_wav(tmp_path / "s.wav", seconds=0.1, rate=8000, channels=2))
        assert data.shape == (800, 2)

    def test_rejects_non_16_bit(self, tmp_path):
        """8-bit WAV loads fine in some players and silently misplays in others."""
        path = write_wav(tmp_path / "eight.wav", rate=8000, width=1)
        with pytest.raises(VoiceError, match="16-bit"):
            read_wav(path)


class TestVoicePlayer:
    def test_plays_the_clip_for_a_matched_jutsu(self, tmp_path):
        write_wav(tmp_path / "chidori.wav")
        backend = FakeBackend()
        player = VoicePlayer(tmp_path, ["Chidori"], backend=backend)
        assert player.play("Chidori") is True
        assert backend.played == ["chidori.wav"]

    def test_clips_are_decoded_before_the_camera_starts(self, tmp_path):
        """Loading inside the frame loop costs a visible hitch on the cast frame."""
        write_wav(tmp_path / "chidori.wav")
        backend = FakeBackend()
        VoicePlayer(tmp_path, ["Chidori"], backend=backend)
        assert backend.loaded == ["chidori.wav"]

    def test_jutsu_without_a_clip_is_silent_not_fatal(self, tmp_path):
        backend = FakeBackend()
        player = VoicePlayer(tmp_path, ["Chidori"], backend=backend)
        assert player.play("Chidori") is False
        assert player.missing == ["Chidori"]
        assert backend.played == []

    def test_unknown_jutsu_is_silent(self, tmp_path):
        write_wav(tmp_path / "chidori.wav")
        player = VoicePlayer(tmp_path, ["Chidori"], backend=FakeBackend())
        assert player.play("Rasengan") is False

    def test_lookup_survives_a_different_spelling(self, tmp_path):
        """A caller outside the demo may hold the name in whatever case it typed it."""
        write_wav(tmp_path / "fireball_jutsu.wav")
        player = VoicePlayer(tmp_path, ["Fireball Jutsu"], backend=FakeBackend())
        assert player.play("fireball jutsu") is True

    def test_a_broken_clip_disables_only_itself(self, tmp_path):
        """One unreadable file must not take the other clips down with it."""
        write_wav(tmp_path / "chidori.wav")
        (tmp_path / "fireball_jutsu.wav").write_bytes(b"not a wav")

        class PickyBackend(FakeBackend):
            def load(self, path):
                with wave.open(str(path), "rb"):
                    pass
                return super().load(path)

        player = VoicePlayer(tmp_path, ["Chidori", "Fireball Jutsu"], backend=PickyBackend())
        assert player.play("Chidori") is True
        assert player.play("Fireball Jutsu") is False
        assert [name for name, _ in player.broken] == ["Fireball Jutsu"]

    def test_playback_failure_does_not_raise_into_the_frame_loop(self, tmp_path, capsys):
        write_wav(tmp_path / "chidori.wav")
        player = VoicePlayer(tmp_path, ["Chidori"], backend=FakeBackend(fail_on_play=True))
        assert player.play("Chidori") is False
        assert "voice disabled" in capsys.readouterr().err

    def test_failure_is_reported_once_not_every_cast(self, tmp_path, capsys):
        write_wav(tmp_path / "chidori.wav")
        player = VoicePlayer(tmp_path, ["Chidori"], backend=FakeBackend(fail_on_play=True))
        for _ in range(5):
            player.play("Chidori")
        assert capsys.readouterr().err.count("voice disabled") == 1

    def test_no_backend_leaves_a_usable_player(self, tmp_path):
        """A machine with no audio device at all still runs the demo."""
        write_wav(tmp_path / "chidori.wav")
        player = VoicePlayer(tmp_path, ["Chidori"], backend=None, prefer=None)
        player.backend = None
        player.clips.clear()
        assert player.play("Chidori") is False
        player.stop()               # must not raise

    def test_stop_is_safe_with_no_backend(self, tmp_path):
        player = VoicePlayer(tmp_path, [], backend=None)
        player.backend = None
        player.stop()

    def test_stop_reaches_the_backend(self, tmp_path):
        backend = FakeBackend()
        VoicePlayer(tmp_path, [], backend=backend).stop()
        assert backend.stops == 1

    def test_describe_mentions_the_missing_folder(self, tmp_path):
        player = VoicePlayer(tmp_path / "empty", ["Chidori"], backend=FakeBackend())
        assert "06_voice.py" in player.describe()

    def test_describe_lists_loaded_clips(self, tmp_path):
        write_wav(tmp_path / "chidori.wav")
        player = VoicePlayer(tmp_path, ["Chidori"], backend=FakeBackend())
        assert "Chidori" in player.describe() and "fake" in player.describe()


class TestOpenBackend:
    def test_unknown_name_raises(self):
        with pytest.raises(VoiceError, match="unknown audio backend"):
            open_backend("gramophone")

    def test_no_preference_never_raises(self):
        """Whatever this machine has -- or has not -- the demo must still start."""
        open_backend()

    def test_missing_command_player_raises_with_its_name(self, monkeypatch):
        monkeypatch.setattr("handsign.voice.shutil.which", lambda _: None)
        with pytest.raises(VoiceError, match="ffplay"):
            CommandBackend("ffplay")


class TestTimeParsing:
    @pytest.mark.parametrize("text,seconds", [
        ("12.5", 12.5),
        ("90", 90.0),
        ("1:03", 63.0),
        ("1:03.25", 63.25),
        ("0:01:03.5", 63.5),
    ])
    def test_accepts_plain_seconds_and_timestamps(self, text, seconds):
        assert voice_script.parse_time(text) == pytest.approx(seconds)

    def test_rejects_nonsense(self):
        with pytest.raises(Exception):
            voice_script.parse_time("start")


class TestTrimArgs:
    def test_no_range_passes_nothing(self):
        assert voice_script.trim_args(None, None) == []

    def test_end_becomes_a_duration_not_a_timestamp(self):
        """`-ss` before `-i` makes `-to` relative to the seek point; `-t` is unambiguous."""
        assert voice_script.trim_args(2.0, 5.0) == ["-ss", "2.000", "-t", "3.000"]

    def test_start_alone_runs_to_the_end(self):
        assert voice_script.trim_args(1.5, None) == ["-ss", "1.500"]

    def test_backwards_range_is_rejected(self):
        with pytest.raises(voice_script.VoiceBuildError):
            voice_script.trim_args(4.0, 2.0)


class TestScanSegments:
    """`--scan` inverts ffmpeg's silence report; the inversion is the part that can be wrong."""

    def fake_run(self, stderr):
        return lambda *args, **kwargs: stderr

    def test_gap_between_two_silences_is_a_segment(self, monkeypatch):
        monkeypatch.setattr(voice_script, "run_ffmpeg", self.fake_run(
            "  Duration: 00:00:05.00, start: 0.000\n"
            "[silencedetect] silence_start: 0.000\n"
            "[silencedetect] silence_end: 1.20 | silence_duration: 1.20\n"
            "[silencedetect] silence_start: 3.40\n"
            "[silencedetect] silence_end: 5.00 | silence_duration: 1.60\n"
        ))
        assert voice_script.scan_segments("ffmpeg", Path("x.mp4")) == [(1.20, 3.40)]

    def test_audio_running_to_the_end_of_file_is_a_segment(self, monkeypatch):
        monkeypatch.setattr(voice_script, "run_ffmpeg", self.fake_run(
            "  Duration: 00:00:04.00, start: 0.000\n"
            "[silencedetect] silence_start: 0.000\n"
            "[silencedetect] silence_end: 2.50 | silence_duration: 2.50\n"
        ))
        assert voice_script.scan_segments("ffmpeg", Path("x.mp4")) == [(2.50, 4.00)]

    def test_file_that_opens_loud_starts_a_segment_at_zero(self, monkeypatch):
        monkeypatch.setattr(voice_script, "run_ffmpeg", self.fake_run(
            "  Duration: 00:00:04.00, start: 0.000\n"
            "[silencedetect] silence_start: 1.75\n"
        ))
        assert voice_script.scan_segments("ffmpeg", Path("x.mp4")) == [(0.0, 1.75)]

    def test_silence_throughout_yields_nothing(self, monkeypatch):
        monkeypatch.setattr(voice_script, "run_ffmpeg", self.fake_run(
            "  Duration: 00:00:04.00, start: 0.000\n"
            "[silencedetect] silence_start: 0.000\n"
        ))
        assert voice_script.scan_segments("ffmpeg", Path("x.mp4")) == []


class TestMeasurePeak:
    def test_reads_the_peak_ffmpeg_reports(self, monkeypatch):
        monkeypatch.setattr(voice_script, "run_ffmpeg",
                            lambda *a, **k: "[Parsed_volumedetect] max_volume: -12.7 dB\n")
        assert voice_script.measure_peak("ffmpeg", Path("x.mp4"), None, None) == -12.7

    def test_a_video_with_no_audio_track_is_an_error(self, monkeypatch):
        """Silently writing an empty clip would only show up on stage."""
        monkeypatch.setattr(voice_script, "run_ffmpeg", lambda *a, **k: "Stream #0:0 Video\n")
        with pytest.raises(voice_script.VoiceBuildError, match="no audio"):
            voice_script.measure_peak("ffmpeg", Path("x.mp4"), None, None)


class TestExtractedClipIsPlayable:
    """The format contract between 06_voice.py and the backends that need plain PCM."""

    def test_written_clip_loads_through_the_player(self, tmp_path, monkeypatch):
        destination = tmp_path / "voice" / "chidori.wav"

        def fake_run(ffmpeg, args):
            write_wav(destination, seconds=0.2, rate=voice_script.SAMPLE_RATE)
            return ""

        monkeypatch.setattr(voice_script, "run_ffmpeg", fake_run)
        voice_script.extract("ffmpeg", Path("in.mp4"), destination, 1.0, 2.0, -3.0, fade=True)

        data, rate = read_wav(destination)
        assert rate == voice_script.SAMPLE_RATE
        assert voice_script.wav_duration(destination) == pytest.approx(0.2, abs=0.01)
        assert find_clip(destination.parent, "Chidori") == destination

    def test_an_empty_result_is_reported(self, tmp_path, monkeypatch):
        """An out-of-range --start yields a 44-byte header and no samples."""
        destination = tmp_path / "chidori.wav"

        def fake_run(ffmpeg, args):
            write_wav(destination, seconds=0.0)
            return ""

        monkeypatch.setattr(voice_script, "run_ffmpeg", fake_run)
        with pytest.raises(voice_script.VoiceBuildError, match="empty"):
            voice_script.extract("ffmpeg", Path("in.mp4"), destination, 90.0, 92.0, 0.0,
                                 fade=False)


class TestExtractFilters:
    """Filter graph assembly, checked without running ffmpeg."""

    def capture(self, monkeypatch, tmp_path, start, end, gain, fade):
        seen = {}

        def fake_run(ffmpeg, args):
            seen["args"] = args
            write_wav(tmp_path / "out.wav", seconds=0.1)
            return ""

        monkeypatch.setattr(voice_script, "run_ffmpeg", fake_run)
        voice_script.extract("ffmpeg", Path("in.mp4"), tmp_path / "out.wav",
                             start, end, gain, fade)
        args = seen["args"]
        return args, (args[args.index("-af") + 1] if "-af" in args else "")

    def test_output_is_mono_16_bit_pcm(self, monkeypatch, tmp_path):
        args, _ = self.capture(monkeypatch, tmp_path, None, None, 0.0, False)
        assert args[args.index("-ac") + 1] == "1"
        assert args[args.index("-c:a") + 1] == "pcm_s16le"
        assert args[args.index("-ar") + 1] == str(voice_script.SAMPLE_RATE)

    def test_gain_is_applied(self, monkeypatch, tmp_path):
        _, filters = self.capture(monkeypatch, tmp_path, None, None, 6.5, False)
        assert "volume=6.50dB" in filters

    def test_no_gain_means_no_filter(self, monkeypatch, tmp_path):
        _, filters = self.capture(monkeypatch, tmp_path, None, None, 0.0, False)
        assert filters == ""

    def test_fade_out_is_placed_at_the_end_of_the_cut(self, monkeypatch, tmp_path):
        _, filters = self.capture(monkeypatch, tmp_path, 1.0, 3.0, 0.0, True)
        assert "afade=t=in" in filters
        assert f"afade=t=out:st={2.0 - voice_script.FADE_OUT_S:.3f}" in filters

    def test_open_ended_cut_skips_the_fade_out(self, monkeypatch, tmp_path):
        """Without an --end there is no known end position to fade at."""
        _, filters = self.capture(monkeypatch, tmp_path, 1.0, None, 0.0, True)
        assert "afade=t=out" not in filters

    def test_very_short_cut_skips_the_fade_out(self, monkeypatch, tmp_path):
        """A 60 ms fade on a 40 ms clip would fade in from and out to nothing."""
        _, filters = self.capture(monkeypatch, tmp_path, 1.0, 1.04, 0.0, True)
        assert "afade=t=out" not in filters
