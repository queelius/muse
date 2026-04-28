"""Tests for audio_generation codec.

WAV is stdlib, always available. FLAC uses soundfile (already in
muse[audio]). MP3 + Opus go through pydub + ffmpeg; tests assert
clean UnsupportedFormatError when either dep is missing.
"""
import io
import wave
from unittest.mock import patch

import numpy as np
import pytest

from muse.modalities.audio_generation.codec import (
    UnsupportedFormatError,
    content_type_for,
    encode_flac,
    encode_mp3,
    encode_opus,
    encode_wav,
)


def _mono(samples=4410, sr=44100, freq=440.0):
    """1-D float32 sine wave."""
    t = np.arange(samples, dtype=np.float32) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _stereo(samples=4410, sr=44100):
    """(samples, 2) float32 stereo: left = sine, right = inverted sine."""
    left = _mono(samples, sr, 440.0)
    right = -left
    return np.stack([left, right], axis=1)


def test_encode_wav_mono_returns_riff_with_headers():
    out = encode_wav(_mono(), 44100, channels=1)
    assert isinstance(out, bytes)
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"
    # Round-trip via wave to check params.
    with wave.open(io.BytesIO(out), "rb") as r:
        assert r.getnchannels() == 1
        assert r.getframerate() == 44100
        assert r.getsampwidth() == 2


def test_encode_wav_stereo_round_trips():
    out = encode_wav(_stereo(), 44100, channels=2)
    with wave.open(io.BytesIO(out), "rb") as r:
        assert r.getnchannels() == 2
        assert r.getframerate() == 44100
        assert r.getnframes() == 4410


def test_encode_wav_clipping_does_not_overflow():
    """Out-of-range samples are clipped, not wrapped."""
    audio = np.array([3.0, -3.0, 0.0, 0.5], dtype=np.float32)
    out = encode_wav(audio, 8000, channels=1)
    with wave.open(io.BytesIO(out), "rb") as r:
        frames = r.readframes(r.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    # 3.0 clamped to +1.0 -> 32767 (just below 32768 ceiling)
    assert pcm[0] == 32767
    # -3.0 clamped to -1.0 -> -32768
    assert pcm[1] == -32768


def test_encode_wav_sample_rate_honored():
    out = encode_wav(_mono(samples=2000, sr=22050), 22050, channels=1)
    with wave.open(io.BytesIO(out), "rb") as r:
        assert r.getframerate() == 22050


def test_encode_wav_rejects_mismatched_shape_for_channels():
    """1-D + channels=2 is incoherent; raise."""
    with pytest.raises(ValueError, match="channels"):
        encode_wav(_mono(), 44100, channels=2)


def test_encode_flac_returns_bytes_when_soundfile_available():
    """Skip if soundfile not installed locally."""
    sf = pytest.importorskip("soundfile")
    out = encode_flac(_mono(), 44100, channels=1)
    assert isinstance(out, bytes)
    # FLAC stream marker is "fLaC".
    assert out[:4] == b"fLaC"


def test_encode_flac_raises_when_soundfile_missing():
    with patch(
        "muse.modalities.audio_generation.codec._try_import_soundfile",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="soundfile"):
            encode_flac(_mono(), 44100, channels=1)


def test_encode_mp3_raises_when_pydub_missing():
    with patch(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="pydub"):
            encode_mp3(_mono(), 44100, channels=1)


def test_encode_mp3_raises_when_ffmpeg_missing(monkeypatch):
    """If pydub is installed but ffmpeg is not on PATH, error cleanly."""
    fake_pydub = type("p", (), {})
    fake_pydub.AudioSegment = type("AudioSegment", (), {})
    with patch(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        return_value=fake_pydub,
    ):
        monkeypatch.setattr(
            "muse.modalities.audio_generation.codec.shutil.which",
            lambda _: None,
        )
        with pytest.raises(UnsupportedFormatError, match="ffmpeg"):
            encode_mp3(_mono(), 44100, channels=1)


def test_encode_opus_raises_when_pydub_missing():
    with patch(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="pydub"):
            encode_opus(_mono(), 44100, channels=1)


def test_encode_mp3_calls_pydub_when_available(monkeypatch):
    """When deps are present, encode_mp3 hands off to pydub.AudioSegment."""
    captured = {}

    class FakeSegment:
        @classmethod
        def from_wav(cls, buf):
            captured["wav_len"] = len(buf.getvalue())
            return cls()

        def export(self, out, *, format, bitrate):
            captured["format"] = format
            captured["bitrate"] = bitrate
            out.write(b"FAKEMP3")

    fake_pydub = type("p", (), {"AudioSegment": FakeSegment})
    monkeypatch.setattr(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        lambda: fake_pydub,
    )
    monkeypatch.setattr(
        "muse.modalities.audio_generation.codec.shutil.which",
        lambda _: "/usr/bin/ffmpeg",
    )
    out = encode_mp3(_mono(), 44100, channels=1)
    assert out == b"FAKEMP3"
    assert captured["format"] == "mp3"
    assert captured["bitrate"] == "128k"
    assert captured["wav_len"] > 0


def test_encode_opus_calls_pydub_when_available(monkeypatch):
    captured = {}

    class FakeSegment:
        @classmethod
        def from_wav(cls, buf):
            return cls()

        def export(self, out, *, format, bitrate):
            captured["format"] = format
            captured["bitrate"] = bitrate
            out.write(b"FAKEOPUS")

    fake_pydub = type("p", (), {"AudioSegment": FakeSegment})
    monkeypatch.setattr(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        lambda: fake_pydub,
    )
    monkeypatch.setattr(
        "muse.modalities.audio_generation.codec.shutil.which",
        lambda _: "/usr/bin/ffmpeg",
    )
    out = encode_opus(_mono(), 44100, channels=1)
    assert out == b"FAKEOPUS"
    assert captured["format"] == "opus"
    assert captured["bitrate"] == "64k"


def test_unsupported_format_error_is_exception_subclass():
    """Mirrors v0.18.0 image_animation pattern."""
    assert issubclass(UnsupportedFormatError, Exception)


def test_content_type_for_each_format():
    assert content_type_for("wav") == "audio/wav"
    assert content_type_for("mp3") == "audio/mpeg"
    assert content_type_for("opus") == "audio/ogg"
    assert content_type_for("flac") == "audio/flac"


def test_content_type_for_unknown_format_raises():
    with pytest.raises(KeyError):
        content_type_for("aac")
