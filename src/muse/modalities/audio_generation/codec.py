"""Audio encoding for audio/generation responses.

Pure functions: numpy audio array + sample_rate + channels -> bytes.

WAV uses stdlib `wave`; always available.
FLAC uses `soundfile` (already in muse[audio] for Kokoro).
MP3 / Opus go through `pydub` + `ffmpeg`. Lazy-imported and
swappable; if either dep is missing, raises UnsupportedFormatError so
the route layer can convert to a clean 400 envelope.

The audio array shape is `(samples,)` for mono or `(samples, channels)`
for multi-channel. Codec normalizes both into the right interleaved
PCM layout for the WAV format and into the right shape for soundfile.
"""
from __future__ import annotations

import io
import shutil
import wave
from typing import Any

import numpy as np


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def _to_int16_pcm(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to int16 PCM samples."""
    scaled = np.clip(audio, -1.0, 1.0) * 32768.0
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def _normalize_shape(audio: np.ndarray, channels: int) -> np.ndarray:
    """Return audio reshaped to (samples, channels) for multi-channel
    or (samples,) for mono. The wave + soundfile encoders both want
    interleaved samples; this helper makes mono and stereo paths share
    the same downstream code.
    """
    if channels == 1:
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2 and audio.shape[1] == 1:
            return audio[:, 0]
        raise ValueError(
            f"audio shape {audio.shape} incompatible with channels=1"
        )
    if audio.ndim == 1:
        raise ValueError(
            f"audio is 1-D but channels={channels}; cannot demux"
        )
    if audio.shape[1] != channels:
        raise ValueError(
            f"audio shape {audio.shape} incompatible with channels={channels}"
        )
    return audio


def encode_wav(
    audio: np.ndarray, sample_rate: int, channels: int = 1,
) -> bytes:
    """Convert float32 [-1, 1] audio to a 16-bit PCM WAV bytestring.

    audio shape: (samples,) for mono OR (samples, channels) for multi.
    Always available (stdlib `wave`).
    """
    a = _normalize_shape(audio, channels)
    pcm = _to_int16_pcm(a)
    if channels > 1:
        # Interleave: wave expects bytes laid out frame-by-frame.
        pcm_bytes = pcm.tobytes()  # numpy is C-contiguous: row-major
    else:
        pcm_bytes = pcm.tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def encode_flac(
    audio: np.ndarray, sample_rate: int, channels: int = 1,
) -> bytes:
    """Encode FLAC via soundfile.

    Raises UnsupportedFormatError when `soundfile` is missing.
    """
    sf = _try_import_soundfile()
    if sf is None:
        raise UnsupportedFormatError(
            "flac response_format requires soundfile; "
            "install via `pip install soundfile` or use wav"
        )
    a = _normalize_shape(audio, channels).astype(np.float32, copy=False)
    buf = io.BytesIO()
    sf.write(buf, a, sample_rate, format="FLAC")
    return buf.getvalue()


def encode_mp3(
    audio: np.ndarray, sample_rate: int, channels: int = 1,
) -> bytes:
    """Encode MP3 by writing WAV in-memory then transcoding via pydub.

    Raises UnsupportedFormatError when pydub or ffmpeg is missing.
    """
    return _wav_to(audio, sample_rate, channels, fmt="mp3")


def encode_opus(
    audio: np.ndarray, sample_rate: int, channels: int = 1,
) -> bytes:
    """Encode Opus (OGG container) via pydub.

    Raises UnsupportedFormatError when pydub or ffmpeg is missing.
    """
    return _wav_to(audio, sample_rate, channels, fmt="opus")


def _wav_to(
    audio: np.ndarray, sample_rate: int, channels: int, *, fmt: str,
) -> bytes:
    """Internal helper: WAV via stdlib, then pydub/ffmpeg to the target."""
    pydub = _try_import_pydub()
    if pydub is None:
        raise UnsupportedFormatError(
            f"{fmt} response_format requires pydub; "
            f"install via `pip install pydub` or use wav/flac"
        )
    if shutil.which("ffmpeg") is None:
        raise UnsupportedFormatError(
            f"{fmt} response_format requires ffmpeg on PATH; "
            f"install ffmpeg or use wav/flac"
        )
    wav_bytes = encode_wav(audio, sample_rate, channels)
    seg = pydub.AudioSegment.from_wav(io.BytesIO(wav_bytes))
    out = io.BytesIO()
    if fmt == "mp3":
        seg.export(out, format="mp3", bitrate="128k")
    elif fmt == "opus":
        # pydub uses ffmpeg for opus inside an OGG container.
        seg.export(out, format="opus", bitrate="64k")
    else:
        raise ValueError(f"_wav_to: unknown fmt {fmt!r}")
    return out.getvalue()


def content_type_for(response_format: str) -> str:
    """Map response_format -> Content-Type header value."""
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "flac": "audio/flac",
    }[response_format]


def _try_import_soundfile() -> Any:
    try:
        import soundfile
        return soundfile
    except ImportError:
        return None


def _try_import_pydub() -> Any:
    try:
        import pydub
        return pydub
    except ImportError:
        return None
