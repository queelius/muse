"""Audio encoding for audio/generation responses.

Pure functions: numpy audio array + sample_rate + channels -> bytes.

WAV uses stdlib `wave` via the shared `_common.audio_codec` writer; always
available. FLAC uses `soundfile` (already in museq[audio] for Kokoro).
Opus uses the shared owned-ffmpeg encoder (the same one audio/speech
uses); MP3 goes through `pydub` + `ffmpeg`. A missing dependency raises
UnsupportedFormatError so the route layer can return a clean 400.

The audio array shape is `(samples,)` for mono or `(samples, channels)`
for multi-channel. Codec normalizes both into the right interleaved
PCM layout for the WAV format and into the right shape for soundfile.
"""
from __future__ import annotations

import io
import shutil
from typing import Any

import numpy as np

from muse.modalities._common.audio_codec import (
    FfmpegUnavailableError,
    pcm16_wav_bytes,
    wav_bytes_to_opus,
)


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


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
    return pcm16_wav_bytes(_normalize_shape(audio, channels), sample_rate, channels)


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
    """Encode Opus (OGG container) with the shared owned-ffmpeg encoder.

    Raises UnsupportedFormatError when ffmpeg is missing; any other
    encoding failure raises AudioFormatError.
    """
    wav = encode_wav(audio, sample_rate, channels)
    try:
        return wav_bytes_to_opus(wav)
    except FfmpegUnavailableError as exc:
        raise UnsupportedFormatError(
            "opus response_format requires ffmpeg on PATH; "
            "install ffmpeg or use wav/flac"
        ) from exc


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
