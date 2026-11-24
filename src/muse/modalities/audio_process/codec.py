"""Audio decoding and encoding for ``audio/process``.

Decodes WAV via stdlib ``wave`` (8/16/24/32-bit PCM) with a ``soundfile``
fallback for IEEE-float WAV, FLAC, OGG, and other libsndfile formats.
Encodes WAV via shared ``_common.audio_codec``, FLAC via ``soundfile``, and
Opus via owned ``ffmpeg``.

``wave`` lives here (never in ``runtime.py`` or ``models/audio_tools.py``) so
the bundled-script ``pip_extras`` static audit only inspects runtime DSP
imports.
"""
from __future__ import annotations

import base64
import io
import wave
from typing import Any

import numpy as np

from muse.modalities._common.audio_codec import (
    FfmpegUnavailableError,
    pcm16_wav_bytes,
    wav_bytes_to_opus,
)
from muse.modalities._common.audio_decode import AudioDecodeError
from muse.modalities.audio_process.protocol import AudioProcessResult

ALLOWED_OUTPUT_FORMATS = ("wav", "flac", "opus")


class UnsupportedFormatError(Exception):
    """Raised when an output_format requires deps that aren't installed."""


def _try_import_soundfile() -> Any:
    try:
        import soundfile
        return soundfile
    except ImportError:
        return None


def _decode_pcm24(raw: bytes) -> np.ndarray:
    """Decode little-endian signed 24-bit PCM bytes into float32 in [-1, 1]."""
    if len(raw) % 3 != 0:
        raise AudioDecodeError("corrupt 24-bit WAV payload")
    triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    # Sign-extend the 3rd byte (MSB) into a 32-bit signed int.
    signed = (
        triplets[:, 0].astype(np.int32)
        | (triplets[:, 1].astype(np.int32) << 8)
        | (triplets[:, 2].astype(np.int32) << 16)
    )
    signed = np.where(signed >= 0x800000, signed - 0x1000000, signed)
    return (signed.astype(np.float32) / 8388608.0)


def _decode_wav_stdlib(data: bytes) -> tuple[np.ndarray, int]:
    """Decode uncompressed PCM WAV via stdlib ``wave``."""
    with wave.open(io.BytesIO(data), "rb") as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if framerate <= 0 or nchannels <= 0:
        raise AudioDecodeError("invalid WAV header")
    if not raw:
        raise AudioDecodeError("audio file contains no frames")

    if sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        samples = _decode_pcm24(raw)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioDecodeError(f"unsupported WAV sample width: {sampwidth} bytes")

    if nchannels > 1:
        usable = (samples.size // nchannels) * nchannels
        if usable == 0:
            raise AudioDecodeError("audio file contains no complete frames")
        samples = samples[:usable].reshape(-1, nchannels)

    return np.ascontiguousarray(samples, dtype=np.float32), int(framerate)


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode raw uploaded bytes into ``(float32_audio, sample_rate)``.

    Tries stdlib ``wave`` first for PCM WAV payloads, then falls back to
    ``soundfile`` for FLAC/OGG/float-WAV/etc. Raises :class:`AudioDecodeError`
    when the bytes cannot be decoded.
    """
    if not data:
        raise AudioDecodeError("audio payload is empty")

    # Fast path: stdlib wave for standard RIFF/WAVE PCM files.
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            return _decode_wav_stdlib(data)
        except (wave.Error, EOFError, AudioDecodeError):
            # Fall through to soundfile (e.g. IEEE float WAV / extensible WAV).
            pass

    sf = _try_import_soundfile()
    if sf is not None:
        try:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            raise AudioDecodeError("could not decode audio file") from exc
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            raise AudioDecodeError("audio file contains no frames")
        return np.ascontiguousarray(arr, dtype=np.float32), int(sr)

    raise AudioDecodeError(
        "could not decode audio file; provide a PCM WAV file or install soundfile for FLAC/OGG"
    )


def _normalize_shape(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2 and audio.shape[1] == 1:
            return audio[:, 0]
        raise ValueError(f"audio shape {audio.shape} incompatible with channels=1")
    if audio.ndim == 1 or audio.shape[1] != channels:
        raise ValueError(f"audio shape {audio.shape} incompatible with channels={channels}")
    return audio


def encode_wav(audio: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Encode float32 [-1, 1] audio to 16-bit PCM WAV bytes."""
    return pcm16_wav_bytes(_normalize_shape(audio, channels), sample_rate, channels)


def encode_flac(audio: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Encode FLAC via ``soundfile``."""
    sf = _try_import_soundfile()
    if sf is None:
        raise UnsupportedFormatError(
            "flac output_format requires soundfile; install via `pip install soundfile` or use wav"
        )
    arr = _normalize_shape(audio, channels).astype(np.float32, copy=False)
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="FLAC")
    return buf.getvalue()


def encode_opus(audio: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Encode Opus (OGG container) via owned ``ffmpeg``."""
    wav_data = encode_wav(audio, sample_rate, channels)
    try:
        return wav_bytes_to_opus(wav_data)
    except FfmpegUnavailableError as exc:
        raise UnsupportedFormatError(
            "opus output_format requires ffmpeg on PATH; install ffmpeg or use wav/flac"
        ) from exc


def encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    channels: int,
    output_format: str,
) -> bytes:
    """Dispatch audio encoding by ``output_format`` (wav | flac | opus)."""
    if output_format == "wav":
        return encode_wav(audio, sample_rate, channels)
    if output_format == "flac":
        return encode_flac(audio, sample_rate, channels)
    if output_format == "opus":
        return encode_opus(audio, sample_rate, channels)
    raise UnsupportedFormatError(f"unknown output_format: {output_format!r}")


def content_type_for(output_format: str) -> str:
    """Map ``output_format`` to its HTTP Content-Type."""
    return {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "opus": "audio/ogg",
    }[output_format]


def encode_json_response(
    result: AudioProcessResult,
    *,
    model_id: str,
    output_format: str,
    audio_bytes: bytes,
) -> dict[str, Any]:
    """Build the JSON response envelope for ``POST /v1/audio/process``."""
    return {
        "object": "audio.process",
        "model": model_id,
        "format": output_format,
        "b64_audio": base64.b64encode(audio_bytes).decode("ascii"),
        "metrics": result.metrics,
    }
