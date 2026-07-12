"""Bounded 16kHz mono decoding for audio-alignment runtimes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from muse.modalities.audio_alignment.protocol import (
    AudioAlignmentDecodeError,
    AudioAlignmentDurationExceededError,
)


AudioDecoder: Any = None


def _ensure_decoder() -> None:
    global AudioDecoder
    if AudioDecoder is not None:
        return
    try:
        from torchcodec.decoders import AudioDecoder as _AudioDecoder
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TorchCodec audio decoder is unavailable; run `muse pull` and "
            "ensure FFmpeg shared libraries are installed"
        ) from exc
    AudioDecoder = _AudioDecoder


def _metadata_duration(metadata: Any) -> float | None:
    """Read the best duration exposed across TorchCodec releases."""
    for name in (
        "duration_seconds",
        "duration_seconds_from_content",
        "duration_seconds_from_header",
    ):
        value = getattr(metadata, name, None)
        if value is not None and float(value) >= 0:
            return float(value)
    begin = getattr(metadata, "begin_stream_seconds", None)
    end = getattr(metadata, "end_stream_seconds", None)
    if begin is not None and end is not None and float(end) >= float(begin):
        return float(end) - float(begin)
    return None


@dataclass(frozen=True)
class DecodedAudio:
    """One bounded mono waveform, retained on CPU until inference."""

    waveform: Any
    sample_rate: int
    duration_seconds: float


def _decode_range(
    decoder: Any,
    *,
    start_seconds: float,
    stop_seconds: float,
    allow_eof: bool,
) -> Any | None:
    try:
        return decoder.get_samples_played_in_range(
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
    except RuntimeError as exc:
        no_frames = "no audio frames were decoded" in str(exc).lower()
        if allow_eof and no_frames:
            return None
        raise AudioAlignmentDecodeError(
            "audio decoder failed while reading input"
        ) from exc


def decode_audio(
    audio_path: str,
    *,
    sample_rate: int = 16000,
    max_duration_seconds: float = 300.0,
) -> DecodedAudio:
    """Decode at most ``max_duration_seconds`` and probe for overflow."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")
    _ensure_decoder()
    try:
        decoder = AudioDecoder(
            audio_path,
            sample_rate=int(sample_rate),
            num_channels=1,
        )
    except RuntimeError as exc:
        raise AudioAlignmentDecodeError(
            "audio decoder could not open input"
        ) from exc
    reported_duration = _metadata_duration(decoder.metadata)
    one_sample = 1.0 / sample_rate
    if (
        reported_duration is not None
        and reported_duration > max_duration_seconds + one_sample
    ):
        raise AudioAlignmentDurationExceededError(
            maximum_seconds=max_duration_seconds,
            actual_seconds=reported_duration,
        )

    samples = _decode_range(
        decoder,
        start_seconds=0.0,
        stop_seconds=max_duration_seconds,
        allow_eof=False,
    )
    waveform = samples.data
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    sample_count = int(waveform.shape[-1])
    if sample_count <= 0:
        raise AudioAlignmentDecodeError("audio decoder returned no samples")
    duration = sample_count / sample_rate

    if duration >= max_duration_seconds - one_sample:
        overflow = _decode_range(
            decoder,
            start_seconds=max_duration_seconds,
            stop_seconds=max_duration_seconds + 0.1,
            allow_eof=True,
        )
        if overflow is not None and int(overflow.data.shape[-1]) > 0:
            raise AudioAlignmentDurationExceededError(
                maximum_seconds=max_duration_seconds,
            )

    return DecodedAudio(
        waveform=waveform,
        sample_rate=int(sample_rate),
        duration_seconds=duration,
    )
