"""Bounded mono TorchCodec decoding shared by the audio modalities.

Used by audio/alignment, audio/quality and the Qwen3-ASR transcription
runtime. Failures are typed so routes never guess from message text:

- `AudioDecodeError`: the upload is not decodable audio (routes map it to
  415).
- `AudioDurationExceededError`: the audio is longer than the request's cap
  (413).
- `AudioDecoderUnavailableError`: TorchCodec or FFmpeg is missing in the
  model's venv, a server problem (500).

TorchCodec is imported lazily into the `AudioDecoder` sentinel; tests
replace the sentinel with a fake.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


AudioDecoder: Any = None


class AudioDecodeError(ValueError):
    """The uploaded payload could not be decoded as supported audio."""


class AudioDecoderUnavailableError(RuntimeError):
    """The audio decoder cannot be imported in this environment."""


class AudioDurationExceededError(ValueError):
    """Decoded audio exceeds the amount of work allowed for one request."""

    def __init__(
        self,
        *,
        maximum_seconds: float,
        actual_seconds: float | None = None,
    ) -> None:
        self.maximum_seconds = float(maximum_seconds)
        self.actual_seconds = (
            None if actual_seconds is None else float(actual_seconds)
        )
        detail = (
            f" ({self.actual_seconds:.3f}s received)"
            if self.actual_seconds is not None
            else ""
        )
        super().__init__(
            f"audio duration exceeds {self.maximum_seconds:.3f}s{detail}"
        )


def _ensure_decoder() -> None:
    global AudioDecoder
    if AudioDecoder is not None:
        return
    try:
        from torchcodec.decoders import AudioDecoder as _AudioDecoder
    except Exception as exc:  # noqa: BLE001
        raise AudioDecoderUnavailableError(
            "TorchCodec audio decoder is unavailable; run `muse pull` and "
            "ensure FFmpeg shared libraries are installed"
        ) from exc
    AudioDecoder = _AudioDecoder


def _open_decoder(audio_path: str, sample_rate: int) -> Any:
    _ensure_decoder()
    try:
        return AudioDecoder(
            audio_path, sample_rate=int(sample_rate), num_channels=1,
        )
    except RuntimeError as exc:
        raise AudioDecodeError("audio decoder could not open input") from exc


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


def _decode_range(
    decoder: Any,
    *,
    start_seconds: float,
    stop_seconds: float,
    allow_eof: bool,
) -> Any | None:
    """Decode one range, normalizing TorchCodec's EOF exception.

    Returns None at EOF when `allow_eof`; any other decoder failure becomes
    `AudioDecodeError` (without the codec's message, which can carry paths).
    """
    try:
        return decoder.get_samples_played_in_range(
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
    except RuntimeError as exc:
        no_frames = "no audio frames were decoded" in str(exc).lower()
        if allow_eof and no_frames:
            return None
        raise AudioDecodeError(
            "audio decoder failed while reading input"
        ) from exc


def _mono(waveform: Any) -> Any:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def _check_reported_duration(
    decoder: Any, maximum_seconds: float, tolerance: float,
) -> float | None:
    reported = _metadata_duration(decoder.metadata)
    if reported is not None and reported > maximum_seconds + tolerance:
        raise AudioDurationExceededError(
            maximum_seconds=maximum_seconds, actual_seconds=reported,
        )
    return reported


def _probe_overflow(
    decoder: Any, maximum_seconds: float, probe_seconds: float,
) -> None:
    """Reject audio that has any sample past `maximum_seconds`.

    Container metadata is not trusted as the only guard.
    """
    overflow = _decode_range(
        decoder,
        start_seconds=maximum_seconds,
        stop_seconds=maximum_seconds + probe_seconds,
        allow_eof=True,
    )
    if overflow is not None and int(overflow.data.shape[-1]) > 0:
        raise AudioDurationExceededError(maximum_seconds=maximum_seconds)


@dataclass(frozen=True)
class DecodedAudio:
    """One bounded mono waveform, retained on CPU until inference."""

    waveform: Any
    sample_rate: int
    duration_seconds: float


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
    decoder = _open_decoder(audio_path, sample_rate)
    one_sample = 1.0 / sample_rate
    _check_reported_duration(decoder, max_duration_seconds, one_sample)

    samples = _decode_range(
        decoder,
        start_seconds=0.0,
        stop_seconds=max_duration_seconds,
        allow_eof=False,
    )
    waveform = _mono(samples.data)
    sample_count = int(waveform.shape[-1])
    if sample_count <= 0:
        raise AudioDecodeError("audio decoder returned no samples")
    duration = sample_count / sample_rate

    if duration >= max_duration_seconds - one_sample:
        _probe_overflow(decoder, max_duration_seconds, 0.1)

    return DecodedAudio(
        waveform=waveform,
        sample_rate=int(sample_rate),
        duration_seconds=duration,
    )


@dataclass(frozen=True)
class AudioWindow:
    """One decoded mono range, retained on CPU until inference."""

    waveform: Any
    sample_rate: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class WindowedAudio:
    """Decode fixed ranges without ever materializing the complete waveform."""

    def __init__(
        self,
        audio_path: str,
        *,
        sample_rate: int = 16000,
        window_seconds: float = 10.0,
        max_duration_seconds: float = 600.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self._decoder = _open_decoder(audio_path, self.sample_rate)
        self.source_duration_seconds = _check_reported_duration(
            self._decoder, self.max_duration_seconds, 1e-6,
        )

    def __iter__(self) -> Iterator[AudioWindow]:
        start = 0.0
        saw_samples = False
        reached_limit = False
        one_sample = 1.0 / self.sample_rate

        while start < self.max_duration_seconds - one_sample:
            stop = min(
                start + self.window_seconds,
                self.max_duration_seconds,
            )
            samples = _decode_range(
                self._decoder,
                start_seconds=start,
                stop_seconds=stop,
                allow_eof=saw_samples,
            )
            if samples is None:
                break
            waveform = _mono(samples.data)
            sample_count = int(waveform.shape[-1])
            if sample_count <= 0:
                break

            saw_samples = True
            decoded_seconds = sample_count / self.sample_rate
            sample_pts = getattr(samples, "pts_seconds", None)
            sample_start = (
                start if sample_pts is None else float(sample_pts)
            )
            sample_end = min(sample_start + decoded_seconds, stop)
            yield AudioWindow(
                waveform=waveform,
                sample_rate=self.sample_rate,
                start_seconds=sample_start,
                end_seconds=sample_end,
            )

            # Encoder delay can make the first compressed range shorter than
            # requested even though its samples still reach ``stop``. Use the
            # decoded presentation range, rather than fallible container
            # duration metadata, to distinguish that case from a true EOF.
            if sample_end + one_sample < stop:
                break
            start = stop
            reached_limit = (
                start >= self.max_duration_seconds - one_sample
            )

        if not saw_samples:
            raise AudioDecodeError("audio decoder returned no samples")

        # If every allowed range was full, probe just beyond the boundary.
        if reached_limit:
            _probe_overflow(
                self._decoder,
                self.max_duration_seconds,
                min(self.window_seconds, 0.1),
            )
