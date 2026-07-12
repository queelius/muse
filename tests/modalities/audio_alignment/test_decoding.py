from types import SimpleNamespace

import pytest
import torch

from muse.modalities.audio_alignment import (
    AudioAlignmentDecodeError,
    AudioAlignmentDurationExceededError,
)
from muse.modalities.audio_alignment import decoding


def _decoder(
    monkeypatch,
    *,
    actual_seconds,
    reported_seconds=None,
    channels=1,
    raise_at_eof=False,
):
    instances = []

    class _Decoder:
        def __init__(self, path, *, sample_rate, num_channels):
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.metadata = SimpleNamespace(duration_seconds=(
                actual_seconds
                if reported_seconds is None
                else reported_seconds
            ))
            self.calls = []
            instances.append(self)

        def get_samples_played_in_range(self, *, start_seconds, stop_seconds):
            self.calls.append((start_seconds, stop_seconds))
            if raise_at_eof and start_seconds >= actual_seconds:
                raise RuntimeError("No audio frames were decoded")
            duration = max(
                0.0, min(stop_seconds, actual_seconds) - start_seconds,
            )
            count = round(duration * self.sample_rate)
            return SimpleNamespace(data=torch.ones(channels, count))

    monkeypatch.setattr(decoding, "AudioDecoder", _Decoder)
    return instances


def test_decodes_one_bounded_mono_waveform(monkeypatch):
    instances = _decoder(monkeypatch, actual_seconds=2, channels=2)
    result = decoding.decode_audio(
        "clip.mp3", sample_rate=10, max_duration_seconds=5,
    )
    assert result.duration_seconds == 2.0
    assert result.waveform.shape == (1, 20)
    assert instances[0].num_channels == 1
    assert instances[0].calls == [(0.0, 5)]


def test_rejects_duration_from_metadata_before_decoding(monkeypatch):
    instances = _decoder(monkeypatch, actual_seconds=12)
    with pytest.raises(AudioAlignmentDurationExceededError, match="10.000s"):
        decoding.decode_audio(
            "long.mp3", sample_rate=10, max_duration_seconds=10,
        )
    assert instances[0].calls == []


def test_probes_past_limit_when_metadata_underreports(monkeypatch):
    instances = _decoder(
        monkeypatch, actual_seconds=11, reported_seconds=10,
    )
    with pytest.raises(AudioAlignmentDurationExceededError):
        decoding.decode_audio(
            "misleading.mp3", sample_rate=10, max_duration_seconds=10,
        )
    assert instances[0].calls[-1] == (10, 10.1)


def test_empty_decode_fails_clearly(monkeypatch):
    _decoder(monkeypatch, actual_seconds=0)
    with pytest.raises(AudioAlignmentDecodeError, match="no samples"):
        decoding.decode_audio(
            "empty.wav", sample_rate=10, max_duration_seconds=10,
        )


def test_decoder_runtime_failure_becomes_typed_input_error(monkeypatch):
    class _Decoder:
        def __init__(self, path, *, sample_rate, num_channels):
            self.metadata = SimpleNamespace(duration_seconds=1)

        def get_samples_played_in_range(self, **kwargs):
            raise RuntimeError("codec-specific private detail")

    monkeypatch.setattr(decoding, "AudioDecoder", _Decoder)
    with pytest.raises(AudioAlignmentDecodeError) as exc_info:
        decoding.decode_audio(
            "corrupt.wav", sample_rate=10, max_duration_seconds=10,
        )
    assert "private detail" not in str(exc_info.value)


def test_exact_limit_treats_no_frames_as_eof(monkeypatch):
    instances = _decoder(
        monkeypatch, actual_seconds=10, raise_at_eof=True,
    )
    result = decoding.decode_audio(
        "ten.wav", sample_rate=10, max_duration_seconds=10,
    )
    assert result.duration_seconds == 10
    assert instances[0].calls[-1] == (10, 10.1)
