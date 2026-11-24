"""Unit tests for the pure-DSP pipeline in ``muse.modalities.audio_process.dsp``."""
from __future__ import annotations

import json
import math
from unittest.mock import patch

import numpy as np
import pytest

from muse.modalities.audio_process.dsp import PCM16_CEILING, process_audio

EXPECTED_METRIC_KEYS = {
    "duration_s",
    "raw_duration_s",
    "sample_rate_hz",
    "channels",
    "frames",
    "peak",
    "peak_dbfs",
    "rms",
    "rms_dbfs",
    "raw_peak_dbfs",
    "clipped_samples",
    "trimmed_leading_ms",
    "trimmed_trailing_ms",
    "looped",
}


def _sine(freq_hz: float, duration_s: float, sr: int = 44100, amp: float = 0.5) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_metrics_schema_and_json_serializable():
    clip = _sine(440.0, 0.5, sr=16000, amp=0.4)
    res = process_audio(clip, 16000)
    assert set(res.metrics.keys()) == EXPECTED_METRIC_KEYS
    # Must serialize with strict JSON (no NaN/Inf).
    serialized = json.dumps(res.metrics, allow_nan=False)
    assert serialized
    assert res.audio.dtype == np.float32
    assert res.audio.ndim == 1
    assert res.channels == 1
    assert res.sample_rate == 16000


def test_trim_silence_and_micro_fades():
    sr = 16000
    lead_silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    tone = _sine(440.0, 0.4, sr=sr, amp=0.5)
    tail_silence = np.zeros(int(0.4 * sr), dtype=np.float32)
    clip = np.concatenate([lead_silence, tone, tail_silence])

    res = process_audio(
        clip,
        sr,
        trim_silence=True,
        trim_threshold_db=-42.0,
        pad_lead_ms=20.0,
        pad_tail_ms=65.0,
        fade_in_ms=5.0,
        fade_out_ms=15.0,
        target_peak_dbfs=-3.0,
    )
    assert res.metrics["looped"] is False
    # Leading ~300ms minus ~20ms pad => ~280ms trimmed.
    assert 250.0 <= res.metrics["trimmed_leading_ms"] <= 300.0
    # Trailing ~400ms minus ~65ms pad => ~335ms trimmed.
    assert 310.0 <= res.metrics["trimmed_trailing_ms"] <= 350.0
    assert res.metrics["duration_s"] < res.metrics["raw_duration_s"]
    # Micro-fades pull exact endpoints to 0.0.
    assert abs(float(res.audio[0])) < 1e-6
    assert abs(float(res.audio[-1])) < 1e-6


def test_pure_silence_is_not_trimmed_to_empty_and_metrics_finite():
    sr = 16000
    clip = np.zeros(sr // 2, dtype=np.float32)
    res = process_audio(clip, sr, trim_silence=True, target_peak_dbfs=-3.0)
    assert res.audio.shape == (sr // 2,)
    assert res.metrics["frames"] == sr // 2
    assert res.metrics["trimmed_leading_ms"] == 0.0
    assert res.metrics["trimmed_trailing_ms"] == 0.0
    assert math.isfinite(res.metrics["peak_dbfs"])
    assert math.isfinite(res.metrics["rms_dbfs"])
    assert math.isfinite(res.metrics["raw_peak_dbfs"])
    json.dumps(res.metrics, allow_nan=False)


def test_dc_offset_removal():
    sr = 16000
    clip = _sine(220.0, 0.5, sr=sr, amp=0.25) + 0.35
    res = process_audio(
        clip, sr, remove_dc=True, trim_silence=False,
        fade_in_ms=0.0, fade_out_ms=0.0, target_peak_dbfs=None,
    )
    assert abs(float(np.mean(res.audio))) < 1e-4

    res_keep = process_audio(
        clip, sr, remove_dc=False, trim_silence=False,
        fade_in_ms=0.0, fade_out_ms=0.0, target_peak_dbfs=None,
    )
    assert float(np.mean(res_keep.audio)) == pytest.approx(0.35, abs=1e-3)


def test_peak_normalization_and_pcm16_ceiling():
    sr = 16000
    quiet = _sine(440.0, 0.25, sr=sr, amp=0.1)
    res = process_audio(
        quiet, sr, trim_silence=False, fade_in_ms=0.0, fade_out_ms=0.0,
        target_peak_dbfs=-3.0,
    )
    expected_peak = 10.0 ** (-3.0 / 20.0)
    assert res.metrics["peak"] == pytest.approx(expected_peak, abs=1e-3)
    assert res.metrics["peak_dbfs"] == pytest.approx(-3.0, abs=0.05)
    assert res.metrics["clipped_samples"] == 0
    assert float(np.max(np.abs(res.audio))) <= PCM16_CEILING

    # Even at target_peak_dbfs=0.0, ceiling is hard-enforced <= 0.999 with zero clipping.
    res_zero = process_audio(
        quiet, sr, trim_silence=False, fade_in_ms=0.0, fade_out_ms=0.0,
        target_peak_dbfs=0.0,
    )
    assert float(np.max(np.abs(res_zero.audio))) <= PCM16_CEILING
    assert res_zero.metrics["clipped_samples"] == 0

    # Hot signal without normalization gets hard-clamped to PCM16_CEILING and reports clipped_samples.
    hot = _sine(440.0, 0.1, sr=sr, amp=1.5)
    res_hot = process_audio(
        hot, sr, trim_silence=False, fade_in_ms=0.0, fade_out_ms=0.0,
        target_peak_dbfs=None,
    )
    assert float(np.max(np.abs(res_hot.audio))) <= PCM16_CEILING
    assert res_hot.metrics["clipped_samples"] > 0


def test_equal_power_crossfade_loop_seamless_and_constant_energy():
    sr = 16000
    # Choose a sine frequency whose period divides the final loop length (1.5s) exactly:
    # raw = 2.0s, loop_fade_s = 0.5s => out = 1.5s = 24000 samples.
    # At 100 Hz, 1.5s = 150 full cycles -> phase wraps cleanly.
    raw = _sine(100.0, 2.0, sr=sr, amp=0.5)
    res = process_audio(
        raw,
        sr,
        loop_fade_s=0.5,
        trim_silence=True,  # ignored in loop mode
        target_peak_dbfs=None,
        remove_dc=False,
    )
    assert res.metrics["looped"] is True
    assert res.metrics["trimmed_leading_ms"] == 0.0
    assert res.metrics["trimmed_trailing_ms"] == 0.0
    assert res.metrics["frames"] == int(1.5 * sr)
    assert res.metrics["duration_s"] == pytest.approx(1.5, abs=1e-4)

    # Check wrap discontinuity between last sample and first sample is tiny (smooth sine step).
    wrap_jump = abs(float(res.audio[0]) - float(res.audio[-1]))
    max_step = float(np.max(np.abs(np.diff(res.audio))))
    assert wrap_jump <= max_step * 1.15

    # Equal-power check on uncorrelated stationary noise: crossfade region RMS ~ body RMS.
    rng = np.random.default_rng(42)
    noise = (0.2 * rng.standard_normal(sr * 2)).astype(np.float32)
    res_noise = process_audio(
        noise, sr, loop_fade_s=0.5, target_peak_dbfs=None, remove_dc=False,
    )
    fade_n = int(0.5 * sr)
    rms_fade = float(np.sqrt(np.mean(res_noise.audio[:fade_n] ** 2)))
    rms_body = float(np.sqrt(np.mean(res_noise.audio[fade_n:] ** 2)))
    assert rms_fade == pytest.approx(rms_body, rel=0.1)


def test_mono_downmix_from_stereo():
    sr = 16000
    left = _sine(440.0, 0.2, sr=sr, amp=0.6)
    right = _sine(440.0, 0.2, sr=sr, amp=0.2)
    stereo = np.stack([left, right], axis=1)

    res_stereo = process_audio(
        stereo, sr, mono=False, trim_silence=False,
        fade_in_ms=0.0, fade_out_ms=0.0, target_peak_dbfs=None,
    )
    assert res_stereo.channels == 2
    assert res_stereo.audio.ndim == 2
    assert res_stereo.audio.shape[1] == 2

    res_mono = process_audio(
        stereo, sr, mono=True, trim_silence=False,
        fade_in_ms=0.0, fade_out_ms=0.0, target_peak_dbfs=None,
    )
    assert res_mono.channels == 1
    assert res_mono.audio.ndim == 1
    assert res_mono.metrics["peak"] == pytest.approx(0.4, abs=1e-2)


def test_resampling_scipy_and_linear_fallback():
    src_sr = 44100
    dst_sr = 22050
    clip = _sine(440.0, 0.5, sr=src_sr, amp=0.5)

    res_poly = process_audio(
        clip, src_sr, target_sample_rate=dst_sr,
        trim_silence=False, fade_in_ms=0.0, fade_out_ms=0.0,
        target_peak_dbfs=None,
    )
    assert res_poly.sample_rate == dst_sr
    assert res_poly.metrics["sample_rate_hz"] == dst_sr
    assert res_poly.metrics["frames"] == pytest.approx(int(0.5 * dst_sr), abs=4)

    with patch(
        "muse.modalities.audio_process.dsp._try_import_resample_poly",
        return_value=None,
    ):
        res_lin = process_audio(
            clip, src_sr, target_sample_rate=dst_sr,
            trim_silence=False, fade_in_ms=0.0, fade_out_ms=0.0,
            target_peak_dbfs=None,
        )
    assert res_lin.sample_rate == dst_sr
    assert res_lin.metrics["frames"] == int(0.5 * dst_sr)
    assert res_lin.audio.dtype == np.float32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"target_sample_rate": -16000},
        {"pad_lead_ms": -1.0},
        {"pad_tail_ms": -1.0},
        {"fade_in_ms": -1.0},
        {"fade_out_ms": -1.0},
        {"loop_fade_s": -0.5},
        {"target_peak_dbfs": 1.0},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs):
    clip = _sine(440.0, 0.1, sr=16000)
    sr = kwargs.pop("sample_rate", 16000)
    with pytest.raises(ValueError):
        process_audio(clip, sr, **kwargs)


def test_empty_and_invalid_shape_raise_value_error():
    with pytest.raises(ValueError, match="empty"):
        process_audio(np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        process_audio(np.zeros((2, 2, 2), dtype=np.float32), 16000)
