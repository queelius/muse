"""Pure-DSP audio mastering and post-processing pipeline (numpy + scipy).

Transforms float32 audio arrays (``(frames,)`` mono or ``(frames, channels)``
multi-channel) into game-ready clips:

1. Optional mono downmix.
2. Optional sample-rate resampling (``scipy.signal.resample_poly`` with a
   linear interpolation fallback when scipy is absent).
3. Optional DC-offset removal.
4. Either seamless equal-power crossfade looping (``loop_fade_s > 0``) OR
   silence trimming + click-free micro-fades.
5. Optional peak normalization with a hard PCM16-safe ceiling (``<= 0.999``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Hard ceiling just below 1.0 so float -> int16 PCM conversion
# (``* 32768`` clipped to ``[-32768, 32767]``) never clips.
PCM16_CEILING = 0.999
# Largest float32 value strictly <= PCM16_CEILING (avoids float32 rounding
# 0.999 up to 0.9990000128746033 when converted back to Python float).
_PCM16_CEILING_F32 = np.float32(
    np.nextafter(np.float32(PCM16_CEILING), np.float32(0.0))
    if float(np.float32(PCM16_CEILING)) > PCM16_CEILING
    else PCM16_CEILING
)

# Floor for dBFS calculations so silence produces a finite JSON-safe float.
_DBFS_FLOOR_AMP = 1e-6  # -120.0 dBFS


@dataclass
class ProcessedAudio:
    """Output of :func:`process_audio` with detailed measurement metrics."""

    audio: np.ndarray
    sample_rate: int
    channels: int
    metrics: dict[str, Any]


def _amp_to_dbfs(amplitude: float) -> float:
    """Convert linear amplitude >= 0 to dBFS, clamped at -120.0 dBFS."""
    return float(20.0 * math.log10(max(float(amplitude), _DBFS_FLOOR_AMP)))


def _try_import_resample_poly() -> Any:
    try:
        from scipy.signal import resample_poly
        return resample_poly
    except ImportError:
        return None


def _resample_2d(work: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample ``(frames, channels)`` float32 audio from ``src_sr`` to ``dst_sr``."""
    if src_sr == dst_sr:
        return work
    resample_poly = _try_import_resample_poly()
    if resample_poly is not None:
        g = math.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return resample_poly(work, up, down, axis=0).astype(np.float32, copy=False)

    # Pure-numpy linear interpolation fallback when scipy is absent.
    src_len = work.shape[0]
    dst_len = max(1, int(round(src_len * dst_sr / src_sr)))
    if src_len == 1:
        return np.repeat(work, dst_len, axis=0).astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    out = np.empty((dst_len, work.shape[1]), dtype=np.float32)
    for ch in range(work.shape[1]):
        out[:, ch] = np.interp(new_x, old_x, work[:, ch]).astype(np.float32)
    return out


def _equal_power_loop(work: np.ndarray, fade_samples: int) -> tuple[np.ndarray, bool]:
    """Fold the last ``fade_samples`` frames into the first ``fade_samples``
    frames using equal-power sin/cos weights so end -> start joins seamlessly.
    """
    n = work.shape[0]
    if n < 2 or fade_samples < 1:
        return work, False
    k = min(fade_samples, n // 2)
    if k < 1:
        return work, False

    head = work[:k]
    body = work[k : n - k]
    tail = work[n - k :]

    if k == 1:
        # Single-sample crossfade: start directly from tail[0] for continuity.
        crossfaded = tail.copy()
    else:
        t = np.linspace(0.0, 0.5 * math.pi, num=k, endpoint=True, dtype=np.float32)
        fade_in = np.sin(t)[:, np.newaxis]
        fade_out = np.cos(t)[:, np.newaxis]
        crossfaded = head * fade_in + tail * fade_out

    out = np.concatenate([crossfaded, body], axis=0).astype(np.float32, copy=False)
    return out, True


def _trim_silence(
    work: np.ndarray,
    sr: int,
    *,
    threshold_db: float,
    pad_lead_ms: float,
    pad_tail_ms: float,
) -> tuple[np.ndarray, float, float]:
    """Trim leading and trailing samples below ``threshold_db`` while keeping
    requested lead/tail padding. Pure silence is preserved untouched.
    """
    thresh = 10.0 ** (threshold_db / 20.0)
    envelope = np.max(np.abs(work), axis=1)
    active = np.flatnonzero(envelope >= thresh)
    if active.size == 0:
        return work, 0.0, 0.0

    lead_pad = int(round(pad_lead_ms * sr / 1000.0))
    tail_pad = int(round(pad_tail_ms * sr / 1000.0))
    start = max(0, int(active[0]) - lead_pad)
    end = min(work.shape[0], int(active[-1]) + 1 + tail_pad)

    trimmed_leading_ms = float(start * 1000.0 / sr)
    trimmed_trailing_ms = float((work.shape[0] - end) * 1000.0 / sr)
    return work[start:end].copy(), trimmed_leading_ms, trimmed_trailing_ms


def _apply_micro_fades(
    work: np.ndarray,
    sr: int,
    *,
    fade_in_ms: float,
    fade_out_ms: float,
) -> np.ndarray:
    """Apply click-free linear fade-in and fade-out ramps at clip boundaries."""
    n = work.shape[0]
    if n < 2 or (fade_in_ms <= 0.0 and fade_out_ms <= 0.0):
        return work

    out = work.copy()
    max_ramp = n // 2

    if fade_in_ms > 0.0:
        fi = min(int(round(fade_in_ms * sr / 1000.0)), max_ramp)
        if fi == 1:
            out[0, :] = 0.0
        elif fi > 1:
            ramp = np.linspace(0.0, 1.0, num=fi, endpoint=True, dtype=np.float32)[:, np.newaxis]
            out[:fi] *= ramp

    if fade_out_ms > 0.0:
        fo = min(int(round(fade_out_ms * sr / 1000.0)), max_ramp)
        if fo == 1:
            out[-1, :] = 0.0
        elif fo > 1:
            ramp = np.linspace(1.0, 0.0, num=fo, endpoint=True, dtype=np.float32)[:, np.newaxis]
            out[-fo:] *= ramp

    return out


def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    mono: bool = False,
    target_sample_rate: int | None = None,
    remove_dc: bool = True,
    trim_silence: bool = True,
    trim_threshold_db: float = -42.0,
    pad_lead_ms: float = 20.0,
    pad_tail_ms: float = 65.0,
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 15.0,
    loop_fade_s: float = 0.0,
    target_peak_dbfs: float | None = -3.0,
) -> ProcessedAudio:
    """Run the deterministic mastering pipeline on a float32 audio array.

    Args:
        audio: 1-D ``(frames,)`` mono or 2-D ``(frames, channels)`` audio array.
        sample_rate: Input sample rate in Hz (must be positive).
        mono: Downmix multi-channel audio to mono by channel averaging.
        target_sample_rate: Optional output sample rate in Hz.
        remove_dc: Subtract per-channel mean offset.
        trim_silence: Trim leading/trailing silence (ignored when ``loop_fade_s > 0``).
        trim_threshold_db: Amplitude threshold in dBFS for silence trimming.
        pad_lead_ms: Leading padding retained before first active sample (ms).
        pad_tail_ms: Trailing padding retained after last active sample (ms).
        fade_in_ms: Fade-in ramp length (ms) in non-loop mode.
        fade_out_ms: Fade-out ramp length (ms) in non-loop mode.
        loop_fade_s: When > 0, apply equal-power tail-into-head crossfade looping.
        target_peak_dbfs: Optional peak normalization target in dBFS (<= 0.0).

    Returns:
        :class:`ProcessedAudio` with float32 audio and a metrics dict.
    """
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be positive")
    if target_sample_rate is not None and int(target_sample_rate) <= 0:
        raise ValueError("target_sample_rate must be positive")
    if not math.isfinite(trim_threshold_db):
        raise ValueError("trim_threshold_db must be finite")
    if pad_lead_ms < 0.0 or not math.isfinite(pad_lead_ms):
        raise ValueError("pad_lead_ms must be non-negative and finite")
    if pad_tail_ms < 0.0 or not math.isfinite(pad_tail_ms):
        raise ValueError("pad_tail_ms must be non-negative and finite")
    if fade_in_ms < 0.0 or not math.isfinite(fade_in_ms):
        raise ValueError("fade_in_ms must be non-negative and finite")
    if fade_out_ms < 0.0 or not math.isfinite(fade_out_ms):
        raise ValueError("fade_out_ms must be non-negative and finite")
    if loop_fade_s < 0.0 or not math.isfinite(loop_fade_s):
        raise ValueError("loop_fade_s must be non-negative and finite")
    if target_peak_dbfs is not None:
        if not math.isfinite(target_peak_dbfs) or target_peak_dbfs > 0.0:
            raise ValueError("target_peak_dbfs must be <= 0.0 and finite")

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        work = arr[:, np.newaxis].copy()
    elif arr.ndim == 2:
        if arr.shape[1] < 1:
            raise ValueError("audio must have at least one channel")
        work = arr.copy()
    else:
        raise ValueError(f"expected 1-D or 2-D audio array, got shape {arr.shape}")

    if work.shape[0] == 0:
        raise ValueError("audio is empty")

    if not np.all(np.isfinite(work)):
        work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)

    raw_frames = int(work.shape[0])
    raw_duration_s = float(raw_frames / sr)
    raw_peak = float(np.max(np.abs(work)))
    raw_peak_dbfs = _amp_to_dbfs(raw_peak)

    # 1. Optional mono downmix
    if mono and work.shape[1] > 1:
        work = np.mean(work, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)

    # 2. Optional sample-rate resampling
    if target_sample_rate is not None and int(target_sample_rate) != sr:
        dst_sr = int(target_sample_rate)
        work = _resample_2d(work, sr, dst_sr)
        sr = dst_sr

    # 3. DC-offset removal
    if remove_dc:
        dc = np.mean(work, axis=0, keepdims=True, dtype=np.float64)
        work = (work.astype(np.float64) - dc).astype(np.float32)

    # 4. Seamless equal-power crossfade loop OR silence trim + micro-fades
    trimmed_leading_ms = 0.0
    trimmed_trailing_ms = 0.0
    looped = False

    if loop_fade_s > 0.0:
        fade_samples = int(round(loop_fade_s * sr))
        work, looped = _equal_power_loop(work, fade_samples)
    else:
        if trim_silence:
            work, trimmed_leading_ms, trimmed_trailing_ms = _trim_silence(
                work,
                sr,
                threshold_db=trim_threshold_db,
                pad_lead_ms=pad_lead_ms,
                pad_tail_ms=pad_tail_ms,
            )
        work = _apply_micro_fades(
            work,
            sr,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
        )

    # 5. Optional peak normalization + hard PCM16-safe ceiling
    if target_peak_dbfs is not None:
        cur_peak = float(np.max(np.abs(work)))
        if cur_peak > 1e-8:
            target_linear = min(10.0 ** (target_peak_dbfs / 20.0), float(_PCM16_CEILING_F32))
            gain = target_linear / cur_peak
            work = (work * gain).astype(np.float32, copy=False)

    clipped_samples = int(np.count_nonzero(np.abs(work) > PCM16_CEILING + 1e-6))
    work = np.clip(work, -_PCM16_CEILING_F32, _PCM16_CEILING_F32).astype(
        np.float32, copy=False
    )

    frames = int(work.shape[0])
    channels = int(work.shape[1])
    peak = float(np.max(np.abs(work)))
    rms = float(math.sqrt(float(np.mean(work.astype(np.float64) ** 2))))

    metrics: dict[str, Any] = {
        "duration_s": float(frames / sr),
        "raw_duration_s": raw_duration_s,
        "sample_rate_hz": sr,
        "channels": channels,
        "frames": frames,
        "peak": peak,
        "peak_dbfs": _amp_to_dbfs(peak),
        "rms": rms,
        "rms_dbfs": _amp_to_dbfs(rms),
        "raw_peak_dbfs": raw_peak_dbfs,
        "clipped_samples": clipped_samples,
        "trimmed_leading_ms": trimmed_leading_ms,
        "trimmed_trailing_ms": trimmed_trailing_ms,
        "looped": looped,
    }

    if channels == 1:
        out_audio = np.ascontiguousarray(work[:, 0], dtype=np.float32)
    else:
        out_audio = np.ascontiguousarray(work, dtype=np.float32)

    return ProcessedAudio(
        audio=out_audio,
        sample_rate=sr,
        channels=channels,
        metrics=metrics,
    )
