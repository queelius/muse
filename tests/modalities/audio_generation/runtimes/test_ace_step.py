"""Tests for AceStepRuntime (fully mocked ACEStepPipeline).

ACE-Step's real pipeline writes audio FILES to disk and returns paths
(unlike StableAudio, which returns in-memory arrays). These tests mock
the pipeline so it writes a real temp WAV via the stdlib ``wave`` module,
and mock ``soundfile.read`` with a wave-backed reader so the runtime's
path-resolution + read-back + temp-cleanup logic is exercised end to end
WITHOUT requiring the ``soundfile`` package to be installed in the dev
venv. The real SDK contract (return shape, manual_seeds format,
instrumental convention) is verified on the GPU box before release.
"""
from __future__ import annotations

import os
import tempfile
import types
import wave

import numpy as np
import pytest

import muse.modalities.audio_generation.runtimes.ace_step as ace_mod
from muse.modalities.audio_generation.runtimes.ace_step import AceStepRuntime
from muse.modalities.audio_generation.protocol import AudioGenerationResult


def _write_wav(path: str, *, seconds: float = 2.0, sr: int = 48000, channels: int = 2) -> None:
    """Write a real (silent) PCM WAV, mirroring what ACE-Step writes to save_path."""
    n = int(seconds * sr)
    if channels > 1:
        frames = np.zeros((n, channels), dtype="<i2")
    else:
        frames = np.zeros(n, dtype="<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames.tobytes())


def _wave_read(path, dtype="float32", always_2d=False):
    """Stand-in for soundfile.read: decode a PCM WAV to float32 via stdlib wave."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch)
    return data, sr


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Patch torch + ACEStepPipeline + soundfile sentinels.

    The fake pipe records its init/call kwargs and writes a real 48kHz
    stereo WAV to the save_path it is handed, returning [save_path] like
    the real SDK. ``calls`` exposes the recorded kwargs to assertions.
    """
    monkeypatch.setattr(ace_mod, "torch", types.SimpleNamespace())
    monkeypatch.setattr(ace_mod, "sf", types.SimpleNamespace(read=_wave_read))

    calls: dict = {}

    class FakePipe:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["call"] = kwargs
            save_path = kwargs["save_path"]
            _write_wav(save_path, seconds=2.0, sr=48000, channels=2)
            return [save_path]

    monkeypatch.setattr(ace_mod, "ACEStepPipeline", FakePipe)
    return calls


def _runtime(device="cuda"):
    return AceStepRuntime(
        model_id="ace-step-v1-3.5b",
        hf_repo="ACE-Step/ACE-Step-v1-3.5B",
        local_dir="/fake/weights",
        device=device,
    )


def test_construct_passes_checkpoint_dir(fake_pipeline):
    _runtime()
    assert fake_pipeline["init"]["checkpoint_dir"] == "/fake/weights"


def test_cuda_device_passes_device_id_zero(fake_pipeline):
    # A cuda pin is honored: ACE-Step indexes the GPU by integer id.
    _runtime(device="cuda")
    assert fake_pipeline["init"]["device_id"] == 0


def test_indexed_cuda_device_maps_to_its_index(fake_pipeline):
    # A future "cuda:1" pin targets GPU 1, not 0 (override genuinely honored).
    _runtime(device="cuda:1")
    assert fake_pipeline["init"]["device_id"] == 1


def test_cpu_device_pin_rejected_clearly(fake_pipeline):
    # ACE-Step is GPU-only; a cpu pin must fail loudly at load, not silently
    # run on cuda:0 (the v0.48.0 override-is-honored contract).
    with pytest.raises(RuntimeError, match="GPU-only"):
        _runtime(device="cpu")


def test_mps_device_pin_rejected_clearly(fake_pipeline):
    with pytest.raises(RuntimeError, match="GPU-only"):
        _runtime(device="mps")


def test_construct_maps_bf16_to_acestep_dtype_string(fake_pipeline):
    # Default dtype "bf16" -> ACE-Step's expected "bfloat16" string.
    _runtime()
    assert fake_pipeline["init"]["dtype"] == "bfloat16"


def test_generate_returns_audio_result(fake_pipeline):
    res = _runtime().generate("pop, upbeat", duration=2.0)
    assert isinstance(res, AudioGenerationResult)
    assert res.sample_rate == 48000
    assert res.channels == 2
    assert res.audio.dtype == np.float32
    assert res.audio.ndim == 2


def test_call_uses_wav_format(fake_pipeline):
    _runtime().generate("pop")
    assert fake_pipeline["call"]["format"] == "wav"


def test_lyrics_forwarded_to_pipe(fake_pipeline):
    _runtime().generate("pop", lyrics="[verse]\nhello world")
    assert fake_pipeline["call"]["lyrics"] == "[verse]\nhello world"


def test_no_lyrics_defaults_to_instrumental(fake_pipeline):
    _runtime().generate("ambient piano")
    assert fake_pipeline["call"]["lyrics"] == "[instrumental]"


def test_blank_lyrics_treated_as_instrumental(fake_pipeline):
    _runtime().generate("ambient", lyrics="   \n  ")
    assert fake_pipeline["call"]["lyrics"] == "[instrumental]"


def test_metadata_records_instrumental_flag(fake_pipeline):
    inst = _runtime().generate("ambient")
    assert inst.metadata["instrumental"] is True
    sung = _runtime().generate("pop", lyrics="[chorus] la la")
    assert sung.metadata["instrumental"] is False


def test_duration_clamped_to_max(fake_pipeline):
    _runtime().generate("x", duration=999.0)
    assert fake_pipeline["call"]["audio_duration"] == 240.0


def test_duration_defaults_when_none(fake_pipeline):
    _runtime().generate("x")
    # default_duration default is 60s.
    assert fake_pipeline["call"]["audio_duration"] == 60.0


def test_steps_and_guidance_forwarded(fake_pipeline):
    _runtime().generate("x", steps=40, guidance=12.0)
    assert fake_pipeline["call"]["infer_step"] == 40
    assert fake_pipeline["call"]["guidance_scale"] == 12.0


def test_seed_forwarded_as_manual_seeds_string(fake_pipeline):
    _runtime().generate("x", seed=42)
    assert fake_pipeline["call"]["manual_seeds"] == "42"


def test_no_seed_passes_none_manual_seeds(fake_pipeline):
    _runtime().generate("x")
    assert fake_pipeline["call"]["manual_seeds"] is None


def test_temp_dir_cleaned_up(fake_pipeline, tmp_path, monkeypatch):
    # Force mkdtemp under tmp_path; assert nothing leaks after generate.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _runtime().generate("x")
    assert list(tmp_path.iterdir()) == [], "temp dir leaked after generate"


def test_falls_back_to_save_path_when_returned_path_missing(fake_pipeline, monkeypatch):
    # A pipe that writes to save_path but returns a bogus nonexistent path:
    # the runtime should fall back to the save_path it provided.
    calls: dict = {}

    class WrongReturnPipe:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            _write_wav(kwargs["save_path"], seconds=1.0, sr=48000, channels=2)
            return ["/definitely/not/here.wav"]

    monkeypatch.setattr(ace_mod, "ACEStepPipeline", WrongReturnPipe)
    res = _runtime().generate("x")
    assert res.sample_rate == 48000
    assert res.channels == 2
