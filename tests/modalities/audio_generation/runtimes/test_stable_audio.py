"""Tests for StableAudioRuntime (diffusers.StableAudioPipeline wrapper).

The runtime defers heavy imports (torch, diffusers). Tests patch the
sentinels directly; _ensure_deps short-circuits when patched.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import muse.modalities.audio_generation.runtimes.stable_audio as sa_mod
from muse.modalities.audio_generation import AudioGenerationResult
from muse.modalities.audio_generation.runtimes.stable_audio import (
    StableAudioRuntime,
    _normalize_pipeline_output,
)


def _patched_runtime(audios_return=None, *, sample_rate=44100, **runtime_kwargs):
    """Construct a StableAudioRuntime with the diffusers pipeline mocked.

    Returns the (runtime, pipe, pipeline_class, torch). The fake_torch
    is also installed on the module via monkey-set so subsequent
    generate() calls still see it. Cleanup is the caller's job (most
    tests don't care because each test gets a fresh module-level None
    via the `_reset_sa_mod` fixture).
    """
    if audios_return is None:
        # Default: 2-second mono float32 sine wave at 44100Hz.
        n = sample_rate * 2
        t = np.arange(n, dtype=np.float32) / sample_rate
        audios_return = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    fake_pipe = MagicMock()
    fake_pipe.sample_rate = sample_rate
    fake_call_result = MagicMock()
    fake_call_result.audios = [audios_return]
    fake_pipe.return_value = fake_call_result
    fake_pipe.to.return_value = fake_pipe

    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"

    kwargs = dict(
        model_id="test-stable",
        hf_repo="org/repo",
        local_dir=None,
        device="cpu",
    )
    kwargs.update(runtime_kwargs)

    # Install on the module directly so generate() (called outside the
    # patch context) still sees the fakes.
    sa_mod.StableAudioPipeline = fake_pipeline_class
    sa_mod.torch = fake_torch
    rt = StableAudioRuntime(**kwargs)
    return rt, fake_pipe, fake_pipeline_class, fake_torch


@pytest.fixture(autouse=True)
def _reset_sa_mod():
    """Restore sentinels after each test so leakage doesn't poison neighbors."""
    yield
    sa_mod.torch = None
    sa_mod.StableAudioPipeline = None


def test_runtime_constructs_with_local_dir_preference():
    """Runtime prefers local_dir over hf_repo as the source path."""
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe
    with patch.object(sa_mod, "StableAudioPipeline", fake_pipeline_class), \
            patch.object(sa_mod, "torch", MagicMock()):
        StableAudioRuntime(
            model_id="m", hf_repo="org/repo", local_dir="/tmp/cache/abc",
            device="cpu",
        )
    args, kwargs = fake_pipeline_class.from_pretrained.call_args
    assert args[0] == "/tmp/cache/abc"


def test_runtime_falls_back_to_hf_repo_when_no_local_dir():
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe
    with patch.object(sa_mod, "StableAudioPipeline", fake_pipeline_class), \
            patch.object(sa_mod, "torch", MagicMock()):
        StableAudioRuntime(
            model_id="m", hf_repo="org/repo", local_dir=None, device="cpu",
        )
    args, _ = fake_pipeline_class.from_pretrained.call_args
    assert args[0] == "org/repo"


def test_runtime_raises_when_diffusers_missing():
    """Simulate missing diffusers by stubbing _ensure_deps so it leaves
    the StableAudioPipeline sentinel as None."""
    with patch.object(sa_mod, "StableAudioPipeline", None), \
            patch.object(sa_mod, "_ensure_deps", lambda: None), \
            patch.object(sa_mod, "torch", MagicMock()):
        with pytest.raises(RuntimeError, match="diffusers"):
            StableAudioRuntime(
                model_id="m", hf_repo="org/repo", local_dir=None,
                device="cpu",
            )


def test_generate_returns_audio_generation_result():
    rt, fake_pipe, _, _ = _patched_runtime()
    out = rt.generate("ambient piano", duration=2.0)
    assert isinstance(out, AudioGenerationResult)
    assert out.sample_rate == 44100
    assert out.channels == 1
    assert isinstance(out.audio, np.ndarray)
    assert out.audio.shape == (88200,)


def test_generate_calls_pipeline_with_audio_end_in_s():
    rt, fake_pipe, _, _ = _patched_runtime()
    rt.generate("hello", duration=7.5, steps=20, guidance=5.0)
    _, kwargs = fake_pipe.call_args
    assert kwargs["prompt"] == "hello"
    assert kwargs["audio_end_in_s"] == 7.5
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["guidance_scale"] == 5.0


def test_generate_applies_capability_defaults_when_omitted():
    rt, fake_pipe, _, _ = _patched_runtime(
        default_duration=8.0, default_steps=42, default_guidance=6.0,
    )
    rt.generate("hello")
    _, kwargs = fake_pipe.call_args
    assert kwargs["audio_end_in_s"] == 8.0
    assert kwargs["num_inference_steps"] == 42
    assert kwargs["guidance_scale"] == 6.0


def test_generate_clamps_duration_to_max():
    """Requested duration above max_duration is clamped."""
    rt, fake_pipe, _, _ = _patched_runtime(max_duration=20.0)
    rt.generate("hello", duration=999.0)
    _, kwargs = fake_pipe.call_args
    assert kwargs["audio_end_in_s"] == 20.0


def test_generate_clamps_duration_to_min():
    rt, fake_pipe, _, _ = _patched_runtime(min_duration=2.0)
    rt.generate("hello", duration=0.1)
    _, kwargs = fake_pipe.call_args
    assert kwargs["audio_end_in_s"] == 2.0


def test_generate_passes_negative_prompt_when_set():
    rt, fake_pipe, _, _ = _patched_runtime()
    rt.generate("hello", negative_prompt="bad audio")
    _, kwargs = fake_pipe.call_args
    assert kwargs["negative_prompt"] == "bad audio"


def test_generate_omits_negative_prompt_when_none():
    rt, fake_pipe, _, _ = _patched_runtime()
    rt.generate("hello")
    _, kwargs = fake_pipe.call_args
    assert "negative_prompt" not in kwargs


def test_generate_seed_creates_torch_generator():
    rt, fake_pipe, _, fake_torch = _patched_runtime()
    rt.generate("hello", seed=42)
    fake_torch.Generator.assert_called_once_with(device="cpu")
    fake_torch.Generator.return_value.manual_seed.assert_called_once_with(42)
    _, kwargs = fake_pipe.call_args
    assert "generator" in kwargs


def test_generate_no_seed_skips_generator():
    rt, fake_pipe, _, _ = _patched_runtime()
    rt.generate("hello")
    _, kwargs = fake_pipe.call_args
    assert "generator" not in kwargs


def test_generate_metadata_includes_provenance():
    rt, fake_pipe, _, _ = _patched_runtime()
    out = rt.generate("ambient", duration=3.0, steps=10, guidance=4.0, seed=7)
    assert out.metadata["prompt"] == "ambient"
    assert out.metadata["steps"] == 10
    assert out.metadata["guidance"] == 4.0
    assert out.metadata["seed"] == 7
    assert out.metadata["model"] == "test-stable"


def test_generate_handles_stereo_pipeline_output():
    """diffusers emits (channels, samples) for stereo; runtime transposes
    to (samples, channels)."""
    n = 4410
    stereo = np.stack(
        [
            np.linspace(0, 1, n, dtype=np.float32),
            np.linspace(1, 0, n, dtype=np.float32),
        ],
        axis=0,
    )  # (2, 4410)
    rt, _, _, _ = _patched_runtime(audios_return=stereo, sample_rate=44100)
    out = rt.generate("hello")
    assert out.channels == 2
    assert out.audio.shape == (4410, 2)


def test_generate_handles_torch_tensor_output(monkeypatch):
    """Pipeline returning a torch.Tensor still produces a numpy array."""
    n = 4410
    audio = np.linspace(0, 1, n, dtype=np.float32)

    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr
        def detach(self):
            return self
        def cpu(self):
            return self
        def float(self):
            return self
        def numpy(self):
            return self._arr

    fake_pipe = MagicMock()
    fake_pipe.sample_rate = 44100
    fake_pipe.return_value = MagicMock(audios=[FakeTensor(audio)])
    fake_pipe.to.return_value = fake_pipe
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe

    fake_torch = MagicMock()
    fake_torch.Tensor = FakeTensor
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.float16 = "float16"

    with patch.object(sa_mod, "StableAudioPipeline", fake_pipeline_class), \
            patch.object(sa_mod, "torch", fake_torch):
        rt = StableAudioRuntime(
            model_id="m", hf_repo="org/repo", local_dir=None,
            device="cpu",
        )
        out = rt.generate("hello")
    assert isinstance(out.audio, np.ndarray)
    assert out.audio.shape == (4410,)


def test_select_device_auto_with_no_torch_returns_cpu():
    with patch.object(sa_mod, "torch", None):
        from muse.modalities.audio_generation.runtimes.stable_audio import (
            _select_device,
        )
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda_when_available():
    fake = MagicMock()
    fake.cuda.is_available.return_value = True
    with patch.object(sa_mod, "torch", fake):
        from muse.modalities.audio_generation.runtimes.stable_audio import (
            _select_device,
        )
        assert _select_device("auto") == "cuda"


def test_select_device_explicit_passes_through():
    with patch.object(sa_mod, "torch", MagicMock()):
        from muse.modalities.audio_generation.runtimes.stable_audio import (
            _select_device,
        )
        assert _select_device("cpu") == "cpu"
        assert _select_device("cuda") == "cuda"


def test_normalize_pipeline_output_mono():
    a = np.zeros(1000, dtype=np.float32)
    out, ch = _normalize_pipeline_output(a)
    assert ch == 1
    assert out.shape == (1000,)


def test_normalize_pipeline_output_stereo_channel_first():
    a = np.zeros((2, 1000), dtype=np.float32)
    out, ch = _normalize_pipeline_output(a)
    assert ch == 2
    assert out.shape == (1000, 2)


def test_normalize_pipeline_output_already_samples_first():
    """Some pipelines emit (samples, channels) directly."""
    a = np.zeros((1000, 2), dtype=np.float32)
    out, ch = _normalize_pipeline_output(a)
    assert ch == 2
    assert out.shape == (1000, 2)


def test_normalize_pipeline_output_3d_raises():
    a = np.zeros((1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        _normalize_pipeline_output(a)


def test_runtime_to_called_when_device_not_cpu():
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.float16 = "float16"
    with patch.object(sa_mod, "StableAudioPipeline", fake_pipeline_class), \
            patch.object(sa_mod, "torch", fake_torch):
        StableAudioRuntime(
            model_id="m", hf_repo="org/repo", local_dir=None,
            device="auto",
        )
    fake_pipe.to.assert_called_once_with("cuda")


def test_runtime_to_skipped_when_device_cpu():
    fake_pipe = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe
    with patch.object(sa_mod, "StableAudioPipeline", fake_pipeline_class), \
            patch.object(sa_mod, "torch", MagicMock()):
        StableAudioRuntime(
            model_id="m", hf_repo="org/repo", local_dir=None, device="cpu",
        )
    fake_pipe.to.assert_not_called()


def test_runtime_falls_back_to_default_sample_rate_when_pipe_missing_attr():
    fake_pipe = MagicMock(spec=[])  # no sample_rate attr
    fake_pipe.return_value = MagicMock(
        audios=[np.zeros(2205, dtype=np.float32)],
    )
    # MagicMock with spec=[] still has __call__ but no sample_rate.
    fake_pipe.to = MagicMock(return_value=fake_pipe)
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe
    with patch.object(sa_mod, "StableAudioPipeline", fake_pipeline_class), \
            patch.object(sa_mod, "torch", MagicMock()):
        rt = StableAudioRuntime(
            model_id="m", hf_repo="org/repo", local_dir=None,
            device="cpu", default_sample_rate=22050,
        )
        out = rt.generate("hello")
    assert out.sample_rate == 22050
