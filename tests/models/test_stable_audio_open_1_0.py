"""Tests for the bundled stable-audio-open-1.0 script (fully mocked).

Module-level imports of `muse.models.stable_audio_open_1_0` are
DELIBERATELY avoided: another test in the suite
(test_discovery.test_discovery_robust_to_broken_deps) pops
`muse.models.*` from sys.modules and re-imports them, which means a
top-level `import muse.models.stable_audio_open_1_0 as sa_script`
captures a stale module reference once that test has run. We re-resolve
the live module inside helpers/each test instead.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.modalities.audio_generation import (
    AudioGenerationModel,
    AudioGenerationResult,
)


def _sa_script():
    """Resolve the live module each call so test_discovery's sys.modules
    eviction doesn't leave us holding a stale reference."""
    import importlib
    return importlib.import_module("muse.models.stable_audio_open_1_0")


def _manifest():
    return _sa_script().MANIFEST


def _mock_pipe_call(samples=441000, sr=44100):
    """Fake pipeline call result with .audios attribute."""
    audio = np.zeros(samples, dtype=np.float32)
    return MagicMock(audios=[audio])


def _patched_setup(*, with_cuda=False, with_sample_rate=44100):
    """Install a fake pipeline + torch on the live module. Returns
    (sa_script, fake_pipe, fake_pipeline_class, fake_torch)."""
    sa = _sa_script()
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe
    fake_pipe.sample_rate = with_sample_rate
    fake_pipe.return_value = _mock_pipe_call()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipe

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = with_cuda
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"

    sa.StableAudioPipeline = fake_pipeline_class
    sa.torch = fake_torch
    return sa, fake_pipe, fake_pipeline_class, fake_torch


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    yield
    sa = _sa_script()
    sa.torch = None
    sa.StableAudioPipeline = None


def test_manifest_has_required_fields():
    m = _manifest()
    assert m["model_id"] == "stable-audio-open-1.0"
    assert m["modality"] == "audio/generation"
    assert m["hf_repo"] == "stabilityai/stable-audio-open-1.0"
    assert "pip_extras" in m
    assert "license" in m
    assert m["license"] == "Apache 2.0"


def test_manifest_pip_extras_declares_torch_diffusers_transformers():
    extras_str = " ".join(_manifest()["pip_extras"])
    assert "torch" in extras_str
    assert "diffusers" in extras_str
    assert "transformers" in extras_str
    assert "soundfile" in extras_str


def test_manifest_capabilities_have_music_and_sfx_flags():
    caps = _manifest()["capabilities"]
    assert caps["supports_music"] is True
    assert caps["supports_sfx"] is True


def test_manifest_capabilities_have_duration_bounds():
    caps = _manifest()["capabilities"]
    assert caps["min_duration"] >= 0.5
    assert caps["max_duration"] <= 47.0
    assert caps["default_duration"] == 10.0


def test_manifest_capabilities_default_steps_and_guidance():
    caps = _manifest()["capabilities"]
    assert caps["default_steps"] == 50
    assert caps["default_guidance"] == 7.0
    assert caps["default_sample_rate"] == 44100


def test_manifest_system_packages_lists_ffmpeg():
    """ffmpeg required for mp3/opus codec; declare so muse pull installs."""
    assert "ffmpeg" in _manifest()["system_packages"]


def test_manifest_allow_patterns_keep_only_fp16_weights():
    patterns = _manifest()["allow_patterns"]
    assert any("fp16" in p for p in patterns)
    assert not any(p.startswith("*.fp32") for p in patterns)


def test_model_class_named_model():
    """Discovery requires a class named exactly `Model`."""
    sa = _sa_script()
    assert sa.Model.__name__ == "Model"


def test_model_id_attribute_matches_manifest():
    sa = _sa_script()
    assert sa.Model.model_id == _manifest()["model_id"]


def test_model_construction_with_local_dir_preference():
    sa, fake_pipe, fake_pipeline_class, _ = _patched_setup()
    sa.Model(local_dir="/tmp/cache/abc", device="cpu")
    args, _ = fake_pipeline_class.from_pretrained.call_args
    assert args[0] == "/tmp/cache/abc"


def test_model_construction_falls_back_to_hf_repo():
    sa, fake_pipe, fake_pipeline_class, _ = _patched_setup()
    sa.Model(device="cpu")
    args, _ = fake_pipeline_class.from_pretrained.call_args
    assert args[0] == _manifest()["hf_repo"]


def test_model_raises_when_diffusers_missing():
    """When the StableAudioPipeline sentinel stays None, init should raise."""
    sa = _sa_script()
    sa.StableAudioPipeline = None
    sa.torch = MagicMock()
    with patch.object(sa, "_ensure_deps", lambda: None):
        with pytest.raises(RuntimeError, match="diffusers"):
            sa.Model(device="cpu")


def test_generate_returns_audio_generation_result():
    sa, fake_pipe, _, _ = _patched_setup()
    fake_pipe.return_value = _mock_pipe_call(samples=44100)
    m = sa.Model(device="cpu")
    out = m.generate("hello", duration=1.0)
    assert isinstance(out, AudioGenerationResult)
    assert out.sample_rate == 44100
    assert out.channels == 1
    assert out.audio.shape == (44100,)


def test_generate_calls_pipeline_with_audio_end_in_s():
    sa, fake_pipe, _, _ = _patched_setup()
    m = sa.Model(device="cpu")
    m.generate("hello", duration=8.5, steps=20, guidance=4.0)
    _, kwargs = fake_pipe.call_args
    assert kwargs["audio_end_in_s"] == 8.5
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["guidance_scale"] == 4.0


def test_generate_applies_capability_defaults_when_omitted():
    sa, fake_pipe, _, _ = _patched_setup()
    m = sa.Model(device="cpu")
    m.generate("hello")
    _, kwargs = fake_pipe.call_args
    caps = _manifest()["capabilities"]
    assert kwargs["audio_end_in_s"] == caps["default_duration"]
    assert kwargs["num_inference_steps"] == caps["default_steps"]
    assert kwargs["guidance_scale"] == caps["default_guidance"]


def test_generate_clamps_duration_to_capability_max():
    sa, fake_pipe, _, _ = _patched_setup()
    m = sa.Model(device="cpu")
    m.generate("hello", duration=999.0)
    _, kwargs = fake_pipe.call_args
    assert kwargs["audio_end_in_s"] == _manifest()["capabilities"]["max_duration"]


def test_generate_seed_creates_torch_generator():
    sa, fake_pipe, _, fake_torch = _patched_setup()
    m = sa.Model(device="cpu")
    m.generate("hello", seed=42)
    fake_torch.Generator.assert_called_once_with(device="cpu")
    fake_torch.Generator.return_value.manual_seed.assert_called_once_with(42)


def test_generate_omits_negative_prompt_when_none():
    sa, fake_pipe, _, _ = _patched_setup()
    m = sa.Model(device="cpu")
    m.generate("hello")
    _, kwargs = fake_pipe.call_args
    assert "negative_prompt" not in kwargs


def test_generate_passes_negative_prompt_when_set():
    sa, fake_pipe, _, _ = _patched_setup()
    m = sa.Model(device="cpu")
    m.generate("hello", negative_prompt="bad")
    _, kwargs = fake_pipe.call_args
    assert kwargs["negative_prompt"] == "bad"


def test_model_satisfies_audio_generation_protocol():
    sa, _, _, _ = _patched_setup()
    m = sa.Model(device="cpu")
    assert isinstance(m, AudioGenerationModel)


def test_default_device_is_cuda_in_capabilities():
    """Stable Audio is too slow on CPU; default to cuda; users override."""
    assert _manifest()["capabilities"]["device"] == "cuda"


def test_to_called_when_device_not_cpu():
    sa, fake_pipe, _, _ = _patched_setup(with_cuda=True)
    sa.Model(device="auto")
    fake_pipe.to.assert_called_once_with("cuda")
