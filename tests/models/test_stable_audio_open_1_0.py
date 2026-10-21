"""Tests for the bundled stable-audio-open-1.0 script (fully mocked).

The script aliases the generic `StableAudioRuntime` as its `Model`;
runtime behavior (output normalization, batched outputs, clamping) is
covered by tests/modalities/audio_generation/runtimes/test_stable_audio.py.
These tests pin the MANIFEST and check that the runtime honors it when
constructed the way `load_backend` does (`model_id` + capabilities
splatted as kwargs).

Module-level imports of `muse.models.stable_audio_open_1_0` are
DELIBERATELY avoided: test_discovery_robust_to_broken_deps pops
`muse.models.*` from sys.modules and re-imports them.
"""
import importlib
from unittest.mock import MagicMock, patch

import numpy as np

from muse.modalities.audio_generation import (
    AudioGenerationModel,
    AudioGenerationResult,
)

_RT = "muse.modalities.audio_generation.runtimes.stable_audio"


def _sa_script():
    return importlib.import_module("muse.models.stable_audio_open_1_0")


def _manifest():
    return _sa_script().MANIFEST


def test_manifest_has_required_fields():
    m = _manifest()
    assert m["model_id"] == "stable-audio-open-1.0"
    assert m["modality"] == "audio/generation"
    assert m["hf_repo"] == "stabilityai/stable-audio-open-1.0"
    assert "pip_extras" in m
    assert m["license"] == "Apache 2.0"


def test_manifest_pip_extras_declares_torch_diffusers_transformers():
    extras_str = " ".join(_manifest()["pip_extras"])
    assert "torch" in extras_str
    assert "diffusers" in extras_str
    assert "transformers" in extras_str
    assert "soundfile" in extras_str


def test_manifest_capabilities():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "cuda"
    assert caps["supports_music"] is True
    assert caps["supports_sfx"] is True
    assert caps["min_duration"] >= 0.5
    assert caps["max_duration"] <= 47.0
    assert caps["default_duration"] == 10.0
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


def test_model_is_stable_audio_runtime():
    from muse.modalities.audio_generation.runtimes.stable_audio import (
        StableAudioRuntime,
    )
    assert _sa_script().Model is StableAudioRuntime


def _construct(**overrides):
    sa = _sa_script()
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe
    fake_pipe.sample_rate = 44100
    fake_pipe.return_value = MagicMock(audios=[np.zeros(441000, np.float32)])
    pipe_cls = MagicMock()
    pipe_cls.from_pretrained.return_value = fake_pipe
    kwargs = {
        "model_id": sa.MANIFEST["model_id"],
        **sa.MANIFEST["capabilities"],
        **overrides,
    }
    with patch(f"{_RT}.StableAudioPipeline", pipe_cls), \
            patch(f"{_RT}.torch", MagicMock()):
        m = sa.Model(hf_repo=sa.MANIFEST["hf_repo"], local_dir="/fake", **kwargs)
    return m, fake_pipe, pipe_cls


def test_manifest_capabilities_drive_runtime_defaults():
    m, fake_pipe, pipe_cls = _construct()
    assert pipe_cls.from_pretrained.call_args.args[0] == "/fake"
    fake_pipe.to.assert_called_once_with("cuda")
    out = m.generate("rain on a tin roof")
    call = fake_pipe.call_args.kwargs
    assert call["audio_end_in_s"] == 10.0
    assert call["num_inference_steps"] == 50
    assert call["guidance_scale"] == 7.0
    assert isinstance(out, AudioGenerationResult)
    assert out.sample_rate == 44100
    assert out.metadata["model"] == "stable-audio-open-1.0"


def test_generate_clamps_duration_to_manifest_max():
    m, fake_pipe, _ = _construct()
    m.generate("long drone", duration=120.0)
    assert fake_pipe.call_args.kwargs["audio_end_in_s"] == 47.0


def test_model_satisfies_audio_generation_protocol():
    m, _, _ = _construct(device="cpu")
    assert isinstance(m, AudioGenerationModel)
