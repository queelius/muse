"""Tests for the bundled ace-step-v1-3.5b script (fully mocked).

The script aliases the shared AceStepRuntime as `Model` (the VLM-bundled
pattern), so behavior is covered by the runtime test. These tests pin
the MANIFEST contract + the alias identity + protocol satisfaction.
"""
from __future__ import annotations

import importlib

from muse.modalities.audio_generation import AudioGenerationModel


def _script():
    """Resolve the live module each call (test_discovery evicts muse.models.*)."""
    return importlib.import_module("muse.models.ace_step_v1_3_5b")


def _manifest():
    return _script().MANIFEST


def test_model_aliases_ace_step_runtime():
    from muse.modalities.audio_generation.runtimes.ace_step import AceStepRuntime
    assert _script().Model is AceStepRuntime


def test_manifest_core_fields():
    m = _manifest()
    assert m["model_id"] == "ace-step-v1-3.5b"
    assert m["modality"] == "audio/generation"
    assert m["hf_repo"] == "ACE-Step/ACE-Step-v1-3.5B"
    assert m["license"] == "Apache 2.0"


def test_manifest_music_only_capabilities():
    caps = _manifest()["capabilities"]
    assert caps["supports_music"] is True
    assert caps["supports_sfx"] is False


def test_manifest_pins_cuda_device():
    # 3.5B; GPU-required. Heavy GPU-only models pin "cuda" (not "auto").
    assert _manifest()["capabilities"]["device"] == "cuda"


def test_manifest_duration_ceiling_is_240():
    caps = _manifest()["capabilities"]
    assert caps["max_duration"] == 240.0
    assert caps["default_sample_rate"] == 48000


def test_pip_extras_declares_acestep_torch_soundfile():
    extras = " ".join(_manifest()["pip_extras"])
    # Distribution name is `ace-step` (hyphen); import name is `acestep`.
    # `acestep @ git+...` fails pip's name-consistency check, so the
    # requirement MUST use the hyphenated distribution name.
    assert "ace-step @ git+" in extras
    assert "torch" in extras
    assert "soundfile" in extras
    assert "numpy" in extras


def test_system_packages_include_ffmpeg():
    assert "ffmpeg" in _manifest()["system_packages"]


def test_instance_satisfies_audio_generation_protocol(monkeypatch):
    import types
    import muse.modalities.audio_generation.runtimes.ace_step as ace_mod

    monkeypatch.setattr(ace_mod, "torch", types.SimpleNamespace())
    monkeypatch.setattr(ace_mod, "sf", types.SimpleNamespace(read=lambda *a, **k: (None, 0)))
    monkeypatch.setattr(ace_mod, "ACEStepPipeline", lambda **kw: object())

    Model = _script().Model
    inst = Model(
        model_id="ace-step-v1-3.5b",
        hf_repo="ACE-Step/ACE-Step-v1-3.5B",
        local_dir="/fake",
        device="cuda",  # ACE-Step is GPU-only; a cpu pin is rejected.
    )
    assert isinstance(inst, AudioGenerationModel)
    assert inst.model_id == "ace-step-v1-3.5b"
