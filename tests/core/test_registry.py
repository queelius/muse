"""Tests for ModalityRegistry: {modality: {model_id: Model}}."""
from dataclasses import dataclass
from typing import Any

import pytest

from muse.core.registry import ModalityRegistry, ModelInfo


@dataclass
class FakeAudioModel:
    model_id: str = "fake-tts"
    sample_rate: int = 16000
    def synthesize(self, text: str) -> Any: ...  # noqa


@dataclass
class FakeImageModel:
    model_id: str = "fake-diffusion"
    default_size: tuple[int, int] = (512, 512)
    def generate(self, prompt: str) -> Any: ...  # noqa


@pytest.fixture
def reg():
    return ModalityRegistry()


def test_register_and_get_by_modality(reg):
    m = FakeAudioModel()
    reg.register("audio/speech", m)
    assert reg.get("audio/speech", "fake-tts") is m


def test_first_registered_becomes_default_per_modality(reg):
    a1 = FakeAudioModel(model_id="tts-1")
    a2 = FakeAudioModel(model_id="tts-2")
    reg.register("audio/speech", a1)
    reg.register("audio/speech", a2)
    assert reg.get("audio/speech") is a1  # default


def test_modalities_are_isolated(reg):
    a = FakeAudioModel()
    i = FakeImageModel()
    reg.register("audio/speech", a)
    reg.register("image/generation", i)
    assert reg.get("audio/speech") is a
    assert reg.get("image/generation") is i
    with pytest.raises(KeyError):
        reg.get("audio/speech", "fake-diffusion")


def test_set_default_overrides_first_registered(reg):
    a1 = FakeAudioModel(model_id="tts-1")
    a2 = FakeAudioModel(model_id="tts-2")
    reg.register("audio/speech", a1)
    reg.register("audio/speech", a2)
    reg.set_default("audio/speech", "tts-2")
    assert reg.get("audio/speech").model_id == "tts-2"


def test_list_models_returns_modelinfo_per_modality(reg):
    reg.register("audio/speech", FakeAudioModel())
    reg.register("image/generation", FakeImageModel())
    audio = reg.list_models("audio/speech")
    assert len(audio) == 1
    assert isinstance(audio[0], ModelInfo)
    assert audio[0].model_id == "fake-tts"
    assert audio[0].modality == "audio/speech"


def test_list_all_spans_modalities(reg):
    reg.register("audio/speech", FakeAudioModel())
    reg.register("image/generation", FakeImageModel())
    all_models = reg.list_all()
    modalities = {m.modality for m in all_models}
    assert modalities == {"audio/speech", "image/generation"}


def test_modalities_lists_registered_keys(reg):
    reg.register("audio/speech", FakeAudioModel())
    assert reg.modalities() == ["audio/speech"]
    reg.register("image/generation", FakeImageModel())
    assert set(reg.modalities()) == {"audio/speech", "image/generation"}


def test_missing_modality_raises(reg):
    with pytest.raises(KeyError, match="no models registered"):
        reg.get("audio/speech")


def test_duplicate_registration_overwrites(reg):
    a1 = FakeAudioModel(model_id="tts")
    a2 = FakeAudioModel(model_id="tts")
    reg.register("audio/speech", a1)
    reg.register("audio/speech", a2)
    assert reg.get("audio/speech", "tts") is a2
    # Default lookup must also reflect the overwrite
    assert reg.get("audio/speech") is a2


def test_get_error_messages_include_available_options(reg):
    """Errors should tell the operator what's actually available."""
    reg.register("audio/speech", FakeAudioModel(model_id="tts-1"))

    # Missing modality -> lists known modalities
    with pytest.raises(KeyError, match="known modalities"):
        reg.get("video.clips")

    # Wrong model_id in a valid modality -> lists available in that modality
    with pytest.raises(KeyError, match="available.*tts-1"):
        reg.get("audio/speech", "tts-99")


def test_register_stores_manifest_on_modelinfo(reg):
    """Manifest passed to register() flows to ModelInfo unchanged."""
    m = FakeAudioModel()
    manifest = {
        "model_id": "fake-tts",
        "modality": "audio/speech",
        "capabilities": {"sample_rate": 16000, "voices": ["a", "b"]},
        "license": "CC0",
    }
    reg.register("audio/speech", m, manifest=manifest)
    info = reg.list_models("audio/speech")[0]
    assert info.manifest == manifest


def test_register_without_manifest_gets_minimal_stub(reg):
    """Registration without a manifest still populates the required keys."""
    reg.register("audio/speech", FakeAudioModel(model_id="anon"))
    info = reg.list_models("audio/speech")[0]
    # Minimal stub so downstream consumers never hit a KeyError on these
    assert info.manifest["model_id"] == "anon"
    assert info.manifest["modality"] == "audio/speech"


def test_registry_exposes_manifest_after_register():
    from unittest.mock import MagicMock
    reg = ModalityRegistry()
    fake = MagicMock(model_id="m1")
    reg.register("text/classification", fake, manifest={
        "model_id": "m1", "capabilities": {"flag_threshold": 0.7},
    })
    m = reg.manifest("text/classification", "m1")
    assert m["capabilities"]["flag_threshold"] == 0.7


def test_registry_manifest_returns_none_for_unknown_model():
    reg = ModalityRegistry()
    assert reg.manifest("text/classification", "nope") is None


# -- v0.34.0 finding #14: per-backend inference lock ---------------------


def test_register_attaches_inference_lock_per_backend():
    """Each registered backend gets its own _inference_lock; two
    backends do NOT share one. Replaces the older module-global
    per-modality lock that serialized unrelated models on the same
    worker."""
    import threading

    class _FakeA:
        model_id = "a"

    class _FakeB:
        model_id = "b"

    reg = ModalityRegistry()
    a, b = _FakeA(), _FakeB()
    reg.register("test/modality", a)
    reg.register("test/modality", b)

    assert hasattr(a, "_inference_lock")
    assert hasattr(b, "_inference_lock")
    assert a._inference_lock is not b._inference_lock
    # Both should be threading.Lock instances (or whatever returns from
    # threading.Lock; the type is private so we acquire/release as a
    # functional check).
    assert a._inference_lock.acquire(blocking=False)
    a._inference_lock.release()
    assert b._inference_lock.acquire(blocking=False)
    b._inference_lock.release()


def test_register_preserves_existing_lock():
    """A backend declaring its own _inference_lock keeps it (idempotent
    attachment). This lets a backend with internal synchronization
    requirements expose the lock it already uses."""
    import threading

    class _FakeWithLock:
        model_id = "x"

        def __init__(self):
            self._inference_lock = threading.Lock()

    reg = ModalityRegistry()
    f = _FakeWithLock()
    original_lock = f._inference_lock
    reg.register("test/modality", f)
    assert f._inference_lock is original_lock
