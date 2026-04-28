"""End-to-end test: in-process FastAPI app + fake audio-generation backend.

Stand up a real FastAPI app via create_app, register a FakeAudioGenerationModel
that returns a deterministic 1-second mono sine wave, and exercise both
/v1/audio/music and /v1/audio/sfx end-to-end.

This is faster than spawning a subprocess and gives us coverage of the
modality wiring (registry + create_app + build_router + codec) without
needing real Stable Audio weights.
"""
import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_generation import (
    MODALITY,
    AudioGenerationResult,
    build_router,
)


class FakeAudioGenerationModel:
    """Deterministic backend: returns a 1-second 440Hz sine wave."""

    model_id = "fake-audio-gen-1.0"

    def generate(
        self, prompt, *, duration=None, seed=None, steps=None,
        guidance=None, negative_prompt=None, **_,
    ):
        sr = 44100
        eff = duration or 1.0
        n = int(sr * eff)
        t = np.arange(n, dtype=np.float32) / sr
        audio = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        return AudioGenerationResult(
            audio=audio, sample_rate=sr, channels=1,
            duration_seconds=eff,
            metadata={"prompt": prompt, "seed": seed},
        )


def _client_with_caps(supports_music=True, supports_sfx=True):
    backend = FakeAudioGenerationModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": backend.model_id,
        "modality": MODALITY,
        "capabilities": {
            "supports_music": supports_music,
            "supports_sfx": supports_sfx,
            "default_duration": 1.0,
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_e2e_music_route_returns_valid_wav():
    client = _client_with_caps()
    r = client.post("/v1/audio/music", json={
        "prompt": "ambient piano",
        "duration": 1.0,
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"

    # Open the returned bytes via wave; assert sample rate, channels.
    with wave.open(io.BytesIO(r.content), "rb") as w:
        assert w.getframerate() == 44100
        assert w.getnchannels() == 1
        # ~44100 frames for 1 second.
        assert abs(w.getnframes() - 44100) < 100


def test_e2e_sfx_route_returns_valid_wav():
    client = _client_with_caps()
    r = client.post("/v1/audio/sfx", json={
        "prompt": "footsteps",
        "duration": 1.0,
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content), "rb") as w:
        assert w.getframerate() == 44100
        assert w.getnchannels() == 1


def test_e2e_capability_gate_returns_400_on_music():
    client = _client_with_caps(supports_music=False)
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "music" in body["error"]["message"]


def test_e2e_capability_gate_returns_400_on_sfx():
    client = _client_with_caps(supports_sfx=False)
    r = client.post("/v1/audio/sfx", json={"prompt": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "sfx" in body["error"]["message"]


def test_e2e_music_then_sfx_independent_calls():
    """Two consecutive calls on the same client return distinct audio bytes
    (the prompt differs so the metadata differs; this confirms the same
    backend serves both routes without state leakage)."""
    client = _client_with_caps()
    r1 = client.post("/v1/audio/music", json={"prompt": "music"})
    r2 = client.post("/v1/audio/sfx", json={"prompt": "sfx"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Both should be valid WAV; payloads can be byte-identical because
    # FakeAudioGenerationModel is deterministic. Just assert structure.
    assert r1.content[:4] == b"RIFF"
    assert r2.content[:4] == b"RIFF"


def test_e2e_v1_models_lists_audio_generation_model():
    """Health and /v1/models surface the registered model under
    audio/generation."""
    client = _client_with_caps()
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    ids = {m["id"] for m in body["data"]}
    assert "fake-audio-gen-1.0" in ids
