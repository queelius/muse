"""Route tests for /v1/audio/music and /v1/audio/sfx.

Both routes share the handler. Tests verify:
- 200 happy path on each route returns audio bytes + content-type.
- 400 capability gate on each route when manifest forbids.
- 400 unsupported response_format triggers UnsupportedFormatError.
- 404 model_not_found envelope on unknown model.
- Pydantic field validation: empty prompt, range checks, format pattern.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_generation import (
    MODALITY,
    AudioGenerationResult,
    build_router,
)
from muse.modalities.audio_generation.codec import UnsupportedFormatError


def _fake_backend(*, model_id="stable-audio-open-1.0", samples=4410, sr=44100, channels=1):
    """Return a MagicMock satisfying AudioGenerationModel.

    Default: 0.1s of silence, mono, 44.1kHz.
    """
    backend = MagicMock()
    backend.model_id = model_id

    if channels == 1:
        audio = np.zeros(samples, dtype=np.float32)
    else:
        audio = np.zeros((samples, channels), dtype=np.float32)

    result = AudioGenerationResult(
        audio=audio, sample_rate=sr, channels=channels,
        duration_seconds=samples / sr,
    )
    backend.generate.return_value = result
    return backend


def _make_client(backend, capabilities=None):
    reg = ModalityRegistry()
    manifest = {
        "model_id": backend.model_id,
        "modality": MODALITY,
        "capabilities": capabilities or {},
    }
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_music_returns_audio_bytes_with_wav_content_type():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "ambient"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WAVE"


def test_sfx_returns_audio_bytes_with_wav_content_type():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/sfx", json={"prompt": "footsteps"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_music_400_when_supports_music_false():
    backend = _fake_backend()
    client = _make_client(backend, capabilities={"supports_music": False})
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "music" in body["error"]["message"]


def test_sfx_400_when_supports_sfx_false():
    backend = _fake_backend()
    client = _make_client(backend, capabilities={"supports_sfx": False})
    r = client.post("/v1/audio/sfx", json={"prompt": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "sfx" in body["error"]["message"]


def test_music_succeeds_when_supports_music_missing_default_true():
    """When the capability flag is not set, default is True."""
    backend = _fake_backend()
    client = _make_client(backend, capabilities={})
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 200


def test_sfx_succeeds_when_supports_sfx_missing_default_true():
    backend = _fake_backend()
    client = _make_client(backend, capabilities={})
    r = client.post("/v1/audio/sfx", json={"prompt": "x"})
    assert r.status_code == 200


def test_music_404_on_unknown_model_envelope():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 404
    body = r.json()
    # OpenAI-style envelope, not FastAPI's {"detail": ...}
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_sfx_404_on_unknown_model_envelope():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/sfx", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_400_on_empty_prompt():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": ""})
    assert r.status_code == 422  # Pydantic min_length


def test_400_on_invalid_response_format():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "response_format": "aac",
    })
    assert r.status_code == 422  # Pydantic pattern


def test_400_on_duration_out_of_range_low():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "duration": 0.1,
    })
    assert r.status_code == 422


def test_400_on_duration_out_of_range_high():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "duration": 999.0,
    })
    assert r.status_code == 422


def test_400_on_steps_out_of_range():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "steps": 1000,
    })
    assert r.status_code == 422


def test_400_on_guidance_out_of_range():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "guidance": 99.0,
    })
    assert r.status_code == 422


def test_negative_seed_rejected():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "seed": -1,
    })
    assert r.status_code == 422


def test_request_kwargs_forwarded_to_backend():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "ambient piano",
        "duration": 8.0, "seed": 42, "steps": 30, "guidance": 5.0,
        "negative_prompt": "noise",
    })
    assert r.status_code == 200
    args, kwargs = backend.generate.call_args
    assert args[0] == "ambient piano"
    assert kwargs["duration"] == 8.0
    assert kwargs["seed"] == 42
    assert kwargs["steps"] == 30
    assert kwargs["guidance"] == 5.0
    assert kwargs["negative_prompt"] == "noise"


def test_music_and_sfx_call_backend_with_same_kwargs():
    """Routes share the handler; only the capability key differs."""
    backend = _fake_backend()
    client = _make_client(backend)
    client.post("/v1/audio/music", json={"prompt": "p"})
    music_kwargs = backend.generate.call_args.kwargs
    backend.reset_mock()
    backend.generate.return_value = backend.generate.return_value
    backend.generate.return_value = AudioGenerationResult(
        audio=np.zeros(4410, dtype=np.float32),
        sample_rate=44100, channels=1, duration_seconds=0.1,
    )
    client.post("/v1/audio/sfx", json={"prompt": "p"})
    sfx_kwargs = backend.generate.call_args.kwargs
    assert music_kwargs == sfx_kwargs


def test_flac_response_when_soundfile_available():
    pytest.importorskip("soundfile")
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "response_format": "flac",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/flac"
    assert r.content[:4] == b"fLaC"


def test_flac_400_when_soundfile_missing():
    backend = _fake_backend()
    client = _make_client(backend)
    with patch(
        "muse.modalities.audio_generation.codec._try_import_soundfile",
        return_value=None,
    ):
        r = client.post("/v1/audio/music", json={
            "prompt": "x", "response_format": "flac",
        })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    assert "soundfile" in r.json()["error"]["message"]


def test_mp3_400_when_pydub_missing():
    backend = _fake_backend()
    client = _make_client(backend)
    with patch(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        return_value=None,
    ):
        r = client.post("/v1/audio/music", json={
            "prompt": "x", "response_format": "mp3",
        })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    assert "pydub" in r.json()["error"]["message"]


def test_internal_error_returns_500_envelope():
    """Backend exception translates to a 500 with OpenAI envelope."""
    backend = _fake_backend()
    backend.generate.side_effect = RuntimeError("CUDA OOM")
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"


def test_response_is_bytes_not_json():
    """No streaming: single Response, not SSE."""
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.headers["content-type"].startswith("audio/")
    assert "content-length" in {k.lower() for k in r.headers}
