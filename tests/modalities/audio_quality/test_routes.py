import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core import config
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_quality import (
    AudioDurationExceededError,
    AudioQualityResult,
    AudioQualityScore,
    MODALITY,
    build_router,
)


class _FakeQualityModel:
    def __init__(self, model_id="quality-fake"):
        self.model_id = model_id
        self.last_path = None
        self.max_duration_seconds = None

    def assess(self, audio_path, *, max_duration_seconds=None):
        self.last_path = audio_path
        self.max_duration_seconds = max_duration_seconds
        return AudioQualityResult(
            scores={
                "naturalness": AudioQualityScore(
                    value=4.1, minimum=1, maximum=5,
                ),
            },
            primary_score="naturalness",
            metadata={"family": "fake"},
        )


def _client(backend):
    registry = ModalityRegistry()
    registry.register(MODALITY, backend)
    app = create_app(
        registry=registry,
        routers={MODALITY: build_router(registry)},
    )
    return TestClient(app)


def test_returns_named_scaled_scores_and_cleans_temp_file():
    backend = _FakeQualityModel()
    response = _client(backend).post(
        "/v1/audio/quality",
        files={"file": ("clip.wav", b"WAV", "audio/wav")},
        data={"model": backend.model_id},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == backend.model_id
    assert body["primary_score"] == "naturalness"
    assert body["scores"]["naturalness"]["value"] == 4.1
    assert backend.last_path is not None
    assert not os.path.exists(backend.last_path)
    assert backend.max_duration_seconds == 600.0


def test_default_model_is_optional_at_worker_route():
    response = _client(_FakeQualityModel()).post(
        "/v1/audio/quality",
        files={"file": ("clip.wav", b"WAV", "audio/wav")},
    )
    assert response.status_code == 200


def test_unknown_model_returns_404():
    response = _client(_FakeQualityModel("real")).post(
        "/v1/audio/quality",
        files={"file": ("clip.wav", b"WAV", "audio/wav")},
        data={"model": "ghost"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_empty_audio_returns_400():
    response = _client(_FakeQualityModel()).post(
        "/v1/audio/quality",
        files={"file": ("clip.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_parameter"


def test_oversized_audio_returns_413(monkeypatch):
    monkeypatch.setenv("MUSE_AUDIO_QUALITY_MAX_BYTES", "4")
    config.reset_config()
    try:
        response = _client(_FakeQualityModel()).post(
            "/v1/audio/quality",
            files={"file": ("clip.wav", b"12345", "audio/wav")},
        )
    finally:
        config.reset_config()
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_non_positive_cap_falls_back(monkeypatch):
    monkeypatch.setenv("MUSE_AUDIO_QUALITY_MAX_BYTES", "0")
    config.reset_config()
    try:
        from muse.modalities.audio_quality.routes import _max_bytes
        assert _max_bytes() == config.SETTINGS_BY_KEY[
            "limits.audio_quality_max_bytes"
        ].default
    finally:
        config.reset_config()


def test_duration_cap_is_forwarded_from_live_config(monkeypatch):
    monkeypatch.setenv("MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS", "42.5")
    config.reset_config()
    backend = _FakeQualityModel()
    try:
        response = _client(backend).post(
            "/v1/audio/quality",
            files={"file": ("clip.wav", b"WAV", "audio/wav")},
        )
    finally:
        config.reset_config()
    assert response.status_code == 200
    assert backend.max_duration_seconds == 42.5


def test_decoded_duration_limit_returns_413():
    backend = MagicMock()
    backend.model_id = "too-long"
    backend.assess.side_effect = AudioDurationExceededError(
        maximum_seconds=600,
        actual_seconds=1200,
    )
    response = _client(backend).post(
        "/v1/audio/quality",
        files={"file": ("chapter.mp3", b"MP3", "audio/mpeg")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert "1200.000s" in response.json()["error"]["message"]


def test_decode_exception_returns_generic_415():
    backend = MagicMock()
    backend.model_id = "broken"
    backend.assess.side_effect = RuntimeError("decoder leaked /secret/path")
    response = _client(backend).post(
        "/v1/audio/quality",
        files={"file": ("clip.wav", b"bad", "audio/wav")},
    )
    assert response.status_code == 415
    assert "/secret/path" not in response.text


def test_backend_exception_returns_generic_500():
    backend = MagicMock()
    backend.model_id = "broken"
    backend.assess.side_effect = RuntimeError("private backend detail")
    response = _client(backend).post(
        "/v1/audio/quality",
        files={"file": ("clip.wav", b"bad", "audio/wav")},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private backend detail" not in response.text
