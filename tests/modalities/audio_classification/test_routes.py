"""Route tests for /v1/audio/classifications."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_classification import (
    AudioClassificationResult,
    MODALITY,
    build_router,
)


class _FakeAudioClassifier:
    def __init__(self, model_id="ast-fake"):
        self.model_id = model_id
        self.last_path: str | None = None

    def classify(self, audio_path):
        self.last_path = audio_path
        return [AudioClassificationResult(
            scores={"speech": 0.7, "music": 0.2, "silence": 0.1},
            multi_label=False,
        )]


def _client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": backend.model_id}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_returns_envelope():
    backend = _FakeAudioClassifier()
    client = _client(backend)
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"FAKE-WAV-BYTES", "audio/wav")},
        data={"model": backend.model_id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == backend.model_id
    assert body["id"].startswith("audio-cls-")
    assert len(body["results"][0]) == 3
    assert body["results"][0][0]["label"] == "speech"


def test_top_k_truncates():
    backend = _FakeAudioClassifier()
    client = _client(backend)
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"x", "audio/wav")},
        data={"top_k": "2"},
    )
    assert r.status_code == 200
    assert len(r.json()["results"][0]) == 2


def test_invalid_top_k_returns_400():
    backend = _FakeAudioClassifier()
    client = _client(backend)
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"x", "audio/wav")},
        data={"top_k": "0"},
    )
    assert r.status_code == 400


def test_unknown_model_returns_404():
    backend = _FakeAudioClassifier(model_id="real")
    client = _client(backend)
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"x", "audio/wav")},
        data={"model": "ghost"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_empty_audio_returns_400():
    backend = _FakeAudioClassifier()
    client = _client(backend)
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["error"]["message"]


def test_oversized_audio_returns_400(monkeypatch):
    """MUSE_AUDIO_CLS_MAX_BYTES env cap rejects oversized uploads."""
    backend = _FakeAudioClassifier()
    client = _client(backend)
    monkeypatch.setenv("MUSE_AUDIO_CLS_MAX_BYTES", "100")
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"x" * 200, "audio/wav")},
    )
    assert r.status_code == 400
    assert "exceeds" in r.json()["error"]["message"]


def test_audio_cls_cap_zero_env_falls_back(monkeypatch):
    """MUSE_AUDIO_CLS_MAX_BYTES=0 must not turn into "reject everything".

    config.get() parses "0" to the literal int 0 (a valid int, not an
    error), so the accessor must guard non-positive values itself and
    fall back to the registry default, mirroring image_input.py's
    _default_max_bytes."""
    from muse.core import config
    monkeypatch.setenv("MUSE_AUDIO_CLS_MAX_BYTES", "0")
    config.reset_config()
    from muse.modalities.audio_classification.routes import _max_bytes
    assert _max_bytes() == config.SETTINGS_BY_KEY["limits.audio_cls_max_bytes"].default
    config.reset_config()


def test_audio_cls_cap_empty_env_falls_back(monkeypatch):
    """MUSE_AUDIO_CLS_MAX_BYTES="" coerces to None (opt_int); the
    accessor must also fall back to the registry default rather than
    returning None or crashing on the len(raw) > cap comparison."""
    from muse.core import config
    monkeypatch.setenv("MUSE_AUDIO_CLS_MAX_BYTES", "")
    config.reset_config()
    from muse.modalities.audio_classification.routes import _max_bytes
    assert _max_bytes() == config.SETTINGS_BY_KEY["limits.audio_cls_max_bytes"].default
    config.reset_config()


def test_runtime_exception_returns_500():
    backend = MagicMock()
    backend.model_id = "broken"
    backend.classify = MagicMock(side_effect=RuntimeError("boom"))
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "broken"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    c = TestClient(app)
    r = c.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    # Finding 1 (v0.58.1 review): the backend exception text must NOT
    # reach the client body; only a generic message does.
    assert "boom" not in body["error"]["message"]


def test_temp_file_passed_to_backend():
    """The route writes the upload to a temp path and passes the path
    to backend.classify; verify that path is non-empty and the temp
    file gets cleaned up after the call."""
    import os
    backend = _FakeAudioClassifier()
    client = _client(backend)
    r = client.post(
        "/v1/audio/classifications",
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert r.status_code == 200
    assert backend.last_path is not None
    # Temp file is unlinked after the call.
    assert not os.path.exists(backend.last_path)
