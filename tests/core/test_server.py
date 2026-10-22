"""Tests for the core FastAPI app factory."""
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


def test_create_app_returns_fastapi():
    app = create_app(registry=ModalityRegistry(), routers={})
    assert isinstance(app, FastAPI)


def test_unhandled_exception_returns_openai_envelope_not_bare_500():
    """Finding 2 (v0.58.1 review): a route whose backend call is NOT
    wrapped in its own try/except must still surface the OpenAI-shape
    error envelope, not FastAPI's bare `{"detail": "Internal Server
    Error"}`. create_app installs one global Exception handler so this
    holds for every mounted router, not just the ones that remembered to
    catch locally."""
    router = APIRouter()

    @router.get("/boom")
    def boom():
        raise RuntimeError("secret /internal/path detail")

    app = create_app(registry=ModalityRegistry(), routers={"x": router})
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert "detail" not in body
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["type"] == "server_error"
    assert "secret /internal/path detail" not in r.text


def test_model_not_found_error_handler_still_wins_over_global_handler():
    """ModelNotFoundError's dedicated handler (404 + its own envelope)
    must take precedence over the new catch-all Exception handler."""
    from muse.core.errors import ModelNotFoundError

    router = APIRouter()

    @router.get("/missing")
    def missing():
        raise ModelNotFoundError(model_id="nope", modality="audio/speech")

    app = create_app(registry=ModalityRegistry(), routers={"x": router})
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/missing")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_root_health_endpoint():
    app = create_app(registry=ModalityRegistry(), routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["modalities"] == []


def test_health_reports_registered_modalities():
    reg = ModalityRegistry()

    class Fake:
        model_id = "fake"
    reg.register("audio/speech", Fake())
    reg.register("image/generation", Fake())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert set(r.json()["modalities"]) == {"audio/speech", "image/generation"}


def test_routers_are_mounted():
    router = APIRouter()

    @router.get("/v1/test/ping")
    def ping():
        return {"ok": True}

    app = create_app(registry=ModalityRegistry(), routers={"test": router})
    client = TestClient(app)
    r = client.get("/v1/test/ping")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_global_v1_models_endpoint_aggregates():
    reg = ModalityRegistry()

    class FakeAudio:
        model_id = "fake-tts"
        sample_rate = 16000
    reg.register("audio/speech", FakeAudio())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "fake-tts" and m["modality"] == "audio/speech" for m in data)


def test_registry_stored_on_app_state():
    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})
    assert app.state.registry is reg


def test_v1_models_registry_fields_win_over_manifest():
    """Authoritative fields (id, modality, object) must never be clobbered by manifest."""
    reg = ModalityRegistry()

    class HostileModel:
        model_id = "real-id"

    # A manifest whose top-level + capabilities both try to override the
    # authoritative fields. The server must ignore all three overrides.
    hostile_manifest = {
        "model_id": "real-id",
        "modality": "audio/speech",
        "id": "IMPOSTOR",           # collides with authoritative "id"
        "object": "evil",           # collides with authoritative "object"
        "capabilities": {
            "id": "ALSO-IMPOSTOR",
            "modality": "wrong",
            "object": "evil-cap",
        },
    }
    reg.register("audio/speech", HostileModel(), manifest=hostile_manifest)

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    data = r.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "real-id"
    assert data[0]["modality"] == "audio/speech"
    assert data[0]["object"] == "model"


def test_v1_models_exposes_capabilities_and_metadata_from_manifest():
    """Capabilities, description, license, hf_repo from the manifest flow to /v1/models."""
    reg = ModalityRegistry()

    class Fake:
        model_id = "kokoro-82m"

    manifest = {
        "model_id": "kokoro-82m",
        "modality": "audio/speech",
        "hf_repo": "hexgrad/Kokoro-82M",
        "description": "Lightweight TTS, 54 voices, 24kHz",
        "license": "Apache 2.0",
        "capabilities": {
            "sample_rate": 24000,
            "voices": ["af_heart", "am_adam"],
        },
    }
    reg.register("audio/speech", Fake(), manifest=manifest)

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    entry = r.json()["data"][0]
    # Top-level metadata
    assert entry["hf_repo"] == "hexgrad/Kokoro-82M"
    assert entry["description"] == "Lightweight TTS, 54 voices, 24kHz"
    assert entry["license"] == "Apache 2.0"
    # Capabilities projected to top level
    assert entry["sample_rate"] == 24000
    assert entry["voices"] == ["af_heart", "am_adam"]


# v0.40.0 lazy-load fields on /v1/models -----------------------------------


def test_v1_models_includes_lazy_load_fields_no_supervisor():
    """Without a SupervisorState wired, /v1/models still emits the v0.40.0
    fields with safe defaults: every registered model is treated as
    loaded (the worker has it in its registry), with no last_loaded_at
    or unservable_reason. This keeps the worker's local view consistent
    when running in isolation (tests, debugging, single-worker mode).
    """
    reg = ModalityRegistry()

    class Fake:
        model_id = "test-model"

    reg.register("audio/speech", Fake(), manifest={"model_id": "test-model"})
    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert "loaded" in entry
    assert "last_loaded_at" in entry
    assert "unservable_reason" in entry
    # Defaults: registered = loaded; no director timestamp on a worker
    # without supervisor state; no unservable_reason on a successfully
    # registered model.
    assert entry["loaded"] is True
    assert entry["last_loaded_at"] is None
    assert entry["unservable_reason"] is None


def test_model_not_found_error_serializes_openai_shape():
    """ModelNotFoundError must produce {error: {...}} not {detail: {error: {...}}}."""
    from muse.core.errors import ModelNotFoundError

    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})

    @app.get("/boom")
    def boom():
        raise ModelNotFoundError(model_id="missing-model", modality="audio/speech")

    client = TestClient(app)
    r = client.get("/boom")
    assert r.status_code == 404
    body = r.json()
    # Top-level key must be "error", not "detail"
    assert "error" in body
    assert "detail" not in body
    err = body["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"
    assert "missing-model" in err["message"]
    assert "audio/speech" in err["message"]


def test_speech_route_with_empty_registry_returns_openai_404():
    """When no audio/speech model is registered, POST /v1/audio/speech must
    return a 404 with the OpenAI error envelope (not FastAPI's generic 404).

    This requires the audio/speech router to be mounted even when no models are
    loaded — otherwise FastAPI returns {detail: 'Not Found'} for an unknown path.
    """
    from muse.modalities.audio_speech.routes import build_router as build_audio_router

    reg = ModalityRegistry()
    router = build_audio_router(reg)
    app = create_app(registry=reg, routers={"audio/speech": router})
    client = TestClient(app)

    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 404
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_images_route_with_empty_registry_returns_openai_404():
    """Same contract for images.generations."""
    from muse.modalities.image_generation.routes import build_router as build_images_router

    reg = ModalityRegistry()
    router = build_images_router(reg)
    app = create_app(registry=reg, routers={"image/generation": router})
    client = TestClient(app)

    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 404
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_validation_error_uses_openai_envelope():
    """422 for invalid input must use OpenAI shape, not FastAPI default."""
    from pydantic import BaseModel, Field

    router = APIRouter()

    class Req(BaseModel):
        value: int = Field(..., ge=0, le=10)

    @router.post("/validate")
    def validate(r: Req):
        return {"ok": True}

    app = create_app(registry=ModalityRegistry(), routers={"v": router})
    client = TestClient(app)
    r = client.post("/validate", json={"value": 999})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# v0.47.3: build_model_entry shared shape (reused by the gateway to list
# enabled-but-unloaded catalog models).
# ---------------------------------------------------------------------------


def test_build_model_entry_shape():
    from muse.core.server import build_model_entry

    manifest = {
        "capabilities": {"sample_rate": 24000},
        "description": "d",
        "license": "MIT",
        "hf_repo": "x/y",
    }
    e = build_model_entry(
        "foo", "audio/speech", manifest,
        loaded=False, last_loaded_at=None, unservable_reason="r",
    )
    assert e["id"] == "foo"
    assert e["modality"] == "audio/speech"
    assert e["object"] == "model"
    assert e["loaded"] is False
    assert e["last_loaded_at"] is None
    assert e["unservable_reason"] == "r"
    assert e["sample_rate"] == 24000
    assert e["description"] == "d"
    assert e["license"] == "MIT"
    assert e["hf_repo"] == "x/y"


def test_build_model_entry_capabilities_cannot_clobber_authoritative_fields():
    from muse.core.server import build_model_entry

    manifest = {
        "capabilities": {
            "id": "EVIL", "modality": "EVIL", "object": "EVIL",
            "loaded": "EVIL", "unservable_reason": "EVIL",
            "created": "EVIL", "owned_by": "EVIL",
        },
    }
    e = build_model_entry(
        "foo", "audio/speech", manifest,
        loaded=True, last_loaded_at="2026-01-01T00:00:00+00:00",
        unservable_reason=None,
    )
    assert e["id"] == "foo"
    assert e["modality"] == "audio/speech"
    assert e["object"] == "model"
    assert e["loaded"] is True
    assert e["last_loaded_at"] == "2026-01-01T00:00:00+00:00"
    assert e["unservable_reason"] is None
    # OpenAI-compat fields cannot be clobbered by capabilities either.
    assert e["created"] == 0
    assert e["owned_by"] == "muse"


def test_build_model_entry_openai_compat_fields():
    from muse.core.server import build_model_entry

    # hf_repo present: owned_by is the org slug (before the '/').
    e = build_model_entry(
        "foo", "audio/speech",
        {"hf_repo": "hexgrad/Kokoro-82M"},
        loaded=False, last_loaded_at=None, unservable_reason=None,
    )
    assert e["object"] == "model"
    assert e["created"] == 0
    assert e["owned_by"] == "hexgrad"


def test_build_model_entry_owned_by_defaults_to_muse_without_hf_repo():
    from muse.core.server import build_model_entry

    # No hf_repo (bundled custom model): owned_by falls back to "muse".
    e = build_model_entry(
        "soprano", "audio/speech", {},
        loaded=False, last_loaded_at=None, unservable_reason=None,
    )
    assert e["owned_by"] == "muse"
    assert e["created"] == 0


def test_build_model_entry_revision_cannot_be_clobbered():
    from muse.core.server import build_model_entry
    e = build_model_entry("m", "audio/speech", {"capabilities": {"revision": "evil"}},
                          loaded=False, last_loaded_at=None, unservable_reason=None, revision="a" * 40)
    assert e["revision"] == "a" * 40
    e = build_model_entry("m", "audio/speech", {"capabilities": {"revision": "evil"}},
                          loaded=False, last_loaded_at=None, unservable_reason=None)
    assert e["revision"] is None
