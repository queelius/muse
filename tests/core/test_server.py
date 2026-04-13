"""Tests for the core FastAPI app factory."""
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


def test_create_app_returns_fastapi():
    app = create_app(registry=ModalityRegistry(), routers={})
    assert isinstance(app, FastAPI)


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
    reg.register("audio.speech", Fake())
    reg.register("images.generations", Fake())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert set(r.json()["modalities"]) == {"audio.speech", "images.generations"}


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
    reg.register("audio.speech", FakeAudio())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "fake-tts" and m["modality"] == "audio.speech" for m in data)


def test_registry_stored_on_app_state():
    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})
    assert app.state.registry is reg
