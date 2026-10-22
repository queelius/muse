"""GET /v1/capabilities on the gateway."""
from __future__ import annotations

import asyncio
import copy
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl import gateway as gw
from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.supervisor import SupervisorState

REV = "c" * 40

CATALOG = {
    "fake-tts": {
        "enabled": True,
        "python_path": sys.executable,
        "local_dir": f"/w/models--org--fake-tts/snapshots/{REV}",
        "pid": 4242,
        "manifest": {
            "model_id": "fake-tts", "backend_path": "muse.fake:Model",
            "modality": "audio/speech",
            "hf_repo": "org/fake-tts", "license": "MIT",
            "description": "A fake voice.",
            "capabilities": {"memory_gb": 0.5, "voices": ["v1"], "n_threads": 2},
        },
    },
    "fake-asr": {
        "enabled": True,
        "python_path": sys.executable,
        "measurements": {"cpu": {"peak_bytes": 2 * 1024 ** 3}},
        "manifest": {
            "model_id": "fake-asr", "backend_path": "muse.fake:Model",
            "modality": "audio/transcription",
            "hf_repo": "org/fake-asr",
            "capabilities": {"device": "cpu"},
        },
    },
    "fake-disabled": {
        "enabled": False,
        "python_path": sys.executable,
        "manifest": {
            "model_id": "fake-disabled", "backend_path": "muse.fake:Model",
            "modality": "audio/speech",
            "hf_repo": "org/fake-disabled", "capabilities": {},
        },
    },
}


class DirectorSpy:
    def __init__(self):
        self.loaded: dict = {}
        self.lock = threading.RLock()
        self.acquire_calls = 0

    def acquire(self, *a, **k):
        self.acquire_calls += 1
        raise AssertionError("capabilities must never acquire")

    def release(self, *a, **k):
        raise AssertionError("capabilities must never release")


class StoreStub:
    def __init__(self, data):
        self.data = data

    def last_cold_load_seconds(self):
        return dict(self.data)


@pytest.fixture(autouse=True)
def _catalog(tmp_path, monkeypatch):
    from muse.core.catalog import (
        _reset_known_models_cache, _reset_read_catalog_cache, _write_catalog,
    )

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    gw._reset_capabilities_cache()
    _write_catalog(copy.deepcopy(CATALOG))
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    gw._reset_capabilities_cache()


@pytest.fixture
def manifest_calls():
    calls: list[str] = []

    def _gm(model_id):
        calls.append(model_id)
        from muse.core.catalog import _read_catalog
        return copy.deepcopy(_read_catalog()[model_id]["manifest"])

    with patch("muse.cli_impl.gateway.get_manifest", side_effect=_gm):
        yield calls


@pytest.fixture
def revalidate_calls():
    calls: list[str] = []

    def _rv(state, model_id, **_):
        calls.append(model_id)
        return "no worker environment" if model_id == "fake-asr" else None

    with patch("muse.cli_impl.gateway.revalidate_servability", side_effect=_rv):
        yield calls


@pytest.fixture
def director_spy():
    return DirectorSpy()


@pytest.fixture
def state(director_spy):
    s = SupervisorState(workers=[], device="cpu")
    s.director = director_spy
    s.telemetry_store = StoreStub({"fake-tts": 12.5})
    return s


@pytest.fixture
def gateway_client(state, manifest_calls, revalidate_calls):
    return TestClient(build_gateway(state=state))


def _models(client, **params):
    r = client.get("/v1/capabilities", params=params)
    assert r.status_code == 200, r.text
    return {m["id"]: m for m in r.json()["models"]}


def test_gateway_capabilities_lists_unloaded_enabled_models(gateway_client):
    r = gateway_client.get("/v1/capabilities")
    assert r.status_code == 200
    doc = r.json()
    assert doc["object"] == "muse.capabilities"
    assert doc["server"]["mcp"]["env"]["MUSE_SERVER"] == "http://testserver"
    assert doc["server"]["links"]["capabilities"] == "/v1/capabilities"
    by_id = {m["id"]: m for m in doc["models"]}
    assert set(by_id) == {"fake-tts", "fake-asr"}

    tts = by_id["fake-tts"]
    assert tts["loaded"] is False
    assert tts["revision"] == REV
    assert tts["components"] == [{"role": "weights", "repo": "org/fake-tts", "revision": REV}]
    assert tts["summary"] == "A fake voice."
    assert tts["capabilities"] == {"memory_gb": 0.5, "voices": ["v1"]}
    assert tts["memory_gb"] == 0.5
    assert tts["last_cold_load_seconds"] == 12.5
    assert tts["queue_depth"] == 0
    assert tts["unservable_reason"] is None

    asr = by_id["fake-asr"]
    # Sized from the probe measurement (sizing ladder), not a declaration.
    assert asr["memory_gb"] == pytest.approx(2.0)
    assert asr["last_cold_load_seconds"] is None
    assert asr["unservable_reason"] == "no worker environment"

    speech = next(m for m in doc["modalities"] if m["modality"] == "audio/speech")
    assert speech["models"] == ["fake-tts"]
    assert speech["routes"] and speech["summary"]


def test_gateway_capabilities_excludes_disabled_and_admin_data(gateway_client):
    body = gateway_client.get("/v1/capabilities").text
    assert "fake-disabled" not in body
    assert "pid" not in body and "admin" not in body.lower().replace("/v1/admin", "")
    assert "python_path" not in body and "n_threads" not in body


def test_queue_depth_null_without_gate(state, manifest_calls, revalidate_calls):
    state.concurrency_gate = None
    client = TestClient(build_gateway(state=state))
    assert all(m["queue_depth"] is None for m in _models(client).values())

    state.concurrency_gate = SimpleNamespace(depths=lambda: {"fake-asr": 3})
    by_id = _models(client)
    assert by_id["fake-asr"]["queue_depth"] == 3
    assert by_id["fake-tts"]["queue_depth"] == 0


def test_resident_model_is_loaded_and_skips_revalidation(
    gateway_client, director_spy, revalidate_calls,
):
    director_spy.loaded["fake-tts"] = SimpleNamespace(loaded_at=1.0, memory_gb=0.5)
    by_id = _models(gateway_client)
    assert by_id["fake-tts"]["loaded"] is True
    assert by_id["fake-tts"]["unservable_reason"] is None
    assert "fake-tts" not in revalidate_calls
    assert by_id["fake-asr"]["loaded"] is False


def test_resident_model_is_listed_even_when_disabled(gateway_client, director_spy):
    director_spy.loaded["fake-disabled"] = SimpleNamespace(loaded_at=1.0)
    by_id = _models(gateway_client)
    assert by_id["fake-disabled"]["loaded"] is True


def test_capabilities_does_not_trigger_load(gateway_client, director_spy):
    gateway_client.get("/v1/capabilities")
    assert director_spy.acquire_calls == 0


def test_unknown_filter_404(gateway_client):
    for params in ({"modality": "video/nope"}, {"model": "nope"}, {"model": "fake-disabled"}):
        r = gateway_client.get("/v1/capabilities", params=params)
        assert r.status_code == 404, params
        assert r.json()["error"]["code"] == "not_found"


def test_filters_slice_the_document(gateway_client):
    doc = gateway_client.get("/v1/capabilities", params={"model": "fake-asr"}).json()
    assert [m["id"] for m in doc["models"]] == ["fake-asr"]
    assert [m["modality"] for m in doc["modalities"]] == ["audio/transcription"]
    doc = gateway_client.get("/v1/capabilities", params={"modality": "audio/speech"}).json()
    assert [m["id"] for m in doc["models"]] == ["fake-tts"]


def test_static_facts_cached_on_catalog_mtime(gateway_client, manifest_calls):
    _models(gateway_client)
    first = len(manifest_calls)
    assert first >= 2
    _models(gateway_client)
    assert len(manifest_calls) == first  # cached

    from muse.core.catalog import _catalog_path, _read_catalog, _write_catalog
    data = copy.deepcopy(_read_catalog())
    data["fake-tts"]["manifest"]["description"] = "A new voice."
    _write_catalog(data)
    st = _catalog_path().stat()
    os.utime(_catalog_path(), ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    by_id = _models(gateway_client)
    assert len(manifest_calls) > first
    assert by_id["fake-tts"]["summary"] == "A new voice."


def test_live_state_recomputed_every_call(gateway_client, state):
    assert _models(gateway_client)["fake-tts"]["last_cold_load_seconds"] == 12.5
    state.telemetry_store = StoreStub({"fake-tts": 3.0})
    assert _models(gateway_client)["fake-tts"]["last_cold_load_seconds"] == 3.0
    state.telemetry_store = None
    assert _models(gateway_client)["fake-tts"]["last_cold_load_seconds"] is None


def test_broken_telemetry_store_degrades(gateway_client, state):
    class Boom:
        def last_cold_load_seconds(self):
            raise RuntimeError("closed")

    state.telemetry_store = Boom()
    assert _models(gateway_client)["fake-tts"]["last_cold_load_seconds"] is None


def test_bad_manifest_degrades_to_safe_fields(gateway_client):
    from muse.core.catalog import CatalogError

    def _gm(model_id):
        raise CatalogError("broken")

    with patch("muse.cli_impl.gateway.get_manifest", side_effect=_gm):
        by_id = _models(gateway_client)
    assert by_id["fake-tts"]["modality"] == "audio/speech"
    assert by_id["fake-tts"]["capabilities"] == {}


def test_build_runs_off_the_event_loop(gateway_client):
    real = asyncio.to_thread
    seen: list[str] = []

    async def spy(fn, *a, **k):
        seen.append(getattr(fn, "__name__", repr(fn)))
        return await real(fn, *a, **k)

    with patch("muse.cli_impl.gateway.asyncio.to_thread", side_effect=spy):
        assert gateway_client.get("/v1/capabilities").status_code == 200
    assert "_capabilities_document" in seen


def test_capabilities_document_callable_on_app_state(gateway_client):
    """Task 9 mounts /llms.txt from this awaitable."""
    fn = gateway_client.app.state.capabilities_document
    assert callable(fn)


def test_static_routes_mode_serves_document(manifest_calls):
    from muse.cli_impl.gateway import WorkerRoute

    app = build_gateway(routes=[WorkerRoute("fake-tts", "http://127.0.0.1:9")])
    by_id = _models(TestClient(app))
    assert by_id["fake-tts"]["loaded"] is True
    assert by_id["fake-asr"]["loaded"] is False
    assert by_id["fake-asr"]["unservable_reason"] is None


def test_catalog_file_key_is_replacement_sensitive(tmp_path, monkeypatch):
    from muse.cli_impl import gateway
    from muse.core import catalog_store

    path = tmp_path / "catalog.json"
    path.write_text("{}")
    os.utime(path, ns=(10**18, 10**18))
    monkeypatch.setattr(catalog_store, "_catalog_path", lambda: path)
    before = gateway._catalog_file_key()
    tmp = tmp_path / "next.json"
    tmp.write_text("{}")
    os.utime(tmp, ns=(10**18, 10**18))
    os.replace(tmp, path)  # same size, same mtime, new inode
    assert gateway._catalog_file_key() != before
