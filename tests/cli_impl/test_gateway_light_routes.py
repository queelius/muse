"""The gateway serves light routes (voice catalog) from disk, ahead of its
proxy: no director acquire, no worker, the proxy's own model errors."""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import wave
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from muse.cli_impl import gateway as gw
from muse.cli_impl.gateway import build_gateway, extract_model_from_request
from muse.cli_impl.supervisor import SupervisorState
from muse.core import config


def _entry(model_id, caps, *, enabled=True):
    return {
        "enabled": enabled,
        "python_path": sys.executable,
        "manifest": {
            "model_id": model_id, "backend_path": "muse.fake:Model",
            "modality": "audio/speech", "hf_repo": f"org/{model_id}",
            "capabilities": caps,
        },
    }


CATALOG = {
    "fake-clone": _entry("fake-clone", {"supports_saved_voices": True, "memory_gb": 3.0}),
    "fake-preset": _entry("fake-preset", {"memory_gb": 0.5}),
    "fake-off": _entry("fake-off", {"supports_saved_voices": True}, enabled=False),
}


class DirectorSpy:
    def __init__(self):
        self.loaded: dict = {}
        self.lock = threading.RLock()
        self.acquire_calls = 0

    def acquire(self, *a, **k):
        self.acquire_calls += 1
        raise AssertionError("light routes must never acquire")

    def release(self, *a, **k):
        raise AssertionError("light routes must never release")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from muse.core.catalog import (
        _reset_known_models_cache, _reset_read_catalog_cache, _write_catalog,
    )

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("MUSE_HOME", str(tmp_path / "home"))
    config.reset_config()
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    gw._reset_capabilities_cache()
    _write_catalog(copy.deepcopy(CATALOG))
    yield tmp_path / "home"
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    gw._reset_capabilities_cache()
    config.reset_config()


@pytest.fixture
def director():
    return DirectorSpy()


@pytest.fixture
def client(director):
    s = SupervisorState(workers=[], device="cpu")
    s.director = director
    return TestClient(build_gateway(state=s))


def _voice(home, name, model="fake-clone", **meta):
    root = home / "voices" / model
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    os.chmod(home / "voices", 0o700)
    os.chmod(root, 0o700)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0\0" * 48000)
    (d / "reference.wav").write_bytes(buf.getvalue())
    (d / "voice.json").write_text(json.dumps({
        "name": name, "transcript": "Hi.", "sample_rate": 24000,
        "duration_seconds": 2.0, **meta}))


def test_list_saved_voices_never_acquires(client, director, _env):
    _voice(_env, "sage", category="narrator/epic")
    r = client.get("/v1/audio/speech/voices?model=fake-clone")
    assert r.status_code == 200, r.text
    assert r.json()["voices"] == ["narrator", "sage"]
    assert director.acquire_calls == 0


def test_get_audio_patch_and_delete_never_acquire(client, director, _env):
    _voice(_env, "sage")
    assert client.get("/v1/audio/speech/voices/sage?model=fake-clone").status_code == 200
    audio = client.get("/v1/audio/speech/voices/sage/audio?model=fake-clone")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    r = client.patch("/v1/audio/speech/voices/sage",
                     json={"model": "fake-clone", "category": "narrator/epic"})
    assert r.status_code == 200 and r.json()["category"] == "narrator/epic"
    r = client.delete("/v1/audio/speech/voices/sage?model=fake-clone")
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert director.acquire_calls == 0


def test_missing_model_is_400_model_required(client):
    r = client.get("/v1/audio/speech/voices")
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_required"


def test_invalid_model_id_is_400_invalid_model(client):
    r = client.get("/v1/audio/speech/voices", params={"model": "bad\nid"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_model"


def test_disabled_model_is_409_model_disabled(client):
    r = client.get("/v1/audio/speech/voices?model=fake-off")
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_disabled"


def test_unknown_model_is_404(client):
    r = client.get("/v1/audio/speech/voices?model=nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_model_of_another_modality_is_404(client, _env):
    from muse.core.catalog import _read_catalog, _reset_read_catalog_cache, _write_catalog

    cat = copy.deepcopy(_read_catalog())
    cat["fake-asr"] = _entry("fake-asr", {})
    cat["fake-asr"]["manifest"]["modality"] = "audio/transcription"
    _write_catalog(cat)
    _reset_read_catalog_cache()
    r = client.get("/v1/audio/speech/voices?model=fake-asr")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_preset_model_list_forwards_through_director(client):
    seen = {}

    async def fake_route(request, full_path, model_id, state, timeout, **kw):
        seen.update(full_path=full_path, model_id=model_id)
        return JSONResponse({"model": model_id, "voices": ["af_heart"]})

    with patch.object(gw, "_route_via_director", side_effect=fake_route):
        r = client.get("/v1/audio/speech/voices?model=fake-preset")
    assert r.status_code == 200 and r.json()["voices"] == ["af_heart"]
    assert seen == {"full_path": "v1/audio/speech/voices", "model_id": "fake-preset"}


def test_preset_model_patch_is_voice_saving_not_supported(client, director):
    r = client.patch("/v1/audio/speech/voices/x", json={"model": "fake-preset", "notes": "n"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "voice_saving_not_supported"
    assert director.acquire_calls == 0


def test_patch_validation_error_is_envelope_not_detail(client, _env):
    _voice(_env, "sage")
    r = client.patch("/v1/audio/speech/voices/sage", json={"model": "fake-clone", "colour": 1})
    assert r.status_code == 422
    body = r.json()
    assert "detail" not in body and body["error"]["code"] == "invalid_request"


def test_openapi_lists_voice_light_routes_with_descriptions(client):
    paths = client.get("/openapi.json").json()["paths"]
    item = paths["/v1/audio/speech/voices/{name}"]
    assert {"get", "patch", "delete"} <= set(item)
    assert "get" in paths["/v1/audio/speech/voices/{name}/audio"]
    assert item["patch"]["tags"] == ["audio/speech"]
    assert "catalog" in item["patch"]["description"]


@pytest.mark.parametrize("method,headers,body", [
    ("PATCH", {"content-type": "application/json"}, b'{"model": "m1"}'),
    ("POST", {"content-type": "application/json"}, b'{"model": "m1"}'),
])
def test_extract_model_reads_json_body_on_patch_and_post(method, headers, body):
    import asyncio

    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {"type": "http", "method": method, "path": "/x", "query_string": b"",
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    assert asyncio.run(extract_model_from_request(Request(scope, receive))) == "m1"
