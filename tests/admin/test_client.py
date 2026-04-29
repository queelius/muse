"""Tests for AdminClient.

httpx is patched at the call site; no real server is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from muse.admin.client import AdminClient, AdminClientError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    return AdminClient(base_url="http://test.example.com", token="tok")


@pytest.fixture
def mock_httpx_client():
    """Patch httpx.Client to a context-managed MagicMock returning the mock."""
    with patch("muse.admin.client.httpx.Client") as cls:
        ctx = MagicMock()
        cls.return_value = ctx
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=None)
        yield ctx


def _ok(status: int, body: dict):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = "ok"
    return r


def _err(status: int, code: str, message: str):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {
        "error": {"code": code, "message": message, "type": "invalid_request_error"},
    }
    r.text = "err"
    return r


class TestEnableDisable:
    def test_enable_calls_post(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(202, {"job_id": "j1", "status": "pending"})
        out = client.enable("kokoro-82m")
        assert out == {"job_id": "j1", "status": "pending"}
        call = mock_httpx_client.request.call_args
        assert call.args[0] == "POST"
        assert call.args[1].endswith("/v1/admin/models/kokoro-82m/enable")
        assert call.kwargs["headers"]["Authorization"] == "Bearer tok"

    def test_disable_calls_post(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(200, {"model_id": "k", "loaded": False})
        out = client.disable("kokoro-82m")
        assert out["loaded"] is False


class TestPullProbe:
    def test_pull_uses_underscore_path_with_body(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(202, {"job_id": "j2", "status": "pending"})
        client.pull("hf://Qwen/Qwen3-9B-GGUF@q4_k_m")
        call = mock_httpx_client.request.call_args
        assert call.args[1].endswith("/v1/admin/models/_/pull")
        assert call.kwargs["json"] == {"identifier": "hf://Qwen/Qwen3-9B-GGUF@q4_k_m"}

    def test_probe_passes_options(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(202, {"job_id": "j3", "status": "pending"})
        client.probe("kokoro-82m", no_inference=True, device="cpu")
        call = mock_httpx_client.request.call_args
        body = call.kwargs["json"]
        assert body["no_inference"] is True
        assert body["device"] == "cpu"


class TestStatusAndViews:
    def test_status_returns_record(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(200, {
            "model_id": "kokoro-82m", "enabled": True, "loaded": True,
            "worker_port": 9001,
        })
        out = client.status("kokoro-82m")
        assert out["worker_port"] == 9001

    def test_workers_aggregates_view(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(200, {"workers": [{"port": 9001}]})
        out = client.workers()
        assert len(out["workers"]) == 1


class TestErrorEnvelope:
    def test_400_raises_AdminClientError(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _err(404, "model_not_found", "ghost")
        with pytest.raises(AdminClientError) as exc:
            client.status("ghost")
        assert exc.value.status == 404
        assert exc.value.code == "model_not_found"

    def test_503_admin_disabled_propagates(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _err(503, "admin_disabled", "no token")
        with pytest.raises(AdminClientError) as exc:
            client.workers()
        assert exc.value.status == 503

    def test_unwraps_detail_envelope(self, client, mock_httpx_client):
        # FastAPI's HTTPException(detail=...) wraps the OpenAI envelope
        # one layer deeper. AdminClient should still extract the code.
        r = MagicMock()
        r.status_code = 401
        r.json.return_value = {
            "detail": {"error": {
                "code": "missing_token", "message": "...",
                "type": "invalid_request_error",
            }},
        }
        r.text = "err"
        mock_httpx_client.request.return_value = r
        with pytest.raises(AdminClientError) as exc:
            client.workers()
        assert exc.value.code == "missing_token"


class TestWait:
    def test_wait_returns_when_done(self, client, mock_httpx_client):
        mock_httpx_client.request.side_effect = [
            _ok(200, {"job_id": "j", "state": "running"}),
            _ok(200, {"job_id": "j", "state": "running"}),
            _ok(200, {"job_id": "j", "state": "done", "result": {"x": 1}}),
        ]
        out = client.wait("j", poll=0.001, timeout=2.0)
        assert out["state"] == "done"

    def test_wait_raises_on_timeout(self, client, mock_httpx_client):
        mock_httpx_client.request.return_value = _ok(200, {"job_id": "j", "state": "running"})
        with pytest.raises(TimeoutError):
            client.wait("j", poll=0.001, timeout=0.05)


class TestTokenResolution:
    def test_constructor_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "env-token")
        c = AdminClient(token="ctor-token")
        assert c.token == "ctor-token"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "env-token")
        c = AdminClient()
        assert c.token == "env-token"

    def test_base_url_resolution_precedence(self, monkeypatch):
        monkeypatch.setenv("MUSE_SERVER", "http://env-host:8080")
        c = AdminClient(base_url="http://ctor-host:9000")
        assert c.base_url == "http://ctor-host:9000"

    def test_base_url_default(self, monkeypatch):
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        c = AdminClient()
        assert c.base_url == "http://localhost:8000"
