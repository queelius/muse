"""Tests for muse.mcp.client.MuseClient.

Probes use a mocked httpx.Client patched at the call site; the discovery
and call surface uses an httpx.MockTransport. No network is touched.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.mcp.client import MuseClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    return MuseClient(server_url="http://test.example.com", admin_token="tok")


@pytest.fixture
def mock_httpx_client():
    """Patch httpx.Client to a context-managed MagicMock returning the mock."""
    with patch("muse.mcp.client.httpx.Client") as cls:
        ctx = MagicMock()
        cls.return_value = ctx
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=None)
        yield ctx


def _ok_json(body: dict, status: int = 200, content_type: str = "application/json"):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.headers = {"content-type": content_type}
    r.content = b""
    r.raise_for_status = MagicMock()
    return r


class TestProbes:
    def test_health(self, client, mock_httpx_client):
        mock_httpx_client.get.return_value = _ok_json({"status": "ok"})
        out = client.health()
        assert out == {"status": "ok"}
        assert mock_httpx_client.get.call_args.args[0].endswith("/health")

    def test_list_models(self, client, mock_httpx_client):
        mock_httpx_client.get.return_value = _ok_json(
            {"data": [{"id": "kokoro-82m"}]},
        )
        out = client.list_models()
        assert out["data"][0]["id"] == "kokoro-82m"


class TestDiscoveryAndCall:
    """The generic surface the generated tools use, over a MockTransport."""

    @staticmethod
    def _client(handler):
        import httpx

        seen = []

        def route(request):
            seen.append(request)
            return handler(request)

        c = MuseClient(server_url="http://muse.test/",
                       transport=httpx.MockTransport(route))
        return c, seen

    def test_capabilities_passes_filters(self):
        import httpx

        c, seen = self._client(lambda r: httpx.Response(200, json={"object": "x"}))
        assert c.capabilities(modality="audio/speech") == {"object": "x"}
        assert seen[-1].url.path == "/v1/capabilities"
        assert dict(seen[-1].url.params) == {"modality": "audio/speech"}
        c.capabilities()
        assert dict(seen[-1].url.params) == {}

    def test_openapi(self):
        import httpx

        c, seen = self._client(lambda r: httpx.Response(200, json={"openapi": "3.1.0"}))
        assert c.openapi()["openapi"] == "3.1.0"
        assert seen[-1].url.path == "/openapi.json"

    def test_call_json_body(self):
        import httpx

        c, seen = self._client(lambda r: httpx.Response(200, content=b"RIFF"))
        r = c.call({"method": "POST", "path": "/v1/audio/speech"},
                   json={"model": "m", "input": "hi"})
        assert r.content == b"RIFF"
        assert seen[-1].method == "POST"
        assert seen[-1].headers["content-type"] == "application/json"

    def test_call_multipart_and_get(self):
        import httpx

        c, seen = self._client(lambda r: httpx.Response(200, json={}))
        c.call({"method": "POST", "path": "/v1/audio/alignments"},
               data={"text": "hi"}, files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert seen[-1].headers["content-type"].startswith("multipart/form-data")
        c.call({"method": "GET", "path": "/languages"}, params={"model": "m"})
        assert seen[-1].method == "GET" and seen[-1].url.params["model"] == "m"

    def test_call_raises_on_error_status(self):
        import httpx

        c, _ = self._client(lambda r: httpx.Response(500, json={}))
        with pytest.raises(httpx.HTTPStatusError):
            c.call({"method": "POST", "path": "/v1/embeddings"}, json={})


class TestServerUrlResolution:
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_SERVER", "http://env-host:8080")
        c = MuseClient(server_url="http://ctor-host:9000")
        assert c.server_url == "http://ctor-host:9000"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_SERVER", "http://env-host:8080")
        c = MuseClient()
        assert c.server_url == "http://env-host:8080"

    def test_default_is_localhost(self, monkeypatch):
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        c = MuseClient()
        assert c.server_url == "http://localhost:8000"

    def test_admin_token_passes_through(self, monkeypatch):
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        c = MuseClient(server_url="http://x", admin_token="t")
        assert c.admin.token == "t"


class TestSearchModels:
    def test_search_uses_resolver(self, client, monkeypatch):
        from types import SimpleNamespace

        def fake_search(query, *, backend, modality, limit):  # noqa: ARG001
            return [
                SimpleNamespace(
                    uri="hf://x/y@q4",
                    modality="chat/completion",
                    size_gb=1.2,
                    downloads=5,
                    license="apache",
                    description="d",
                ),
            ]
        monkeypatch.setattr(
            "muse.core.resolvers.search", fake_search,
        )
        # also patch resolvers_hf import side-effect to a no-op
        monkeypatch.setattr(
            "muse.core.resolvers_hf.HFResolver", object, raising=False,
        )
        out = client.search_models(query="qwen", modality="chat/completion")
        assert out["results"][0]["uri"] == "hf://x/y@q4"

    def test_search_filters_by_max_size(self, client, monkeypatch):
        from types import SimpleNamespace

        def fake_search(query, *, backend, modality, limit):  # noqa: ARG001
            return [
                SimpleNamespace(uri="a", modality="x", size_gb=0.5,
                                downloads=1, license=None, description=""),
                SimpleNamespace(uri="b", modality="x", size_gb=10.0,
                                downloads=1, license=None, description=""),
                SimpleNamespace(uri="c", modality="x", size_gb=None,
                                downloads=1, license=None, description=""),
            ]
        monkeypatch.setattr("muse.core.resolvers.search", fake_search)
        out = client.search_models(query="q", max_size_gb=1.0)
        uris = [r["uri"] for r in out["results"]]
        assert "a" in uris and "c" in uris
        assert "b" not in uris  # filtered out by size cap
