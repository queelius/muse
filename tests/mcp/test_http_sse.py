"""Tests for muse.mcp.server HTTP+SSE mode.

Smoke-test that build_http_app builds a Starlette app with the /mcp
mount in place. Also covers the bearer-token auth middleware added in
v0.45.6.
"""
from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


pytest.importorskip(
    "mcp.server.streamable_http_manager",
    reason="mcp SDK without streamable HTTP support",
)
pytest.importorskip("uvicorn")
pytest.importorskip("starlette")


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    client = MuseClient(server_url="http://test.example.com", admin_token="t")
    return MCPServer(client=client)


class TestBuildHttpApp:
    def test_build_returns_starlette_app(self, server):
        app = server.build_http_app()
        from starlette.applications import Starlette
        assert isinstance(app, Starlette)

    def test_route_mount_at_mcp(self, server):
        app = server.build_http_app()
        # Starlette routes are reachable via app.routes
        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/mcp" in paths

    def test_filter_admin_propagates_to_app(self, server):  # noqa: ARG002
        client = MuseClient(server_url="http://x")
        srv = MCPServer(client=client, filter_kind="admin")
        app = srv.build_http_app()
        from starlette.applications import Starlette
        assert isinstance(app, Starlette)


class TestRunHttpGracefulShutdown:
    """run_http must build its uvicorn.Config with a BOUNDED
    timeout_graceful_shutdown so `muse mcp --http` releases its port on
    Ctrl-C instead of hanging forever on a lingering SSE connection (the
    same port-binding bug fixed for `muse serve` / `muse _worker`)."""

    def test_config_has_bounded_graceful_shutdown(self, server, monkeypatch):
        import uvicorn

        from muse.cli_impl.serve_util import shutdown_grace_seconds

        captured = {}

        class _FakeServer:
            def __init__(self, config):
                self.config = config

            async def serve(self):
                return None

        def _spy_config(app, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(uvicorn, "Config", _spy_config)
        monkeypatch.setattr(uvicorn, "Server", _FakeServer)

        asyncio.run(server.run_http(host="127.0.0.1", port=8099))

        assert captured.get("timeout_graceful_shutdown") == shutdown_grace_seconds()
        assert captured["timeout_graceful_shutdown"] is not None


class TestHttpAuth:
    """Bearer-token middleware for the MCP HTTP transport (v0.45.6)."""

    @pytest.fixture
    def mcp_server(self, monkeypatch):
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        client = MuseClient(server_url="http://unused.example.com")
        return MCPServer(client=client)

    def test_no_token_configured_allows_requests(self, mcp_server, monkeypatch):
        """When MUSE_ADMIN_TOKEN is unset, the server runs in open mode
        (but emits a WARNING). All requests pass through unauthenticated."""
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        app = mcp_server.build_http_app()
        # We can't exercise the full MCP protocol here, but we can confirm
        # that the middleware does NOT return 401 for an unauthenticated request
        # when no token is configured. A non-MCP path would 404 from Starlette.
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/mcp")
            # Open mode: no 401 from our middleware. Status could be anything
            # from the MCP layer (likely 4xx for a bad MCP request), but NOT 401.
            assert resp.status_code != 401

    def test_missing_token_header_returns_401(self, mcp_server):
        """When a token is configured, missing Authorization header -> 401."""
        app = mcp_server.build_http_app(admin_token="super-secret")
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/mcp")
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "missing_token"

    def test_wrong_token_returns_403(self, mcp_server):
        """Wrong bearer token -> 403."""
        app = mcp_server.build_http_app(admin_token="correct-token")
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/mcp", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "invalid_token"

    def test_correct_token_passes_through(self, mcp_server):
        """Correct bearer token passes the middleware (MCP layer responds)."""
        token = "my-valid-token"
        app = mcp_server.build_http_app(admin_token=token)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                "/mcp", headers={"Authorization": f"Bearer {token}"},
            )
        # Middleware passed; Starlette/MCP may 4xx for a bad MCP GET,
        # but must NOT be 401 or 403 (our auth codes).
        assert resp.status_code not in (401, 403)

    def test_token_from_env_var(self, mcp_server, monkeypatch):
        """MUSE_ADMIN_TOKEN env var is honored when no explicit token is passed."""
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "env-token")
        app = mcp_server.build_http_app()  # no explicit token
        with TestClient(app, raise_server_exceptions=False) as client:
            # No header -> 401
            resp = client.get("/mcp")
        assert resp.status_code == 401

    def test_token_never_echoed_in_error_body(self, mcp_server):
        """The configured token must never appear in any error response body."""
        secret_token = "very-secret-abc123"
        app = mcp_server.build_http_app(admin_token=secret_token)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp1 = client.get("/mcp")  # missing header -> 401
            resp2 = client.get(
                "/mcp", headers={"Authorization": "Bearer wrong"},
            )  # wrong token -> 403
        for resp in (resp1, resp2):
            assert secret_token not in resp.text, (
                f"token leaked in {resp.status_code} response body"
            )
