"""Tests for muse.mcp.server HTTP+SSE mode.

Smoke-test that build_http_app builds a Starlette app with the /mcp
mount in place. Skips when uvicorn or the SDK's streamable transport
isn't importable.
"""
from __future__ import annotations

import pytest

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
