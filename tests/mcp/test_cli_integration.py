"""Tests for the `muse mcp` CLI subcommand.

Builds the argparse tree and asserts the subcommand exists, has the
expected flags, and dispatches to the right entry point. The actual
asyncio.run is mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from muse.cli import build_parser


class TestArgparse:
    def test_mcp_subcommand_present(self):
        parser = build_parser()
        # Parse with no args lands on no subcommand (argparse with
        # required=False); ensure 'mcp' parses cleanly.
        args = parser.parse_args(["mcp"])
        assert args.cmd == "mcp"
        # Defaults
        assert args.http is False
        assert args.port == 8088
        assert args.filter_kind == "all"

    def test_mcp_http_flag(self):
        parser = build_parser()
        args = parser.parse_args(["mcp", "--http", "--port", "9999"])
        assert args.http is True
        assert args.port == 9999

    def test_mcp_filter_admin(self):
        parser = build_parser()
        args = parser.parse_args(["mcp", "--filter", "admin"])
        assert args.filter_kind == "admin"

    def test_mcp_filter_inference(self):
        parser = build_parser()
        args = parser.parse_args(["mcp", "--filter", "inference"])
        assert args.filter_kind == "inference"

    def test_mcp_server_and_token_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "mcp",
            "--server", "http://other:8000",
            "--admin-token", "abc",
        ])
        assert args.server == "http://other:8000"
        assert args.admin_token == "abc"


class TestRunMcpServer:
    def test_runs_stdio_by_default(self, monkeypatch):
        from muse.cli_impl.mcp_server import run_mcp_server

        ran_modes: list[str] = []

        def fake_asyncio_run(coro):
            # The coro is run_stdio(); record and close it.
            ran_modes.append(coro.__qualname__)
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        # Mock health to short-circuit
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False,
                port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="all",
            )
        assert rc == 0
        assert any("run_stdio" in m for m in ran_modes)

    def test_runs_http_when_flag_set(self, monkeypatch):
        from muse.cli_impl.mcp_server import run_mcp_server

        ran_modes: list[str] = []

        def fake_asyncio_run(coro):
            ran_modes.append(coro.__qualname__)
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=True,
                port=8088,
                server_url="http://test",
                admin_token="t",
                filter_kind="all",
            )
        assert rc == 0
        assert any("run_http" in m for m in ran_modes)

    def test_unreachable_health_warns_but_continues(self, monkeypatch, capsys):
        from muse.cli_impl.mcp_server import run_mcp_server

        def fake_asyncio_run(coro):
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        # Force health to raise
        with patch(
            "muse.mcp.client.MuseClient.health",
            side_effect=ConnectionError("nope"),
        ):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="all",
            )
        captured = capsys.readouterr()
        assert "warning: could not reach muse server" in captured.err
        assert rc == 0

    def test_admin_filter_without_token_warns(self, monkeypatch, capsys):
        from muse.cli_impl.mcp_server import run_mcp_server

        def fake_asyncio_run(coro):
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="admin",
            )
        captured = capsys.readouterr()
        assert "warning: --filter admin without MUSE_ADMIN_TOKEN" in captured.err
        assert rc == 0

    def test_invalid_filter_returns_2(self, monkeypatch, capsys):
        from muse.cli_impl.mcp_server import run_mcp_server

        # Bypass argparse choices=... by calling run_mcp_server directly.
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="bogus",
            )
        captured = capsys.readouterr()
        assert rc == 2
        assert "error" in captured.err

    def test_keyboard_interrupt_exits_zero(self, monkeypatch):
        from muse.cli_impl.mcp_server import run_mcp_server

        def fake_asyncio_run(coro):
            coro.close()
            raise KeyboardInterrupt

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="all",
            )
        assert rc == 0
