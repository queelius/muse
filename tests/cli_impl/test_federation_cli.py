"""Tests for the `muse federate` CLI command.

Mirrors the existing `muse config` CLI test conventions: invoked
in-process via typer's CliRunner against the real `app`, with
MUSE_CATALOG_DIR pointed at a tmp_path so nothing touches ~/.muse.

`run_coordinator` itself blocks forever on a real invocation (it runs
uvicorn), so these tests monkeypatch `muse.cli_impl.federation.run_coordinator`
to a fast no-op and assert the CLI wires arguments through correctly,
plus the real no-nodes exit-2 path (which returns before ever reaching
uvicorn).
"""
from typer.testing import CliRunner

from muse.core import config as cfg
from muse.cli import app
import muse.cli_impl.federation as fed

runner = CliRunner()


def test_federate_no_nodes_exits_2(monkeypatch, tmp_path):
    """No --node, no --config, and no federation.yaml in the catalog
    dir: `muse federate` must refuse to start a server and exit 2 with
    a clear message."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["federate"])
    assert r.exit_code == 2
    assert "no federation nodes" in r.output.lower()


def test_federate_passes_cli_nodes_through(monkeypatch, tmp_path):
    """`--node http://a:8000` must reach `run_coordinator` as
    `cli_nodes=["http://a:8000"]`; run_coordinator itself is
    monkeypatched to a no-op so the test never starts a real server."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()

    seen = {}

    def fake_run_coordinator(*, host, port, cli_nodes, config_path):
        seen["host"] = host
        seen["port"] = port
        seen["cli_nodes"] = cli_nodes
        seen["config_path"] = config_path
        return 0

    monkeypatch.setattr(fed, "run_coordinator", fake_run_coordinator)

    r = runner.invoke(app, ["federate", "--node", "http://a:8000"])
    assert r.exit_code == 0
    assert seen["cli_nodes"] == ["http://a:8000"]
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 8100
    assert seen["config_path"] is None
