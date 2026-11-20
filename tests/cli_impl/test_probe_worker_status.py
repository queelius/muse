"""Tests for `muse models info`'s worker-status probing (v0.47.4).

`_probe_online_worker_status` prefers the admin API (rich worker detail)
when MUSE_ADMIN_TOKEN is set, then falls back to the PUBLIC /v1/models
loaded state so the info view shows loaded/not-loaded for anyone who can
reach the server, not just operators holding an admin token.
"""
from __future__ import annotations

from unittest.mock import patch


def test_public_loaded_status_reports_loaded():
    from muse.cli_impl import admin_actions as cli

    data = [{"id": "m", "loaded": True}, {"id": "other", "loaded": False}]
    with patch("muse.cli_impl.runtime_state.fetch_public_models", return_value=data):
        assert cli.public_loaded_status("m") == {
            "loaded": True, "detail_source": "public",
        }


def test_public_loaded_status_reports_not_loaded_when_reachable():
    from muse.cli_impl import admin_actions as cli

    # Server reachable, model not listed -> reachable but not resident.
    with patch("muse.cli_impl.runtime_state.fetch_public_models", return_value=[]):
        assert cli.public_loaded_status("m") == {
            "loaded": False, "detail_source": "public",
        }


def test_public_loaded_status_none_when_unreachable():
    from muse.cli_impl import admin_actions as cli

    with patch("muse.cli_impl.runtime_state.fetch_public_models", return_value=None):
        assert cli.public_loaded_status("m") is None


def test_probe_prefers_admin_when_available():
    from muse.cli_impl import admin_actions as cli

    admin_status = {"loaded": True, "worker_port": 9001, "worker_pid": 42}
    with patch.object(cli, "probe_admin_worker_status", return_value=admin_status), \
         patch.object(cli, "public_loaded_status") as pub:
        assert cli.probe_online_worker_status("m") is admin_status
        pub.assert_not_called()


def test_probe_falls_back_to_public_without_admin():
    from muse.cli_impl import admin_actions as cli

    pub_status = {"loaded": True, "detail_source": "public"}
    with patch.object(cli, "probe_admin_worker_status", return_value=None), \
         patch.object(cli, "public_loaded_status", return_value=pub_status):
        assert cli.probe_online_worker_status("m") == pub_status


def test_probe_admin_worker_status_returns_none_without_token(monkeypatch):
    """No admin token -> the admin probe returns None (so the caller falls
    back to the public path) without constructing an AdminClient."""
    from muse.cli_impl import admin_actions as cli

    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    constructed = {"count": 0}

    class _Boom:
        def __init__(self, *a, **kw):
            constructed["count"] += 1

    monkeypatch.setattr("muse.admin.client.AdminClient", _Boom, raising=False)
    assert cli.probe_admin_worker_status("m") is None
    assert constructed["count"] == 0
