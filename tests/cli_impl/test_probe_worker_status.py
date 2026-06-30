"""Tests for `muse models info`'s worker-status probing (v0.47.4).

`_probe_online_worker_status` prefers the admin API (rich worker detail)
when MUSE_ADMIN_TOKEN is set, then falls back to the PUBLIC /v1/models
loaded state so the info view shows loaded/not-loaded for anyone who can
reach the server, not just operators holding an admin token.
"""
from __future__ import annotations

from unittest.mock import patch


def test_public_loaded_status_reports_loaded():
    from muse import cli

    data = [{"id": "m", "loaded": True}, {"id": "other", "loaded": False}]
    with patch("muse.cli_impl.runtime_state.fetch_public_models", return_value=data):
        assert cli._public_loaded_status("m") == {
            "loaded": True, "detail_source": "public",
        }


def test_public_loaded_status_reports_not_loaded_when_reachable():
    from muse import cli

    # Server reachable, model not listed -> reachable but not resident.
    with patch("muse.cli_impl.runtime_state.fetch_public_models", return_value=[]):
        assert cli._public_loaded_status("m") == {
            "loaded": False, "detail_source": "public",
        }


def test_public_loaded_status_none_when_unreachable():
    from muse import cli

    with patch("muse.cli_impl.runtime_state.fetch_public_models", return_value=None):
        assert cli._public_loaded_status("m") is None


def test_probe_prefers_admin_when_available():
    from muse import cli

    admin_status = {"loaded": True, "worker_port": 9001, "worker_pid": 42}
    with patch.object(cli, "_probe_admin_worker_status", return_value=admin_status), \
         patch.object(cli, "_public_loaded_status") as pub:
        assert cli._probe_online_worker_status("m") is admin_status
        pub.assert_not_called()


def test_probe_falls_back_to_public_without_admin():
    from muse import cli

    pub_status = {"loaded": True, "detail_source": "public"}
    with patch.object(cli, "_probe_admin_worker_status", return_value=None), \
         patch.object(cli, "_public_loaded_status", return_value=pub_status):
        assert cli._probe_online_worker_status("m") == pub_status
