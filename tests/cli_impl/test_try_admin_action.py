"""Tests for `muse.cli._try_admin_action`'s enable/disable dispatch.

`_try_admin_action` is the admin-API-first path `models_enable` /
`models_disable` consult before falling back to a catalog-only mutation.
The `enable` branch nested every `return True` inside `if job_id:`, so a
successful admin call (`client.enable()` returned without raising, i.e.
the server already accepted and started the operation) whose response
happened to omit -- or falsy -- `job_id` fell through to the trailing
`return False`. The caller then ran the catalog-only fallback AND
printed a misleading "catalog only" message even though the admin API
had already handled the action.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _with_admin_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "test-token")


def test_enable_missing_job_id_is_still_handled(monkeypatch, capsys):
    """A successful admin `enable()` response without a job_id must
    still be treated as handled (return True), not fall through to the
    catalog fallback.
    """
    from muse.cli import _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"status": "accepted"}  # no job_id key

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        handled = _try_admin_action("enable", "some-model")

    assert handled is True
    fake_client.wait.assert_not_called()


def test_enable_falsy_job_id_is_still_handled(monkeypatch):
    """An empty-string job_id is falsy but still a successful response
    shape; must not be treated as "admin API not used".
    """
    from muse.cli import _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "", "status": "accepted"}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        handled = _try_admin_action("enable", "some-model")

    assert handled is True


def test_enable_with_job_id_still_waits_and_reports(monkeypatch):
    """Sanity check: the normal (job_id present) path is unaffected."""
    from muse.cli import _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.return_value = {"state": "done", "result": {"worker_port": 9001}}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        handled = _try_admin_action("enable", "some-model")

    assert handled is True
    fake_client.wait.assert_called_once()


def test_models_enable_does_not_run_catalog_fallback_when_admin_succeeds(
    monkeypatch, tmp_path,
):
    """End-to-end: `models_enable` must not touch the catalog when the
    admin API already accepted the enable, even if the response has no
    job_id.
    """
    from muse.cli import models_enable

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_client = MagicMock()
    fake_client.enable.return_value = {"status": "accepted"}

    with patch("muse.admin.client.AdminClient", return_value=fake_client), \
         patch("muse.core.catalog.set_enabled") as mock_set_enabled:
        models_enable("some-model")

    mock_set_enabled.assert_not_called()
