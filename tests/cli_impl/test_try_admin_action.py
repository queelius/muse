"""Tests for `muse.cli._try_admin_action`'s enable/disable dispatch.

`_try_admin_action` is the admin-API-first path `models_enable` /
`models_disable` consult before falling back to a catalog-only mutation.
It returns a three-way `AdminActionOutcome` (SUCCESS / FAILED /
NOT_HANDLED), with TIMEOUT as a distinguished FAILED sub-case (the async
job may still complete, so it is genuinely ambiguous rather than a hard
failure).

Regression (pre-fix): `_try_admin_action` returned a plain bool, and
`True` was overloaded to mean both "admin API succeeded" AND "admin API
was consulted but the operation actually failed" (job finished
non-done, job wait timed out, non-503 AdminClientError). Callers did
`if _try_admin_action(...): return`, so a failed admin operation still
exited 0 -- the same logical failure in catalog-only mode exits 2.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer


def _with_admin_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "test-token")


def test_enable_missing_job_id_is_still_success(monkeypatch):
    """A successful admin `enable()` response without a job_id must
    still be treated as SUCCESS, not fall through to the catalog
    fallback.
    """
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"status": "accepted"}  # no job_id key

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.SUCCESS
    fake_client.wait.assert_not_called()


def test_enable_falsy_job_id_is_still_success(monkeypatch):
    """An empty-string job_id is falsy but still a successful response
    shape; must not be treated as "admin API not used".
    """
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "", "status": "accepted"}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.SUCCESS


def test_enable_with_job_id_still_waits_and_reports(monkeypatch):
    """Sanity check: the normal (job_id present) success path is unaffected."""
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.return_value = {"state": "done", "result": {"worker_port": 9001}}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.SUCCESS
    fake_client.wait.assert_called_once()


def test_enable_job_finished_non_done_is_failed(monkeypatch):
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.return_value = {"state": "failed", "error": "boom"}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.FAILED


def test_enable_job_wait_timeout_is_timeout_outcome(monkeypatch):
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.side_effect = TimeoutError()

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.TIMEOUT


def test_disable_success(monkeypatch):
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.disable.return_value = {"worker_terminated": True, "worker_port": 9001}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("disable", "some-model")

    assert outcome is AdminActionOutcome.SUCCESS


def test_non_503_admin_client_error_is_failed(monkeypatch):
    from muse.admin.client import AdminClientError
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.side_effect = AdminClientError(
        status=500, code="internal_error", message="boom", body={},
    )

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.FAILED


def test_503_admin_disabled_is_not_handled(monkeypatch):
    from muse.admin.client import AdminClientError
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.side_effect = AdminClientError(
        status=503, code="admin_disabled", message="admin disabled", body={},
    )

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.NOT_HANDLED


def test_unreachable_admin_api_is_not_handled(monkeypatch):
    from muse.cli import AdminActionOutcome, _try_admin_action

    _with_admin_token(monkeypatch)
    fake_client = MagicMock()
    fake_client.enable.side_effect = ConnectionError("connection refused")

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        outcome = _try_admin_action("enable", "some-model")

    assert outcome is AdminActionOutcome.NOT_HANDLED


def test_no_admin_token_is_not_handled(monkeypatch):
    from muse.cli import AdminActionOutcome, _try_admin_action

    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    outcome = _try_admin_action("enable", "some-model")
    assert outcome is AdminActionOutcome.NOT_HANDLED


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


def test_models_enable_exits_2_when_admin_job_failed(monkeypatch, tmp_path):
    from muse.cli import models_enable

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.return_value = {"state": "failed", "error": "boom"}

    with patch("muse.admin.client.AdminClient", return_value=fake_client), \
         patch("muse.core.catalog.set_enabled") as mock_set_enabled:
        with pytest.raises(typer.Exit) as exc:
            models_enable("some-model")

    assert exc.value.exit_code == 2
    mock_set_enabled.assert_not_called()


def test_models_enable_exits_3_when_admin_job_wait_times_out(monkeypatch, tmp_path):
    from muse.cli import models_enable

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.side_effect = TimeoutError()

    with patch("muse.admin.client.AdminClient", return_value=fake_client), \
         patch("muse.core.catalog.set_enabled") as mock_set_enabled:
        with pytest.raises(typer.Exit) as exc:
            models_enable("some-model")

    assert exc.value.exit_code == 3
    mock_set_enabled.assert_not_called()


def test_models_disable_exits_2_on_non_503_admin_error(monkeypatch, tmp_path):
    from muse.admin.client import AdminClientError
    from muse.cli import models_disable

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_client = MagicMock()
    fake_client.disable.side_effect = AdminClientError(
        status=500, code="internal_error", message="boom", body={},
    )

    with patch("muse.admin.client.AdminClient", return_value=fake_client), \
         patch("muse.core.catalog.set_enabled") as mock_set_enabled:
        with pytest.raises(typer.Exit) as exc:
            models_disable("some-model")

    assert exc.value.exit_code == 2
    mock_set_enabled.assert_not_called()


def test_models_disable_falls_through_to_catalog_when_admin_unreachable(
    monkeypatch, tmp_path,
):
    """A 503 admin_disabled / unreachable admin API must fall through to
    the catalog-only path unchanged.
    """
    from muse.admin.client import AdminClientError
    from muse.core.catalog import pull
    from muse.cli import models_disable

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    fake_client = MagicMock()
    fake_client.disable.side_effect = AdminClientError(
        status=503, code="admin_disabled", message="admin disabled", body={},
    )

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        models_disable("soprano-80m")

    from muse.core.catalog import _read_catalog
    assert _read_catalog()["soprano-80m"]["enabled"] is False


def test_main_returns_2_for_admin_job_failed(monkeypatch, tmp_path):
    from muse.cli import main

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.return_value = {"state": "failed", "error": "boom"}

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        rc = main(["models", "enable", "some-model"])

    assert rc == 2


def test_main_returns_3_for_admin_job_timeout(monkeypatch, tmp_path):
    from muse.cli import main

    _with_admin_token(monkeypatch)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_client = MagicMock()
    fake_client.enable.return_value = {"job_id": "abc123", "status": "pending"}
    fake_client.wait.side_effect = TimeoutError()

    with patch("muse.admin.client.AdminClient", return_value=fake_client):
        rc = main(["models", "enable", "some-model"])

    assert rc == 3
