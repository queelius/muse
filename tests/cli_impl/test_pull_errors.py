"""Tests for muse.cli_impl.pull_errors.friendly_pull_error.

`muse pull` downloads weights via huggingface_hub. A gated/private repo with
no auth raises GatedRepoError / RepositoryNotFoundError / HfHubHTTPError, which
-- uncaught -- dumped a ~200-line traceback for what is really a one-line "log
in and accept access" condition. friendly_pull_error classifies those into a
short actionable message, and returns None for anything it does not recognize
(so genuine bugs still surface with a full traceback).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from muse.cli_impl.pull_errors import friendly_pull_error


def _make(cls, message, status=401):
    """Construct an HF error across constructor-signature variants."""
    try:
        return cls(message, response=MagicMock(status_code=status))
    except TypeError:
        exc = cls(message)
        exc.response = MagicMock(status_code=status)
        return exc


class TestGatedRepo:
    def test_gated_error_returns_actionable_message(self):
        from huggingface_hub.errors import GatedRepoError

        exc = _make(GatedRepoError, "Access to model X is restricted", status=401)
        exc.repo_id = "black-forest-labs/FLUX.1-schnell"

        msg = friendly_pull_error("flux-schnell", exc)

        assert msg is not None
        # names the alias the user typed
        assert "flux-schnell" in msg
        # points at the exact model page to accept access
        assert "https://huggingface.co/black-forest-labs/FLUX.1-schnell" in msg
        # tells them how to authenticate
        assert "hf auth login" in msg or "HF_TOKEN" in msg
        # no raw traceback noise
        assert "Traceback" not in msg

    def test_gated_error_without_repo_id_falls_back_to_identifier(self):
        from huggingface_hub.errors import GatedRepoError

        exc = _make(GatedRepoError, "restricted", status=401)
        msg = friendly_pull_error("some-alias", exc)
        assert msg is not None
        assert "some-alias" in msg


class TestOtherAccessErrors:
    def test_repository_not_found_returns_message(self):
        from huggingface_hub.errors import RepositoryNotFoundError

        exc = _make(RepositoryNotFoundError, "not found", status=404)
        msg = friendly_pull_error("ghost-model", exc)
        assert msg is not None
        assert "ghost-model" in msg
        assert "private" in msg.lower() or "not found" in msg.lower()

    def test_generic_http_401_returns_message(self):
        from huggingface_hub.errors import HfHubHTTPError

        exc = _make(HfHubHTTPError, "unauthorized", status=401)
        msg = friendly_pull_error("locked", exc)
        assert msg is not None
        assert "401" in msg
        assert "hf auth login" in msg or "HF_TOKEN" in msg

    def test_http_500_is_not_treated_as_access_error(self):
        from huggingface_hub.errors import HfHubHTTPError

        exc = _make(HfHubHTTPError, "server error", status=500)
        # A 5xx is a Hub outage, not an auth problem -> let it surface raw.
        assert friendly_pull_error("x", exc) is None


class TestUnrecognized:
    def test_plain_exception_returns_none(self):
        # Not an HF error at all -> return None so the real traceback shows.
        assert friendly_pull_error("x", ValueError("boom")) is None

    def test_keyerror_returns_none(self):
        assert friendly_pull_error("x", KeyError("unknown id")) is None


class TestPullCommandWiring:
    """The `muse pull` command turns a gated-repo error into a clean
    non-zero exit + message, not a ~200-line traceback."""

    def test_gated_pull_exits_1_with_clean_message(self, monkeypatch):
        from typer.testing import CliRunner

        import muse.core.catalog as catalog
        from huggingface_hub.errors import GatedRepoError
        from muse.cli import app

        exc = _make(GatedRepoError, "Access to model X is restricted", status=401)
        exc.repo_id = "black-forest-labs/FLUX.1-schnell"

        def _boom(identifier):
            raise exc

        monkeypatch.setattr(catalog, "pull", _boom)

        result = CliRunner().invoke(app, ["pull", "flux-schnell"])

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "gated" in result.output.lower()
        assert "https://huggingface.co/black-forest-labs/FLUX.1-schnell" in result.output

    def test_unrecognized_pull_error_still_raises(self, monkeypatch):
        from typer.testing import CliRunner

        import muse.core.catalog as catalog
        from muse.cli import app

        def _boom(identifier):
            raise RuntimeError("some genuinely unexpected bug")

        monkeypatch.setattr(catalog, "pull", _boom)

        result = CliRunner().invoke(app, ["pull", "whatever"])

        # Not swallowed: non-zero exit, and the real exception surfaced.
        assert result.exit_code != 0
        assert result.exception is not None
        assert isinstance(result.exception, RuntimeError)
