import pytest
from fastapi import HTTPException

from muse.core import config
from muse.observability.dashboard_auth import check_dashboard_token


def test_no_token_configured_returns_503_dashboard_closed(monkeypatch):
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token(None, None)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"]["code"] == "dashboard_closed"


def test_token_set_neither_supplied_returns_401_missing_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token(None, None)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "missing_token"


def test_token_set_wrong_bearer_returns_403_invalid_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token("Bearer wrongtoken", None)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"]["code"] == "invalid_token"


def test_token_set_wrong_access_token_returns_403_invalid_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token(None, "wrongtoken")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"]["code"] == "invalid_token"


def test_token_set_correct_via_bearer_returns_none(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    assert check_dashboard_token("Bearer secret123", None) is None


def test_token_set_correct_via_access_token_returns_none(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    assert check_dashboard_token(None, "secret123") is None


def test_malformed_bearer_falls_through_to_correct_access_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    # "secret123" without the "Bearer " prefix is malformed as a bearer value;
    # the correct access_token should still be accepted via fallthrough.
    assert check_dashboard_token("secret123", "secret123") is None
