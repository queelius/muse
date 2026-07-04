import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from muse.core import config
from muse.observability.dashboard_auth import check_dashboard_token, require_dashboard_auth


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


@pytest.fixture
def app():
    """A trivial FastAPI app whose only route requires the dashboard token.

    Unlike the admin auth reference suite (tests/admin/test_auth.py), no
    OpenAI-envelope-unwrapping handler is installed here: dashboard_auth has
    no equivalent to muse.admin.errors.install_admin_error_handler. So the
    real wire shape a bare route sees is FastAPI's default HTTPException
    handler, which wraps our {"error": {...}} detail under a top-level
    "detail" key: {"detail": {"error": {...}}}.
    """
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_dashboard_auth)])
    def protected():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestRequireDashboardAuth:
    """Exercises require_dashboard_auth (the FastAPI dependency) through
    real HTTP, so the Header/Query parameter wiring routes will actually
    mount is under test -- not just check_dashboard_token's internal logic.
    """

    def test_no_token_configured_returns_503(self, client, monkeypatch):
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        config.reset_config()
        r = client.get("/protected")
        assert r.status_code == 503
        assert r.json()["detail"]["error"]["code"] == "dashboard_closed"

    def test_token_set_no_credential_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get("/protected")
        assert r.status_code == 401
        assert r.json()["detail"]["error"]["code"] == "missing_token"

    def test_token_set_correct_via_header_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get(
            "/protected", headers={"Authorization": "Bearer secret123"}
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_token_set_correct_via_query_param_returns_200(self, client, monkeypatch):
        """The SSE path: EventSource clients cannot set custom headers, so
        the query param must work standalone with NO Authorization header.
        This is the most important assertion in this module -- it proves
        the Query(...) wiring on require_dashboard_auth actually works
        through real HTTP, not just via a direct Python call."""
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get("/protected?access_token=secret123")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_token_set_wrong_via_header_returns_403(self, client, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get(
            "/protected", headers={"Authorization": "Bearer wrongtoken"}
        )
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "invalid_token"
