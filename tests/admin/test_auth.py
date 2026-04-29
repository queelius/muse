"""Tests for verify_admin_token: the five auth paths.

The token is read from MUSE_ADMIN_TOKEN; without it, every request is
rejected with 503 (closed-by-default). With it, the bearer must match.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token


@pytest.fixture
def app():
    """A trivial FastAPI app whose only route requires the admin token."""
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(verify_admin_token)])
    def protected():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestVerifyAdminToken:
    def test_no_token_configured_returns_503(self, client, monkeypatch):
        """Closed-by-default: missing env var -> 503 admin_disabled."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        r = client.get("/protected")
        assert r.status_code == 503
        body = r.json()
        assert body["detail"]["error"]["code"] == "admin_disabled"

    def test_no_token_configured_even_with_bearer_returns_503(self, client, monkeypatch):
        """A request CAN'T succeed if the env var is unset, even with a header."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        r = client.get("/protected", headers={"Authorization": "Bearer anything"})
        assert r.status_code == 503

    def test_token_set_no_header_returns_401(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected")
        assert r.status_code == 401
        assert r.json()["detail"]["error"]["code"] == "missing_token"

    def test_token_set_malformed_header_returns_401(self, client, monkeypatch):
        """An "Authorization: Token X" or other non-Bearer prefix -> 401."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Token secret"})
        assert r.status_code == 401
        assert r.json()["detail"]["error"]["code"] == "missing_token"

    def test_token_set_wrong_bearer_returns_403(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "invalid_token"

    def test_token_set_correct_bearer_passes(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_token_never_appears_in_403_response(self, client, monkeypatch):
        """The 403 envelope MUST NOT echo the configured secret."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "ultra-secret-12345")
        r = client.get("/protected", headers={"Authorization": "Bearer not-the-token"})
        assert r.status_code == 403
        assert "ultra-secret-12345" not in r.text

    def test_envelope_has_invalid_request_error_type(self, client, monkeypatch):
        """All admin errors land in the OpenAI error_type=invalid_request_error envelope."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected")
        assert r.status_code == 401
        body = r.json()
        assert body["detail"]["error"]["type"] == "invalid_request_error"
