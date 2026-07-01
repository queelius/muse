"""Tests for verify_admin_token: the five auth paths.

The token is read from MUSE_ADMIN_TOKEN; without it, every request is
rejected with 503 (closed-by-default). With it, the bearer must match.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.errors import install_admin_error_handler


@pytest.fixture
def app():
    """A trivial FastAPI app whose only route requires the admin token.

    The OpenAI-envelope-unwrapping handler is installed exactly as the
    gateway installs it, so these tests assert the real wire shape
    (bare {"error": {...}}), not the dependency's raw HTTPException.
    """
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(verify_admin_token)])
    def protected():
        return {"ok": True}

    install_admin_error_handler(app)
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
        assert body["error"]["code"] == "admin_disabled"

    def test_no_token_configured_even_with_bearer_returns_503(self, client, monkeypatch):
        """A request CAN'T succeed if the env var is unset, even with a header."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        r = client.get("/protected", headers={"Authorization": "Bearer anything"})
        assert r.status_code == 503

    def test_token_set_no_header_returns_401(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "missing_token"

    def test_token_set_malformed_header_returns_401(self, client, monkeypatch):
        """An "Authorization: Token X" or other non-Bearer prefix -> 401."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Token secret"})
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "missing_token"

    def test_token_set_wrong_bearer_returns_403(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "invalid_token"

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
        assert body["error"]["type"] == "invalid_request_error"

    def test_whitespace_only_token_treated_as_disabled(self, client, monkeypatch):
        """A whitespace-only MUSE_ADMIN_TOKEN is an accident, not a secret:
        treat it as unset (503 closed-by-default), even with a Bearer header."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "   ")
        r = client.get("/protected", headers={"Authorization": "Bearer    "})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "admin_disabled"

    def test_token_with_trailing_newline_still_matches(self, client, monkeypatch):
        """`MUSE_ADMIN_TOKEN=$(cat tokenfile)` leaves a trailing newline; the
        operator's `Bearer secret` must still authenticate (the env token is
        stripped before comparison)."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret\n")
        r = client.get("/protected", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_non_ascii_bearer_returns_403_not_500(self, monkeypatch):
        """A non-ASCII bearer must be rejected (403), not crash the server.

        HTTP headers arrive latin-1-decoded, so a raw non-ASCII byte in the
        Authorization header reaches the dependency as a non-ASCII str.
        `secrets.compare_digest` raises TypeError on non-ASCII str args; that
        would surface as a 500. We call the dependency directly because the
        httpx test client refuses to send non-ASCII header bytes (the real
        ASGI layer does not).
        """
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        with pytest.raises(HTTPException) as exc:
            verify_admin_token(authorization="Bearer café")
        assert exc.value.status_code == 403
        assert exc.value.detail["error"]["code"] == "invalid_token"

    def test_non_ascii_env_token_does_not_crash(self, monkeypatch):
        """A non-ASCII configured token must also not crash the compare."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "sécret")
        with pytest.raises(HTTPException) as exc:
            verify_admin_token(authorization="Bearer wrong")
        assert exc.value.status_code == 403

    def test_token_uses_constant_time_compare(self, client, monkeypatch):
        """Token comparison MUST go through secrets.compare_digest, not `!=`.

        A naive `!=` short-circuits on first byte mismatch, leaking the
        prefix one byte at a time via response-time variance. Even though
        the timing window is small for an in-process test, the regression
        guard here is structural: assert the constant-time API is the one
        being called.
        """
        from muse.admin import auth as auth_mod
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret-correct")
        calls: list[tuple[str, str]] = []
        original = auth_mod.secrets.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return original(a, b)

        monkeypatch.setattr(auth_mod.secrets, "compare_digest", spy)
        r = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 403
        assert calls, "secrets.compare_digest was not called"
        # The two arguments should be presented bearer + expected env value,
        # UTF-8-encoded to bytes so non-ASCII tokens compare without raising.
        assert calls[0] == (b"wrong", b"secret-correct")
