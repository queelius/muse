"""Tests for install_admin_error_handler.

The handler is installed app-wide (it displaces FastAPI's default
StarletteHTTPException handler), so it MUST leave non-OpenAI-shaped
HTTPExceptions untouched: only details that are already a dict carrying
an "error" key get unwrapped to the bare envelope; everything else keeps
the default {"detail": ...} shape. These tests pin both branches.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from muse.admin.errors import install_admin_error_handler


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/envelope")
    def envelope():
        raise HTTPException(
            status_code=403,
            detail={"error": {"code": "nope", "message": "no", "type": "invalid_request_error"}},
        )

    @app.get("/plain")
    def plain():
        raise HTTPException(status_code=400, detail="bad request")

    @app.get("/dict-no-error")
    def dict_no_error():
        # A dict detail WITHOUT an "error" key must NOT be unwrapped.
        raise HTTPException(status_code=422, detail={"field": "missing"})

    install_admin_error_handler(app)
    return app


def test_envelope_detail_is_unwrapped_to_bare_error():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/envelope")
    assert r.status_code == 403
    body = r.json()
    assert body["error"]["code"] == "nope"
    assert "detail" not in body


def test_plain_string_detail_keeps_default_shape():
    """The delegation branch: a plain-string HTTPException keeps FastAPI's
    default {"detail": ...} shape. Because the handler is app-wide, a
    regression here would silently reshape EVERY non-admin error."""
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/plain")
    assert r.status_code == 400
    assert r.json() == {"detail": "bad request"}


def test_auto_404_keeps_default_shape():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/no-such-route")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}


def test_dict_detail_without_error_key_keeps_default_shape():
    """A dict detail that is NOT an OpenAI envelope (no 'error' key) is
    left wrapped in 'detail' -- only the envelope shape is unwrapped."""
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/dict-no-error")
    assert r.status_code == 422
    assert r.json() == {"detail": {"field": "missing"}}
