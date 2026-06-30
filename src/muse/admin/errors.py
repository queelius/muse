"""Exception handler that unwraps OpenAI-shaped HTTPException details.

The admin auth dependency (`muse.admin.auth.verify_admin_token`) signals
failures by raising `HTTPException(detail={"error": {...}})`. FastAPI's
default HTTPException handler serializes that as
`{"detail": {"error": {...}}}`, which double-wraps the envelope and
diverges from the route-level admin errors (which return a bare
`{"error": {...}}` via JSONResponse).

`install_admin_error_handler` registers an app-level handler that emits
the bare `{"error": {...}}` envelope when an HTTPException's detail is
already OpenAI-shaped, and defers to FastAPI's default handler for every
other HTTPException so plain-string details elsewhere keep their
`{"detail": ...}` shape. The handler is safe-degrading: an includer that
forgets to register it still gets a valid (if double-wrapped) error
response, never a 500.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def install_admin_error_handler(app: FastAPI) -> None:
    """Register the OpenAI-envelope-unwrapping HTTPException handler."""

    async def _handler(request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=detail,
                headers=getattr(exc, "headers", None),
            )
        # Non-envelope HTTPException: preserve FastAPI's default shape.
        return await http_exception_handler(request, exc)

    app.add_exception_handler(StarletteHTTPException, _handler)
