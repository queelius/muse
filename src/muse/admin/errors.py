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

It also registers the handler that renders an `OperationError` raised by
an admin route as its bound (status, code, message) envelope, so routes
can let operation failures propagate instead of converting each one.
An includer that skips `install_admin_error_handler` turns those into
500s, so every app mounting the admin router must call it.

Finally it renders FastAPI's `RequestValidationError` (a non-integer
`{port}`, a non-object JSON body) on `/v1/admin` paths as a 400
`invalid_parameter` envelope instead of the bare `{"detail": [...]}` 422.
Other paths keep FastAPI's default handling.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def install_admin_error_handler(app: FastAPI) -> None:
    """Register the admin HTTPException and OperationError handlers."""

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

    from muse.admin.operations import OperationError
    from muse.core.errors import error_response

    async def _operation_error_handler(request, exc: OperationError):
        return error_response(exc.status, exc.code, exc.message)

    app.add_exception_handler(OperationError, _operation_error_handler)

    async def _validation_handler(request, exc: RequestValidationError):
        if not request.url.path.startswith(ADMIN_PATH_PREFIX):
            return await request_validation_exception_handler(request, exc)
        return error_response(400, "invalid_parameter", _validation_message(exc))

    app.add_exception_handler(RequestValidationError, _validation_handler)


ADMIN_PATH_PREFIX = "/v1/admin"


def _validation_message(exc) -> str:
    """One line naming the first invalid field (never echoes its value)."""
    errors = exc.errors() if hasattr(exc, "errors") else []
    if not errors:
        return "invalid request"
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    msg = first.get("msg", "invalid value")
    return f"invalid {loc or 'request body'}: {msg}"
