"""OpenAI-style error envelopes.

Matches the structure of OpenAI's error responses so clients written
against their API can reuse error-handling code.
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def validation_error_response(exc: RequestValidationError) -> JSONResponse:
    """The 422 ``invalid_request`` envelope for a request that failed FastAPI
    validation (field path and message per error, joined with ``; ``)."""
    details = "; ".join(
        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
        for e in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": details or str(exc),
                "type": "invalid_request_error",
            }
        },
    )


def error_type_for_status(status: int) -> str:
    """Map an HTTP status to the OpenAI error `type`.

    OpenAI tags 5xx envelopes with "server_error" and 4xx with
    "invalid_request_error"; SDKs that branch on error.type expect the
    server-side class for a 500/503, not a client-error label. Single
    source of truth so every envelope builder (core error_response, the
    gateway's OperationError surface, the admin auth dependency) derives
    the type the same way instead of hardcoding it.
    """
    return "server_error" if status >= 500 else "invalid_request_error"


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "type": error_type_for_status(status),
            },
        },
    )


class ModelNotFoundError(HTTPException):
    def __init__(self, model_id: str, modality: str):
        super().__init__(
            status_code=404,
            detail={"error": {
                "code": "model_not_found",
                "message": f"Model {model_id!r} is not available for modality {modality!r}",
                "type": error_type_for_status(404),
            }},
        )
