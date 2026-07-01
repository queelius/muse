"""OpenAI-style error envelopes.

Matches the structure of OpenAI's error responses so clients written
against their API can reuse error-handling code.
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse


def error_response(status: int, code: str, message: str) -> JSONResponse:
    # OpenAI tags 5xx envelopes with type "server_error" and 4xx with
    # "invalid_request_error"; SDKs that branch on error.type expect the
    # server-side class for a 500/503, not a client-error label.
    error_type = "server_error" if status >= 500 else "invalid_request_error"
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "type": error_type}},
    )


class ModelNotFoundError(HTTPException):
    def __init__(self, model_id: str, modality: str):
        super().__init__(
            status_code=404,
            detail={"error": {
                "code": "model_not_found",
                "message": f"Model {model_id!r} is not available for modality {modality!r}",
                "type": "invalid_request_error",
            }},
        )
