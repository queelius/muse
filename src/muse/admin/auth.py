"""Bearer-token verification for admin endpoints.

The token is read from the MUSE_ADMIN_TOKEN environment variable. With
no token configured, every admin request is rejected with 503; this is
the closed-by-default policy. With a token configured, the request must
carry an Authorization: Bearer <token> header matching the env var.

The token is never echoed in error messages or logs.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException

ADMIN_TOKEN_ENV = "MUSE_ADMIN_TOKEN"


def _err(status: int, code: str, message: str) -> HTTPException:
    """Build an OpenAI-shape envelope inside an HTTPException.

    The message text never includes the secret token; only static
    descriptive strings flow through here.
    """
    return HTTPException(
        status_code=status,
        detail={"error": {
            "code": code,
            "message": message,
            "type": "invalid_request_error",
        }},
    )


def verify_admin_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: raise unless caller carries the admin bearer.

    Five paths:
      - env var unset                    -> 503 admin_disabled
      - header missing                   -> 401 missing_token
      - header malformed (no "Bearer ")  -> 401 missing_token
      - header bearer wrong              -> 403 invalid_token
      - header bearer matches            -> return None (route runs)
    """
    expected = os.environ.get(ADMIN_TOKEN_ENV)
    if not expected:
        raise _err(
            503,
            "admin_disabled",
            f"Admin endpoints require the {ADMIN_TOKEN_ENV} env var to be set",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise _err(
            401,
            "missing_token",
            "Authorization: Bearer <token> required",
        )
    presented = authorization[len("Bearer "):]
    if presented != expected:
        raise _err(403, "invalid_token", "Bad admin token")
