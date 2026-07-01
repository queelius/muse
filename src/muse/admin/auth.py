"""Bearer-token verification for admin endpoints.

The token is read from the MUSE_ADMIN_TOKEN environment variable. With
no token configured, every admin request is rejected with 503; this is
the closed-by-default policy. With a token configured, the request must
carry an Authorization: Bearer <token> header matching the env var.

The token is never echoed in error messages or logs.
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException

from muse.core.errors import error_type_for_status

ADMIN_TOKEN_ENV = "MUSE_ADMIN_TOKEN"


def _err(status: int, code: str, message: str) -> HTTPException:
    """Build an OpenAI-shape envelope inside an HTTPException.

    The message text never includes the secret token; only static
    descriptive strings flow through here. The error `type` is derived
    from the status (server_error for the 503 admin_disabled, else
    invalid_request_error) via the shared core.errors helper, so this path
    can't drift from core.errors.error_response.
    """
    return HTTPException(
        status_code=status,
        detail={"error": {
            "code": code,
            "message": message,
            "type": error_type_for_status(status),
        }},
    )


def verify_admin_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: raise unless caller carries the admin bearer.

    Five paths:
      - env var unset / whitespace-only  -> 503 admin_disabled
      - header missing                   -> 401 missing_token
      - header malformed (no "Bearer ")  -> 401 missing_token
      - header bearer wrong              -> 403 invalid_token
      - header bearer matches            -> return None (route runs)
    """
    expected = os.environ.get(ADMIN_TOKEN_ENV)
    # Strip the operator-supplied token: a whitespace-only value is treated
    # as "unset" (closed-by-default), and this defends against the common
    # `MUSE_ADMIN_TOKEN=$(cat tokenfile)` footgun, where a trailing newline
    # would otherwise 403 every legitimate `Bearer <token>` request. The
    # presented token is NOT stripped: it must match the real secret exactly.
    if expected:
        expected = expected.strip()
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
    # Constant-time compare prevents recovering the token byte-by-byte
    # via response-time variance. Both operands are UTF-8-encoded to bytes:
    # secrets.compare_digest raises TypeError on non-ASCII str args, so a
    # bearer carrying non-ASCII bytes (headers arrive latin-1-decoded) would
    # otherwise 500 instead of cleanly 403'ing as a bad token.
    if not secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise _err(403, "invalid_token", "Bad admin token")
