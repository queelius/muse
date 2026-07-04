"""Bearer-token / query-param verification for the observability dashboard.

Mirrors muse.admin.auth's closed-by-default policy: with no admin.token
configured (either the MUSE_ADMIN_TOKEN env var or admin.token in
config.yaml), every dashboard/telemetry request is rejected with 503.
With a token configured, the request must carry it either as an
Authorization: Bearer <token> header (regular HTTP clients) or as an
?access_token=<token> query parameter (SSE EventSource clients, which
cannot set custom headers).

The token is never echoed in error messages or logs.
"""
from __future__ import annotations

import secrets

from fastapi import Header, Query

from muse.admin.auth import _err
from muse.core import config


def check_dashboard_token(bearer: str | None, access_token: str | None) -> None:
    """Raise unless the caller presents a valid dashboard token.

    ``bearer`` is the raw Authorization header value (None, "Bearer <tok>",
    or malformed). ``access_token`` is the raw token from the ?access_token=
    query param (no "Bearer " prefix; what EventSource clients pass).

    Branches, most authoritative first:
      - admin.token unset / whitespace-only  -> 503 dashboard_closed
      - no candidate credential supplied      -> 401 missing_token
      - candidate mismatches expected         -> 403 invalid_token
      - candidate matches                     -> return None (caller proceeds)
    """
    expected = config.get("admin.token")
    # Strip the operator-supplied token: a whitespace-only value is treated
    # as "unset" (closed-by-default), and this defends against the common
    # `MUSE_ADMIN_TOKEN=$(cat tokenfile)` footgun, where a trailing newline
    # would otherwise reject every legitimate request. The presented
    # credential is NOT stripped: it must match the real secret exactly.
    if expected:
        expected = expected.strip()
    if not expected:
        raise _err(
            503,
            "dashboard_closed",
            "Dashboard requires an admin token "
            "(MUSE_ADMIN_TOKEN env or admin.token in config)",
        )

    if bearer and bearer.startswith("Bearer "):
        candidate = bearer[len("Bearer "):]
    elif access_token:
        candidate = access_token
    else:
        candidate = None

    if candidate is None:
        raise _err(
            401,
            "missing_token",
            "Authorization: Bearer <token> header or "
            "?access_token= query param required",
        )

    # Constant-time compare prevents recovering the token byte-by-byte via
    # response-time variance. Both operands are UTF-8-encoded to bytes:
    # secrets.compare_digest raises TypeError on non-ASCII str args, so a
    # candidate carrying non-ASCII bytes (headers arrive latin-1-decoded)
    # would otherwise 500 instead of cleanly 403'ing as a bad token.
    if not secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
        raise _err(403, "invalid_token", "Bad dashboard token")


def require_dashboard_auth(
    authorization: str | None = Header(default=None),
    access_token: str | None = Query(default=None),
) -> None:
    """FastAPI dependency wrapping check_dashboard_token for route use."""
    check_dashboard_token(authorization, access_token)
