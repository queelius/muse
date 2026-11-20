"""Bearer-token verification for admin endpoints.

The token is read via muse.core.config's admin.token setting, which
resolves the MUSE_ADMIN_TOKEN environment variable first and falls back
to admin.token in config.yaml. With no token configured (either source),
every admin request is rejected with 503; this is the closed-by-default
policy. With a token configured, the request must carry an
Authorization: Bearer <token> header matching it. The scheme is matched
case-insensitively (RFC 7235); the credential must match exactly.

`check_bearer` is the one implementation of that check. It is
fastapi-free (it raises `BearerAuthError`), so muse.observability's
dashboard auth and the MCP HTTP transport (whose light install has no
fastapi) can share it. `verify_admin_token` is the FastAPI dependency
built on it, and is the only name here that needs fastapi.

The token is never echoed in error messages or logs.
"""
from __future__ import annotations

import secrets

from muse.core import config

ADMIN_TOKEN_ENV = "MUSE_ADMIN_TOKEN"


class BearerAuthError(Exception):
    """A failed bearer check: an HTTP status, an error code and a message.

    The message is always a static string; it never contains a token.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def bearer_credential(authorization: str | None) -> str | None:
    """The credential of an `Authorization: Bearer <credential>` header.

    The scheme is case-insensitive; the credential is returned exactly as
    sent (not stripped). None when the header is missing or not Bearer.
    """
    if not authorization:
        return None
    scheme, sep, credential = authorization.partition(" ")
    if not sep or scheme.lower() != "bearer":
        return None
    return credential


def check_bearer(
    authorization: str | None,
    *,
    expected: str | None,
    closed_code: str,
    closed_message: str,
    invalid_message: str,
) -> None:
    """Raise `BearerAuthError` unless `authorization` carries `expected`.

    - `expected` unset / whitespace-only -> 503 `closed_code`
    - header missing or not Bearer       -> 401 missing_token
    - credential wrong                   -> 403 invalid_token

    `expected` is stripped: a whitespace-only value counts as unset
    (closed-by-default), and the `MUSE_ADMIN_TOKEN=$(cat tokenfile)`
    trailing newline does not reject every request. The presented
    credential is NOT stripped: it must match the real secret exactly.
    """
    expected = (expected or "").strip()
    if not expected:
        raise BearerAuthError(503, closed_code, closed_message)
    presented = bearer_credential(authorization)
    if presented is None:
        raise BearerAuthError(
            401, "missing_token", "Authorization: Bearer <token> required",
        )
    # Constant-time compare prevents recovering the token byte-by-byte via
    # response-time variance. Both operands are UTF-8-encoded to bytes:
    # secrets.compare_digest raises TypeError on non-ASCII str args, so a
    # bearer carrying non-ASCII bytes (headers arrive latin-1-decoded) would
    # otherwise 500 instead of cleanly 403'ing as a bad token.
    if not secrets.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8"),
    ):
        raise BearerAuthError(403, "invalid_token", invalid_message)


def _err(status: int, code: str, message: str):
    """Build an OpenAI-shape envelope inside a FastAPI HTTPException.

    The message text never includes the secret token; only static
    descriptive strings flow through here. The error `type` is derived
    from the status (server_error for the 503 admin_disabled, else
    invalid_request_error) via the shared core.errors helper, so this path
    can't drift from core.errors.error_response.
    """
    from fastapi import HTTPException

    from muse.core.errors import error_type_for_status

    return HTTPException(
        status_code=status,
        detail={"error": {
            "code": code,
            "message": message,
            "type": error_type_for_status(status),
        }},
    )


def http_bearer_check(authorization: str | None, **kwargs) -> None:
    """`check_bearer` against admin.token, raising the HTTPException form."""
    try:
        check_bearer(authorization, expected=config.get("admin.token"), **kwargs)
    except BearerAuthError as exc:
        raise _err(exc.status, exc.code, exc.message) from None


try:
    from fastapi import Header
except ImportError:  # light MCP install: check_bearer works without fastapi
    Header = None

if Header is not None:
    def verify_admin_token(
        authorization: str | None = Header(default=None),
    ) -> None:
        """FastAPI dependency: raise unless caller carries the admin bearer.

        503 admin_disabled with no token configured, 401 missing_token for
        a missing or non-Bearer header, 403 invalid_token for a wrong one.
        """
        http_bearer_check(
            authorization,
            closed_code="admin_disabled",
            closed_message=(
                "Admin endpoints require an admin token "
                "(MUSE_ADMIN_TOKEN env or admin.token in config)"
            ),
            invalid_message="Bad admin token",
        )
