"""Bearer-token verification for the observability dashboard.

Mirrors muse.admin.auth's closed-by-default policy: with no admin.token
configured (either the MUSE_ADMIN_TOKEN env var or admin.token in
config.yaml), every dashboard/telemetry request is rejected with 503.
With a token configured, the request must carry it as an
Authorization: Bearer <token> header. HEADER-ONLY: there is no
?access_token= query-param fallback here (that path put the admin
token in URLs, which land in access logs / proxy logs / browser
history, and has been removed). The one exception is the SSE log
stream (`GET /v1/telemetry/logs/{model_id}`), which cannot use this
dependency at all -- `EventSource` clients cannot set custom headers --
so that route authenticates inline with a short-lived ticket (see
muse.observability.log_tickets) OR this same Authorization header for
programmatic (curl-style) clients.

The token is never echoed in error messages or logs.
"""
from __future__ import annotations

from fastapi import Header

from muse.admin.auth import http_bearer_check


def check_dashboard_token(bearer: str | None) -> None:
    """Raise unless the caller presents a valid Authorization header.

    ``bearer`` is the raw Authorization header value (None, "Bearer <tok>",
    or malformed). The check is muse.admin.auth's shared `check_bearer`
    (case-insensitive scheme, exact credential, constant-time compare):
      - admin.token unset / whitespace-only  -> 503 dashboard_closed
      - no Bearer header supplied             -> 401 missing_token
      - credential mismatches                 -> 403 invalid_token
      - credential matches                    -> return None
    """
    http_bearer_check(
        bearer,
        closed_code="dashboard_closed",
        closed_message=(
            "Dashboard requires an admin token "
            "(MUSE_ADMIN_TOKEN env or admin.token in config)"
        ),
        invalid_message="Bad dashboard token",
    )


def require_dashboard_auth(
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency wrapping check_dashboard_token for route use."""
    check_dashboard_token(authorization)
