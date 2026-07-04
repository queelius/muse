"""Short-lived, reusable tickets that gate the SSE log-tail endpoint.

Why a ticket at all: the admin token is header-only everywhere else in
this package (see dashboard_auth.py), but `EventSource` (the browser API
behind Server-Sent Events) cannot set custom request headers, so it has
no way to present `Authorization: Bearer <token>`. Rather than fall back
to putting the long-lived admin token in the URL (which lands in access
logs, proxy logs, and browser history), the dashboard exchanges the real
token for a short-lived, random ticket via a header-gated mint endpoint
(`POST /v1/telemetry/logs-ticket`), then opens the `EventSource` with
that ticket in the query string instead. A leaked ticket in a log line
is useless once its TTL elapses.

The ticket is REUSABLE within its TTL (not single-use): `EventSource`
auto-reconnects on transient network blips, and a single-use ticket
would force a re-mint on every reconnect, which is unnecessary friction
for a credential that already expires quickly on its own. It is also
unscoped -- it authorizes the logs SSE surface generally, not one
specific model_id -- because the admin token itself is all-or-nothing,
so a per-model ticket would not narrow the actual privilege boundary.

Stdlib only (`secrets`, `time`, `threading`): this module must stay
import-light so it can be constructed without pulling in fastapi.
"""
from __future__ import annotations

import secrets
import threading
import time


class LogTicketStore:
    """In-memory store of ticket -> expiry timestamp.

    Uses `time.monotonic()` internally (not wall-clock), so it is
    immune to system clock adjustments. Thread-safe via a single lock;
    lazily prunes expired entries on every `validate()` call rather
    than running a background sweep thread.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._tickets: dict[str, float] = {}
        self._lock = threading.Lock()

    def mint(self) -> tuple[str, int]:
        """Create a new ticket. Returns (ticket, expires_in_seconds)."""
        ticket = secrets.token_urlsafe(32)
        expiry = time.monotonic() + self._ttl_seconds
        with self._lock:
            self._tickets[ticket] = expiry
        return ticket, int(self._ttl_seconds)

    def validate(self, ticket: str | None) -> bool:
        """True iff `ticket` exists and has not expired.

        Lazily prunes: an expired ticket is dropped from the store as a
        side effect of being checked, whether or not it was the one
        found. A non-existent, empty, or None ticket returns False
        without raising.
        """
        if not ticket:
            return False
        now = time.monotonic()
        with self._lock:
            expiry = self._tickets.get(ticket)
            if expiry is None:
                return False
            if expiry <= now:
                del self._tickets[ticket]
                return False
            return True
