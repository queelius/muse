"""Shared read of the server's public runtime state for the CLI.

`muse models list` (status glyph) and `muse models info` (header + worker
status) both need to know which models the running server currently has
loaded. That state is exposed by the PUBLIC `GET /v1/models` endpoint --
each entry carries `loaded: bool` since v0.47.3 -- with no admin token
required. This module centralizes the fetch + parse so both call sites
share one implementation and one failure policy.
"""
from __future__ import annotations

import logging

from muse.core import config

# Per-request HTTP loggers emit an INFO line ("HTTP Request: GET ...")
# that would interleave with `muse models list` / `info` output. Raise
# them to WARNING before the call so the CLI surface stays clean.
_NOISY_LOGGERS = ("httpx", "httpcore")


def quiet_noisy_loggers(names: tuple[str, ...] = _NOISY_LOGGERS) -> None:
    """Raise per-request third-party loggers to WARNING for CLI output.

    Skipped entirely when the root logger is at DEBUG (`muse --log-level
    DEBUG ...`), so an operator who asks for DEBUG sees the httpx lines.
    Otherwise it only raises levels, never lowers them: a logger already
    set to ERROR stays at ERROR.
    """
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        return
    for name in names:
        lg = logging.getLogger(name)
        # NOTSET (0) means "inherit"; treat as silenceable.
        if lg.level == logging.NOTSET or lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)


def _base_url() -> str:
    return config.get("client.server_url").rstrip("/")


def fetch_public_models(timeout: float = 2.0) -> list[dict] | None:
    """Return the `/v1/models` `data` list, or None if it can't be read.

    None means "could not read runtime state" (server down, httpx not
    installed, non-200, malformed body). An empty list means "reachable,
    no models." Callers distinguish the two: list treats None like empty
    (every row enabled_unloaded); info uses None to show the offline
    "supervisor unreachable" view.
    """
    base = _base_url()
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    quiet_noisy_loggers()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base}/v1/models")
        if r.status_code != 200:
            return None
        body = r.json()
    except Exception:  # noqa: BLE001
        return None
    data = body.get("data") if isinstance(body, dict) else None
    return data if isinstance(data, list) else None


def loaded_ids(data: list[dict]) -> set[str]:
    """Ids of `/v1/models` entries whose `loaded` is True."""
    out: set[str] = set()
    for m in data:
        if not isinstance(m, dict) or m.get("loaded") is not True:
            continue
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            out.add(mid)
    return out
