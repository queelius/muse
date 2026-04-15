"""`muse search` implementation: thin wrapper over resolvers.search.

Queries a registered resolver (defaults to the only one if exactly one
is registered) for candidate models matching `query`, optionally
filtered by modality / size / sort. Prints a compact aligned table.

Lazy: the HF resolver is only imported when this command runs, so
`muse --help` does not pay the huggingface_hub import cost.
"""
from __future__ import annotations

import logging
import sys

from muse.core.resolvers import ResolverError, search


logger = logging.getLogger(__name__)


# Loggers that emit per-request lines during search and would interleave
# with the table output. Quieted to WARNING during search so the table
# stays readable. Per-request HTTP debug remains accessible via
# `muse --log-level DEBUG search ...` only if the user re-lowers them
# after this call (rare).
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.repocard_data",
    "huggingface_hub.file_download",
)


def _quiet_third_party_logs() -> None:
    """Raise per-request HTTP loggers to WARNING.

    Only raises levels (never lowers them), so users who explicitly
    asked for DEBUG via `muse --log-level DEBUG search ...` and a
    custom logging-config still get whatever they configured.
    """
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        # logging.NOTSET (0) means "inherit"; treat as silenceable.
        # Otherwise only raise, never lower.
        if lg.level == logging.NOTSET or lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)


def run_search(
    *,
    query: str,
    modality: str | None = None,
    limit: int = 20,
    sort: str = "downloads",
    max_size_gb: float | None = None,
    backend: str | None = None,
) -> int:
    """Query resolver(s) for candidate models; print an aligned table.

    Returns 0 on success (including no-results), 2 on resolver error.
    """
    _quiet_third_party_logs()
    try:
        results = list(search(
            query,
            backend=backend,
            modality=modality,
            limit=limit,
            sort=sort,
        ))
    except ResolverError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if max_size_gb is not None:
        results = [
            r for r in results
            if r.size_gb is None or r.size_gb <= max_size_gb
        ]

    if not results:
        print("no results")
        return 0

    for r in results:
        size = f"{r.size_gb:.1f} GB" if r.size_gb else "?"
        downloads = f"{r.downloads:,}" if r.downloads else "?"
        lic = r.license or ""
        desc = r.description or ""
        print(f"  {r.uri:55s}  {size:>9s}  dl={downloads:>12s}  {lic:15s}  {desc}")
    return 0
