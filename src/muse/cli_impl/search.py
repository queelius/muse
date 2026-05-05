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

    if sys.stdout.isatty():
        _render_rich_search(results)
    else:
        _render_plain_search(results)
    return 0


def _render_rich_search(results: list) -> None:
    """Pretty interactive table for search results."""
    from rich import box
    from rich.table import Table

    from muse.cli_impl.console import get_console

    console = get_console()
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        expand=True,
    )
    table.add_column("uri", no_wrap=True, style="cyan")
    table.add_column("size", justify="right", no_wrap=True)
    table.add_column("downloads", justify="right", no_wrap=True)
    table.add_column("license", no_wrap=True)
    table.add_column("description", overflow="ellipsis", no_wrap=True, ratio=1)
    for r in results:
        size = f"{r.size_gb:.1f} GB" if r.size_gb else "?"
        downloads = f"{r.downloads:,}" if r.downloads else "?"
        table.add_row(
            r.uri, size, downloads, r.license or "", r.description or "",
        )
    console.print(table)


def _render_plain_search(results: list) -> None:
    """Plain aligned text for piped / non-TTY output."""
    uri_w = max((len(r.uri) for r in results), default=0)
    size_w = max(
        (len(f"{r.size_gb:.1f} GB" if r.size_gb else "?") for r in results),
        default=0,
    )
    dl_w = max(
        (len(f"{r.downloads:,}" if r.downloads else "?") for r in results),
        default=0,
    )
    lic_w = max((len(r.license or "") for r in results), default=0)
    for r in results:
        size = f"{r.size_gb:.1f} GB" if r.size_gb else "?"
        downloads = f"{r.downloads:,}" if r.downloads else "?"
        print(
            f"  {r.uri:<{uri_w}s}  "
            f"{size:>{size_w}s}  "
            f"dl={downloads:>{dl_w}s}  "
            f"{(r.license or ''):<{lic_w}s}  "
            f"{r.description or ''}"
        )
