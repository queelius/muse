"""`muse serve` — user-facing entry point.

Delegates to the supervisor, which spawns per-venv worker subprocesses
and runs the gateway. The old in-process behavior is gone; any model
that needs to be loaded into a single process now uses `muse _worker`
directly (intended for supervisor use, also fine for debugging).
"""
from __future__ import annotations

from muse.cli_impl.supervisor import run_supervisor


def run_serve(*, host: str, port: int, device: str) -> int:
    """Run the supervisor; the home for future serve-only flags (auth, TLS)."""
    return run_supervisor(host=host, port=port, device=device)
