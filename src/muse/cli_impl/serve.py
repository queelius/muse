"""`muse serve` — user-facing entry point.

Delegates to the supervisor, which spawns per-venv worker subprocesses
and runs the gateway. The old in-process behavior is gone; any model
that needs to be loaded into a single process now uses `muse _worker`
directly (intended for supervisor use, also fine for debugging).
"""
from __future__ import annotations

from muse.cli_impl.supervisor import run_supervisor


def run_serve(*, host: str, port: int, device: str, **_: object) -> int:
    """Thin wrapper delegating to the supervisor.

    Kept as a separate function (instead of pointing `muse.cli._cmd_serve`
    directly at `run_supervisor`) so future ergonomic flags (gateway
    auth, TLS, etc.) have a natural home.

    **_ absorbs deprecated kwargs (modalities, models) that older callers
    might still pass; they're silently ignored.
    """
    return run_supervisor(host=host, port=port, device=device)
