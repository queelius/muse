"""Worker introspection and control routes.

Endpoints (under /v1/admin):

  GET  /workers                  list all workers + their state
  POST /workers/{port}/restart   SIGTERM by port; monitor handles bringup.
                                 Also revives a dead or process-less
                                 worker (status back to unhealthy, restart
                                 budget reset) so the monitor respawns it.

Worker fields surfaced: port, models, pid, uptime_seconds, restart_count,
status. PID is read from spec.process.pid; uptime from monotonic delta
since last_spawn_at. Both are `null` when the process hasn't been spawned
yet (rare; the restart path may briefly pass through a process=None state).
"""
from __future__ import annotations

import signal
import time

from fastapi import APIRouter

from muse.cli_impl.supervisor import (
    _signal_worker_process,
    get_supervisor_state,
)
from muse.core.errors import error_response


def _revive_worker_spec(spec) -> None:
    """Hand a worker back to the auto-restart monitor with a fresh budget.

    The monitor skips `dead` specs, so a dead one goes back to `unhealthy`;
    a spec with no live process is respawned on the monitor's next tick
    (it treats a missing process as exited). Restart and failure counters
    reset so an operator restart re-arms the full automatic budget.

    Local stand-in: batch 4 adds a supervisor-owned revive helper to
    `muse.cli_impl.supervisor`. At merge, replace this body with a call to
    that helper (same semantics: under ``spec.process_lock``, status
    "unhealthy", ``restart_count`` and ``failure_count`` reset).
    """
    with spec.process_lock:
        if spec.status == "dead" or spec.process is None:
            spec.status = "unhealthy"
        spec.restart_count = 0
        spec.failure_count = 0


def _restart_response(port: int, *, signalled: bool) -> dict:
    return {
        "port": port,
        "signal": "SIGTERM" if signalled else None,
        "revived": not signalled,
        "message": "auto-restart monitor will respawn the worker",
        "restart_count_reset": True,
    }


def build_workers_router() -> APIRouter:
    router = APIRouter()

    @router.get("/workers")
    def list_workers():
        state = get_supervisor_state()
        with state.lock:
            now = time.monotonic()
            workers = []
            for spec in state.workers:
                with spec.process_lock:
                    process = spec.process
                    pid = getattr(process, "pid", None) if process else None
                uptime = (
                    max(0.0, now - spec.last_spawn_at)
                    if spec.last_spawn_at else None
                )
                workers.append({
                    "port": spec.port,
                    "models": list(spec.models),
                    "pid": pid if isinstance(pid, int) else None,
                    "uptime_seconds": uptime,
                    "restart_count": spec.restart_count,
                    "status": spec.status,
                })
        return {"workers": workers}

    @router.post("/workers/{port}/restart")
    def restart_worker(port: int):
        state = get_supervisor_state()
        with state.lock:
            spec = next((w for w in state.workers if w.port == port), None)
            if spec is None:
                return error_response(
                    404, "worker_not_found", f"no worker on port {port}",
                )
            with spec.process_lock:
                proc = spec.process
                signalled = False
                if proc is not None:
                    try:
                        signalled = _signal_worker_process(
                            spec, proc, signal.SIGTERM,
                        )
                    except Exception as e:  # noqa: BLE001
                        return error_response(
                            500, "terminate_failed",
                            f"failed to SIGTERM worker on port {port}: {e}",
                        )
                if not signalled and spec.job_id is not None:
                    # No live process (a dead worker has none) and an
                    # admin/director transition owns the spec: leave it.
                    return error_response(
                        409, "worker_busy",
                        f"worker on port {port} is mid-transition; "
                        "retry when it settles",
                    )
                # Signalled: the monitor sees the exit and respawns. Not
                # signalled: the generation already exited, so reviving lets
                # the monitor respawn it (a dead spec would be skipped).
                # Reset the budget only after the exact live process
                # generation was signalled, or found not live.
                _revive_worker_spec(spec)
        return _restart_response(port, signalled=signalled)

    return router
