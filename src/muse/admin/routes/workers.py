"""Worker introspection and control routes.

Endpoints (under /v1/admin):

  GET  /workers                  list all workers + their state
  POST /workers/{port}/restart   SIGTERM by port; monitor handles bringup.
                                 Also revives a dead or process-less
                                 worker through
                                 `supervisor.revive_worker` (status back
                                 to unhealthy, restart budget reset) so
                                 the monitor respawns it; a spec that
                                 helper refuses is 409
                                 worker_not_revivable.

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
    revive_worker,
)
from muse.core.errors import error_response


def _restart_response(
    port: int, *, signalled: bool, revived: bool = False, retired: bool = False,
) -> dict:
    """The restart reply. ``revived`` is true for a spec handed back to the
    monitor (always, when nothing was signalled). ``retired`` marks a
    signalled spec that ``revive_worker`` refused (retained dead): nothing
    will respawn it."""
    message = (
        "worker was signalled but was retired by a disable, eviction or "
        "failed load, so it will not be restarted; re-enable or warm up "
        "the model instead"
        if retired else "auto-restart monitor will respawn the worker"
    )
    return {
        "port": port,
        "signal": "SIGTERM" if signalled else None,
        "revived": revived or not signalled,
        "message": message,
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
        # state.lock is an RLock, and revive_worker takes it (then
        # spec.process_lock) itself: the same lock order as here.
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
                if signalled:
                    # The monitor sees the exit and respawns. Reset the
                    # budget only after the exact live process generation
                    # was signalled: the operator escape hatch for a worker
                    # deep into its automatic restart budget.
                    spec.restart_count = 0
                    spec.failure_count = 0
                    was_dead = spec.status == "dead"
                exited_not_dead = (
                    not signalled and proc is not None and spec.status != "dead"
                )
            if signalled:
                if was_dead:
                    # The monitor skips dead specs; revive one it gave up
                    # on. A False here (a retained-dead spec) is not an
                    # error: the signal was still sent, but nothing will
                    # respawn it, and the reply says so.
                    revived = revive_worker(state, spec)
                    return _restart_response(
                        port, signalled=True, revived=revived,
                        retired=not revived,
                    )
                return _restart_response(port, signalled=True)
            if spec.job_id is not None:
                # No live process and an admin/director transition owns
                # the spec: leave it.
                return error_response(
                    409, "worker_busy",
                    f"worker on port {port} is mid-transition; "
                    "retry when it settles",
                )
            if exited_not_dead:
                # The generation already exited and the monitor is still
                # tracking the spec, so its next tick respawns it; just
                # re-arm the budget.
                with spec.process_lock:
                    spec.restart_count = 0
                    spec.failure_count = 0
                return _restart_response(port, signalled=False)
            if not revive_worker(state, spec):
                # A spec retained dead after a disable, an eviction or a
                # failed load (teardown unproven) was never handed back to
                # the monitor; reviving it would respawn a worker that
                # admission never sized and the idle sweeper never evicts.
                return error_response(
                    409, "worker_not_revivable",
                    f"worker on port {port} was retired by a disable, "
                    "eviction or failed load; re-enable or warm up the "
                    "model instead",
                )
        return _restart_response(port, signalled=False)

    return router
