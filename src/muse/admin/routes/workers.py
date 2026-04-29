"""Worker introspection and control routes.

Endpoints (under /v1/admin):

  GET  /workers                  list all workers + their state
  POST /workers/{port}/restart   SIGTERM by port; monitor handles bringup

Worker fields surfaced: port, models, pid, uptime_seconds, restart_count,
status. PID is read from spec.process.pid; uptime from monotonic delta
since last_spawn_at. Both are `null` when the process hasn't been spawned
yet (rare; the restart path may briefly pass through a process=None state).
"""
from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from muse.cli_impl.supervisor import get_supervisor_state


def _err_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {
            "code": code,
            "message": message,
            "type": "invalid_request_error",
        }},
    )


def build_workers_router() -> APIRouter:
    router = APIRouter()

    @router.get("/workers")
    def list_workers():
        state = get_supervisor_state()
        with state.lock:
            now = time.monotonic()
            workers = []
            for spec in state.workers:
                pid = getattr(spec.process, "pid", None) if spec.process else None
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
                return _err_response(
                    404, "worker_not_found", f"no worker on port {port}",
                )
            proc = spec.process
            if proc is None:
                return _err_response(
                    409, "worker_not_running",
                    f"worker on port {port} is not currently running",
                )
            try:
                proc.terminate()
            except Exception as e:  # noqa: BLE001
                return _err_response(
                    500, "terminate_failed",
                    f"failed to SIGTERM worker on port {port}: {e}",
                )
        return {
            "port": port,
            "signal": "SIGTERM",
            "message": "auto-restart monitor will respawn the worker",
        }

    return router
