"""Orchestrate admin operations against the supervisor.

Each operation reads/mutates SupervisorState (workers, device) under the
state's RLock. Async operations (enable, pull, probe) spawn a daemon
thread tracked by the JobStore; the thread updates the Job as it
progresses. Sync operations (disable, remove) return their result
directly and raise OperationError on user-facing failures.

Subprocess-based ops (pull, probe) shell out to `muse pull <id>` and
`muse models probe <id>` respectively. This keeps clean isolation: the
gateway never imports torch / diffusers / llama-cpp; per-model venvs do.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Any, Callable

from muse.admin.jobs import Job, JobStore
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    _shutdown_workers,
    spawn_worker,
    wait_for_ready,
)
from muse.core.catalog import (
    _read_catalog,
    is_pulled,
    known_models,
    remove as catalog_remove,
    set_enabled,
)
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


class OperationError(Exception):
    """Raised by sync operations on user-facing failures.

    Routes catch this and translate it into an HTTP envelope with the
    bound (status, code, message) without leaking internals. Async
    operations write the same fields into the Job's `error` instead.
    """

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def find_worker_for_model(state: SupervisorState, model_id: str) -> WorkerSpec | None:
    """Return the WorkerSpec hosting `model_id`, or None if no worker hosts it.

    Acquires the state's RLock for the duration of the iteration so
    concurrent enable/disable can't slip a model in or out mid-search.
    """
    with state.lock:
        for spec in state.workers:
            if model_id in spec.models:
                return spec
    return None


def enable_model(
    model_id: str,
    *,
    state: SupervisorState,
    store: JobStore,
    job: Job,
) -> None:
    """Async operation: ensure `model_id` is loaded in some worker.

    Updates `job` with state transitions. Three terminal paths:
      1. Already loaded -> done with spawned_new=False, loaded=True.
      2. Joins an existing venv-group worker -> restart-in-place.
      3. Spawns a brand-new worker for this model's python_path.
    """
    store.update(job.job_id, state="running")
    try:
        catalog_known = known_models()
        if model_id not in catalog_known:
            raise OperationError(
                "model_not_found", f"unknown model {model_id!r}", status=404,
            )
        if not is_pulled(model_id):
            raise OperationError(
                "model_not_pulled",
                f"model {model_id!r} not pulled; run pull first",
                status=409,
            )

        catalog = _read_catalog()
        python_path = catalog[model_id].get("python_path")
        if not python_path:
            raise OperationError(
                "missing_venv",
                f"model {model_id!r} has no per-model venv on record",
                status=409,
            )

        with state.lock:
            set_enabled(model_id, True)

            existing = find_worker_for_model(state, model_id)
            if existing is not None:
                store.update(
                    job.job_id, state="done",
                    result={
                        "model_id": model_id,
                        "worker_port": existing.port,
                        "loaded": True,
                        "spawned_new": False,
                    },
                )
                return

            # Look for a venv-group sibling we can join (restart-in-place
            # adds the new model to the existing worker's load list).
            target = next(
                (s for s in state.workers if s.python_path == python_path),
                None,
            )
            if target is not None:
                target.models = sorted(set(target.models) | {model_id})
                _restart_worker_inplace(target, device=state.device)
                store.update(
                    job.job_id, state="done",
                    result={
                        "model_id": model_id,
                        "worker_port": target.port,
                        "loaded": True,
                        "spawned_new": False,
                    },
                )
                return

            # Spawn a brand-new worker.
            new_port = find_free_port(start=9001, end=9999)
            spec = WorkerSpec(
                models=[model_id],
                python_path=python_path,
                port=new_port,
                device=state.device,
            )
            state.workers.append(spec)
            spawn_worker(spec, device=state.device)
            wait_for_ready(port=spec.port, timeout=120.0)
            spec.status = "running"
            store.update(
                job.job_id, state="done",
                result={
                    "model_id": model_id,
                    "worker_port": spec.port,
                    "loaded": True,
                    "spawned_new": True,
                },
            )
    except OperationError as e:
        store.update(job.job_id, state="failed", error=e.message)
    except Exception as e:  # noqa: BLE001
        logger.exception("enable_model failed")
        store.update(job.job_id, state="failed", error=str(e))


def disable_model(model_id: str, *, state: SupervisorState) -> dict:
    """Sync operation: catalog flip + worker unload.

    Three paths:
      1. model unknown -> OperationError(404).
      2. model not loaded in any worker -> catalog flip only.
      3. model is loaded -> drop it from the worker; if it was the only
         model in that worker, terminate the worker; else restart-in-place
         with the reduced load list.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )

    with state.lock:
        spec = find_worker_for_model(state, model_id)
        try:
            set_enabled(model_id, False)
        except KeyError:
            # Model is in known_models() but not in catalog.json (e.g.
            # bundled-but-not-pulled). Still report a coherent shape.
            pass

        if spec is None:
            return {
                "model_id": model_id,
                "loaded": False,
                "worker_terminated": False,
                "remaining_models_in_worker": [],
            }

        spec.models = [m for m in spec.models if m != model_id]
        if not spec.models:
            _shutdown_workers([spec])
            state.workers = [w for w in state.workers if w.port != spec.port]
            return {
                "model_id": model_id,
                "loaded": False,
                "worker_terminated": True,
                "worker_port": spec.port,
                "remaining_models_in_worker": [],
            }
        _restart_worker_inplace(spec, device=state.device)
        return {
            "model_id": model_id,
            "loaded": False,
            "worker_terminated": False,
            "worker_port": spec.port,
            "remaining_models_in_worker": list(spec.models),
        }


def remove_model(model_id: str, *, state: SupervisorState, purge: bool) -> dict:
    """Sync operation: drop the catalog entry. Refuses if currently loaded.

    Caller must `disable` first when the model is hosted by a worker;
    otherwise the running process holds open file descriptors against
    the venv we're about to delete.
    """
    catalog_known = known_models()
    if model_id not in catalog_known and not is_pulled(model_id):
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )
    if find_worker_for_model(state, model_id) is not None:
        raise OperationError(
            "model_loaded",
            f"model {model_id!r} is currently loaded; disable it first",
            status=409,
        )
    catalog_remove(model_id, purge=purge)
    return {"model_id": model_id, "removed": True, "purged": bool(purge)}


def probe_model(
    model_id: str,
    *,
    no_inference: bool,
    device: str | None,
    store: JobStore,
    job: Job,
) -> None:
    """Async wrapper around `muse models probe <id>`.

    Spawns a subprocess in the supervisor's interpreter (which dispatches
    into the model's per-model venv via the existing probe machinery).
    Captures stdout/stderr into job.log_lines.
    """
    store.update(job.job_id, state="running")
    cmd = [sys.executable, "-m", "muse.cli", "models", "probe", model_id]
    if no_inference:
        cmd.append("--no-inference")
    if device is not None:
        cmd.extend(["--device", device])
    cmd.append("--json")
    _run_subprocess_into_job(cmd, store=store, job=job, success_op="probe")


def pull_model(identifier: str, *, store: JobStore, job: Job) -> None:
    """Async wrapper around `muse pull <identifier>`.

    `identifier` may be a curated alias, a bundled model id, or a
    resolver URI. The subprocess persists the resulting catalog entry
    directly (catalog.json is written under MUSE_CATALOG_DIR), so the
    next `enable` call sees it.
    """
    store.update(job.job_id, state="running")
    cmd = [sys.executable, "-m", "muse.cli", "pull", identifier]
    _run_subprocess_into_job(cmd, store=store, job=job, success_op="pull")


def _run_subprocess_into_job(
    cmd: list[str],
    *,
    store: JobStore,
    job: Job,
    success_op: str,
) -> None:
    """Run `cmd` to completion; capture stdout/stderr into the Job.

    On success: state=done, log_lines populated, result contains the op
    name + return code + stdout. On failure: state=failed, error has the
    return code + stderr.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        log_lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        if proc.returncode == 0:
            store.update(
                job.job_id, state="done",
                log_lines=log_lines,
                result={
                    "op": success_op,
                    "returncode": 0,
                    "stdout": proc.stdout,
                },
            )
        else:
            store.update(
                job.job_id, state="failed",
                log_lines=log_lines,
                error=f"exit {proc.returncode}: {proc.stderr.strip() or 'subprocess failed'}",
            )
    except subprocess.TimeoutExpired:
        store.update(job.job_id, state="failed", error="subprocess timed out")
    except Exception as e:  # noqa: BLE001
        logger.exception("subprocess job failed")
        store.update(job.job_id, state="failed", error=str(e))


def _restart_worker_inplace(spec: WorkerSpec, *, device: str) -> None:
    """Terminate + respawn one worker with its current `spec.models`.

    Used by enable_model (joining a venv group) and disable_model
    (dropping a model from a multi-model worker). Reuses the spec's
    port, python_path, and updated models list. restart_count bumps so
    the auto-restart monitor sees this in its bookkeeping; failure_count
    resets so we don't carry over stale unhealthy state.
    """
    _shutdown_workers([spec])
    spec.process = None
    spec.failure_count = 0
    spec.restart_count += 1
    spawn_worker(spec, device=device)
    wait_for_ready(port=spec.port, timeout=120.0)
    spec.status = "running"


def launch_async(
    op_fn: Callable[..., None],
    *,
    op_name: str,
    model_id: str,
    store: JobStore,
    op_args: tuple = (),
    **kwargs: Any,
) -> Job:
    """Create a Job + spawn a daemon thread that runs op_fn(...).

    `op_fn` must accept (positional args from `op_args`, keyword args
    `job=Job`, `store=JobStore`, **kwargs). The thread is daemonized so
    a Ctrl+C on the supervisor takes it down with the process;
    JobStore.shutdown joins them with a timeout on graceful exit.

    `model_id` is the JobStore label (for /v1/admin/jobs/{id} display);
    if `op_args` is empty, model_id is ALSO passed as the first
    positional argument (the common case for enable_model / probe_model
    / pull_model whose signature starts with `model_id`).
    """
    job = store.create(op_name, model_id)
    if not op_args:
        op_args = (model_id,)
    thread = threading.Thread(
        target=op_fn,
        args=op_args,
        kwargs={"job": job, "store": store, **kwargs},
        daemon=True,
        name=f"muse-admin-{op_name}-{job.job_id}",
    )
    job.thread = thread
    thread.start()
    return job
