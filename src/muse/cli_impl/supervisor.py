"""`muse serve` supervisor: orchestrate workers + run gateway.

Responsibilities (v0.40.0+, lazy load):
  1. Read catalog at boot (only for validation, not for eager spawning).
  2. Construct a LoadDirector and hang it off SupervisorState.
  3. Stamp `unservable_reasons` for enabled catalog rows that lack memory
     data or whose declared memory_gb exceeds device capacity at boot.
  4. Start gateway immediately (zero workers initially). First request
     to a model triggers `LoadDirector.acquire`, which calls back into
     this module's `load_model_into_worker` to spawn the worker.
  5. On shutdown: SIGTERM workers (whatever was loaded by then), wait for
     exit.

A module-level SupervisorState singleton holds the worker list and shared
metadata. Admin endpoints (muse.admin.*) read and mutate the state under
its RLock; the auto-restart monitor reads `state.workers` directly. The
state is registered by `run_supervisor` and cleared on its way out.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn

from muse.core.catalog import _read_catalog
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


@dataclass
class WorkerSpec:
    """Everything needed to spawn and supervise one worker subprocess.

    Fields mutated by the monitor thread (after startup):
      - process: replaced on restart
      - restart_count: total restart attempts (caps at _MAX_RESTARTS)
      - failure_count: consecutive unhealthy polls
      - last_spawn_at: time.monotonic() of most recent spawn (for backoff)
      - status: pending -> running -> unhealthy -> dead
    """
    models: list[str]
    python_path: str
    port: int
    device: str = "auto"
    process: object = field(default=None)
    restart_count: int = 0
    failure_count: int = 0
    last_spawn_at: float = 0.0
    status: str = "pending"
    # Job that owns the in-flight transition (if any). Set when the
    # admin enable / restart-in-place op claims the spec; cleared
    # when the op finishes. Other concurrent enable requests for the
    # same model coalesce onto this job_id rather than launching a
    # duplicate spawn.
    job_id: str | None = None


@dataclass
class SupervisorState:
    """Runtime state shared across the supervisor and admin endpoints.

    `workers` is the live list of spawned WorkerSpec records. Admin
    operations (enable/disable) mutate it; the monitor thread reads it.
    Under lazy load, the LoadDirector also adds and removes WorkerSpec
    records via `load_model_into_worker` and `unload_model_from_worker`
    in `muse.admin.operations`.

    `device` is the supervisor-wide device flag (cuda/cpu/auto/mps).
    Admin-spawned workers inherit it unless their MANIFEST capability
    pins a specific device.

    `started_at` is monotonic seconds at supervisor boot; admin uptime
    queries can subtract this to report worker uptimes.

    `director` is the LoadDirector singleton. Populated by
    `run_supervisor` after construction; admin endpoints reach the
    director through `state.director`. None outside of a running
    supervisor (tests building bare states get a coherent default).

    `unservable_reasons` is a per-model-id map populated at boot by
    `validate_catalog_at_boot`. Maps model_id to a string explaining
    why the model cannot be served (no memory data, exceeds device
    capacity, etc). The gateway short-circuits 503 for these models
    before calling `director.acquire`; `/v1/models` surfaces the
    reason to clients.

    `lock` is a reentrant lock guarding all mutations of `workers` +
    `unservable_reasons`. v1 uses one global lock; per-model locks
    are deferred until contention becomes measurable.
    """
    workers: list[WorkerSpec] = field(default_factory=list)
    device: str = "auto"
    started_at: float = field(default_factory=time.monotonic)
    director: "Any | None" = None
    unservable_reasons: dict[str, str] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


# Module-level singleton; admin routes reach this through
# get_supervisor_state. Tests build their own SupervisorState instances
# and either set it via set_supervisor_state or pass it directly.
_state: "SupervisorState | None" = None


def get_supervisor_state() -> SupervisorState:
    """Return the active SupervisorState, or an empty sentinel.

    The sentinel is fresh on every call when nothing is set; this means
    admin endpoints loaded outside a running supervisor (e.g. unit tests
    spinning up the gateway in isolation) get a coherent empty state
    instead of a None that crashes the routes.
    """
    return _state if _state is not None else SupervisorState()


def set_supervisor_state(state: SupervisorState) -> None:
    """Register a SupervisorState as the active singleton."""
    global _state
    _state = state


def clear_supervisor_state() -> None:
    """Test hook + supervisor shutdown: drop the active singleton."""
    global _state
    _state = None


def plan_workers(port_start: int = 9001, port_end: int = 9999) -> list[WorkerSpec]:
    """Read catalog, group by venv, allocate ports.

    Returns one WorkerSpec per unique venv (identified by python_path).
    Pre-worker catalog entries (missing python_path) are logged + skipped.
    """
    catalog = _read_catalog()

    # Group by python_path. Preserve insertion order for determinism.
    groups: dict[str, list[str]] = {}
    for model_id, entry in catalog.items():
        python = entry.get("python_path")
        if not python:
            logger.warning(
                "skipping pre-worker catalog entry %r - no python_path; "
                "re-run `muse pull %s` to create its venv",
                model_id, model_id,
            )
            continue
        # Default True covers legacy entries without the field
        # (also backfilled by _read_catalog's setdefault)
        if not entry.get("enabled", True):
            logger.info(
                "skipping disabled model %r (use `muse models enable %s` to re-enable)",
                model_id, model_id,
            )
            continue
        groups.setdefault(python, []).append(model_id)

    specs: list[WorkerSpec] = []
    used_ports: set[int] = set()
    for python_path, models in groups.items():
        # Allocate a free port, avoiding collisions with ports already
        # assigned to earlier specs in this planning pass.
        while True:
            port = find_free_port(start=port_start, end=port_end)
            if port not in used_ports:
                used_ports.add(port)
                break
            port_start = port + 1
        specs.append(WorkerSpec(
            models=sorted(models),
            python_path=python_path,
            port=port,
        ))
    return specs


def spawn_worker(spec: WorkerSpec, *, device: str) -> None:
    """Start a worker subprocess using its venv's Python.

    Persists `device` onto the spec so the monitor thread can respawn
    with the same settings on restart. Records last_spawn_at for the
    backoff timer in _attempt_restart.
    """
    spec.device = device
    cmd = [
        spec.python_path, "-m", "muse.cli", "_worker",
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        "--device", device,
    ]
    for m in spec.models:
        cmd.extend(["--model", m])
    logger.info("spawning worker: %s", " ".join(cmd))
    spec.process = subprocess.Popen(cmd)
    spec.last_spawn_at = time.monotonic()


def wait_for_ready(
    *, port: int, timeout: float = 60.0, poll_interval: float = 0.5,
) -> None:
    """Block until http://127.0.0.1:<port>/health returns 200, or timeout.

    Raises TimeoutError if the worker never becomes ready. Polls with
    short sleeps so slow workers (big model loads) still get through.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as e:
            last_err = e
        time.sleep(poll_interval)
    raise TimeoutError(
        f"worker on port {port} did not become ready within {timeout}s "
        f"(last error: {last_err})"
    )


def _wait_for_first_ready(
    specs: list[WorkerSpec],
    *,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> WorkerSpec:
    """Block until ANY spec's /health returns 200; return that spec.

    Round-robin polling across all specs in each tick. A worker that
    responds 200 first wins, regardless of position in the list. This
    means the gateway can boot as soon as the fastest worker is up,
    independent of slower workers buried earlier in the list.

    Raises TimeoutError if no spec passes within timeout. The caller
    treats this as a fatal supervisor-bringup failure: every spawned
    worker is sick, gateway should not start.
    """
    if not specs:
        raise ValueError("_wait_for_first_ready called with no specs")
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        for spec in specs:
            try:
                r = httpx.get(
                    f"http://127.0.0.1:{spec.port}/health", timeout=2.0,
                )
                if r.status_code == 200:
                    return spec
            except httpx.HTTPError as e:
                last_err = e
        time.sleep(poll_interval)
    raise TimeoutError(
        f"no worker became ready within {timeout}s "
        f"(spawned {len(specs)}; last error: {last_err})"
    )


def _promote_workers(
    specs: list[WorkerSpec],
    state: SupervisorState,
    *,
    timeout: float = 120.0,
) -> None:
    """Background-thread target: poll /health for late-loading workers.

    For each spec, polls /health via wait_for_ready. On success, sets
    spec.status = 'running' under state.lock. On timeout, sets
    'unhealthy' so the auto-restart monitor's next tick triggers a
    respawn.

    Failures here are non-fatal: the gateway is already serving the
    first-ready worker, so a slow late-boot doesn't keep clients
    from reaching healthy workers.
    """
    for spec in specs:
        try:
            wait_for_ready(port=spec.port, timeout=timeout)
            with state.lock:
                spec.status = "running"
            logger.info(
                "late-promote: worker on port %d (%s) ready",
                spec.port, spec.models,
            )
        except TimeoutError:
            with state.lock:
                spec.status = "unhealthy"
            logger.warning(
                "late-promote: worker on port %d did not become ready in %ds; "
                "auto-restart monitor will retry",
                spec.port, timeout,
            )


def check_worker_health(*, port: int, timeout: float = 2.0) -> bool:
    """Single /health poll. Returns True iff the worker responds 200.

    Swallows all httpx errors; they indicate "unhealthy" for our purposes.
    Used by the monitor thread's periodic liveness check.
    """
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# Monitor defaults (module constants; not CLI-configurable in this iteration)
_MONITOR_INTERVAL = 5.0
_FAILURE_THRESHOLD = 3
_MAX_RESTARTS = 10
_BACKOFF_CAP = 30.0  # seconds
_BACKOFF_BASE = 1.0


def _attempt_restart(
    spec: WorkerSpec,
    *,
    stop_event: "threading.Event",
    max_restarts: int = _MAX_RESTARTS,
    backoff_base: float = _BACKOFF_BASE,
    backoff_cap: float = _BACKOFF_CAP,
    ready_timeout: float = 60.0,
) -> None:
    """Terminate existing process if alive, wait backoff, respawn.

    Mutates spec.process, spec.restart_count, spec.failure_count, spec.status.
    Marks spec.status = "dead" if restart_count reaches max_restarts.
    Returns early if stop_event fires during backoff.
    """
    if spec.restart_count >= max_restarts:
        logger.error(
            "worker on port %d: exhausted %d restart attempts; marking dead",
            spec.port, max_restarts,
        )
        spec.status = "dead"
        return

    # Exponential backoff, capped
    backoff = min(backoff_base * (2 ** spec.restart_count), backoff_cap)
    logger.warning(
        "worker on port %d: restart attempt %d after %.1fs backoff",
        spec.port, spec.restart_count + 1, backoff,
    )
    # wait() returns True if event was set during the wait (skip restart)
    if stop_event.wait(backoff):
        return

    # Terminate existing process if still alive (best-effort)
    if spec.process is not None and spec.process.poll() is None:
        try:
            spec.process.terminate()
            try:
                spec.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                spec.process.kill()
        except Exception as e:
            logger.warning("worker on port %d: terminate failed: %s", spec.port, e)

    # Respawn. Always bump restart_count so we can't loop forever.
    spec.restart_count += 1
    try:
        spawn_worker(spec, device=spec.device)
        wait_for_ready(port=spec.port, timeout=ready_timeout)
        spec.failure_count = 0
        spec.status = "running"
        logger.info("worker on port %d: successfully restarted", spec.port)
    except (subprocess.SubprocessError, TimeoutError) as e:
        logger.error("worker on port %d: restart failed: %s", spec.port, e)
        spec.status = "unhealthy"


def _monitor_workers(
    specs: list[WorkerSpec],
    stop_event: "threading.Event",
    *,
    interval: float = _MONITOR_INTERVAL,
    failure_threshold: int = _FAILURE_THRESHOLD,
    max_restarts: int = _MAX_RESTARTS,
) -> None:
    """Poll each worker; restart after `failure_threshold` consecutive failures.

    Exits when stop_event is set. Called from the monitor daemon thread
    started by run_supervisor (Task B4).
    """
    while not stop_event.is_set():
        for spec in specs:
            if stop_event.is_set():
                return
            if spec.status == "dead":
                continue

            # Process-death detection is unambiguous; short-circuit
            if spec.process is not None and spec.process.poll() is not None:
                logger.warning(
                    "worker on port %d: process exited with code %s",
                    spec.port, spec.process.returncode,
                )
                spec.failure_count = failure_threshold
            else:
                healthy = check_worker_health(port=spec.port)
                if healthy:
                    spec.failure_count = 0
                    spec.status = "running"
                    continue
                spec.failure_count += 1
                if spec.status == "running":
                    spec.status = "unhealthy"
                logger.info(
                    "worker on port %d: unhealthy (%d/%d consecutive failures)",
                    spec.port, spec.failure_count, failure_threshold,
                )

            if spec.failure_count >= failure_threshold:
                _attempt_restart(spec, stop_event=stop_event, max_restarts=max_restarts)

        # Sleep with early-exit if stop_event fires
        if stop_event.wait(interval):
            return


def _shutdown_workers(specs: list[WorkerSpec], grace: float = 5.0) -> None:
    """SIGTERM all workers; SIGKILL any that don't exit within `grace` seconds."""
    for spec in specs:
        if spec.process is None:
            continue
        try:
            spec.process.terminate()
        except Exception as e:
            logger.warning("failed to SIGTERM worker on port %d: %s", spec.port, e)

    for spec in specs:
        if spec.process is None:
            continue
        try:
            spec.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning("worker on port %d did not exit in %ds; killing", spec.port, grace)
            spec.process.kill()
        except Exception as e:
            logger.warning("error waiting for worker on port %d: %s", spec.port, e)


class _MemoryProbeAdapter:
    """Thin adapter wrapping `muse.core.memory_probe` module functions
    as bound methods on an object.

    LoadDirector accepts any object with `.gpu_free_gb()` and
    `.cpu_free_gb()`. The memory_probe module exposes those as
    free functions; this adapter satisfies the duck-typed contract
    without forcing a class definition into memory_probe.py
    (deferred to a future refactor; for now an adapter at the
    composition seam is the smaller change).
    """

    def gpu_free_gb(self, device_id: int = 0) -> float | None:
        from muse.core import memory_probe
        return memory_probe.gpu_free_gb(device_id)

    def cpu_free_gb(self) -> float:
        from muse.core import memory_probe
        return memory_probe.cpu_free_gb()


def _has_memory_data(catalog_entry: dict) -> tuple[bool, float, str]:
    """Return (has_data, declared_memory_gb, device).

    Two sources of memory data, in order of preference:
      1. `manifest.capabilities.memory_gb` annotation (hand-set or
         from a script's MANIFEST).
      2. `measurements.<device>.peak_bytes` from a probe run.

    `device` is read from `manifest.capabilities.device` and lowercased.
    Falls back to "cpu" when absent (the catalog default).

    `has_data` is True when either source is present; False when both
    are absent. The boot validation flags False entries as unservable
    with the probe-prompt reason.
    """
    manifest = catalog_entry.get("manifest", {}) or {}
    capabilities = manifest.get("capabilities", {}) or {}
    device = str(capabilities.get("device", "cpu")).lower() or "cpu"
    declared = capabilities.get("memory_gb")

    measurements = catalog_entry.get("measurements", {}) or {}
    # Probe records key by the resolved device (e.g. "cpu" / "cuda")
    # so we look up by the same key. "gpu" alias normalizes to "cuda"
    # to match what the probe writes.
    measurement_key = "cuda" if device == "gpu" else device
    measured = (measurements.get(measurement_key) or {}).get("peak_bytes")

    if declared is not None:
        try:
            declared_gb = float(declared)
        except (TypeError, ValueError):
            declared_gb = 0.0
        return True, declared_gb, device
    if measured is not None and measured > 0:
        try:
            measured_gb = float(measured) / (1024 ** 3)
        except (TypeError, ValueError):
            measured_gb = 0.0
        return True, measured_gb, device

    return False, 0.0, device


def validate_catalog_at_boot(
    state: SupervisorState,
    *,
    memory_probe: "Any | None" = None,
    gpu_headroom_gb: float = 1.0,
    cpu_headroom_gb: float = 2.0,
) -> None:
    """Walk the enabled catalog and stamp unservable_reasons.

    For each `enabled: true` row in the catalog:
      - If the row has neither `manifest.capabilities.memory_gb` nor
        `measurements.<device>.peak_bytes`, mark it
        "no memory estimate; run muse models probe".
      - If the row's declared memory_gb exceeds free at boot
        (live free minus headroom), mark it "exceeds device capacity".
      - GPU rows with no live VRAM info (pynvml unavailable) are
        treated as exceeding capacity until probe data lands.

    The result is stored in `state.unservable_reasons`; the gateway and
    `/v1/models` consult this dict to short-circuit 503 before calling
    the director.

    `memory_probe` defaults to the production adapter; tests inject a
    MagicMock with the desired return values.
    """
    if memory_probe is None:
        memory_probe = _MemoryProbeAdapter()

    catalog = _read_catalog()
    cpu_free_gb = float(memory_probe.cpu_free_gb())
    cpu_available_gb = max(0.0, cpu_free_gb - cpu_headroom_gb)

    gpu_free = memory_probe.gpu_free_gb()
    if gpu_free is None:
        gpu_available_gb = None
    else:
        gpu_available_gb = max(0.0, float(gpu_free) - gpu_headroom_gb)

    for model_id, entry in catalog.items():
        if not entry.get("enabled", True):
            continue
        # Skip pre-worker entries; they cannot load anyway.
        if not entry.get("python_path"):
            continue

        has_data, declared_gb, device = _has_memory_data(entry)
        if not has_data:
            state.unservable_reasons[model_id] = (
                "no memory estimate; run `muse models probe` to populate"
            )
            continue

        # Pick the relevant device's available pool.
        if device in ("cuda", "gpu"):
            available_gb = gpu_available_gb
            if available_gb is None:
                # pynvml unavailable; we cannot say it fits. Until probe
                # data lands or pynvml installs, mark unservable so
                # callers get a 503 instead of a load attempt that ends
                # in a crashed worker.
                state.unservable_reasons[model_id] = (
                    "exceeds device capacity (no GPU info available; "
                    "install nvidia-ml-py / pynvml or set memory budget)"
                )
                continue
        else:
            available_gb = cpu_available_gb

        if declared_gb > available_gb:
            state.unservable_reasons[model_id] = (
                f"exceeds device capacity ({declared_gb:.1f} GB > "
                f"{available_gb:.1f} GB available on {device})"
            )


def _build_load_director(state: SupervisorState) -> "Any":
    """Construct a LoadDirector wired to the supervisor's enable/disable.

    `enable_fn` and `disable_fn` are thin wrappers around the new
    `load_model_into_worker` and `unload_model_from_worker` operations
    in `muse.admin.operations`. Those operations spawn / terminate
    workers WITHOUT touching the catalog's `enabled` flag - lazy load
    is "is there a worker for this model right now?", orthogonal to
    the catalog's "is this model in service?" flag. Reusing the
    existing `enable_model` / `disable_model` ops would re-couple the
    two states, defeating the v0.40.0 design.

    Imported lazily to break the cycle: supervisor.py is imported by
    admin.operations on its way up, so an unconditional top-level
    import would loop.
    """
    from muse.admin.operations import (
        load_model_into_worker,
        unload_model_from_worker,
    )
    from muse.cli_impl.load_director import LoadDirector

    def enable_fn(model_id: str) -> int:
        return load_model_into_worker(model_id, state=state)

    def disable_fn(model_id: str) -> None:
        unload_model_from_worker(model_id, state=state)

    return LoadDirector(
        enable_fn=enable_fn,
        disable_fn=disable_fn,
        memory_probe=_MemoryProbeAdapter(),
    )


def run_supervisor(*, host: str, port: int, device: str) -> int:
    """Entry point for `muse serve` (v0.40.0+: lazy load).

    Boot sequence:
      1. Construct a SupervisorState with an empty worker list.
      2. Construct a LoadDirector and hang it off state.director.
      3. Run validate_catalog_at_boot to stamp unservable_reasons.
      4. Start the auto-restart monitor (it watches state.workers, which
         is empty at boot but will fill via director.acquire).
      5. Start the gateway. First request per model triggers the
         director's enable_fn, which spawns the worker.
      6. On shutdown: SIGTERM whatever workers are loaded (could be 0).

    No worker spawn at boot. No first-ready wait. The gateway is
    reachable instantly. Cold-start latency moves from boot to first
    request per model.

    Registers a SupervisorState singleton so admin endpoints under
    `/v1/admin/*` can inspect and mutate the worker list. The monitor
    thread reads `state.workers` directly, so director-triggered worker
    spawns + admin-triggered enable/disable + auto-restart all show up
    in one consistent live list.
    """
    from muse.cli_impl.gateway import build_gateway

    state = SupervisorState(workers=[], device=device)
    state.director = _build_load_director(state)
    set_supervisor_state(state)

    # Validate the catalog. Models with no memory data or memory > free
    # at boot get a 503 reason stamped on state.unservable_reasons.
    validate_catalog_at_boot(state)
    if state.unservable_reasons:
        for mid, reason in sorted(state.unservable_reasons.items()):
            logger.warning("unservable model %r: %s", mid, reason)

    stop_event = threading.Event()

    # The monitor thread reads `state.workers` (a live reference), so
    # workers spawned later via the director's enable_fn show up on the
    # next polling tick without extra coordination. Started always (not
    # gated on a non-empty worker list, since lazy load means workers
    # arrive later).
    monitor_thread = threading.Thread(
        target=_monitor_workers,
        args=(state.workers, stop_event),
        daemon=True,
        name="muse-monitor",
    )
    monitor_thread.start()
    logger.info(
        "auto-restart monitor running (interval=%.1fs, threshold=%d, budget=%d)",
        _MONITOR_INTERVAL, _FAILURE_THRESHOLD, _MAX_RESTARTS,
    )

    try:
        # Build gateway with a live SupervisorState reference. Routes
        # are derived per-request from state.workers (running-only) so
        # director-spawned workers join the routing table without an
        # app rebuild.
        app = build_gateway(state=state)

        logger.info(
            "starting gateway on %s:%d (lazy load: %d unservable model(s))",
            host, port, len(state.unservable_reasons),
        )
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        # Tell the monitor to stop BEFORE killing workers. Otherwise the
        # monitor could spawn a restart while we're terminating processes.
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        # Whatever workers were loaded by the director get torn down here.
        # Empty list is a no-op.
        _shutdown_workers(state.workers)
        clear_supervisor_state()
        # Best-effort: join admin job threads so a Ctrl+C during a pull
        # doesn't leave dangling daemons. Import lazily to keep the
        # supervisor's startup path free of admin concerns.
        try:
            from muse.admin.jobs import get_default_store
            get_default_store().shutdown()
        except Exception as e:  # noqa: BLE001
            logger.warning("admin job-store shutdown failed: %s", e)
    return 0
