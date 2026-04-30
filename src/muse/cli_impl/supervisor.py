"""`muse serve` supervisor: orchestrate workers + run gateway.

Responsibilities (across E1-E4):
  1. Read catalog (E1)
  2. Group models by venv (same python_path = same worker) (E1)
  3. Allocate a local port per worker (E1)
  4. Spawn worker subprocesses (E2)
  5. Wait for each worker's /health to become responsive (E2)
  6. Build gateway routes + run gateway uvicorn (E3)
  7. On shutdown: SIGTERM workers, wait for exit (E3)

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


@dataclass
class SupervisorState:
    """Runtime state shared across the supervisor and admin endpoints.

    `workers` is the live list of spawned WorkerSpec records. Admin
    operations (enable/disable) mutate it; the monitor thread reads it.

    `device` is the supervisor-wide device flag (cuda/cpu/auto/mps).
    Admin-spawned workers inherit it unless their MANIFEST capability
    pins a specific device.

    `started_at` is monotonic seconds at supervisor boot; admin uptime
    queries can subtract this to report worker uptimes.

    `lock` is a reentrant lock guarding all mutations of `workers`. v1
    uses one global lock; per-model locks are deferred until contention
    becomes measurable.
    """
    workers: list[WorkerSpec] = field(default_factory=list)
    device: str = "auto"
    started_at: float = field(default_factory=time.monotonic)
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


def run_supervisor(*, host: str, port: int, device: str) -> int:
    """Entry point for `muse serve`.

    Plans workers from catalog, spawns them, waits for the FIRST one to
    pass /health, then starts the gateway. Remaining workers promote on
    a background daemon thread; their models join /v1/models when ready.
    This means the gateway is reachable in seconds even when a slow
    worker (large GGUF, big diffusion model) is still loading.

    Registers a SupervisorState singleton so admin endpoints under
    `/v1/admin/*` can inspect and mutate the worker list. The monitor
    thread reads `state.workers` directly so admin-triggered worker
    additions/removals are picked up on the next polling tick without
    extra synchronization.
    """
    from muse.cli_impl.gateway import build_gateway

    specs = plan_workers()
    if not specs:
        logger.warning(
            "no pulled models with a venv - server will start empty. "
            "Pull a model first: `muse pull <model-id>`"
        )

    state = SupervisorState(workers=specs, device=device)
    set_supervisor_state(state)

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None

    try:
        for spec in specs:
            spawn_worker(spec, device=device)

        if specs:
            # Wait for the FIRST worker only. With one spec the loop
            # terminates immediately; with N it returns the moment any
            # single worker passes /health. The remaining workers
            # promote on the boot thread (below) so the gateway can
            # serve fast workers while slow ones load.
            first_ready = _wait_for_first_ready(specs)
            with state.lock:
                first_ready.status = "running"
            logger.info(
                "first worker ready on port %d (%s); gateway will start now; "
                "remaining %d worker(s) will promote in the background",
                first_ready.port, first_ready.models, len(specs) - 1,
            )

            remaining = [s for s in specs if s is not first_ready]
            if remaining:
                threading.Thread(
                    target=_promote_workers,
                    args=(remaining, state),
                    daemon=True,
                    name="muse-late-boot",
                ).start()

            # Auto-restart monitor reads state.workers (a live
            # reference) so admin-triggered enable/disable plus the
            # late-boot thread's promotions all show up here.
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

        # Build gateway with a live SupervisorState reference. Routes
        # are derived per-request from state.workers (running-only) so
        # late-promoting workers join the routing table without an
        # app rebuild.
        app = build_gateway(state=state)

        logger.info("starting gateway on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        # Tell the monitor to stop BEFORE killing workers. Otherwise the
        # monitor could spawn a restart while we're terminating processes.
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        # boot_thread is daemon=True so it dies with the process; it
        # uses short httpx timeouts so it exits naturally on its own.
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
