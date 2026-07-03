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
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from muse.cli_impl.serve_util import run_uvicorn

from muse.cli_impl.idle_sweeper import IdleSweeper
from muse.core import config
from muse.core.catalog import _read_catalog, get_manifest
from muse.core.venv import find_free_port

from muse.core.memory_probe import declared_device

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

    `stop_event` is the supervisor-wide shutdown signal. Set on
    KeyboardInterrupt / SIGTERM in `run_supervisor`'s cleanup; consumed
    by the auto-restart monitor and the idle sweeper so a single
    Ctrl+C unblocks every supervisor-owned daemon thread at once.
    A bare default state (e.g. one returned by `get_supervisor_state`
    when nothing is registered) gets a fresh Event so admin or test
    code that touches `state.stop_event` doesn't crash on None.

    `idle_sweeper` and `idle_sweeper_thread` hold the v0.40.1 idle-
    timeout sweeper after `run_supervisor` boots it. Exposed on the
    state so tests can introspect the sweeper and so future admin
    endpoints can read its tick metadata without a module-level
    singleton lookup.
    """
    workers: list[WorkerSpec] = field(default_factory=list)
    device: str = "auto"
    started_at: float = field(default_factory=time.monotonic)
    director: "Any | None" = None
    unservable_reasons: dict[str, str] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    idle_sweeper: "IdleSweeper | None" = None
    idle_sweeper_thread: "threading.Thread | None" = None
    # #319 same-model cold-load coalescing (v0.51.0). model_id -> asyncio.Future
    # gate. The FIRST request for a cold model becomes the loader (dispatches
    # one off-loop director.acquire); concurrent requests for the SAME model
    # await the gate on the event loop (no thread), so only one thread parks
    # per model-load instead of N-1. Touched ONLY from the gateway's single
    # event loop, so a plain dict is safe (the loader election is await-free).
    cold_load_gates: dict = field(default_factory=dict)


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
    except (subprocess.SubprocessError, TimeoutError, OSError) as e:
        # OSError covers FileNotFoundError / PermissionError from Popen when
        # the venv python is missing or non-executable (e.g. the venv was
        # deleted, or its python symlink broke on a system upgrade). Without
        # catching it, the exception escapes _monitor_workers and kills the
        # monitor daemon thread, silently disabling health-monitoring and
        # auto-restart for ALL workers (M10).
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

    Concurrency: `specs` is the live `state.workers` list shared with
    admin operations (enable/disable) that may call `state.workers.remove`
    under `state.lock` while the monitor is iterating. To avoid
    `RuntimeError: list changed size during iteration`, we snapshot the
    list at the top of each poll tick with `list(specs)`. The snapshot
    holds a reference to each WorkerSpec (not a copy), so in-place
    mutations to spec fields (status, failure_count, etc.) are visible
    to both the monitor and admin operations without extra coordination.
    A spec removed from `state.workers` during the tick may still be
    iterated in that tick; its process is already being torn down by the
    operation that removed it, so any restart the monitor would trigger
    is harmless (the spec will not be re-added to state.workers).
    """
    while not stop_event.is_set():
        for spec in list(specs):  # snapshot: safe against concurrent remove()
            if stop_event.is_set():
                return
            if spec.status == "dead":
                continue

            # Skip specs in the middle of an admin- or director-driven
            # transition. job_id is set when an in-flight operation has
            # claimed the spec (enable_model, load_model_into_worker,
            # restart-in-place). The owning operation is responsible for
            # the spawn / readiness wait; the monitor must not race it
            # by polling /health (which fails until the worker binds the
            # port) and triggering a duplicate restart. The owning op
            # clears job_id on success or marks the spec dead on failure.
            if spec.job_id is not None:
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


def _weights_size_gb(catalog_entry: dict) -> float:
    """On-disk size of a model's downloaded weights, in GB (0.0 if unknown).

    Last resort in the sizing ladder: a model that was never probed and
    declares no `memory_gb` can still be sized from the bytes already on
    disk, so it loads on demand (evicting LRU as needed) instead of being
    503'd "no memory estimate". Sums regular-file sizes under the entry's
    `local_dir` (an HF snapshot dir whose weight files are symlinks into
    the blob store; `os.path.getsize` follows them). Returns 0.0 when
    `local_dir` is absent, missing, or unreadable.

    This UNDERestimates live runtime (no activations / KV cache); the
    LoadDirector's observed-peak writeback self-heals the estimate upward
    after the first real load, and the auto-restart monitor recovers a
    worker that an initial under-estimate happens to OOM.

    GGUF exception: a GGUF snapshot dir routinely holds several quant
    variants of one model (q3/q4/q5/q8/f16), but only the declared
    `capabilities.gguf_file` actually loads. Summing the whole tree would
    OVERestimate wildly (a 4B q4 whose repo ships six quants sums to ~15 GB
    vs its ~2.6 GB weight), and overestimation is the dangerous direction:
    it 503s a servable model as "exceeds device capacity". So when a
    specific `gguf_file` is declared, size from that one file, falling back
    to the tree walk only when it is absent on disk (stale path).
    """
    local_dir = catalog_entry.get("local_dir")
    if not local_dir:
        return 0.0
    capabilities = (catalog_entry.get("manifest") or {}).get("capabilities") or {}
    gguf_file = capabilities.get("gguf_file")
    if gguf_file:
        try:
            return os.path.getsize(os.path.join(local_dir, gguf_file)) / (1024 ** 3)
        except OSError:
            # Declared file missing/unreadable: fall through to the tree walk
            # rather than returning 0.0 and stamping the model unservable.
            pass
    total = 0
    try:
        for root, _dirs, files in os.walk(local_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:
        return 0.0
    return total / (1024 ** 3)


def _has_memory_data(catalog_entry: dict) -> tuple[bool, float, str]:
    """Return (has_data, memory_gb, device).

    Sizing ladder, in order of preference:
      1. `manifest.capabilities.memory_gb` annotation (hand-set or
         from a script's MANIFEST).
      2. `measurements.<device>.peak_bytes` from a probe run / self-healed
         lazy-load observation.
      3. on-disk weights size summed from the entry's `local_dir`.

    `device` is read from `manifest.capabilities.device` and lowercased.
    Falls back to "auto" when absent, matching the worker's own default
    (see muse.core.memory_probe.declared_device).

    `has_data` is True when ANY source is present; False only when the
    model declares nothing, was never probed, AND has no weights on disk.
    The boot validation flags False entries as unservable with the
    probe-prompt reason. Because pulled models always have weights on
    disk, that 503 path is effectively reserved for pre-worker / removed
    entries.
    """
    manifest = catalog_entry.get("manifest", {}) or {}
    capabilities = manifest.get("capabilities", {}) or {}
    device = declared_device(capabilities)
    declared = capabilities.get("memory_gb")

    measurements = catalog_entry.get("measurements", {}) or {}
    # Probe records key by the resolved device (e.g. "cpu" / "cuda")
    # so we look up by the same key. "gpu" alias normalizes to "cuda"
    # to match what the probe writes.
    measurement_key = "cuda" if device == "gpu" else device
    measured = (measurements.get(measurement_key) or {}).get("peak_bytes")

    # Bundled models have no persisted manifest in catalog.json, so `device`
    # falls back to "cpu" above even when the probe ran on cuda. If the
    # manifest-derived device has no measurement but the catalog has one for
    # another device, use it and adopt the measurement's own recorded device
    # so the capacity check below picks the right memory pool. Without this,
    # `muse models probe` never clears a bundled GPU model's "no memory
    # estimate" flag (the probe writes measurements.cuda; the lookup reads
    # measurements.cpu).
    #
    # Gate on `declared is None`: this recovery only matters when there is no
    # declared memory_gb (the bundled-model case). A model that DOES declare
    # memory_gb already trusts its manifest device below, so we must not let a
    # stale cross-device measurement (e.g. a CPU probe of a declared-cuda model
    # pulled with --no-probe) overwrite that device and mis-size the GPU model
    # against the CPU pool.
    if measured is None and declared is None:
        for dev_key, rec in measurements.items():
            rec = rec or {}
            peak = rec.get("peak_bytes")
            if peak:
                measured = peak
                device = str(rec.get("device") or dev_key).lower() or device
                break

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

    # Last resort: size the model from the bytes already on disk so a
    # never-probed model still loads on demand instead of 503'ing.
    weights_gb = _weights_size_gb(catalog_entry)
    if weights_gb > 0:
        return True, weights_gb, device

    return False, 0.0, device


def _available_pools(
    memory_probe: "Any",
    *,
    gpu_headroom_gb: float,
    cpu_headroom_gb: float,
) -> tuple[float, "float | None"]:
    """Live (CPU, GPU) available pools in GB, each minus its headroom.

    GPU is None when no live VRAM info is available (pynvml absent / AMD /
    driver mismatch). Shared by boot validation and the request-path
    re-check so both size capacity from the SAME live readings.
    """
    cpu_free_gb = float(memory_probe.cpu_free_gb())
    cpu_available_gb = max(0.0, cpu_free_gb - cpu_headroom_gb)
    gpu_free = memory_probe.gpu_free_gb()
    if gpu_free is None:
        gpu_available_gb = None
    else:
        gpu_available_gb = max(0.0, float(gpu_free) - gpu_headroom_gb)
    return cpu_available_gb, gpu_available_gb


def _servability_reason(
    entry: dict,
    *,
    cpu_available_gb: float,
    gpu_available_gb: "float | None",
) -> "str | None":
    """The unservable reason for one catalog entry, or None if servable.

    Single source of truth for boot validation AND the live request-path
    re-check (`revalidate_servability`), so the two verdicts never drift.
    Applies the sizing ladder (`_has_memory_data`) then a device-capacity
    check against the caller-supplied available pools.

    Returns:
      - "no memory estimate ..." when the model is not sizable at all
        (no annotation, no probe, no weights on disk).
      - "exceeds device capacity ..." when sized but it does not fit the
        device's available pool (or a cuda model on a host with no live
        VRAM info). This is a HARD stop: the gateway 503s without deferring
        to the director, because a model that does not fit even an empty
        working set can only make the director evict everything and 503.
      - None when sizable AND it fits.
    """
    has_data, sized_gb, device = _has_memory_data(entry)
    if not has_data:
        return "no memory estimate; run `muse models probe` to populate"
    # Resolve the manifest "auto"/"" convention to the concrete pool this
    # model loads on, mirroring the runtime select_device + the
    # LoadDirector: a GPU is present iff we have live VRAM info
    # (gpu_available_gb is not None). Without this, an auto-device model on
    # a GPU host would be sized against the (large) CPU pool and never
    # 503 even when it cannot fit VRAM -- deferring an impossible model
    # into the director's evict-everything-then-fail path.
    if device in ("auto", ""):
        device = "cuda" if gpu_available_gb is not None else "cpu"
    if device in ("cuda", "gpu"):
        if gpu_available_gb is None:
            return (
                "exceeds device capacity (no GPU info available; "
                "install nvidia-ml-py / pynvml or set memory budget)"
            )
        available_gb = gpu_available_gb
    else:
        available_gb = cpu_available_gb
    if sized_gb > available_gb:
        return (
            f"exceeds device capacity ({sized_gb:.1f} GB > "
            f"{available_gb:.1f} GB available on {device})"
        )
    return None


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
    cpu_available_gb, gpu_available_gb = _available_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
    )

    for model_id, entry in catalog.items():
        if not entry.get("enabled", True):
            continue
        # Skip pre-worker entries; they cannot load anyway.
        if not entry.get("python_path"):
            continue

        reason = _servability_reason(
            entry,
            cpu_available_gb=cpu_available_gb,
            gpu_available_gb=gpu_available_gb,
        )
        if reason is not None:
            state.unservable_reasons[model_id] = reason


def revalidate_servability(
    state: SupervisorState,
    model_id: str,
    *,
    memory_probe: "Any | None" = None,
    gpu_headroom_gb: float = 1.0,
    cpu_headroom_gb: float = 2.0,
) -> str | None:
    """Re-derive one model's unservable verdict against the LIVE catalog.

    `validate_catalog_at_boot` stamps `state.unservable_reasons` once at
    boot. That snapshot goes stale two ways: a `muse models probe` (or a
    manifest edit, or weights simply landing on disk) makes a previously
    unsizable model sizable; or memory frees up so a model that did not fit
    at boot now does. This re-reads the (mtime-cached) catalog for ONE model
    and re-runs the SAME `_servability_reason` boot uses -- the estimate AND
    the device-capacity check against LIVE free memory -- then updates the
    stamp, so the gateway reflects reality WITHOUT a supervisor restart.

    Crucially this preserves a genuine "exceeds device capacity" stamp: a
    model that cannot fit even an empty working set is NOT cleared just
    because it became sizable. Clearing it would route an impossible request
    into the director, whose eviction loop would tear down the whole idle
    working set before 503'ing. The gateway 503s such a model directly.

    Scoped to one model: no full-catalog walk. Reads live memory via the
    probe (defaults to the production adapter; tests inject a MagicMock).
    Returns the current reason (None when now servable). Mutations to
    `state.unservable_reasons` are made under `state.lock`; the probe read
    and `_servability_reason` run outside the lock.
    """
    if memory_probe is None:
        memory_probe = _MemoryProbeAdapter()
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    if entry is None:
        # Removed from the catalog since boot. Clear the now-stale stamp and
        # return None so the gateway falls through to get_manifest, which
        # 404s `model_not_found` if truly gone (or serves a bundled fallback)
        # -- rather than 503'ing with a reason that names a model that no
        # longer exists.
        with state.lock:
            state.unservable_reasons.pop(model_id, None)
        return None
    cpu_available_gb, gpu_available_gb = _available_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
    )
    reason = _servability_reason(
        entry,
        cpu_available_gb=cpu_available_gb,
        gpu_available_gb=gpu_available_gb,
    )
    with state.lock:
        if reason is None:
            state.unservable_reasons.pop(model_id, None)
        else:
            state.unservable_reasons[model_id] = reason
        return reason


def backfill_manifest_memory(manifest: dict, model_id: str) -> dict:
    """Return a copy of `manifest` sized (and device-pinned) from the catalog.

    Two backfills, both drawn from the model's catalog entry:

    1. **memory_gb** -- The LoadDirector sizes loads (and drives LRU eviction)
       from `capabilities.memory_gb`. A probed-only or never-probed model
       declares none, so without this the director would treat it as 0 GB
       ("fits anywhere", never evicting). We fill it from the sizing ladder
       (`_has_memory_data`: probe measurement, else on-disk weights size). An
       explicit declared `memory_gb` always wins.

    2. **device** -- An operator `set-device` pin (catalog `device_override`)
       decides where the worker actually loads, mirroring load_backend's
       tier-1 precedence. We fold it into `capabilities.device` so the
       director sizes, admits, and evicts against the pool the worker will
       load on. Without this a cuda model pinned to cpu makes the director
       needlessly evict GPU models to make room for a host-RAM load, and the
       inverse pin over-commits VRAM. The override fires regardless of the
       memory backfill (a model may declare memory_gb yet still be pinned).

    The input is never mutated; a copy is made lazily only when a backfill
    actually changes something.
    """
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    out = manifest
    caps = manifest.get("capabilities", {}) or {}

    if entry is not None and caps.get("memory_gb") is None:
        gb: float | None = None
        # A LoRA entry's own dir holds only the adapter (tens of MB), so
        # the weights-on-disk fallback would grossly undersize the load.
        # When it has no probe measurement of its own, size it from its
        # muse-id base entry instead. A probed LoRA entry measured the
        # real base+adapter peak; prefer that.
        if caps.get("lora_adapter") and not (entry.get("measurements") or {}):
            base = caps.get("base_model")
            base_entry = (
                catalog.get(base) if base and "/" not in base else None
            )
            if base_entry is not None:
                has_b, gb_b, _d = _has_memory_data(base_entry)
                if has_b and gb_b > 0:
                    gb = gb_b
        if gb is None:
            has_data, gb_own, _device = _has_memory_data(entry)
            if has_data and gb_own > 0:
                gb = gb_own
        if gb is not None:
            out = dict(out)
            out_caps = dict(caps)
            out_caps["memory_gb"] = gb
            out["capabilities"] = out_caps

    override = (entry or {}).get("device_override")
    if override:
        if out is manifest:
            out = dict(out)
        out_caps = dict(out.get("capabilities", {}) or {})
        out_caps["device"] = override
        out["capabilities"] = out_caps

    return out


def build_load_director(
    *,
    enable_fn: Callable[[str], int],
    disable_fn: Callable[[str], None],
    memory_probe: Any,
) -> "Any":
    """Construct a LoadDirector with config-derived budgets/headroom.

    This is the v0.5x doc-drift fix: `MUSE_GPU_BUDGET_GB`,
    `MUSE_CPU_BUDGET_GB`, `MUSE_GPU_HEADROOM_GB`, `MUSE_CPU_HEADROOM_GB`
    were documented as active env knobs but `LoadDirector.__init__`
    only ever saw its own hardcoded defaults (None, None, 1.0, 2.0)
    because nothing passed them in. Extracted as a standalone factory
    (rather than inlined at the one call site) so the config wiring is
    independently unit-testable without spinning up a full
    SupervisorState.

    Defaults match today's hardcoded LoadDirector.__init__ values, so a
    deployment that sets nothing sees identical behavior; the knobs
    simply start working for operators who do set them.
    """
    from muse.cli_impl.load_director import LoadDirector

    return LoadDirector(
        enable_fn=enable_fn,
        disable_fn=disable_fn,
        memory_probe=memory_probe,
        gpu_budget_gb=config.get("server.gpu_budget_gb"),
        cpu_budget_gb=config.get("server.cpu_budget_gb"),
        gpu_headroom_gb=config.get("server.gpu_headroom_gb"),
        cpu_headroom_gb=config.get("server.cpu_headroom_gb"),
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

    def enable_fn(model_id: str) -> int:
        return load_model_into_worker(model_id, state=state)

    def disable_fn(model_id: str) -> None:
        unload_model_from_worker(model_id, state=state)

    return build_load_director(
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

    # The auto-restart monitor and the idle sweeper share state.stop_event
    # so a single Ctrl+C / SIGTERM unblocks both at once. Allocate a
    # fresh Event for this supervisor lifecycle (replacing the dataclass
    # default) so a re-entered run_supervisor in the same process always
    # gets a clean unset-state event.
    stop_event = threading.Event()
    state.stop_event = stop_event

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

    # Idle-timeout sweeper (v0.40.1). Per-model idle eviction runs on a
    # background thread that shares stop_event with the monitor; the
    # sweeper reads loaded-set entries via the director's public surface
    # and unloads anything past its `capabilities.idle_timeout_seconds`.
    sweep_interval = config.get("server.idle_sweep_interval_seconds")
    # Global default idle timeout, applied to models that declare no
    # per-model capabilities.idle_timeout_seconds. The registry default
    # is 600.0s (v0.5x); an operator who wants the old "never idle-evict"
    # behavior sets MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS=0 (or negative)
    # explicitly -- the "<=0 disables" guard below still applies. A bad
    # / unparseable env value can no longer crash boot: the registry
    # itself warns and falls back to its default (see Config.get).
    _raw_default_idle = config.get("server.idle_timeout_seconds")
    default_idle_timeout: float | None = (
        _raw_default_idle if _raw_default_idle is not None and _raw_default_idle > 0 else None
    )
    sweeper = IdleSweeper(
        director=state.director,
        catalog_lookup=get_manifest,
        interval_seconds=sweep_interval,
        default_idle_timeout_seconds=default_idle_timeout,
        stop_event=stop_event,
    )
    sweeper_thread = sweeper.start()
    state.idle_sweeper = sweeper
    state.idle_sweeper_thread = sweeper_thread
    logger.info(
        "idle sweeper running (interval=%.1fs, default_idle_timeout=%s)",
        sweep_interval,
        f"{default_idle_timeout:.0f}s" if default_idle_timeout else "off",
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
        # run_uvicorn sets a BOUNDED timeout_graceful_shutdown so the first
        # Ctrl-C exits within a fixed window even when a connection lingers
        # (SSE stream / long inference / idle keep-alive). uvicorn.run's
        # default (None) waits forever, stranding port 8000 and forcing the
        # operator to kill the process before restarting.
        run_uvicorn(app, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        # Tell the monitor + idle sweeper to stop BEFORE killing workers.
        # Otherwise the monitor could spawn a restart while we're
        # terminating processes, and the sweeper could try to evict a
        # model whose worker is mid-shutdown.
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        if sweeper_thread is not None:
            sweeper_thread.join(timeout=5.0)
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
