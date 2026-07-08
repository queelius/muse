"""Memory accounting routes.

GET /v1/admin/memory returns aggregate CPU/GPU usage plus a per-model
breakdown derived from each enabled model's `measurements` block in
catalog.json. The breakdown is grouped by device:
  - GPU: models whose effective device is cuda/mps -> read measurements.cuda
  - CPU: models whose effective device is cpu -> read measurements.cpu

A model declaring device="auto" (the default for CUDA-safe bundled
models) is resolved to the side it actually loads on, mirroring the
runtime's select_device("auto"): the supervisor's --device flag when set,
else live GPU detection. So an auto-device model shows under GPU on a GPU
host and under CPU on a CPU-only host.

Live system totals come from psutil (CPU) and pynvml (GPU). Both are
optional: when neither is installed, the corresponding section is null.

`recent_decisions` (v0.40.0+) surfaces the LoadDirector's last 20
load/evict decisions for operator visibility into lazy-load behavior.
Empty list when no director is bound (supervisor not booted, or running
the gateway in isolation for tests).

Each per-model breakdown entry also carries `refcount` (v0.54.4+) when a
LoadDirector is bound: the live in-flight refcount from the director's
loaded set. The eviction candidate filter is `refcount == 0`, so this is
the field that decides idle-evictable (0) vs pinned by an in-flight
request (> 0). It lets an operator VERIFY a 503 that reports "no evictable
candidates (all loaded models have refcount > 0)" against reality rather
than trusting the message. The key is omitted (not 0) when no director is
bound, so "unknown" is never mistaken for "evictable".
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from muse.cli_impl.supervisor import get_supervisor_state
from muse.core.catalog import _read_catalog, known_models

logger = logging.getLogger(__name__)


def build_memory_router() -> APIRouter:
    router = APIRouter()

    @router.get("/memory")
    def memory_status():
        gpu = _gpu_summary()
        cpu = _cpu_summary()
        return {
            "gpu": gpu,
            "cpu": cpu,
            "recent_decisions": _recent_decisions(),
        }

    return router


def _recent_decisions() -> list[dict]:
    """Return the LoadDirector's recent decisions as JSON-serializable dicts.

    Returns an empty list when no director is bound to the supervisor
    state (supervisor not yet booted, or test harness driving routes
    in isolation). The deque is iterated under the director's lock so
    a concurrent append cannot interleave with our snapshot.

    Each DecisionLogEntry's `timestamp` field is wall-clock seconds
    (time.time(), recorded inside the director). We convert it to an
    ISO-8601 string (UTC) here so the wire shape is operator-friendly
    and round-trips through JSON cleanly.
    """
    state = get_supervisor_state()
    director = state.director
    if director is None:
        return []

    out: list[dict] = []
    with director.lock:
        for entry in director.recent_decisions:
            ts = entry.timestamp
            iso = (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if isinstance(ts, (int, float))
                else str(ts)
            )
            out.append({
                "timestamp": iso,
                "model_id": entry.model_id,
                "action": entry.action,
                "memory_gb": entry.memory_gb,
                "free_before_gb": entry.free_before_gb,
                "free_after_gb": entry.free_after_gb,
                "reason": entry.reason,
                "evicted": list(entry.evicted),
            })
    return out


def _enabled_loaded_model_ids() -> set[str]:
    """Models that should count toward live memory use.

    Counts everything currently hosted by a worker (state.workers).
    Disabled-but-loaded shouldn't happen with the current state model,
    but we err on the side of describing what's actually consuming
    memory rather than what's marked enabled in the catalog.
    """
    state = get_supervisor_state()
    with state.lock:
        ids = set()
        for spec in state.workers:
            ids.update(spec.models)
    return ids


def _loaded_refcounts() -> dict[str, int]:
    """Live per-model refcount from the LoadDirector's loaded set.

    The eviction candidate filter is `refcount == 0`, so this is the field
    that decides whether a loaded model is idle-evictable (0) or pinned by
    an in-flight request (> 0). Surfacing it lets an operator VERIFY a 503
    that reports "no evictable candidates (all loaded models have refcount
    > 0)" against reality instead of taking the (hardcoded) message at its
    word. Read under the director lock so a concurrent acquire/release
    can't interleave. Returns {} when no director is bound (gateway in
    isolation, or pre-boot), which the caller treats as "unknown" -> the
    refcount key is omitted rather than falsely reported as 0.
    """
    state = get_supervisor_state()
    director = getattr(state, "director", None)
    if director is None:
        return {}
    with director.lock:
        return {
            model_id: entry.refcount
            for model_id, entry in director.loaded.items()
        }


def _resolve_auto_side() -> str:
    """Which memory side ('cpu' | 'gpu') an auto-device model loads on.

    Mirrors the runtime's select_device('auto'): a model with
    device='auto' (the default for CUDA-safe bundled models) loads on the
    GPU when one is available, else CPU. We honor the supervisor's
    --device flag first (the operator's explicit choice), then fall back
    to live GPU detection for an 'auto' supervisor. 'mps' groups with the
    CPU side because the GPU summary is pynvml/CUDA-only.
    """
    state = get_supervisor_state()
    dev = (getattr(state, "device", None) or "auto").lower()
    if dev in ("cuda", "gpu"):
        return "gpu"
    if dev in ("cpu", "mps"):
        return "cpu"
    try:
        from muse.core import memory_probe
        return "gpu" if memory_probe.gpu_free_gb() is not None else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _per_model_breakdown(device_key: str, target_device_label: str) -> list[dict]:
    """Return [{model_id, weights_gb, peak_gb}] for currently loaded models
    whose capabilities.device matches the target side (CPU vs GPU).

    `device_key` is the measurements.<key> bucket (cuda/auto/mps/cpu).
    `target_device_label` is "cpu" or "gpu" for capability-side filtering.
    """
    catalog = _read_catalog()
    catalog_known = known_models()
    loaded = _enabled_loaded_model_ids()
    refcounts = _loaded_refcounts()
    out: list[dict] = []
    for model_id in sorted(loaded):
        entry = catalog.get(model_id) or {}
        cap_device = "auto"
        if model_id in catalog_known:
            cap_device = (catalog_known[model_id].extra or {}).get("device", "auto")
        cap_device = (cap_device or "auto").lower()
        if cap_device == "auto":
            # Resolve to the side this model actually loads on so an
            # auto-device model is accounted under GPU on a GPU host and
            # CPU on a CPU-only host (mirrors runtime select_device).
            cap_device = "cpu" if _resolve_auto_side() == "cpu" else "cuda"
        on_cpu = cap_device == "cpu"
        if target_device_label == "gpu" and on_cpu:
            continue
        if target_device_label == "cpu" and not on_cpu:
            continue
        measurements = entry.get("measurements") or {}
        m = measurements.get(device_key) or measurements.get("auto") or {}
        weights_b = m.get("weights_bytes") or 0
        peak_b = m.get("peak_bytes") or 0
        annotation = (catalog_known[model_id].extra or {}).get("memory_gb") if (
            model_id in catalog_known
        ) else None
        record: dict = {"model_id": model_id}
        if weights_b:
            record["weights_gb"] = round(weights_b / (1024**3), 3)
        if peak_b:
            record["peak_gb"] = round(peak_b / (1024**3), 3)
        if not weights_b and annotation is not None:
            try:
                record["annotated_gb"] = float(annotation)
            except (TypeError, ValueError):
                pass
        if model_id in refcounts:
            record["refcount"] = refcounts[model_id]
        out.append(record)
    return out


def _gpu_summary() -> dict | None:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            return {
                "device": "cuda:0",
                "used_gb": round(used_gb, 2),
                "total_gb": round(total_gb, 2),
                "headroom_gb": round(total_gb - used_gb, 2),
                "models": _per_model_breakdown("cuda", "gpu"),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001
        logger.debug("pynvml failed: %s", e)
        return None


def _cpu_summary() -> dict | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / (1024**3)
        total_gb = vm.total / (1024**3)
        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "models": _per_model_breakdown("cpu", "cpu"),
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("psutil failed: %s", e)
        return None
