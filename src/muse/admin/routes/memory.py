"""Memory accounting routes.

GET /v1/admin/memory returns aggregate CPU/GPU usage plus a per-model
breakdown derived from each enabled model's `measurements` block in
catalog.json. The breakdown is grouped by device:
  - GPU: models whose capabilities.device != "cpu" -> read measurements.cuda
  - CPU: models whose capabilities.device == "cpu" -> read measurements.cpu

Live system totals come from psutil (CPU) and pynvml (GPU). Both are
optional: when neither is installed, the corresponding section is null.
"""
from __future__ import annotations

import logging

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
        return {"gpu": gpu, "cpu": cpu}

    return router


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


def _per_model_breakdown(device_key: str, target_device_label: str) -> list[dict]:
    """Return [{model_id, weights_gb, peak_gb}] for currently loaded models
    whose capabilities.device matches the target side (CPU vs GPU).

    `device_key` is the measurements.<key> bucket (cuda/auto/mps/cpu).
    `target_device_label` is "cpu" or "gpu" for capability-side filtering.
    """
    catalog = _read_catalog()
    catalog_known = known_models()
    loaded = _enabled_loaded_model_ids()
    out: list[dict] = []
    for model_id in sorted(loaded):
        entry = catalog.get(model_id) or {}
        cap_device = "auto"
        if model_id in catalog_known:
            cap_device = (catalog_known[model_id].extra or {}).get("device", "auto")
        cap_device = (cap_device or "auto").lower()
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
