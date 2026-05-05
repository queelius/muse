"""FastAPI application factory.

Modality routers are mounted via `create_app(registry, routers=...)`.
Each modality supplies its own APIRouter; core adds /health and /v1/models
(aggregated across modalities).

v0.40.0 added three lazy-load surface fields to each /v1/models entry:
`loaded`, `last_loaded_at`, and `unservable_reason`. They are sourced
from the running supervisor's `SupervisorState` when one is registered
(via `muse.cli_impl.supervisor.set_supervisor_state`), with safe
defaults when no supervisor is bound (workers running in isolation,
unit tests, single-worker debug mode).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)


def create_app(
    *,
    registry: ModalityRegistry,
    routers: Mapping[str, APIRouter],
    title: str = "Muse",
) -> FastAPI:
    """Build a FastAPI app with shared /health + /v1/models endpoints.

    `routers` maps modality-name → APIRouter. Each router is mounted
    with its own internal paths (e.g. /v1/audio/speech).
    """
    app = FastAPI(title=title)

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": details or str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "modalities": registry.modalities(),
            "models": [info.model_id for info in registry.list_all()],
        }

    @app.get("/v1/models")
    def list_models():
        # v0.40.0: lazy-load surface. director_status is keyed by
        # model_id and either contains a {"loaded": True, ...} entry or
        # is missing the model entirely (unloaded). When the supervisor
        # state is empty (worker running in isolation), every registered
        # model is treated as loaded with no timestamp -- the worker has
        # the model in its registry by definition.
        director_status, unservable_reasons = _supervisor_view()
        data = []
        for info in registry.list_all():
            entry: dict = {}
            # Splat capabilities (sample_rate, voices, default_size, ...) first
            entry.update(info.manifest.get("capabilities", {}))
            # Then top-level manifest metadata when present
            for k in ("description", "license", "hf_repo"):
                if k in info.manifest:
                    entry[k] = info.manifest[k]
            # Authoritative fields written last so nothing in the manifest
            # or capabilities can clobber id/modality/object.
            entry["id"] = info.model_id
            entry["modality"] = info.modality
            entry["object"] = "model"
            # v0.40.0 lazy-load fields. Always present on every entry so
            # SDKs / clients can rely on the shape.
            loaded_meta = director_status.get(info.model_id)
            if loaded_meta is None and director_status is _NO_DIRECTOR:
                # No supervisor wiring: treat the worker's own registry
                # as the source of truth -- if it's registered, it's
                # loaded -- but we cannot derive a meaningful
                # last_loaded_at because no monotonic baseline exists.
                entry["loaded"] = True
                entry["last_loaded_at"] = None
            elif loaded_meta is not None and loaded_meta.get("loaded"):
                entry["loaded"] = True
                entry["last_loaded_at"] = _format_loaded_at(loaded_meta)
            else:
                # The director is bound and reports this model as not
                # currently loaded.
                entry["loaded"] = False
                entry["last_loaded_at"] = None
            entry["unservable_reason"] = unservable_reasons.get(info.model_id)
            data.append(entry)
        return {"object": "list", "data": data}

    for name, router in routers.items():
        logger.info("mounting modality router %s", name)
        app.include_router(router)

    app.state.registry = registry
    return app


# Sentinel marking "no supervisor state was registered, treat registered
# = loaded as the only available signal." Distinguished from an empty
# dict so we don't conflate "supervisor present, nothing loaded" with
# "no supervisor wiring at all."
_NO_DIRECTOR: dict[str, Any] = {}


def _supervisor_view() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return (director_status, unservable_reasons) for /v1/models.

    The lookup is lazy and best-effort: if the cli_impl.supervisor
    module isn't importable (e.g. during a partial-install lint run) or
    no SupervisorState has been registered, we fall back to the
    "no director" sentinel + empty unservable_reasons. The /v1/models
    handler interprets that pair as "treat every registered model as
    loaded with no timestamp" -- the right semantic for a worker
    running outside a supervisor.

    Director status is sourced from `director.status()` and enriched
    in-place with `loaded_at` read directly from `director.loaded`
    under the director lock. The director's status() shape is fixed
    (worker_port + last_touched_at + refcount) and shared with admin
    routes; we read the LoadEntry's loaded_at separately so /v1/models
    can render `last_loaded_at` without changing the status() contract.
    """
    try:
        from muse.cli_impl.supervisor import get_supervisor_state
    except Exception:  # noqa: BLE001
        return _NO_DIRECTOR, {}
    try:
        state = get_supervisor_state()
    except Exception:  # noqa: BLE001
        return _NO_DIRECTOR, {}
    director = getattr(state, "director", None)
    if director is None:
        # Supervisor state exists but no director was bound (e.g. the
        # supervisor is mid-bringup, or running the gateway in a unit
        # test). Treat as no-director so registered models still report
        # loaded=True. The unservable_reasons map can still be honored
        # because it's catalog-derived, not director-derived.
        reasons = dict(getattr(state, "unservable_reasons", {}) or {})
        return _NO_DIRECTOR, reasons
    try:
        raw_status = director.status() or {}
    except Exception:  # noqa: BLE001
        # A transient director.status() failure must not flip every
        # registered model to loaded=False -- that would mislead clients
        # into thinking the worker has unloaded everything when in
        # reality the director just had a hiccup. Fall back to the
        # no-director sentinel so the handler's "registered = loaded"
        # branch fires for each model.
        reasons = dict(getattr(state, "unservable_reasons", {}) or {})
        return _NO_DIRECTOR, reasons
    # Defensive copy: status() may return a cached/internal dict in some
    # implementations. We mutate per-entry below to insert loaded_at, so
    # take ownership of the outer dict + each inner dict before writing.
    director_status: dict[str, dict[str, Any]] = {
        mid: dict(meta) if isinstance(meta, dict) else meta
        for mid, meta in raw_status.items()
    }
    # Enrich with loaded_at from the underlying LoadEntry. Reading
    # `director.loaded` under the director lock keeps the snapshot
    # consistent with the status() return; status() releases its lock
    # between calls so a concurrent eviction could remove the entry,
    # but that's fine -- the dict.get returns None and we render
    # last_loaded_at: null.
    try:
        with director.lock:
            for mid, meta in director_status.items():
                if not isinstance(meta, dict):
                    continue
                entry = director.loaded.get(mid)
                if entry is not None:
                    meta["loaded_at"] = getattr(entry, "loaded_at", None)
    except Exception:  # noqa: BLE001
        # Director without `lock` or `loaded` attrs (e.g. a test fake):
        # leave director_status as-is. Tests can pre-populate loaded_at
        # in their fake status() if they need it asserted.
        pass
    reasons = dict(getattr(state, "unservable_reasons", {}) or {})
    return director_status, reasons


def _format_loaded_at(loaded_meta: dict[str, Any]) -> str | None:
    """Render LoadEntry.loaded_at (monotonic seconds) as ISO-8601 UTC.

    monotonic time has no fixed origin; the conversion to wall-clock
    uses the offset between current monotonic and current wall-clock,
    which is stable enough for a short-lived UI render. For audit
    purposes the exact timestamp is approximate (tens of milliseconds
    accuracy); for "is this fresh or old?" UX it's plenty.

    Returns None when no loaded_at is present (older director schemas
    or non-loaded entries).
    """
    loaded_at_mono = loaded_meta.get("loaded_at")
    if loaded_at_mono is None:
        return None
    try:
        loaded_at_mono = float(loaded_at_mono)
    except (TypeError, ValueError):
        return None
    # Convert monotonic offset to wall-clock by subtracting the elapsed
    # monotonic from current wall-clock.
    delta = time.monotonic() - loaded_at_mono
    wall_when_loaded = time.time() - delta
    try:
        return datetime.fromtimestamp(
            wall_when_loaded, tz=timezone.utc,
        ).isoformat()
    except (OSError, ValueError, OverflowError):
        return None
