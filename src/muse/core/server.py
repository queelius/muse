"""FastAPI application factory.

Modality routers are mounted via `create_app(registry, routers=...)`.
Each modality supplies its own APIRouter; core adds /health and /v1/models
(aggregated across modalities).

Each /v1/models entry carries the lazy-load fields `loaded`,
`last_loaded_at` and `unservable_reason`. This app runs in a worker, which
reports its registered models as loaded with null timestamp and reason;
the gateway joins the real load timestamp from its LoadDirector.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from muse.core.errors import ModelNotFoundError, error_response
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

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        # Finding 2 (v0.58.1 review): several routes call a backend
        # without wrapping the call in their own try/except, so a backend
        # exception used to escape as FastAPI's bare
        # `{"detail": "Internal Server Error"}` -- neither the OpenAI
        # envelope shape muse documents nor a shape that avoids leaking
        # exception text. This catch-all is the structural fix: log the
        # real exception server-side (path/CUDA/model detail is
        # operationally useful) and return a generic OpenAI-shape 500 to
        # every client, regardless of which route raised.
        #
        # `ModelNotFoundError` (an HTTPException subclass) and
        # `RequestValidationError` are handled by their own
        # `@app.exception_handler` above; Starlette's ExceptionMiddleware
        # matches those specific handlers before an unhandled exception
        # ever reaches this generic one, so this handler only ever sees
        # genuinely unhandled backend/runtime failures.
        logger.exception(
            "unhandled exception in %s %s", request.method, request.url.path,
        )
        return error_response(
            500, "internal_error",
            "internal server error; see server logs",
        )

    @app.get("/health")
    def health():
        body = {
            "status": "ok",
            "modalities": registry.modalities(),
            "models": [info.model_id for info in registry.list_all()],
        }
        # A backend may expose `usable`; False means its native state is
        # uncertain (e.g. a native call failed mid-inference). Reporting
        # 503 lets the supervisor's monitor replace this worker instead of
        # serving from that state.
        unusable = [
            info.model_id for info in registry.list_all()
            if getattr(registry.get(info.modality, info.model_id), "usable", True) is False
        ]
        if unusable:
            body["status"] = "unusable"
            body["unusable_models"] = unusable
            return JSONResponse(status_code=503, content=body)
        return body

    @app.get("/v1/models")
    def list_models():
        # A worker's registry holds only models it has loaded, so every
        # entry is loaded. The gateway (which owns the LoadDirector) fills
        # in `last_loaded_at` and lists enabled-but-unloaded catalog
        # models; `unservable_reason` is a supervisor verdict, null here.
        data = [
            build_model_entry(
                info.model_id, info.modality, info.manifest,
                loaded=True, last_loaded_at=None, unservable_reason=None,
            )
            for info in registry.list_all()
        ]
        return {"object": "list", "data": data}

    for name, router in routers.items():
        logger.info("mounting modality router %s", name)
        app.include_router(router)

    app.state.registry = registry
    return app


def build_model_entry(
    model_id: str,
    modality: str,
    manifest: Mapping[str, Any],
    *,
    loaded: bool,
    last_loaded_at: str | None,
    unservable_reason: str | None,
    revision: str | None = None,
) -> dict:
    """Shape one /v1/models entry from a manifest + lazy-load state.

    Capabilities (sample_rate, voices, default_size, ...) and select
    top-level manifest metadata are splatted first; the authoritative
    fields (id / modality / object + the lazy-load surface) are written
    last so nothing in the manifest or capabilities can clobber them.

    Shared by `create_app`'s /v1/models (per-worker registry view) and the
    gateway's /v1/models (which also lists enabled-but-unloaded catalog
    models) so both render an identical entry shape.
    """
    entry: dict = {}
    entry.update(manifest.get("capabilities", {}) or {})
    for k in ("description", "license", "hf_repo"):
        if k in manifest:
            entry[k] = manifest[k]
    entry["id"] = model_id
    entry["modality"] = modality
    entry["object"] = "model"
    # OpenAI-compat fields. `created` is a unix epoch the SDK expects on
    # every model object; muse has no per-model creation time, so we emit
    # a stable 0 rather than a wall-clock now() (which would be
    # nondeterministic and meaningless). `owned_by` is the HF org slug
    # (the part before the '/'), falling back to "muse" for bundled
    # custom models that declare no hf_repo. Written after the splat so
    # capabilities can't clobber them.
    entry["created"] = 0
    hf_repo = manifest.get("hf_repo")
    entry["owned_by"] = (
        hf_repo.split("/", 1)[0] if isinstance(hf_repo, str) and "/" in hf_repo
        else "muse"
    )
    entry["loaded"] = loaded
    entry["last_loaded_at"] = last_loaded_at
    entry["unservable_reason"] = unservable_reason
    # Written last so a manifest capability named "revision" cannot win.
    entry["revision"] = revision
    return entry


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
