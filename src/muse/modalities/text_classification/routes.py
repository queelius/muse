"""FastAPI routes for /v1/moderations.

OpenAI-compat shape:
  request:  {"input": str | list[str], "model"?: str, "threshold"?: float}
  response: {"id", "model", "results": [{"flagged", "categories", "category_scores"}]}

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.text_classification.codec import (
    encode_moderations, _resolve_threshold, _resolve_safe_labels,
)

# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/classification"


logger = logging.getLogger(__name__)


class _ModerationsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    threshold: float | None = None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/moderations")
    async def moderations(req: _ModerationsRequest):
        if req.threshold is not None and not (0.0 <= req.threshold <= 1.0):
            return error_response(
                400, "invalid_parameter",
                f"threshold must be in [0, 1]; got {req.threshold}",
            )

        if isinstance(req.input, str) and not req.input:
            return error_response(
                400, "invalid_parameter", "input must not be empty",
            )
        if isinstance(req.input, list) and (
            not req.input or any(not s for s in req.input)
        ):
            return error_response(
                400, "invalid_parameter",
                "input must be a non-empty string or list of non-empty strings",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        # Resolve effective model_id for the response envelope. Prefer
        # the backend's own model_id (set on instantiation) so that the
        # response reflects what actually answered, not the request's
        # model field which may have been None.
        effective_id = backend.model_id
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}
        threshold = _resolve_threshold(req.threshold, capabilities)
        safe_labels = _resolve_safe_labels(capabilities)

        # backend.classify is sync (transformers pipeline); offload to a
        # thread so a slow inference doesn't block the event loop and
        # starve sibling /health, /v1/models, or other in-flight
        # moderation requests on the same worker.
        results = await asyncio.to_thread(backend.classify, req.input)
        body = encode_moderations(
            results, model_id=effective_id, threshold=threshold,
            safe_labels=safe_labels,
        )
        return JSONResponse(content=body)

    return router
