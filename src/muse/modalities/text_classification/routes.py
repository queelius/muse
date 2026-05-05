"""FastAPI routes for /v1/moderations and /v1/text/classifications.

Two routes share this modality:

  POST /v1/moderations
    OpenAI-compat shape; reduces classifier scores to flag/categories.
    Request:  {"input": str | list[str], "model"?: str, "threshold"?: float}
    Response: {"id", "model", "results": [{"flagged", "categories",
               "category_scores"}]}

  POST /v1/text/classifications
    Returns the full label distribution (sentiment, intent, zero-shot,
    etc.). Capability-gated dispatch:
    - candidate_labels present and supports_zero_shot=True ->
        runtime.classify_zero_shot(...)
    - candidate_labels absent and supports_classification=True ->
        runtime.classify(...)
    - mismatch returns 400 with the OpenAI error envelope.
    Request:  {"input": str | list[str], "model"?: str,
               "candidate_labels"?: list[str], "top_k"?: int,
               "multi_label"?: bool}
    Response: {"id", "model", "results": [[{"label", "score"}, ...], ...]}

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.text_classification.codec import (
    encode_classifications, encode_moderations,
    _resolve_safe_labels, _resolve_threshold,
)

# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/classification"


logger = logging.getLogger(__name__)


# Defaults are conservative; tunable via env so a power user with a
# big GPU and a known-safe pipeline can lift them, and a sysadmin
# fronting a public muse can tighten them. The point isn't a perfect
# limit, just a finite one: without these, a single curl with
# `input: [...]` can OOM the worker by trying to materialize a
# multi-million-string batch. Code paths that hit these caps return
# 400 so the client knows the request was rejected (vs. silently
# truncated).
#
# Read per-request so changes via env take effect immediately and so
# tests that patch env don't need to reload the module. Matches the
# v0.34.0 MUSE_IMAGE_INPUT_MAX_BYTES pattern.
def _max_batch() -> int:
    try:
        return int(os.environ.get("MUSE_MODERATIONS_MAX_BATCH", "1024"))
    except ValueError:
        return 1024


def _max_chars_per_item() -> int:
    try:
        return int(
            os.environ.get("MUSE_MODERATIONS_MAX_CHARS_PER_ITEM", "100000")
        )
    except ValueError:
        return 100000


def _max_candidate_labels() -> int:
    try:
        return int(os.environ.get("MUSE_CLASSIFICATIONS_MAX_LABELS", "200"))
    except ValueError:
        return 200


class _ModerationsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    threshold: float | None = None


# /v1/text/classifications batches share the same env caps as
# /v1/moderations (operators don't have to remember per-route knobs);
# additionally, candidate_labels has its own cap because each label
# adds a forward pass through the NLI head.


class _ClassificationsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    candidate_labels: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=1000)
    multi_label: bool = False


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

        items = [req.input] if isinstance(req.input, str) else req.input
        if len(items) > _max_batch():
            return error_response(
                400, "invalid_parameter",
                f"input batch size {len(items)} exceeds "
                f"MUSE_MODERATIONS_MAX_BATCH={_max_batch()}",
            )
        too_long = next(
            (i for i, s in enumerate(items) if len(s) > _max_chars_per_item()),
            None,
        )
        if too_long is not None:
            return error_response(
                400, "invalid_parameter",
                f"input[{too_long}] exceeds "
                f"MUSE_MODERATIONS_MAX_CHARS_PER_ITEM={_max_chars_per_item()}",
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
        # moderation requests on the same worker. Per-backend lock
        # serializes calls into one model without blocking siblings on
        # the same worker (added in v0.34.0 Task A).
        def _call_moderation():
            with backend._inference_lock:
                return backend.classify(req.input)

        try:
            results = await asyncio.to_thread(_call_moderation)
        except Exception as e:  # noqa: BLE001
            logger.exception("moderations classify failed")
            return error_response(500, "internal_error", str(e))
        body = encode_moderations(
            results, model_id=effective_id, threshold=threshold,
            safe_labels=safe_labels,
        )
        return JSONResponse(content=body)

    @router.post("/v1/text/classifications")
    async def classifications(req: _ClassificationsRequest):
        # Input validation mirrors /v1/moderations.
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

        items = [req.input] if isinstance(req.input, str) else req.input
        if len(items) > _max_batch():
            return error_response(
                400, "invalid_parameter",
                f"input batch size {len(items)} exceeds "
                f"MUSE_MODERATIONS_MAX_BATCH={_max_batch()}",
            )
        too_long = next(
            (i for i, s in enumerate(items) if len(s) > _max_chars_per_item()),
            None,
        )
        if too_long is not None:
            return error_response(
                400, "invalid_parameter",
                f"input[{too_long}] exceeds "
                f"MUSE_MODERATIONS_MAX_CHARS_PER_ITEM={_max_chars_per_item()}",
            )

        # Validate candidate_labels shape if present.
        if req.candidate_labels is not None:
            if not req.candidate_labels:
                return error_response(
                    400, "invalid_parameter",
                    "candidate_labels must be a non-empty list when provided",
                )
            if any(not isinstance(s, str) or not s.strip()
                   for s in req.candidate_labels):
                return error_response(
                    400, "invalid_parameter",
                    "candidate_labels must contain non-empty strings",
                )
            if len(req.candidate_labels) > _max_candidate_labels():
                return error_response(
                    400, "invalid_parameter",
                    f"candidate_labels count {len(req.candidate_labels)} "
                    f"exceeds MUSE_CLASSIFICATIONS_MAX_LABELS="
                    f"{_max_candidate_labels()}",
                )

        # Resolve the backend.
        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        effective_id = backend.model_id
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}

        # Capability gate: dispatch by candidate_labels presence.
        if req.candidate_labels is not None:
            if not capabilities.get("supports_zero_shot"):
                return error_response(
                    400, "zero_shot_not_supported",
                    f"model {effective_id!r} does not advertise "
                    f"supports_zero_shot=True; omit candidate_labels or "
                    f"use a zero-shot model",
                )
            if not hasattr(backend, "classify_zero_shot"):
                # Manifest claims supports_zero_shot but the loaded
                # runtime can't actually do it. This is a misconfigured
                # curated entry, not a user error; surface as 500.
                logger.error(
                    "model %s has supports_zero_shot=True but runtime "
                    "lacks classify_zero_shot method",
                    effective_id,
                )
                return error_response(
                    500, "internal_error",
                    f"model {effective_id!r} configured for zero-shot but "
                    f"its runtime does not support classify_zero_shot",
                )

            def _call_zs():
                with backend._inference_lock:
                    return backend.classify_zero_shot(
                        req.input,
                        candidate_labels=req.candidate_labels,
                        multi_label=req.multi_label,
                    )

            try:
                results = await asyncio.to_thread(_call_zs)
            except Exception as e:  # noqa: BLE001
                logger.exception("classify_zero_shot failed")
                return error_response(500, "internal_error", str(e))
        else:
            if not capabilities.get("supports_classification"):
                return error_response(
                    400, "candidate_labels_required",
                    f"model {effective_id!r} does not advertise "
                    f"supports_classification=True; pass candidate_labels "
                    f"or use a fine-tuned classifier",
                )

            def _call_cls():
                with backend._inference_lock:
                    return backend.classify(req.input)

            try:
                results = await asyncio.to_thread(_call_cls)
            except Exception as e:  # noqa: BLE001
                logger.exception("classify failed")
                return error_response(500, "internal_error", str(e))

        body = encode_classifications(
            results, model_id=effective_id, top_k=req.top_k,
        )
        return JSONResponse(content=body)

    return router
