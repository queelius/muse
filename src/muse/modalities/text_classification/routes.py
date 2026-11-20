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

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.inference import cap, check_text_batch, run_locked
from muse.modalities.text_classification.codec import (
    encode_classifications, encode_moderations,
    _resolve_safe_labels, _resolve_threshold,
)

# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/classification"


logger = logging.getLogger(__name__)


# Finite caps (tunable via env / config.yaml, read per request) so one
# request cannot OOM the worker with a multi-million-string batch; a
# request over a cap gets a 400 rather than being truncated.
_MAX_BATCH_KEY = "limits.moderations_max_batch"
_MAX_CHARS_KEY = "limits.moderations_max_chars_per_item"
_MAX_LABELS_KEY = "limits.classifications_max_labels"

# Zero-shot runs one NLI forward pass per (input, label) pair under the
# model lock; bound the product, not just each factor.
MAX_ZERO_SHOT_PAIRS = 16_384


def _validate_input(value: str | list[str]) -> JSONResponse | None:
    if isinstance(value, str) and not value:
        return error_response(
            400, "invalid_parameter", "input must not be empty",
        )
    if isinstance(value, list) and (not value or any(not s for s in value)):
        return error_response(
            400, "invalid_parameter",
            "input must be a non-empty string or list of non-empty strings",
        )
    items = [value] if isinstance(value, str) else value
    return check_text_batch(
        items, max_batch_key=_MAX_BATCH_KEY, max_chars_key=_MAX_CHARS_KEY,
    )


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

        invalid = _validate_input(req.input)
        if invalid is not None:
            return invalid

        backend = registry.get_or_404(MODALITY, req.model)

        # Resolve effective model_id for the response envelope. Prefer
        # the backend's own model_id (set on instantiation) so that the
        # response reflects what actually answered, not the request's
        # model field which may have been None.
        effective_id = backend.model_id
        capabilities = registry.capabilities(MODALITY, effective_id)
        threshold = _resolve_threshold(req.threshold, capabilities)
        safe_labels = _resolve_safe_labels(capabilities)
        # A zero-shot pipeline has no classify(); say so instead of a 500.
        # A manifest that omits the flag (older pulls) still dispatches.
        if capabilities.get("supports_classification") is False:
            return error_response(
                400, "classification_not_supported",
                f"model {effective_id!r} declares "
                f"supports_classification=False (a zero-shot model); use "
                f"/v1/text/classifications with candidate_labels",
            )
        if not hasattr(backend, "classify"):
            return error_response(
                400, "classification_not_supported",
                f"model {effective_id!r}'s runtime has no classify() method; "
                f"use /v1/text/classifications with candidate_labels",
            )

        results = await run_locked(
            backend, lambda: backend.classify(req.input),
            label="moderations classify",
            message="moderation backend failed; see server logs",
        )
        if isinstance(results, JSONResponse):
            return results
        body = encode_moderations(
            results, model_id=effective_id, threshold=threshold,
            safe_labels=safe_labels,
        )
        return JSONResponse(content=body)

    @router.post("/v1/text/classifications")
    async def classifications(req: _ClassificationsRequest):
        invalid = _validate_input(req.input)
        if invalid is not None:
            return invalid
        n_items = 1 if isinstance(req.input, str) else len(req.input)

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
            max_labels = cap(_MAX_LABELS_KEY)
            if len(req.candidate_labels) > max_labels:
                return error_response(
                    400, "invalid_parameter",
                    f"candidate_labels count {len(req.candidate_labels)} "
                    f"exceeds MUSE_CLASSIFICATIONS_MAX_LABELS={max_labels}",
                )
            pairs = n_items * len(req.candidate_labels)
            if pairs > MAX_ZERO_SHOT_PAIRS:
                return error_response(
                    400, "invalid_parameter",
                    f"inputs x candidate_labels = {pairs} exceeds the "
                    f"zero-shot work limit of {MAX_ZERO_SHOT_PAIRS} pairs",
                )

        # Resolve the backend.
        backend = registry.get_or_404(MODALITY, req.model)

        effective_id = backend.model_id
        capabilities = registry.capabilities(MODALITY, effective_id)

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

            results = await run_locked(
                backend,
                lambda: backend.classify_zero_shot(
                    req.input,
                    candidate_labels=req.candidate_labels,
                    multi_label=req.multi_label,
                ),
                label="classify_zero_shot",
                message=(
                    "zero-shot classification backend failed; see server logs"
                ),
            )
        else:
            if not capabilities.get("supports_classification"):
                return error_response(
                    400, "candidate_labels_required",
                    f"model {effective_id!r} does not advertise "
                    f"supports_classification=True; pass candidate_labels "
                    f"or use a fine-tuned classifier",
                )

            results = await run_locked(
                backend, lambda: backend.classify(req.input),
                label="classify",
                message="classification backend failed; see server logs",
            )
        if isinstance(results, JSONResponse):
            return results

        body = encode_classifications(
            results, model_id=effective_id, top_k=req.top_k,
        )
        return JSONResponse(content=body)

    return router
