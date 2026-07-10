"""FastAPI routes for /v1/summarize.

Cohere-compat shape:
  request:  {"text": str, "length"?: "short"|"medium"|"long",
             "format"?: "paragraph"|"bullets", "model"?: str}
  response: {"id", "model", "summary", "usage": {...},
             "meta": {"length", "format", ...}}

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
from pydantic import BaseModel, Field

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.text_summarization.codec import (
    encode_summarization_response,
)


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/summarization"


logger = logging.getLogger(__name__)


# Defaults are conservative. A request with text=10MB can OOM the worker
# trying to tokenize it; cap is tunable via env so power users with big
# GPUs can lift it.
#
# Read per-request via muse.core.config so changes take effect without
# a restart (matches the text_classification / text_translation pattern).
def _max_text_chars() -> int:
    return config.get("limits.summarize_max_text_chars")


_VALID_LENGTHS = ("short", "medium", "long")
_VALID_FORMATS = ("paragraph", "bullets")


class _SummarizationRequest(BaseModel):
    text: str
    length: str = Field(default="medium")
    format: str = Field(default="paragraph")
    model: str | None = None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/summarize")
    async def summarize(req: _SummarizationRequest):
        if not req.text:
            return error_response(
                400, "invalid_parameter", "text must not be empty",
            )
        max_text_chars = _max_text_chars()
        if len(req.text) > max_text_chars:
            return error_response(
                400, "invalid_parameter",
                f"text exceeds MUSE_SUMMARIZE_MAX_TEXT_CHARS={max_text_chars}",
            )
        if req.length not in _VALID_LENGTHS:
            return error_response(
                400, "invalid_parameter",
                f"length must be one of {list(_VALID_LENGTHS)}; got {req.length!r}",
            )
        if req.format not in _VALID_FORMATS:
            return error_response(
                400, "invalid_parameter",
                f"format must be one of {list(_VALID_FORMATS)}; got {req.format!r}",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        # backend.summarize is sync (transformers generate); offload so a
        # slow inference doesn't block sibling /health, /v1/models, or
        # other in-flight requests on the same worker. Per-backend lock
        # serializes calls into one model without blocking siblings on
        # the same worker (H2 fix, v0.45.7).
        def _summarize():
            with backend._inference_lock:
                return backend.summarize(req.text, req.length, req.format)

        try:
            result = await asyncio.to_thread(_summarize)
        except Exception:  # noqa: BLE001
            # Log the real exception server-side but never leak it to the
            # client: str(e) can carry internal filesystem paths, CUDA
            # driver text, or other backend-implementation detail.
            logger.exception("summarize failed")
            return error_response(
                500, "internal_error",
                "summarization backend failed; see server logs",
            )
        body = encode_summarization_response(result)
        return JSONResponse(content=body)

    return router
