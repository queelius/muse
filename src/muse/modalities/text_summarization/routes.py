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
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
_MAX_TEXT_CHARS = int(os.environ.get("MUSE_SUMMARIZE_MAX_TEXT_CHARS", "100000"))


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
        if len(req.text) > _MAX_TEXT_CHARS:
            return error_response(
                400, "invalid_parameter",
                f"text exceeds MUSE_SUMMARIZE_MAX_TEXT_CHARS={_MAX_TEXT_CHARS}",
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
        # other in-flight requests on the same worker.
        result = await asyncio.to_thread(
            backend.summarize, req.text, req.length, req.format,
        )
        body = encode_summarization_response(result)
        return JSONResponse(content=body)

    return router
