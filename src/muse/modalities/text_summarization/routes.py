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

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.inference import cap, run_locked
from muse.modalities.text_summarization.codec import (
    encode_summarization_response,
)


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/summarization"


logger = logging.getLogger(__name__)


# Finite cap: a 10 MB text could OOM the worker in the tokenizer. Tunable
# via env / config.yaml; read per request with config.get_positive.
_MAX_TEXT_CHARS_KEY = "limits.summarize_max_text_chars"


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
        max_text_chars = cap(_MAX_TEXT_CHARS_KEY)
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

        backend = registry.get_or_404(MODALITY, req.model)

        result = await run_locked(
            backend, lambda: backend.summarize(req.text, req.length, req.format),
            label="summarize",
            message="summarization backend failed; see server logs",
        )
        if isinstance(result, JSONResponse):
            return result
        body = encode_summarization_response(result)
        return JSONResponse(content=body)

    return router
