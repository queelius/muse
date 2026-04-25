"""FastAPI router for /v1/embeddings.

Follows OpenAI's /v1/embeddings contract:
  - `input`: str | list[str] (required, at least one non-empty)
  - `model`: str (optional; uses modality default if absent)
  - `encoding_format`: "float" (default) | "base64"
  - `dimensions`: int (optional; backend-dependent truncation)
  - `user`: str (optional; accepted for compat, ignored)

Response shape:
  {
    "object": "list",
    "data": [
      {"object": "embedding", "embedding": [...] | "base64...", "index": 0},
      ...
    ],
    "model": "...",
    "usage": {"prompt_tokens": N, "total_tokens": N}
  }
"""
from __future__ import annotations

import asyncio
import logging
import os
from threading import Lock
from typing import Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.embedding_text.codec import embedding_to_base64

logger = logging.getLogger(__name__)

MODALITY = "embedding/text"
_inference_lock = Lock()

# See text_classification/routes for the rationale (finite caps to
# prevent worker OOM on giant batches). Embedding caps are higher
# because RAG ingestion legitimately wants thousands at a time, but
# they're still finite.
_MAX_BATCH = int(os.environ.get("MUSE_EMBEDDINGS_MAX_BATCH", "2048"))
_MAX_CHARS_PER_ITEM = int(
    os.environ.get("MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM", "100000")
)


class EmbeddingsRequest(BaseModel):
    # Union types in pydantic v2 accept str OR list[str].
    # We validate non-emptiness below.
    input: Union[str, list[str]]
    model: str | None = None
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: int | None = Field(default=None, ge=1, le=8192)
    user: str | None = None  # OpenAI compat; ignored

    @field_validator("input")
    @classmethod
    def _input_nonempty(cls, v):
        if isinstance(v, str):
            if not v:
                raise ValueError("input string cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("input list cannot be empty")
            if any(not isinstance(s, str) or not s for s in v):
                raise ValueError("input list must contain non-empty strings")
        return v


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["embedding/text"])

    @router.post("/embeddings")
    async def embeddings(req: EmbeddingsRequest):
        items = [req.input] if isinstance(req.input, str) else req.input
        if len(items) > _MAX_BATCH:
            return error_response(
                400, "invalid_parameter",
                f"input batch size {len(items)} exceeds "
                f"MUSE_EMBEDDINGS_MAX_BATCH={_MAX_BATCH}",
            )
        too_long = next(
            (i for i, s in enumerate(items) if len(s) > _MAX_CHARS_PER_ITEM),
            None,
        )
        if too_long is not None:
            return error_response(
                400, "invalid_parameter",
                f"input[{too_long}] exceeds "
                f"MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM={_MAX_CHARS_PER_ITEM}",
            )

        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        def _call():
            with _inference_lock:
                return model.embed(req.input, dimensions=req.dimensions)

        result = await asyncio.to_thread(_call)

        data = []
        for i, vec in enumerate(result.embeddings):
            if req.encoding_format == "base64":
                embedding_field = embedding_to_base64(vec)
            else:
                embedding_field = vec
            data.append({
                "object": "embedding",
                "embedding": embedding_field,
                "index": i,
            })

        return {
            "object": "list",
            "data": data,
            "model": result.model_id,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "total_tokens": result.prompt_tokens,
            },
        }

    return router
