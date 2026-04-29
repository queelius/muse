"""FastAPI routes for /v1/images/embeddings.

Wire shape mirrors /v1/embeddings:
  request:  {"input": str | list[str], "model"?: str,
             "encoding_format"?: "float" | "base64",
             "dimensions"?: int, "user"?: str}
  response: {"object": "list",
             "data": [{"object": "embedding", "embedding": ..., "index": i}],
             "model": str,
             "usage": {"prompt_tokens": 0, "total_tokens": 0}}

Each `input` entry must be a `data:image/...;base64,...` URL or an
`http(s)://...` URL. The route layer decodes each into a PIL.Image via
the shared `decode_image_input` helper from the image_generation
modality, batches the list, and forwards to the registered backend.

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging
import os
from threading import Lock
from typing import Union

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.image_embedding.codec import embedding_to_base64
from muse.modalities.image_generation.image_input import decode_image_input


logger = logging.getLogger(__name__)


MODALITY = "image/embedding"
_inference_lock = Lock()


# Conservative caps: a request with input=[10MB image] x 100 entries
# would OOM the worker decoding the batch. Tunable via env so power
# users with big GPUs can lift them.
_MAX_BATCH = int(os.environ.get("MUSE_IMAGE_EMBEDDINGS_MAX_BATCH", "64"))


class _ImageEmbeddingsRequest(BaseModel):
    """OpenAI-shape request for /v1/images/embeddings.

    `input` is either a single image URL (data: or http(s)://) or a
    list of them. Pydantic v2's Union accepts both shapes; non-empty
    is enforced in a validator below.
    """
    input: Union[str, list[str]]
    model: str | None = None
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: int | None = Field(default=None, ge=1, le=4096)
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
    router = APIRouter()

    @router.post("/v1/images/embeddings")
    async def images_embeddings(req: _ImageEmbeddingsRequest):
        items = [req.input] if isinstance(req.input, str) else req.input
        if len(items) > _MAX_BATCH:
            return error_response(
                400, "invalid_parameter",
                f"input batch size {len(items)} exceeds "
                f"MUSE_IMAGE_EMBEDDINGS_MAX_BATCH={_MAX_BATCH}",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        # Decode each input entry into a PIL.Image. Failures surface as
        # 400s with the OpenAI-shape error envelope; one bad input
        # nukes the whole batch (consistent with /v1/embeddings).
        try:
            images = [decode_image_input(item) for item in items]
        except ValueError as e:
            return error_response(
                400, "invalid_parameter",
                f"image decode failed: {e}",
            )

        # backend.embed is sync (transformers forward); offload so a slow
        # inference doesn't block sibling /health, /v1/models, or other
        # in-flight requests on the same worker.
        def _call():
            with _inference_lock:
                return backend.embed(images, dimensions=req.dimensions)

        result = await asyncio.to_thread(_call)

        data = []
        for i, vec in enumerate(result.embeddings):
            embedding_field = (
                embedding_to_base64(vec)
                if req.encoding_format == "base64"
                else vec
            )
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
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return router
