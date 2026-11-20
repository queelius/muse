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
`http(s)://...` URL. The route layer decodes them concurrently (at most
8 at a time, all within one 60 s deadline by default; both are
settings) into PIL.Images via the
shared `muse.modalities._common.images.decode_image_input`, batches the
list, and forwards to the registered backend. The response `model` is
the registry id of the serving model.

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
from typing import Any, Union

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.images import (
    ImageFetchError,
    ImageInputError,
    close_decoded_images,
    decode_image_input,
    image_error_response,
    validate_total_image_pixels,
)
from muse.modalities._common.inference import offload_call
from muse.modalities.image_embedding.codec import encode_embedding_list


MODALITY = "image/embedding"

# URL inputs are fetched concurrently, bounded, under one overall deadline
# (each fetch also has its own 30 s timeout).
def _decode_concurrency() -> int:
    """Inputs decoded at once (MUSE_IMAGE_EMBEDDINGS_DECODE_CONCURRENCY)."""
    return int(config.get_positive("limits.image_embeddings_decode_concurrency"))


def _decode_deadline_seconds() -> float:
    """Whole-batch decode deadline (MUSE_IMAGE_EMBEDDINGS_DECODE_DEADLINE_SECONDS)."""
    return float(config.get_positive("limits.image_embeddings_decode_deadline_seconds"))



# Conservative caps: a request with input=[10MB image] x 100 entries
# would OOM the worker decoding the batch. Tunable via env so power
# users with big GPUs can lift them.
#
# Read per-request via muse.core.config so changes take effect without
# a restart (matches the text_classification / text_translation pattern).
def _max_batch() -> int:
    return config.get("limits.image_embeddings_max_batch")


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


async def _decode_all(items: list[str]) -> list[Any]:
    """Decode every input concurrently; closes all of them on any failure.

    Raises the first input's ImageInputError (or any other exception);
    ImageFetchError when the batch misses the overall deadline.
    """
    gate = asyncio.Semaphore(_decode_concurrency())
    deadline = _decode_deadline_seconds()

    async def _one(item: str) -> Any:
        async with gate:
            return await decode_image_input(item)

    tasks = [asyncio.ensure_future(_one(item)) for item in items]

    def _decoded() -> list[Any]:
        return [
            t.result() for t in tasks
            if t.done() and not t.cancelled() and t.exception() is None
        ]

    async def _release() -> None:
        # Stop the rest, then close what was decoded. A decode cancelled
        # mid-flight closes its own image.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        close_decoded_images(_decoded())

    try:
        done, pending = await asyncio.wait(
            tasks, timeout=deadline,
            return_when=asyncio.FIRST_EXCEPTION,
        )
    except BaseException:  # cancelled: the client is gone
        await _release()
        raise
    failed = next(
        (t for t in tasks if t in done and t.exception() is not None), None,
    )
    if failed is None and not pending:
        return [t.result() for t in tasks]
    await _release()
    if failed is not None:
        raise failed.exception()
    raise ImageFetchError(
        f"decoding {len(items)} inputs exceeded the "
        f"{deadline:g} s deadline (MUSE_IMAGE_EMBEDDINGS_DECODE_DEADLINE_SECONDS)"
    )


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/images/embeddings")
    async def images_embeddings(req: _ImageEmbeddingsRequest):
        items = [req.input] if isinstance(req.input, str) else req.input
        max_batch = _max_batch()
        if len(items) > max_batch:
            return error_response(
                400, "invalid_parameter",
                f"input batch size {len(items)} exceeds "
                f"MUSE_IMAGE_EMBEDDINGS_MAX_BATCH={max_batch}",
            )

        backend = registry.get_or_404(MODALITY, req.model)

        # One bad input nukes the whole batch (consistent with
        # /v1/embeddings).
        try:
            images = await _decode_all(items)
        except ImageInputError as e:
            return image_error_response(e)
        try:
            validate_total_image_pixels(images)
        except ImageInputError as e:
            close_decoded_images(images)
            return image_error_response(e)

        result = await offload_call(
            lambda: backend.embed(images, dimensions=req.dimensions),
            lock=backend._inference_lock,
            inputs=images,
        )

        # The registry id, not whatever id the runtime reports.
        return encode_embedding_list(
            result.embeddings, model_id=backend.model_id,
            encoding_format=req.encoding_format,
        )

    return router
