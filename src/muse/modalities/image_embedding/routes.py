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

Stub in Task A so importing the package + discovery works; replaced in
Task C with the full implementation.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


# MODALITY defined locally to avoid the __init__ circular import; sibling
# modalities all do this.
MODALITY = "image/embedding"


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Stub router; the real implementation lands in Task C."""
    return APIRouter()
