"""FastAPI routes for /v1/audio/embeddings.

Wire shape mirrors /v1/audio/transcriptions for the request side
(multipart/form-data with a `file` part) and /v1/embeddings for the
response envelope:

  request:  multipart/form-data with `file` (one or more), `model`,
            `encoding_format`, `user`
  response: {"object": "list",
             "data": [{"object": "embedding", "embedding": ..., "index": i}],
             "model": str,
             "usage": {"prompt_tokens": 0, "total_tokens": 0}}

The `file` upload(s) carry raw audio bytes (wav/mp3/flac/ogg/...).
The route layer enforces a size cap, then forwards the raw bytes to
the registered backend; the runtime decodes via librosa and
resamples on the way in.

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError (exception handler). 400/413/415 use
error_response() so the bare {"error": ...} envelope reaches the
client without FastAPI's {"detail": ...} wrap.

This is a Task A stub; replaced by the real implementation in Task C.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


MODALITY = "audio/embedding"


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Stub router; Task C replaces with the real multipart endpoint."""
    return APIRouter()
