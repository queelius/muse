"""FastAPI routes for /v1/audio/embeddings.

Wire shape mirrors /v1/audio/transcriptions on the request side
(multipart/form-data with `file` part(s)) and /v1/embeddings on the
response side (OpenAI shape with `data: [{embedding, index, object}],
model, usage`).

Each request carries one or more `file` parts whose bytes decode to
audio (wav/mp3/flac/ogg/...) via librosa (in the runtime). The route
layer:
  - Validates the `file` field (non-empty bytes per file)
  - Enforces a size cap via MUSE_AUDIO_EMBEDDINGS_MAX_BYTES
  - Resolves the registered backend
  - Hands the raw bytes list to backend.embed(...)
  - Encodes the response per `encoding_format` (float | base64)

Multipart batching: repeating the `file` field name produces a list
of UploadFile objects. Single-file uploads remain the common path.

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400/413/415 use error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.uploads import (
    looks_like_audio_decode_error,
    read_upload,
)
from muse.modalities.audio_embedding.codec import encode_embedding_list


logger = logging.getLogger(__name__)


MODALITY = "audio/embedding"

_MAX_FILES = 32
_MAX_TOTAL_UPLOAD_BYTES = 64 * 1024 * 1024



def _max_upload_bytes() -> int:
    """Per-file byte cap via muse.core.config (env:
    MUSE_AUDIO_EMBEDDINGS_MAX_BYTES). Default 50MB; tunable via env."""
    return config.get_positive("limits.audio_embeddings_max_bytes")


_VALID_FORMATS = {"float", "base64"}


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/embeddings")
    async def audio_embeddings(
        file: list[UploadFile] = File(...),
        model: str | None = Form(None),
        encoding_format: str = Form("float"),
        user: str | None = Form(None),  # OpenAI compat; ignored
    ):
        if encoding_format not in _VALID_FORMATS:
            return error_response(
                400, "invalid_parameter",
                f"encoding_format must be one of {sorted(_VALID_FORMATS)}; "
                f"got {encoding_format!r}",
            )

        if not file:
            return error_response(
                400, "invalid_parameter",
                "at least one `file` part is required",
            )
        if len(file) > _MAX_FILES:
            return error_response(
                413, "payload_too_large",
                f"at most {_MAX_FILES} audio files are allowed per request",
            )

        max_bytes = _max_upload_bytes()
        audio_bytes_list: list[bytes] = []
        total_bytes = 0
        for idx, upload in enumerate(file):
            data = await read_upload(
                upload, max_bytes=max_bytes, label=f"file[{idx}]",
                limit=f"MUSE_AUDIO_EMBEDDINGS_MAX_BYTES={max_bytes}",
            )
            if isinstance(data, JSONResponse):
                return data
            total_bytes += len(data)
            if total_bytes > _MAX_TOTAL_UPLOAD_BYTES:
                return error_response(
                    413, "payload_too_large",
                    "aggregate audio upload exceeds "
                    f"{_MAX_TOTAL_UPLOAD_BYTES} bytes",
                )
            audio_bytes_list.append(data)

        backend = registry.get_or_404(MODALITY, model)

        # backend.embed is sync (transformers forward + librosa decode);
        # offload so a slow inference doesn't block sibling /health,
        # /v1/models, or other in-flight requests on the same worker.
        def _call():
            with backend._inference_lock:
                return backend.embed(audio_bytes_list)

        try:
            result = await asyncio.to_thread(_call)
        except Exception as e:  # noqa: BLE001
            # librosa decode failures map to 415 (same heuristic as
            # audio_transcription); the raw exception text stays in the
            # server log, not the client body.
            if looks_like_audio_decode_error(e):
                logger.warning(
                    "backend.embed raised a decoder-looking exception; "
                    "returning 415. Error: %s", e,
                )
                return error_response(
                    415, "unsupported_media_type",
                    "audio decode failed; provide a supported audio file",
                )
            # Deliberately re-raise everything else so a real bug
            # surfaces instead of being masked as a 415. This used to
            # escape as FastAPI's bare `{"detail": "Internal Server
            # Error"}`; create_app's global Exception handler (added
            # v0.58.1) now catches it and returns the OpenAI-shape 500
            # envelope, logging the real exception server-side.
            raise

        return encode_embedding_list(
            result.embeddings, model_id=result.model_id,
            encoding_format=encoding_format,
        )

    return router
