"""FastAPI routes for /v1/audio/classifications.

Multipart in (file + optional model + optional top_k), JSON out
mirroring /v1/text/classifications. Spools the upload to a temp file
before invoking librosa (which expects a path).

Per-backend _inference_lock (v0.34.0) serializes calls. Size cap via
MUSE_AUDIO_CLS_MAX_BYTES env (default 50MB), read per-request.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.uploads import (
    read_upload,
    run_on_spooled_upload,
    upload_suffix,
)
from muse.modalities.audio_classification.codec import (
    encode_audio_classifications,
)


MODALITY = "audio/classification"


logger = logging.getLogger(__name__)


def _max_bytes() -> int:
    """Read the audio-classification byte cap via muse.core.config
    (env: MUSE_AUDIO_CLS_MAX_BYTES) per call, so operators can change
    the cap without a server restart. Garbled values fall back to the
    registry default (50MB) with a logged warning. A resolved value
    that is None (empty env) or non-positive (parseable but nonsensical
    as a byte cap) also falls back to the registry default, mirroring
    image_generation/image_input.py's _default_max_bytes guard."""
    return config.get_positive("limits.audio_cls_max_bytes")


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/classifications")
    async def audio_classifications(
        file: UploadFile = File(...),
        model: str | None = Form(None),
        top_k: int | None = Form(None),
    ):
        if top_k is not None and top_k < 1:
            return error_response(
                400, "invalid_parameter",
                f"top_k must be >= 1; got {top_k}",
            )

        cap = _max_bytes()
        raw = await read_upload(
            file, max_bytes=cap, limit=f"MUSE_AUDIO_CLS_MAX_BYTES={cap}",
        )
        if isinstance(raw, JSONResponse):
            return raw

        backend = registry.get_or_404(MODALITY, model)

        def _call(path: str):
            with backend._inference_lock:
                return backend.classify(path)

        try:
            results = await run_on_spooled_upload(
                raw, _call, suffix=upload_suffix(file.filename),
            )
        except Exception:  # noqa: BLE001
            # Log the real exception server-side but never leak it to the
            # client: str(e) can carry internal filesystem paths, CUDA
            # driver text, or other backend-implementation detail.
            logger.exception("audio classify failed")
            return error_response(
                500, "internal_error",
                "audio classification backend failed; see server logs",
            )

        body = encode_audio_classifications(
            results, model_id=backend.model_id, top_k=top_k,
        )
        return JSONResponse(content=body)

    return router
