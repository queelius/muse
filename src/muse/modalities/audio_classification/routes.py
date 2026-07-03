"""FastAPI routes for /v1/audio/classifications.

Multipart in (file + optional model + optional top_k), JSON out
mirroring /v1/text/classifications. Saves the upload to a temp file
before invoking librosa (which expects a path).

Per-backend _inference_lock (v0.34.0) serializes calls. Size cap via
MUSE_AUDIO_CLS_MAX_BYTES env (default 50MB), read per-request.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
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
    n = config.get("limits.audio_cls_max_bytes")
    if n is None or n <= 0:
        return config.SETTINGS_BY_KEY["limits.audio_cls_max_bytes"].default
    return n


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

        # Bounded read so a malicious giant upload doesn't OOM the worker.
        cap = _max_bytes()
        raw = await file.read(cap + 1)
        if not raw:
            return error_response(400, "invalid_parameter", "empty audio file")
        if len(raw) > cap:
            return error_response(
                400, "invalid_parameter",
                f"audio bytes exceeds max ({len(raw)} > {cap}); "
                f"raise MUSE_AUDIO_CLS_MAX_BYTES if intentional",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        # librosa needs a filesystem path; write the upload to a temp
        # file. Using delete=False so the inference thread can read it
        # after the with-block; we clean up explicitly.
        suffix = Path(file.filename or "audio").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        def _call():
            with backend._inference_lock:
                return backend.classify(tmp_path)

        try:
            results = await asyncio.to_thread(_call)
        except Exception as e:  # noqa: BLE001
            logger.exception("audio classify failed")
            return error_response(500, "internal_error", str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        body = encode_audio_classifications(
            results, model_id=backend.model_id, top_k=top_k,
        )
        return JSONResponse(content=body)

    return router
