"""FastAPI route for single-ended audio-quality assessment."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.uploads import (
    looks_like_audio_decode_error,
    read_upload,
    run_on_spooled_upload,
    upload_suffix,
)
from muse.modalities.audio_quality.codec import encode_audio_quality
from muse.modalities.audio_quality.protocol import AudioDurationExceededError


MODALITY = "audio/quality"
logger = logging.getLogger(__name__)


def _max_bytes() -> int:
    """Return the live upload cap, falling back on invalid opt-int values."""
    return config.get_positive("limits.audio_quality_max_bytes")


def _max_duration_seconds() -> float:
    """Return the live decoded-duration cap with a safe fallback."""
    return float(config.get_positive("limits.audio_quality_max_duration_seconds"))


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/quality")
    async def audio_quality(
        file: UploadFile = File(...),
        model: str | None = Form(None),
    ):
        cap = _max_bytes()
        raw = await read_upload(
            file, max_bytes=cap, limit=f"MUSE_AUDIO_QUALITY_MAX_BYTES={cap}",
        )
        if isinstance(raw, JSONResponse):
            return raw

        backend = registry.get_or_404(MODALITY, model)

        def _call(path: str):
            with backend._inference_lock:
                return backend.assess(
                    path,
                    max_duration_seconds=_max_duration_seconds(),
                )

        try:
            result = await run_on_spooled_upload(
                raw, _call, suffix=upload_suffix(file.filename),
            )
        except AudioDurationExceededError as exc:
            return error_response(
                413,
                "payload_too_large",
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            if looks_like_audio_decode_error(exc):
                logger.warning("audio quality decode failed", exc_info=True)
                return error_response(
                    415,
                    "unsupported_media_type",
                    "audio decode failed; provide a supported audio file",
                )
            logger.exception("audio quality assessment failed")
            return error_response(
                500,
                "internal_error",
                "audio quality backend failed; see server logs",
            )

        return JSONResponse(content=encode_audio_quality(
            result, model_id=backend.model_id,
        ))

    return router
