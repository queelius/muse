"""FastAPI route for single-ended audio-quality assessment."""
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
from muse.modalities.audio_quality.codec import encode_audio_quality
from muse.modalities.audio_quality.protocol import AudioDurationExceededError


MODALITY = "audio/quality"
logger = logging.getLogger(__name__)


def _max_bytes() -> int:
    """Return the live upload cap, falling back on invalid opt-int values."""
    value = config.get("limits.audio_quality_max_bytes")
    if value is None or value <= 0:
        return config.SETTINGS_BY_KEY["limits.audio_quality_max_bytes"].default
    return value


def _max_duration_seconds() -> float:
    """Return the live decoded-duration cap with a safe fallback."""
    key = "limits.audio_quality_max_duration_seconds"
    value = config.get(key)
    if value is None or value <= 0:
        return config.SETTINGS_BY_KEY[key].default
    return float(value)


def _looks_like_decode_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in (
        "audioread",
        "decode",
        "decoder",
        "ffmpeg",
        "format not recognised",
        "format not recognized",
        "no backend available",
        "soundfile",
        "torchcodec",
    ))


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/quality")
    async def audio_quality(
        file: UploadFile = File(...),
        model: str | None = Form(None),
    ):
        cap = _max_bytes()
        raw = await file.read(cap + 1)
        if not raw:
            return error_response(400, "invalid_parameter", "empty audio file")
        if len(raw) > cap:
            return error_response(
                413,
                "payload_too_large",
                f"audio file exceeds MUSE_AUDIO_QUALITY_MAX_BYTES={cap}",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        suffix = Path(file.filename or "audio").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        def _call():
            with backend._inference_lock:
                return backend.assess(
                    tmp_path,
                    max_duration_seconds=_max_duration_seconds(),
                )

        try:
            result = await asyncio.to_thread(_call)
        except AudioDurationExceededError as exc:
            return error_response(
                413,
                "payload_too_large",
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            if _looks_like_decode_error(exc):
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
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return JSONResponse(content=encode_audio_quality(
            result, model_id=backend.model_id,
        ))

    return router
