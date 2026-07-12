"""FastAPI route for reference-text forced alignment."""
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
from muse.modalities.audio_alignment.codec import encode_audio_alignment
from muse.modalities.audio_alignment.protocol import (
    AudioAlignmentDecodeError,
    AudioAlignmentDurationExceededError,
    UnalignableTextError,
    UnsupportedAlignmentLanguageError,
)


MODALITY = "audio/alignment"
logger = logging.getLogger(__name__)


def _positive_setting(key: str) -> int | float:
    value = config.get(key)
    if value is None or value <= 0:
        return config.SETTINGS_BY_KEY[key].default
    return value


def _max_bytes() -> int:
    return int(_positive_setting("limits.audio_alignment_max_bytes"))


def _max_duration_seconds() -> float:
    return float(_positive_setting(
        "limits.audio_alignment_max_duration_seconds"
    ))


def _max_text_chars() -> int:
    return int(_positive_setting("limits.audio_alignment_max_text_chars"))


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/alignments")
    async def audio_alignment(
        file: UploadFile = File(...),
        text: str = Form(...),
        model: str | None = Form(None),
        language: str | None = Form(None),
    ):
        reference_text = text.strip()
        if not reference_text:
            return error_response(
                400, "invalid_parameter", "reference text must not be empty"
            )
        text_cap = _max_text_chars()
        if len(text) > text_cap:
            return error_response(
                400,
                "invalid_parameter",
                f"reference text exceeds MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS="
                f"{text_cap}",
            )

        cap = _max_bytes()
        raw = await file.read(cap + 1)
        if not raw:
            return error_response(400, "invalid_parameter", "empty audio file")
        if len(raw) > cap:
            return error_response(
                413,
                "payload_too_large",
                f"audio file exceeds MUSE_AUDIO_ALIGNMENT_MAX_BYTES={cap}",
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
                return backend.align(
                    tmp_path,
                    reference_text,
                    language=language,
                    max_duration_seconds=_max_duration_seconds(),
                )

        try:
            result = await asyncio.to_thread(_call)
        except AudioAlignmentDurationExceededError as exc:
            return error_response(413, "payload_too_large", str(exc))
        except AudioAlignmentDecodeError:
            logger.warning("audio alignment decode failed", exc_info=True)
            return error_response(
                415,
                "unsupported_media_type",
                "audio decode failed; provide a supported audio file",
            )
        except (UnsupportedAlignmentLanguageError, UnalignableTextError) as exc:
            return error_response(400, "invalid_parameter", str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("audio alignment failed")
            return error_response(
                500,
                "internal_error",
                "audio alignment backend failed; see server logs",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return JSONResponse(content=encode_audio_alignment(
            result, model_id=backend.model_id,
        ))

    return router
