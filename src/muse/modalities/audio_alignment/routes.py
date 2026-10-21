"""FastAPI route for reference-text forced alignment."""
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
from muse.modalities.audio_alignment.codec import encode_audio_alignment
from muse.modalities.audio_alignment.protocol import (
    AudioAlignmentDecodeError,
    AudioAlignmentDurationExceededError,
    UnalignableTextError,
    UnsupportedAlignmentLanguageError,
)


MODALITY = "audio/alignment"
logger = logging.getLogger(__name__)


def _max_bytes() -> int:
    return int(config.get_positive("limits.audio_alignment_max_bytes"))


def _max_duration_seconds() -> float:
    return float(config.get_positive(
        "limits.audio_alignment_max_duration_seconds"
    ))


def _max_text_chars() -> int:
    return int(config.get_positive("limits.audio_alignment_max_text_chars"))


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
        raw = await read_upload(
            file, max_bytes=cap, limit=f"MUSE_AUDIO_ALIGNMENT_MAX_BYTES={cap}",
        )
        if isinstance(raw, JSONResponse):
            return raw

        backend = registry.get_or_404(MODALITY, model)

        def _call(path: str):
            with backend._inference_lock:
                return backend.align(
                    path,
                    reference_text,
                    language=language,
                    max_duration_seconds=_max_duration_seconds(),
                )

        try:
            result = await run_on_spooled_upload(
                raw, _call, suffix=upload_suffix(file.filename),
            )
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

        return JSONResponse(content=encode_audio_alignment(
            result, model_id=backend.model_id,
        ))

    return router
