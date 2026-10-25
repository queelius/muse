"""FastAPI routes for /v1/audio/transcriptions and /v1/audio/translations.

First muse modality with multipart/form-data uploads. Pattern: FastAPI
UploadFile + Form fields, spooled to a temp file via
muse.modalities._common.uploads, passed as path to the backend.

Size cap: MUSE_ASR_MAX_MB env var (default 100). 4x OpenAI's 25 MB
since we're self-hosted.

Error envelopes follow muse's OpenAI-compat convention. 404
(unknown model) raises ModelNotFoundError, which is caught by a
handler registered in core.server.create_app. 400 (bad parameter),
413 (payload too large), and 415 (decoder failure) use
core.errors.error_response() to return a JSONResponse directly
with the {"error": {code, message, type}} envelope. This
returned-vs-raised split exists because FastAPI's default
HTTPException handler wraps detail under {"detail": ...}, not
{"error": ...}, and muse only registers a handler for
ModelNotFoundError and RequestValidationError.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.uploads import (
    looks_like_audio_decode_error,
    read_upload,
    run_on_spooled_upload,
    upload_suffix,
)
from muse.modalities.audio_transcription.codec import encode_transcription
from muse.modalities.audio_transcription.protocol import InvalidLanguageError
from muse.modalities.audio_alignment.protocol import (
    AudioAlignmentDecodeError, AudioAlignmentDurationExceededError,
)


MODALITY = "audio/transcription"


logger = logging.getLogger(__name__)

VALID_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


def _max_upload_bytes() -> int:
    mb = config.get("limits.asr_max_mb")
    return mb * 1024 * 1024


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    async def _handle(
        *,
        task: Literal["transcribe", "translate"],
        file: UploadFile,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float,
        timestamp_granularities: list[str],
        vad_filter: bool,
    ) -> Response:
        if response_format not in VALID_FORMATS:
            return error_response(
                400,
                "invalid_parameter",
                f"response_format must be one of {sorted(VALID_FORMATS)}; "
                f"got {response_format!r}",
            )

        backend = registry.get_or_404(MODALITY, model)
        caps = registry.capabilities(MODALITY, model)
        want_words = "word" in timestamp_granularities
        if want_words and not caps.get("supports_word_timestamps", True):
            return error_response(
                400, "word_timestamps_not_supported",
                "This model has no word timestamps; use /v1/audio/alignments.",
            )
        if (
            response_format in {"verbose_json", "srt", "vtt"}
            and not caps.get("supports_segment_timestamps", True)
        ):
            return error_response(
                400, "segment_timestamps_not_supported",
                "This model returns transcripts only; use /v1/audio/alignments for timestamps.",
            )
        if task == "translate" and not caps.get("supports_translation", True):
            return error_response(
                400, "translation_not_supported", "This model does not support translation.",
            )

        max_bytes = _max_upload_bytes()
        data = await read_upload(
            file, max_bytes=max_bytes, label="file",
            limit=f"MUSE_ASR_MAX_MB={max_bytes // (1024 * 1024)}",
        )
        if isinstance(data, JSONResponse):
            return data

        # Model inference is sync and CPU/GPU-bound; run it off-loop so a
        # long transcription doesn't block /health, the gateway's
        # per-worker probing, or sibling requests.
        def _transcribe(path: str):
            with backend._inference_lock:
                return backend.transcribe(
                    path,
                    task=task,
                    language=None if task == "translate" else language,
                    prompt=prompt,
                    temperature=temperature,
                    word_timestamps=want_words,
                    vad_filter=vad_filter,
                )

        try:
            result = await run_on_spooled_upload(
                data, _transcribe, suffix=upload_suffix(file.filename, default=""),
            )
        except InvalidLanguageError as e:
            return error_response(400, "invalid_language", str(e))
        except AudioAlignmentDurationExceededError as e:
            return error_response(413, "payload_too_large", str(e))
        except AudioAlignmentDecodeError:
            return error_response(
                415, "unsupported_media_type",
                "audio decode failed; provide a supported audio file",
            )
        except Exception as e:  # noqa: BLE001
            # PyAV/ffmpeg decode failures map to 415; the raw exception
            # text stays in the server log, not the client body.
            if looks_like_audio_decode_error(e):
                logger.warning(
                    "backend.transcribe raised a decoder-looking exception; "
                    "returning 415. Error: %s", e,
                )
                return error_response(
                    415,
                    "unsupported_media_type",
                    "audio decode failed; provide a supported audio file",
                )
            # Deliberately re-raise everything else so a real bug
            # surfaces instead of being masked as a 415. create_app's
            # global Exception handler (v0.58.1) returns the OpenAI-shape
            # 500 envelope and logs the real exception server-side.
            raise

        body, content_type = encode_transcription(
            result, response_format, include_words=want_words,
        )
        return Response(content=body, media_type=content_type)

    @router.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: list[str] = Form(
            default_factory=list, alias="timestamp_granularities[]",
        ),
        vad_filter: bool = Form(False),
    ):
        return await _handle(
            task="transcribe",
            file=file, model=model, language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            vad_filter=vad_filter,
        )

    @router.post("/v1/audio/translations")
    async def translations(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: list[str] = Form(
            default_factory=list, alias="timestamp_granularities[]",
        ),
        vad_filter: bool = Form(False),
    ):
        return await _handle(
            task="translate",
            file=file, model=model, language=None, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            vad_filter=vad_filter,
        )

    return router

