"""FastAPI routes for /v1/audio/transcriptions and /v1/audio/translations.

First muse modality with multipart/form-data uploads. Pattern: FastAPI
UploadFile + Form fields, saved to a tempfile, passed as path to the
backend. If a second multipart modality (images/edits, audio input)
lands, factor out into muse.modalities._common.uploads (TODO in
CLAUDE.md).

Size cap: MUSE_ASR_MAX_MB env var (default 100). 4x OpenAI's 25 MB
since we're self-hosted.

Error envelopes match the muse OpenAI-compat convention via
ModelNotFoundError (404) and HTTPException with detail={'error': ...}
(400, 413, 415). FastAPI's exception handler unwraps detail.error.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.audio_transcription.codec import encode_transcription


MODALITY = "audio/transcription"


logger = logging.getLogger(__name__)

VALID_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


def _max_upload_bytes() -> int:
    mb = int(os.environ.get("MUSE_ASR_MAX_MB", "100"))
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

        max_bytes = _max_upload_bytes()
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            return error_response(
                413,
                "payload_too_large",
                f"file exceeds MUSE_ASR_MAX_MB={max_bytes // (1024 * 1024)}",
            )
        if not data:
            return error_response(400, "invalid_parameter", "file is empty")

        # registry.get raises KeyError; convert to ModelNotFoundError for
        # the OpenAI-compat envelope (see embedding_text/routes.py).
        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model, modality=MODALITY)

        want_words = "word" in timestamp_granularities

        with tempfile.NamedTemporaryFile(
            suffix=_suffix_for_upload(file.filename), delete=True,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                result = backend.transcribe(
                    tmp.name,
                    task=task,
                    language=None if task == "translate" else language,
                    prompt=prompt,
                    temperature=temperature,
                    word_timestamps=want_words,
                    vad_filter=vad_filter,
                )
            except Exception as e:  # noqa: BLE001
                # PyAV/ffmpeg decode failures land here; we don't try to
                # distinguish finely (message surface is good enough).
                msg = str(e).lower()
                if "decoder" in msg or "format" in msg or "ffmpeg" in msg:
                    return error_response(
                        415,
                        "unsupported_media_type",
                        f"audio decode failed: {e}",
                    )
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


def _suffix_for_upload(filename: str | None) -> str:
    """Preserve the upload's suffix so ffmpeg/PyAV can sniff the format."""
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1]
