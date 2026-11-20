"""FastAPI router for /v1/audio/speech.

Ports narro/server.py's TTS handlers to muse's per-modality router
pattern. The router is built with a registry reference so handlers
look up backends by name.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SkipValidation
from sse_starlette.sse import EventSourceResponse

from muse.modalities.audio_speech.codec import (
    AudioFormatError,
    audio_to_wav_bytes,
    float_to_pcm16,
    wav_bytes_to_opus,
)
from muse.modalities.audio_speech.pronunciations import (
    PronunciationError, SpokenText, match_pronunciations, parse_pronunciations,
    pronunciations_digest, render_spoken,
)
from muse.modalities.audio_speech.protocol import TimestampsUnavailableError
from muse.modalities.audio_speech.word_timing import TimedSpan, assemble_words
from muse.core.errors import error_response
from muse.modalities._common.streaming import threaded_sse_events
from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)

MODALITY = "audio/speech"
MAX_INPUT_LENGTH = 50_000
_STREAM_QUEUE_DEPTH = 8
_STREAM_POLL_SECONDS = 0.05
_STREAM_CLEANUP_GRACE_SECONDS = 0.5


def _stream_error_event() -> dict[str, str]:
    """Return a stable mid-stream failure without backend details."""
    payload = {
        "error": {
            "code": "streaming_failed",
            "message": "audio streaming backend failed; see server logs",
            "type": "server_error",
        },
    }
    return {"event": "error", "data": json.dumps(payload)}



class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    model: str | None = None
    voice: str | None = None
    response_format: str = Field(default="wav", pattern="^(wav|opus)$")
    stream: bool = False
    speed: float = 1.0
    align: bool = False
    timestamps: bool = False
    # Keep raw values so our parser returns the pronunciation-specific 400.
    pronunciations: SkipValidation[list | None] = Field(
        default=None,
        description=(
            "Optional pronunciation overrides, a list of objects. Each entry has "
            "exactly one target, either `term` (string; every whole-word, "
            "case-insensitive match in `input`) or `spans` (list of [start, end] "
            "character offsets into `input`), and exactly one replacement, either "
            "`ipa` (IPA phonemes, on models that support it) or `respell` (ordinary "
            "words spelled as they sound). Example: "
            "[{\"term\": \"Kutta\", \"respell\": \"koo-tah\"}, "
            "{\"spans\": [[0, 5]], \"ipa\": \"ˈrʊŋə\"}]."
        ),
    )


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=["audio/speech"])

    @router.get("/speech/voices")
    def list_voices(model: str | None = None):
        m = registry.get_or_404(MODALITY, model)
        voices = getattr(m, "voices", [])
        return {"model": m.model_id, "voices": voices}

    @router.post("/speech")
    async def speech(req: SpeechRequest):
        model = registry.get_or_404(MODALITY, req.model)

        capabilities = registry.capabilities(MODALITY, model.model_id)
        if not math.isfinite(req.speed):
            return error_response(400, "invalid_parameter", "speed must be a finite number")
        if req.timestamps and req.stream:
            return error_response(
                400, "timestamps_not_streamable",
                "Word timestamps are not available while streaming",
            )
        if req.stream and req.response_format != "wav":
            return error_response(
                400, "format_not_streamable",
                "Streaming returns base64 PCM16 chunks; response_format "
                f"{req.response_format!r} is only available without stream",
            )
        if req.stream and req.align:
            return error_response(
                400, "alignment_not_streamable",
                "Paragraph alignment (align) is not available while streaming",
            )
        if req.timestamps and not capabilities.get("supports_word_timestamps"):
            return _timestamps_not_supported()

        spoken = SpokenText(req.input)
        pronunciations = None
        if req.pronunciations is not None:
            try:
                entries = parse_pronunciations(req.pronunciations, req.input)
            except PronunciationError as exc:
                return error_response(400, "invalid_pronunciation", str(exc))
            renderer = getattr(model, "render_ipa", None)
            for entry in entries:
                if entry.kind == "ipa" and (
                    "ipa" not in (capabilities.get("pronunciation_kinds") or [])
                    or not callable(renderer)
                ):
                    target = (f"term {entry.term!r}" if entry.term is not None else
                              f"spans {[list(span) for span in entry.spans]}")
                    return error_response(
                        400, "pronunciation_not_supported",
                        f"Pronunciation entry {entry.index} ({target}): "
                        "IPA is not supported by this model",
                    )
            replacements, applied = match_pronunciations(req.input, entries)
            try:
                spoken = render_spoken(req.input, replacements, renderer)
            except PronunciationError as exc:
                return error_response(400, "invalid_pronunciation", str(exc))
            if len(spoken.text) > MAX_INPUT_LENGTH:
                return error_response(
                    400, "invalid_pronunciation",
                    f"Rendered text is too long ({len(spoken.text)} characters from "
                    f"{len(req.input)} input characters; the limit is {MAX_INPUT_LENGTH})",
                )
            pronunciations = {
                "digest": pronunciations_digest(req.pronunciations), "applied": applied,
            }

        if req.stream:
            headers = _provenance_headers(model, req, {}, pronunciations)
            return await _stream(model, req, spoken=spoken.text, headers=headers)
        return await _non_stream(model, req, spoken, pronunciations)

    return router


def _timestamps_not_supported() -> Response:
    return error_response(
        400, "timestamps_not_supported",
        "Native word timestamps are unavailable for this model or language; "
        "use /v1/audio/alignments",
    )


def _voice(model, req: SpeechRequest, metadata: dict) -> str | None:
    return metadata.get("voice") or req.voice or getattr(model, "default_voice", None)


def _provenance_headers(
    model, req: SpeechRequest, metadata: dict, pronunciations: dict | None,
) -> dict[str, str]:
    headers = {"X-Muse-Speed": str(float(req.speed))}
    voice = _voice(model, req, metadata)
    if voice and all(" " <= char <= "~" for char in voice):
        headers["X-Muse-Voice"] = voice
    if pronunciations is not None:
        headers["X-Muse-Pronunciations-Digest"] = pronunciations["digest"]
    return headers


async def _non_stream(
    model, req: SpeechRequest, spoken: SpokenText,
    pronunciations: dict | None,
) -> Response:

    def _synth():
        with model._inference_lock:
            kwargs = {"voice": req.voice, "speed": req.speed, "align": req.align}
            if req.timestamps:
                kwargs["timestamps"] = True
            result = model.synthesize(spoken.text, **kwargs)
        if not req.timestamps:
            return result, None
        # Span mapping and word assembly run off the event loop.
        spans = [
            TimedSpan(*spoken.to_input_span(span.char_start, span.char_end), span.start, span.end)
            for span in ((result.metadata or {}).get("token_spans") or [])
        ]
        return result, assemble_words(req.input, spans, result.duration)

    try:
        result, words = await asyncio.to_thread(_synth)
    except TimestampsUnavailableError:
        if req.timestamps:
            return _timestamps_not_supported()
        raise

    try:
        body, media = await asyncio.to_thread(
            _encode, result, req.response_format,
        )
    except AudioFormatError:
        # Log the real exception server-side but never leak it to the
        # client: str(e) can carry internal filesystem paths, CUDA
        # driver text, or other backend-implementation detail.
        logger.exception("wav encoding failed")
        return error_response(
            500, "encoding_failed",
            "audio encoding backend failed; see server logs",
        )

    metadata = result.metadata or {}
    headers = _provenance_headers(model, req, metadata, pronunciations)
    audio_sha256 = hashlib.sha256(body).hexdigest()
    headers["X-Muse-Audio-SHA256"] = audio_sha256
    if req.align and result.metadata and "alignment" in result.metadata:
        headers["X-Alignment"] = json.dumps(result.metadata["alignment"])

    if req.timestamps:
        measured = sum(word["measured"] for word in words)
        payload = {
            "object": "audio.speech", "model": model.model_id,
            "voice": _voice(model, req, metadata), "speed": req.speed,
            "audio": {
                "format": "opus" if media == "audio/ogg" else "wav",
                "b64_json": base64.b64encode(body).decode(),
                "sha256": audio_sha256, "sample_rate": result.sample_rate,
                "duration_seconds": result.duration,
            },
            "words": words,
            "timing": {
                "source": "native", "measured_words": measured,
                "estimated_words": len(words) - measured,
                "calibration": metadata.get("timing_calibration"),
            },
        }
        if pronunciations is not None:
            payload["pronunciations"] = pronunciations
        return JSONResponse(payload, headers=headers)

    return Response(content=body, media_type=media, headers=headers)


def _encode(result, response_format: str) -> tuple[bytes, str]:
    """WAV, then the ffmpeg Opus transcode when asked; runs off-loop.

    Opus falls back to WAV when ffmpeg cannot produce it. A WAV failure
    raises AudioFormatError.
    """
    wav = audio_to_wav_bytes(result.audio, result.sample_rate)
    if response_format != "opus":
        return wav, "audio/wav"
    try:
        return wav_bytes_to_opus(wav), "audio/ogg"
    except AudioFormatError:
        logger.warning("opus encoding unavailable; falling back to wav")
        return wav, "audio/wav"


async def _stream(
    model, req: SpeechRequest, *, spoken: str,
    headers: dict[str, str],
) -> EventSourceResponse:
    events = threaded_sse_events(
        lock=model._inference_lock,
        open_stream=lambda: model.synthesize_stream(
            spoken, voice=req.voice, speed=req.speed,
        ),
        # Look float_to_pcm16 up at call time so tests can patch it.
        encode=lambda chunk: {
            "data": base64.b64encode(float_to_pcm16(chunk.audio).tobytes()).decode(),
        },
        done_events=({"event": "done", "data": ""},),
        error_event=_stream_error_event,
        label="audio-speech",
        queue_depth=_STREAM_QUEUE_DEPTH,
        poll_seconds=_STREAM_POLL_SECONDS,
        cleanup_grace_seconds=_STREAM_CLEANUP_GRACE_SECONDS,
    )
    return EventSourceResponse(events, headers=headers)
