"""FastAPI router for /v1/audio/speech.

Ports narro/server.py's TTS handlers to muse's per-modality router
pattern. The router is built with a registry reference so handlers
look up backends by name.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from threading import Lock

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from muse.audio.speech.codec import AudioFormatError, audio_to_wav_bytes, wav_bytes_to_opus
from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)

MODALITY = "audio.speech"
MAX_INPUT_LENGTH = 50_000
_inference_lock = Lock()


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    model: str | None = None
    voice: str | None = None
    response_format: str = Field(default="wav", pattern="^(wav|opus)$")
    stream: bool = False
    speed: float = 1.0
    align: bool = False


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=["audio.speech"])

    @router.get("/speech/voices")
    def list_voices(model: str | None = None):
        try:
            m = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)
        voices = getattr(m, "voices", [])
        return {"model": m.model_id, "voices": voices}

    @router.post("/speech")
    async def speech(req: SpeechRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)

        if req.stream:
            return await _stream(model, req)
        return await _non_stream(model, req)

    return router


async def _non_stream(model, req: SpeechRequest) -> Response:
    def _synth():
        with _inference_lock:
            return model.synthesize(
                req.input,
                voice=req.voice,
                speed=req.speed,
                align=req.align,
            )

    result = await asyncio.to_thread(_synth)

    try:
        wav = audio_to_wav_bytes(result.audio, result.sample_rate)
    except AudioFormatError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.response_format == "opus":
        try:
            body = wav_bytes_to_opus(wav)
            media = "audio/ogg"
        except AudioFormatError:
            logger.warning("opus encoding unavailable; falling back to wav")
            body = wav
            media = "audio/wav"
    else:
        body = wav
        media = "audio/wav"

    headers: dict[str, str] = {}
    if req.align and result.metadata and "alignment" in result.metadata:
        headers["X-Alignment"] = json.dumps(result.metadata["alignment"])

    return Response(content=body, media_type=media, headers=headers)


async def _stream(model, req: SpeechRequest) -> EventSourceResponse:
    async def event_gen():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _produce():
            try:
                with _inference_lock:
                    for chunk in model.synthesize_stream(
                        req.input, voice=req.voice, speed=req.speed,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        loop.run_in_executor(None, _produce)

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                logger.error("stream producer raised: %s", item)
                yield {"event": "error", "data": str(item)}
                break
            pcm = (np.clip(item.audio, -1.0, 1.0) * 32767).astype(np.int16)
            yield {"data": base64.b64encode(pcm.tobytes()).decode()}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())
