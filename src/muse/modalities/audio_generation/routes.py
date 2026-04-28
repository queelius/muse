"""FastAPI routes for /v1/audio/music and /v1/audio/sfx.

Both routes share the same body shape and codec; the only difference
is the capability key consulted. Music routes gate on
`capabilities.supports_music`; SFX routes gate on
`capabilities.supports_sfx`. When the flag is missing, the default is
True (assume the model supports the kind unless stated otherwise).

Response is raw audio bytes, mirroring /v1/audio/speech: Content-Type
set per response_format, no JSON envelope.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.audio_generation.codec import (
    UnsupportedFormatError,
    content_type_for,
    encode_flac,
    encode_mp3,
    encode_opus,
    encode_wav,
)


MODALITY = "audio/generation"


logger = logging.getLogger(__name__)


class AudioGenerationRequest(BaseModel):
    """Body shared by /v1/audio/music and /v1/audio/sfx.

    Same body shape as /v1/audio/speech, except `prompt` replaces
    `input` (this is generation, not TTS), and audio-generation
    extensions like `duration`, `steps`, `guidance`, `negative_prompt`
    are exposed.
    """
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    duration: float | None = Field(default=None, ge=0.5, le=120.0)
    seed: int | None = Field(default=None, ge=0)
    response_format: str = Field(default="wav", pattern="^(wav|mp3|opus|flac)$")
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    negative_prompt: str | None = None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=["audio/generation"])

    @router.post("/music")
    async def music(req: AudioGenerationRequest):
        return await _handle(registry, req, kind="music")

    @router.post("/sfx")
    async def sfx(req: AudioGenerationRequest):
        return await _handle(registry, req, kind="sfx")

    return router


async def _handle(registry: ModalityRegistry, req: AudioGenerationRequest, *, kind: str):
    """Shared handler for both /music and /sfx.

    `kind` is "music" or "sfx" and gates on the corresponding manifest
    capability flag. Default True so v1 manifests without the flag set
    are not silently broken.
    """
    try:
        model = registry.get(MODALITY, req.model)
    except KeyError:
        raise ModelNotFoundError(
            model_id=req.model or "<default>", modality=MODALITY,
        )

    effective_id = getattr(model, "model_id", None) or (req.model or "<default>")
    manifest = registry.manifest(MODALITY, effective_id) or {}
    capabilities = manifest.get("capabilities") or {}
    cap_key = "supports_music" if kind == "music" else "supports_sfx"
    if not capabilities.get(cap_key, True):
        return error_response(
            400, "invalid_parameter",
            f"model {effective_id!r} does not support {kind} generation",
        )

    def _call():
        kwargs = {
            "duration": req.duration,
            "seed": req.seed,
            "steps": req.steps,
            "guidance": req.guidance,
            "negative_prompt": req.negative_prompt,
        }
        return model.generate(req.prompt, **kwargs)

    try:
        result = await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        logger.exception("audio/generation %s call failed", kind)
        return error_response(500, "internal_error", str(e))

    try:
        body = _encode(req.response_format, result)
    except UnsupportedFormatError as e:
        return error_response(400, "invalid_parameter", str(e))

    return Response(
        content=body,
        media_type=content_type_for(req.response_format),
    )


def _encode(response_format: str, result) -> bytes:
    if response_format == "wav":
        return encode_wav(result.audio, result.sample_rate, result.channels)
    if response_format == "flac":
        return encode_flac(result.audio, result.sample_rate, result.channels)
    if response_format == "mp3":
        return encode_mp3(result.audio, result.sample_rate, result.channels)
    if response_format == "opus":
        return encode_opus(result.audio, result.sample_rate, result.channels)
    raise UnsupportedFormatError(f"unknown response_format: {response_format!r}")
