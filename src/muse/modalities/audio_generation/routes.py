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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.inference import run_locked
from muse.modalities.audio_generation.codec import (
    UnsupportedFormatError,
    content_type_for,
    encode_flac,
    encode_mp3,
    encode_opus,
    encode_wav,
)
from muse.modalities.audio_generation.protocol import AudioGenerationResult


MODALITY = "audio/generation"


logger = logging.getLogger(__name__)


class AudioGenerationRequest(BaseModel):
    """Body shared by /v1/audio/music and /v1/audio/sfx.

    Same body shape as /v1/audio/speech, except `prompt` replaces
    `input` (this is generation, not TTS), and audio-generation
    extensions like `duration`, `steps`, `guidance`, `negative_prompt`
    are exposed. Optional mastering flags delegate to
    ``muse.modalities.audio_process.dsp.process_audio`` when enabled;
    all default off so existing behavior is unchanged.
    """
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    # Ceiling raised to 240s in v0.49.0 for ACE-Step's long clips; the
    # per-model runtime still clamps to its own capabilities.max_duration
    # (Stable Audio remains 47s).
    duration: float | None = Field(default=None, ge=0.5, le=240.0)
    seed: int | None = Field(default=None, ge=0)
    response_format: str = Field(default="wav", pattern="^(wav|mp3|opus|flac)$")
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    negative_prompt: str | None = None
    # Optional structured lyrics for song models (ACE-Step). None/empty =>
    # instrumental. Models that ignore lyrics (Stable Audio) simply drop it.
    lyrics: str | None = Field(default=None, max_length=8000)
    # Optional post-generation mastering flags (all default off).
    trim_silence: bool = False
    loop_fade_s: float = Field(0.0, ge=0.0, le=30.0)
    target_peak_dbfs: float | None = Field(None, ge=-48.0, le=0.0)
    mono: bool = False
    sample_rate: int | None = Field(None, ge=8000, le=96000)


def _has_mastering(req: AudioGenerationRequest) -> bool:
    return (
        req.trim_silence
        or req.loop_fade_s > 0.0
        or req.target_peak_dbfs is not None
        or req.mono
        or req.sample_rate is not None
    )


def _apply_mastering(
    result: AudioGenerationResult,
    req: AudioGenerationRequest,
) -> AudioGenerationResult:
    """Apply optional post-generation mastering via ``audio_process.dsp``."""
    from muse.modalities.audio_process.dsp import process_audio

    processed = process_audio(
        result.audio,
        result.sample_rate,
        mono=req.mono,
        target_sample_rate=req.sample_rate,
        trim_silence=req.trim_silence,
        fade_in_ms=5.0 if req.trim_silence else 0.0,
        fade_out_ms=15.0 if req.trim_silence else 0.0,
        loop_fade_s=req.loop_fade_s,
        target_peak_dbfs=req.target_peak_dbfs,
    )
    meta = dict(result.metadata)
    meta["mastering"] = processed.metrics
    return AudioGenerationResult(
        audio=processed.audio,
        sample_rate=processed.sample_rate,
        channels=processed.channels,
        duration_seconds=processed.metrics["duration_s"],
        metadata=meta,
    )


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
    model = registry.get_or_404(MODALITY, req.model)

    effective_id = model.model_id
    capabilities = registry.capabilities(MODALITY, effective_id)
    cap_key = "supports_music" if kind == "music" else "supports_sfx"
    if not capabilities.get(cap_key, True):
        return error_response(
            400, "invalid_parameter",
            f"model {effective_id!r} does not support {kind} generation",
        )

    result = await run_locked(
        model,
        lambda: model.generate(
            req.prompt,
            duration=req.duration,
            seed=req.seed,
            steps=req.steps,
            guidance=req.guidance,
            negative_prompt=req.negative_prompt,
            lyrics=req.lyrics,
        ),
        label=f"audio/generation {kind} call",
        message="audio generation backend failed; see server logs",
    )
    if isinstance(result, JSONResponse):
        return result

    # Optional mastering + encoding (pydub/ffmpeg, soundfile FLAC) run off-loop
    # and outside the model lock: up to 240 s of stereo audio would otherwise
    # stall /health.
    def _postprocess_and_encode() -> bytes:
        final = _apply_mastering(result, req) if _has_mastering(req) else result
        return _encode(req.response_format, final)

    try:
        body = await asyncio.to_thread(_postprocess_and_encode)
    except (UnsupportedFormatError, ValueError) as e:
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
