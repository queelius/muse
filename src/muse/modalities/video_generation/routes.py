"""POST /v1/video/generations.

Wire contract documented in
docs/superpowers/specs/2026-04-28-video-generation-modality-design.md.

video/generation is muse's narrative-clip sibling to image/animation:
longer durations, single play (no `loop` field), mp4 default,
transformer-based backbones (Wan, CogVideoX). The two modalities
deliberately don't overlap: short looping animations go to
/v1/images/animations; multi-second narrative clips go here.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.video_generation.codec import (
    UnsupportedFormatError,
    encode_frames_b64,
    encode_mp4,
    encode_webm,
)


MODALITY = "video/generation"

logger = logging.getLogger(__name__)


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.5, le=30.0)
    fps: int | None = Field(default=None, ge=1, le=60)
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")
    seed: int | None = None
    negative_prompt: str | None = Field(default=None, max_length=4000)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    response_format: str = Field(
        default="mp4", pattern="^(mp4|webm|frames_b64)$",
    )
    n: int = Field(default=1, ge=1, le=2)


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/video", tags=["video/generation"])

    @router.post("/generations")
    async def generations(req: VideoGenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs = {
                "negative_prompt": req.negative_prompt,
                "duration_seconds": req.duration_seconds,
                "fps": req.fps,
                "width": width,
                "height": height,
                "steps": req.steps,
                "guidance": req.guidance,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            r = await asyncio.to_thread(_call_one, i)
            results.append(r)

        data = []
        for r in results:
            try:
                encoded = _encode(req.response_format, r)
            except UnsupportedFormatError as e:
                return error_response(400, "invalid_parameter", str(e))
            if req.response_format == "frames_b64":
                # encoded is list[str]; expand into per-frame data entries.
                # For n>1 the per-result frame lists are appended in order;
                # a future revision may add explicit per-result grouping.
                for s in encoded:
                    data.append({"b64_json": s})
            else:
                data.append({
                    "b64_json": base64.b64encode(encoded).decode("ascii"),
                })

        # Use the first result for top-level metadata (n=1 is common)
        head = results[0]
        body = {
            "data": data,
            "model": model.model_id,
            "metadata": {
                "frames": len(head.frames),
                "fps": head.fps,
                "duration_seconds": head.duration_seconds,
                "format": req.response_format,
                "size": [head.width, head.height],
            },
        }
        return JSONResponse(content=body)

    return router


def _encode(fmt, result):
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "webm":
        return encode_webm(result.frames, result.fps)
    if fmt == "frames_b64":
        return encode_frames_b64(result.frames)
    raise ValueError(f"unknown format: {fmt}")
