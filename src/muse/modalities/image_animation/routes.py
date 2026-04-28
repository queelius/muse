"""POST /v1/images/animations.

Wire contract documented in docs/superpowers/specs/2026-04-27-image-animation-modality-design.md.
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
from muse.modalities.image_animation.codec import (
    encode_webp, encode_gif, encode_mp4, encode_frames_b64,
    UnsupportedFormatError,
)
from muse.modalities.image_generation.image_input import decode_image_input


MODALITY = "image/animation"

logger = logging.getLogger(__name__)


class AnimationsRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    frames: int | None = Field(default=None, ge=4, le=64)
    fps: int | None = Field(default=None, ge=1, le=30)
    loop: bool = True
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    response_format: str = Field(default="webp", pattern="^(webp|gif|mp4|frames_b64)$")
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/animation"])

    @router.post("/animations")
    async def animations(req: AnimationsRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )
        manifest = registry.manifest(MODALITY, model.model_id) or {}
        capabilities = manifest.get("capabilities") or {}

        # Image-to-animation gate
        init_image = None
        if req.image is not None:
            if not capabilities.get("supports_image_to_animation"):
                return error_response(
                    400, "invalid_parameter",
                    f"model {model.model_id!r} does not support image-to-animation; "
                    f"use a model with supports_image_to_animation=True",
                )
            try:
                init_image = decode_image_input(req.image)
            except ValueError as e:
                return error_response(
                    400, "invalid_parameter", f"image decode failed: {e}",
                )

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs = {
                "negative_prompt": req.negative_prompt,
                "frames": req.frames,
                "fps": req.fps,
                "width": width, "height": height,
                "steps": req.steps, "guidance": req.guidance,
                "init_image": init_image,
                "strength": req.strength,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            r = await asyncio.to_thread(_call_one, i)
            results.append(r)

        # Encode each result according to response_format
        data = []
        for r in results:
            try:
                encoded = _encode(req.response_format, r, loop=req.loop)
            except UnsupportedFormatError as e:
                return error_response(400, "invalid_parameter", str(e))
            if req.response_format == "frames_b64":
                # encoded is list[str]; expand into per-frame data entries
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
                "duration_seconds": round(len(head.frames) / max(head.fps, 1), 3),
                "format": req.response_format,
                "size": [head.width, head.height],
            },
        }
        return JSONResponse(content=body)

    return router


def _encode(fmt, result, *, loop):
    if fmt == "webp":
        return encode_webp(result.frames, result.fps, loop=loop)
    if fmt == "gif":
        return encode_gif(result.frames, result.fps, loop=loop)
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "frames_b64":
        return encode_frames_b64(result.frames)
    raise ValueError(f"unknown format: {fmt}")
