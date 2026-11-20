"""POST /v1/images/animations.

Wire contract documented in docs/superpowers/specs/2026-04-27-image-animation-modality-design.md.
"""
from __future__ import annotations

import base64
import threading

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.fields import parse_bounded_size
from muse.modalities._common.images import (
    ImageInputError,
    decode_image_input,
    image_error_response,
)
from muse.modalities._common.inference import offload_call, produce_each
from muse.modalities.image_animation.codec import (
    UnsupportedFormatError,
    encode_frames_b64,
    encode_gif,
    encode_mp4,
    encode_webp,
    require_imageio,
)


MODALITY = "image/animation"

_MAX_SIDE = 2048
_MAX_PIXELS_PER_FRAME = 2_097_152
_MAX_REQUEST_PIXEL_FRAMES = 128 * 1024 * 1024
_DEFAULT_FRAMES = 16


class AnimationsRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    frames: int = Field(default=_DEFAULT_FRAMES, ge=4, le=64)
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

    @field_validator("size")
    @classmethod
    def _bounded_size(cls, value: str | None) -> str | None:
        if value is not None:
            parse_bounded_size(
                value, max_side=_MAX_SIDE, max_pixels=_MAX_PIXELS_PER_FRAME,
            )
        return value

    @model_validator(mode="after")
    def _bounded_workload(self):
        if self.size is None:
            return self
        width, height = map(int, self.size.split("x"))
        # ``frames`` has an API-level default, so the admitted count is the
        # exact value forwarded to every backend rather than an estimate of a
        # custom backend's potentially larger private default.
        pixel_frames = width * height * self.frames * self.n
        if pixel_frames > _MAX_REQUEST_PIXEL_FRAMES:
            raise ValueError(
                "requested n × frames × width × height exceeds the animation "
                f"budget of {_MAX_REQUEST_PIXEL_FRAMES} pixel-frames"
            )
        return self


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/animation"])

    @router.post("/animations")
    async def animations(req: AnimationsRequest):
        model = registry.get_or_404(MODALITY, req.model)
        capabilities = registry.capabilities(MODALITY, model.model_id)

        # Reject a format whose encoder is missing before the GPU work.
        try:
            require_imageio(req.response_format)
        except UnsupportedFormatError as e:
            return error_response(400, "invalid_parameter", str(e))

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
                init_image = await decode_image_input(req.image)
            except ImageInputError as e:
                return image_error_response(e, field="image")

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _produce(seed_offset: int):
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

        def _encode_result(r):
            encoded = _encode(req.response_format, r, loop=req.loop)
            if req.response_format == "frames_b64":
                # encoded is list[str]; expand into per-frame data entries
                entries = [{"b64_json": s} for s in encoded]
            else:
                entries = [{"b64_json": base64.b64encode(encoded).decode("ascii")}]
            meta = {
                "frames": len(r.frames),
                "fps": r.fps,
                "duration_seconds": round(len(r.frames) / max(r.fps, 1), 3),
                "format": req.response_format,
                "size": [r.width, r.height],
            }
            return entries, meta

        stop = threading.Event()
        try:
            encoded = await offload_call(
                lambda: produce_each(
                    req.n, _produce, _encode_result,
                    lock=model._inference_lock, stop=stop,
                ),
                inputs=[init_image] if init_image is not None else [],
                stop=stop,
            )
        except UnsupportedFormatError as e:
            return error_response(400, "invalid_parameter", str(e))

        # Use the first result for top-level metadata (n=1 is common).
        body = {
            "data": [entry for entries, _meta in encoded for entry in entries],
            "model": model.model_id,
            "metadata": encoded[0][1],
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
