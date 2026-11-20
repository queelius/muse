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

import base64
import threading

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.fields import parse_bounded_size
from muse.modalities._common.inference import offload_call, produce_each
from muse.modalities.video_generation.codec import (
    UnsupportedFormatError,
    encode_frames_b64,
    encode_mp4,
    encode_webm,
    require_imageio,
)


MODALITY = "video/generation"

_MAX_SIDE = 2048
_MAX_PIXELS_PER_FRAME = 2_097_152
_MAX_REQUEST_PIXEL_FRAMES = 256 * 1024 * 1024

# frames_b64 inlines every frame as a base64 PNG in one JSON body. At the
# request caps (duration<=30s, fps<=60) that can reach ~1800 frames, i.e. a
# multi-hundred-MB response that the worker must hold in memory and serialize
# at once. mp4/webm are container-compressed and have no such ceiling, so the
# cap applies only to frames_b64. Tunable for power users; clips above it
# should use response_format=mp4/webm.
#
# Read per-request via muse.core.config so an operator's env change takes
# effect on the next request, not at server restart. Matches the
# MUSE_IMAGE_INPUT_MAX_BYTES / MUSE_MODERATIONS_MAX_BATCH pattern.
def _max_frames_b64() -> int:
    return config.get("limits.video_max_frames_b64")


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
        duration = self.duration_seconds if self.duration_seconds is not None else 5.0
        fps = self.fps if self.fps is not None else 8
        pixel_frames = width * height * duration * fps * self.n
        if pixel_frames > _MAX_REQUEST_PIXEL_FRAMES:
            raise ValueError(
                "requested n × duration × fps × width × height exceeds the video "
                f"budget of {_MAX_REQUEST_PIXEL_FRAMES} pixel-frames"
            )
        return self


class _FramesCapExceeded(Exception):
    def __init__(self, total: int):
        super().__init__(total)
        self.total = total


def _frames_cap_error(total: int, max_frames: int):
    return error_response(
        400, "invalid_parameter",
        f"frames_b64 would emit {total} frames, exceeding "
        f"MUSE_VIDEO_MAX_FRAMES_B64={max_frames}; request a "
        f"shorter clip or use response_format=mp4 or webm "
        f"(no frame cap).",
    )


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/video", tags=["video/generation"])

    @router.post("/generations")
    async def generations(req: VideoGenerationRequest):
        model = registry.get_or_404(MODALITY, req.model)

        # Reject what can be rejected before minutes of GPU work: a format
        # whose encoder is missing, and a frames_b64 clip whose frame count
        # is already known to exceed the cap.
        try:
            require_imageio(req.response_format)
        except UnsupportedFormatError as e:
            return error_response(400, "invalid_parameter", str(e))
        max_frames = _max_frames_b64()
        frames_b64 = req.response_format == "frames_b64"
        if frames_b64 and req.duration_seconds is not None and req.fps is not None:
            predicted = max(1, round(req.duration_seconds * req.fps)) * req.n
            if predicted > max_frames:
                return _frames_cap_error(predicted, max_frames)

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _produce(seed_offset: int):
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

        emitted = 0

        def _encode_result(r):
            nonlocal emitted
            if frames_b64:
                # frames_b64 inlines every frame; guard the response size
                # (model defaults decide the count when duration/fps are
                # omitted, so it is only known now).
                emitted += len(r.frames)
                if emitted > max_frames:
                    raise _FramesCapExceeded(emitted)
                # For n>1 the per-result frame lists are appended in order.
                entries = [{"b64_json": s} for s in encode_frames_b64(r.frames)]
            else:
                encoded = _encode(req.response_format, r)
                entries = [{"b64_json": base64.b64encode(encoded).decode("ascii")}]
            meta = {
                "frames": len(r.frames),
                "fps": r.fps,
                "duration_seconds": r.duration_seconds,
                "format": req.response_format,
                "size": [r.width, r.height],
            }
            return entries, meta

        # A backend failure, or an encoder handed zero frames (a plain
        # ValueError), is a server fault: the app's global handler returns
        # the generic 500 envelope and logs the detail.
        stop = threading.Event()
        try:
            encoded = await offload_call(
                lambda: produce_each(
                    req.n, _produce, _encode_result,
                    lock=model._inference_lock, stop=stop,
                ),
                stop=stop,
            )
        except _FramesCapExceeded as e:
            return _frames_cap_error(e.total, max_frames)
        except UnsupportedFormatError as e:
            return error_response(400, "invalid_parameter", str(e))

        # Use the first result for top-level metadata (n=1 is common)
        body = {
            "data": [entry for entries, _meta in encoded for entry in entries],
            "model": model.model_id,
            "metadata": encoded[0][1],
        }
        return JSONResponse(content=body)

    return router


def _encode(fmt, result):
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "webm":
        return encode_webm(result.frames, result.fps)
    raise ValueError(f"unknown format: {fmt}")
