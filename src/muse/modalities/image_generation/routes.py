"""FastAPI router for /v1/images/generations.

Follows OpenAI's /v1/images/generations contract:
  - `prompt` (required, 1-4000 chars)
  - `n` number of images (1-10)
  - `size` "WIDTHxHEIGHT" (64-2048 per side)
  - `response_format` "b64_json" (default) | "url" (data URL)
  - muse extensions: `model`, `seed`, `steps`, `guidance`, `negative_prompt`
  - img2img extensions (since v0.17.0): `image` (data URL or http(s) URL),
    `strength` (0.0 to 1.0). When `image` is set, the runtime routes through
    AutoPipelineForImage2Image. Requires the selected model to advertise
    `capabilities.supports_img2img: True` in its manifest.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from threading import Lock

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.image_generation.codec import to_bytes, to_data_url
from muse.modalities.image_generation.image_input import decode_image_input

logger = logging.getLogger(__name__)

MODALITY = "image/generation"
_inference_lock = Lock()


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="512x512", pattern=r"^\d+x\d+$")
    response_format: str = Field(default="b64_json", pattern="^(b64_json|url)$")
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: str) -> str:
        w, h = map(int, v.split("x"))
        if w < 64 or h < 64 or w > 2048 or h > 2048:
            raise ValueError(f"size {v} out of supported range (64-2048 per side)")
        return v


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/generation"])

    @router.post("/generations")
    async def generations(req: GenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)

        # Resolve the effective model id (the registry default when req.model is None)
        # so capability-gate errors and decode-failure errors name a real model.
        effective_id = getattr(model, "model_id", None) or (req.model or "<default>")

        # img2img gating: if `image` is supplied, the selected model must
        # declare `supports_img2img: True` in its manifest, and the input
        # must decode to a PIL.Image.
        init_image = None
        if req.image is not None:
            manifest = registry.manifest(MODALITY, effective_id) or {}
            if not manifest.get("capabilities", {}).get("supports_img2img"):
                return error_response(
                    400,
                    "invalid_parameter",
                    f"model {effective_id!r} does not support img2img; "
                    f"use one of the diffusers-resolved models or sd-turbo",
                )
            try:
                init_image = decode_image_input(req.image)
            except ValueError as e:
                return error_response(
                    400,
                    "invalid_parameter",
                    f"image decode failed: {e}",
                )

        width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "width": width,
                "height": height,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps,
                "guidance": req.guidance,
                "init_image": init_image,
                "strength": req.strength,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            with _inference_lock:
                return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            result = await asyncio.to_thread(_call_one, i)
            results.append(result)

        data = []
        for r in results:
            entry = {"revised_prompt": r.metadata.get("prompt", req.prompt)}
            if req.response_format == "url":
                entry["url"] = to_data_url(r.image, fmt="png")
            else:
                entry["b64_json"] = base64.b64encode(to_bytes(r.image, fmt="png")).decode()
            data.append(entry)

        return {"created": int(time.time()), "data": data}

    return router
