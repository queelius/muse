"""FastAPI router for /v1/images/generations.

Follows OpenAI's /v1/images/generations contract:
  - `prompt` (required, 1-4000 chars)
  - `n` number of images (1-10)
  - `size` "WIDTHxHEIGHT" (64-2048 per side) or "auto"/omitted (model default)
  - gpt-image fields: `quality`, `output_format`, `output_compression`,
    `background` (opaque/auto only), `moderation` (accepted, ignored)
  - `response_format` "b64_json" (default) | "url" (data URL)
  - muse extensions: `model`, `seed`, `steps`, `guidance`, `negative_prompt`
  - img2img extensions (since v0.17.0): `image` (data URL or http(s) URL),
    `strength` (0.0 to 1.0). When `image` is set, the runtime routes through
    AutoPipelineForImage2Image. Requires the selected model to advertise
    `capabilities.supports_img2img: True` in its manifest.
  - LoRA extension: `lora_scale` (0.0 to 2.0). Requires the selected model
    to advertise `capabilities.lora_adapter: True` in its manifest.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.codec import to_bytes, to_data_url
from muse.modalities.image_generation.options import (
    OptionError,
    check_background,
    check_output,
    ignored_fields,
    parse_size,
    resolve_quality,
    snap_size,
    snaps,
)
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
    decode_image_input,
    validate_total_image_pixels,
)

logger = logging.getLogger(__name__)

MODALITY = "image/generation"


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str | None = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    response_format: str = Field(default="b64_json", pattern="^(b64_json|url)$")
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    lora_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    quality: str | None = None
    output_format: str | None = None
    output_compression: int | None = None
    background: str | None = None
    moderation: str | None = None
    style: str | None = None

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: str | None) -> str | None:
        if v is None or v == "auto":
            return v
        w, h = map(int, v.split("x"))
        if w < 64 or h < 64 or w > 2048 or h > 2048:
            raise ValueError(f"size {v} out of supported range (64-2048 per side)")
        return v


def _build_response(
    results: list,
    *,
    prompt: str | None,
    response_format: str,
    output_format: str,
    compression: int | None,
    quality: str,
    background: str,
    ignored: list[str],
) -> dict:
    """Shared envelope for generations/edits/variations.

    A None `prompt` omits `revised_prompt`. `size` is what the first result
    actually measured, not what was requested.
    """
    data = []
    for r in results:
        entry: dict = {}
        if prompt is not None:
            entry["revised_prompt"] = r.metadata.get("prompt", prompt)
        if response_format == "url":
            entry["url"] = to_data_url(r.image, fmt=output_format, compression=compression)
        else:
            entry["b64_json"] = base64.b64encode(
                to_bytes(r.image, fmt=output_format, compression=compression),
            ).decode()
        data.append(entry)
    return {
        "created": int(time.time()),
        "data": data,
        "size": f"{results[0].width}x{results[0].height}",
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "metadata": {"ignored": ignored},
    }


MAX_EDIT_IMAGES = 16


def _resolve_options(
    caps,
    *,
    size,
    quality,
    output_format,
    output_compression,
    background,
    moderation,
    negative_prompt,
    guidance,
    steps,
    style=None,
) -> dict:
    """Resolve the gpt-image options shared by generations and edits.

    Raises OptionError; callers turn it into a 400 envelope.
    """
    width, height = parse_size(size, caps)
    quality_name, preset = resolve_quality(quality, caps)
    if snaps(caps):
        width, height = snap_size(width, height, caps, preset)
    fmt = check_output(output_format, output_compression)
    bg = check_background(background)
    ignored = ignored_fields(
        {
            "moderation": moderation,
            "style": style,
            "quality": quality,
            "negative_prompt": negative_prompt,
            "guidance": guidance,
            "output_compression": output_compression,
            "output_format": fmt,
        },
        caps,
    )
    return {
        "width": width,
        "height": height,
        "quality": quality_name,
        "output_format": fmt,
        "background": bg,
        "steps": steps if steps is not None else preset.get("steps"),
        "ignored": ignored,
    }


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/generation"])

    @router.post("/generations")
    async def generations(req: GenerationRequest):
        model = registry.get_or_404(MODALITY, req.model)

        # Resolve the effective model id (the registry default when req.model is None)
        # so capability-gate errors and decode-failure errors name a real model.
        effective_id = getattr(model, "model_id", None) or (req.model or "<default>")

        # lora_scale only applies to LoRA adapter models; reject early
        # for others (mirrors the supports_img2img capability gate).
        if req.lora_scale is not None:
            if not registry.capabilities(MODALITY, effective_id).get("lora_adapter"):
                return error_response(
                    400,
                    "lora_not_supported",
                    f"model {effective_id!r} is not a LoRA adapter model; "
                    f"lora_scale only applies to LoRA models",
                )

        # img2img gating: if `image` is supplied, the selected model must
        # declare `supports_img2img: True` in its manifest, and the input
        # must decode to a PIL.Image.
        init_image = None
        if req.image is not None:
            if not registry.capabilities(MODALITY, effective_id).get("supports_img2img"):
                return error_response(
                    400,
                    "invalid_parameter",
                    f"model {effective_id!r} does not support img2img; "
                    f"use one of the diffusers-resolved models or sd-turbo",
                )
            try:
                init_image = await decode_image_input(req.image)
            except ValueError as e:
                return error_response(
                    400,
                    "invalid_parameter",
                    f"image decode failed: {e}",
                )

        caps = registry.capabilities(MODALITY, effective_id)
        try:
            opts = _resolve_options(
                caps,
                size=req.size,
                quality=req.quality,
                output_format=req.output_format,
                output_compression=req.output_compression,
                background=req.background,
                moderation=req.moderation,
                style=req.style,
                negative_prompt=req.negative_prompt,
                guidance=req.guidance,
                steps=req.steps,
            )
        except OptionError as e:
            close_decoded_images([init_image] if init_image is not None else [])
            return error_response(400, e.code, e.message)
        width, height = opts["width"], opts["height"]
        steps = opts["steps"]
        output_format = opts["output_format"]

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "width": width,
                "height": height,
                "negative_prompt": req.negative_prompt,
                "steps": steps,
                "guidance": req.guidance,
                "init_image": init_image,
                "strength": req.strength,
                "lora_scale": req.lora_scale,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            with model._inference_lock:
                return model.generate(req.prompt, **kwargs)

        input_images = [init_image] if init_image is not None else []
        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                input_images
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for i in range(req.n):
                result = await run_native_offload(
                    lambda i=i: _call_one(i),
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            # The native ownership helper releases request inputs and any
            # eventual result only after the running backend call settles.
            raise
        except BaseException:
            close_decoded_images(
                input_images
                + [getattr(result, "image", None) for result in results]
            )
            raise

        try:
            return _build_response(
                results,
                prompt=req.prompt,
                response_format=req.response_format,
                output_format=output_format,
                compression=req.output_compression,
                quality=opts["quality"],
                background=opts["background"],
                ignored=opts["ignored"],
            )
        finally:
            close_decoded_images(
                input_images
                + [getattr(r, "image", None) for r in results]
            )

    # ------- /v1/images/edits (inpainting or multi-reference, multipart) -------

    @router.post("/edits")
    async def edits(
        image: list[UploadFile] | None = File(None),
        image_list: list[UploadFile] | None = File(None, alias="image[]"),
        mask: UploadFile | None = File(None),
        prompt: str = Form(...),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: str = Form("b64_json"),
        quality: str | None = Form(None),
        output_format: str | None = Form(None),
        output_compression: int | None = Form(None),
        background: str | None = Form(None),
        moderation: str | None = Form(None),
        style: str | None = Form(None),
        negative_prompt: str | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        seed: int | None = Form(None),
    ):
        # Validate Form fields by hand. Pydantic's BaseModel can't validate
        # a multipart body directly; we mirror the GenerationRequest range
        # checks here so /edits returns the same 400 envelopes.
        if not (1 <= len(prompt) <= 4000):
            return error_response(
                400, "invalid_parameter",
                "prompt must be 1 to 4000 characters",
            )
        if not (1 <= n <= 10):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 10]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'b64_json' or 'url'",
            )
        if steps is not None and not (1 <= steps <= 100):
            return error_response(
                400, "invalid_parameter", "steps must be in [1, 100]",
            )
        if guidance is not None and not (0.0 <= guidance <= 20.0):
            return error_response(
                400, "invalid_parameter", "guidance must be in [0, 20]",
            )

        # `image` parts first, then `image[]` parts, in arrival order.
        parts = list(image or []) + list(image_list or [])
        if not parts:
            return error_response(
                400, "invalid_parameter",
                "at least one image is required (parts 'image' or 'image[]')",
            )
        if len(parts) > MAX_EDIT_IMAGES:
            return error_response(
                400, "too_many_images",
                f"at most {MAX_EDIT_IMAGES} images are accepted; got {len(parts)}",
            )
        if mask is not None and len(parts) > 1:
            return error_response(
                400, "invalid_parameter",
                "a mask applies to exactly one image; send a single image with a mask",
            )

        backend = registry.get_or_404(MODALITY, model)
        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        caps = registry.capabilities(MODALITY, effective_id)

        if mask is not None:
            if not caps.get("supports_inpainting"):
                return error_response(
                    400, "invalid_parameter",
                    f"model {effective_id!r} does not support inpainting",
                )
        else:
            if not caps.get("supports_reference_images"):
                return error_response(
                    400, "reference_images_not_supported",
                    f"model {effective_id!r} cannot edit from reference images "
                    f"without a mask; see GET /v1/capabilities?modality=image/generation "
                    f"for models that support reference images",
                )
            max_refs = caps.get("max_reference_images")
            if max_refs is not None and len(parts) > int(max_refs):
                return error_response(
                    400, "too_many_images",
                    f"model {effective_id!r} accepts at most {int(max_refs)} "
                    f"reference images; got {len(parts)}",
                )

        try:
            opts = _resolve_options(
                caps,
                size=size,
                quality=quality,
                output_format=output_format,
                output_compression=output_compression,
                background=background,
                moderation=moderation,
                style=style,
                negative_prompt=negative_prompt,
                guidance=guidance,
                steps=steps,
            )
        except OptionError as e:
            return error_response(400, e.code, e.message)
        width, height = opts["width"], opts["height"]
        if mask is not None:
            # backend.inpaint takes none of these; report them as ignored.
            sent = {"steps": steps, "seed": seed, "guidance": guidance,
                    "negative_prompt": negative_prompt}
            opts["ignored"] = sorted(
                set(opts["ignored"]) | {k for k, v in sent.items() if v is not None}
            )

        decoded_images = []
        try:
            for part in parts:
                decoded_images.append(await decode_image_file(part))
            mask_image = None
            if mask is not None:
                mask_image = await decode_image_file(mask)
                decoded_images.append(mask_image)
            validate_total_image_pixels(decoded_images)
        except ValueError as e:
            close_decoded_images(decoded_images)
            return error_response(
                400, "invalid_parameter", f"image decode failed: {e}",
            )
        except BaseException:
            close_decoded_images(decoded_images)
            raise

        if mask_image is None:
            # The reference pipeline's image processor rejects these shapes.
            for idx, im in enumerate(decoded_images):
                w, h = im.size
                if min(w, h) < 64 or max(w, h) / min(w, h) > 8:
                    close_decoded_images(decoded_images)
                    return error_response(
                        400, "invalid_parameter",
                        f"reference image {idx} is {w}x{h}; each side must be "
                        f"at least 64 px and the aspect ratio at most 8:1",
                    )

        def _call_one(seed_offset: int):
            with backend._inference_lock:
                if mask_image is not None:
                    return backend.inpaint(
                        prompt,
                        width=width,
                        height=height,
                        init_image=decoded_images[0],
                        mask_image=mask_image,
                    )
                kwargs: dict = {
                    "images": decoded_images,
                    "width": width,
                    "height": height,
                    "steps": opts["steps"],
                    "guidance": guidance,
                    "negative_prompt": negative_prompt,
                }
                if seed is not None:
                    kwargs["seed"] = seed + seed_offset
                return backend.edit(prompt, **kwargs)

        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                decoded_images
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for i in range(n):
                result = await run_native_offload(
                    lambda i=i: _call_one(i),
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            raise
        except BaseException:
            close_decoded_images(
                decoded_images
                + [getattr(result, "image", None) for result in results]
            )
            raise

        try:
            return _build_response(
                results,
                prompt=prompt,
                response_format=response_format,
                output_format=opts["output_format"],
                compression=output_compression,
                quality=opts["quality"],
                background=opts["background"],
                ignored=opts["ignored"],
            )
        finally:
            close_decoded_images(
                decoded_images
                + [getattr(r, "image", None) for r in results]
            )

    # ---------------- /v1/images/variations (multipart) ----------------

    @router.post("/variations")
    async def variations(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str = Form("512x512"),
        response_format: str = Form("b64_json"),
    ):
        if not (1 <= n <= 10):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 10]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'b64_json' or 'url'",
            )
        try:
            w, h = map(int, size.split("x"))
        except Exception:  # noqa: BLE001
            return error_response(
                400, "invalid_parameter",
                f"size must be 'WIDTHxHEIGHT'; got {size!r}",
            )
        if not (64 <= w <= 2048 and 64 <= h <= 2048):
            return error_response(
                400, "invalid_parameter",
                f"size {size} out of supported range (64-2048 per side)",
            )

        backend = registry.get_or_404(MODALITY, model)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        if not registry.capabilities(MODALITY, effective_id).get("supports_variations"):
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support variations",
            )

        try:
            init_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(
                400, "invalid_parameter", f"image decode failed: {e}",
            )

        def _call_one():
            kwargs: dict = {
                "width": w,
                "height": h,
                "init_image": init_image,
            }
            with backend._inference_lock:
                return backend.vary(**kwargs)

        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                [init_image]
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for _ in range(n):
                result = await run_native_offload(
                    _call_one,
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            raise
        except BaseException:
            close_decoded_images(
                [init_image]
                + [getattr(result, "image", None) for result in results]
            )
            raise

        # Variations envelope: no revised_prompt (no prompt).
        try:
            return _build_response(
                results,
                prompt=None,
                response_format=response_format,
                output_format="png",
                compression=None,
                quality="auto",
                background="auto",
                ignored=[],
            )
        finally:
            close_decoded_images(
                [init_image]
                + [getattr(r, "image", None) for r in results]
            )

    return router
