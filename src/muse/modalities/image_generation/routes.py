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
    to advertise `capabilities.lora_adapter: True` in its manifest. Also a
    Form field on /v1/images/edits (reference edits; ignored with a mask).
  - `capabilities.prompt_prefix` (e.g. a LoRA trigger phrase) is prepended
    to the prompt unless already present; generations and edits report
    `metadata.prompt_prefix_applied` for models that declare it.
"""
from __future__ import annotations

import base64
import threading
import time

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.fields import (
    FieldError,
    check_choice,
    check_length,
    check_range,
    field_error_response,
)
from muse.modalities._common.images import (
    ImageInputError,
    close_decoded_images,
    decode_image_input,
    image_error_response,
    read_and_decode_uploads,
)
from muse.modalities._common.inference import offload_call, produce_each
from muse.modalities.image_generation.codec import to_bytes, to_data_url
from muse.modalities.image_generation.options import (
    OptionError,
    apply_prompt_prefix,
    check_background,
    check_output,
    ignored_fields,
    parse_size,
    resolve_quality,
    snap_size,
)

MODALITY = "image/generation"

# Field ranges shared by the JSON route (Pydantic) and the multipart twins.
MAX_PROMPT_CHARS = 4000
MAX_N = 10
STEPS_RANGE = (1, 100)
GUIDANCE_RANGE = (0.0, 20.0)
LORA_SCALE_RANGE = (0.0, 2.0)
RESPONSE_FORMATS = ("b64_json", "url")


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=MAX_N)
    size: str | None = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    response_format: str = Field(default="b64_json", pattern="^(b64_json|url)$")
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=STEPS_RANGE[0], le=STEPS_RANGE[1])
    guidance: float | None = Field(
        default=None, ge=GUIDANCE_RANGE[0], le=GUIDANCE_RANGE[1],
    )
    seed: int | None = None
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    lora_scale: float | None = Field(
        default=None, ge=LORA_SCALE_RANGE[0], le=LORA_SCALE_RANGE[1],
    )
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


def _check_common_fields(
    *, n: int, response_format: str, steps=None, guidance=None, lora_scale=None,
) -> None:
    """The multipart twins of GenerationRequest's range checks."""
    check_range("n", n, 1, MAX_N)
    check_choice("response_format", response_format, RESPONSE_FORMATS)
    check_range("steps", steps, *STEPS_RANGE)
    check_range("guidance", guidance, *GUIDANCE_RANGE)
    check_range("lora_scale", lora_scale, *LORA_SCALE_RANGE)


def _entry_encoder(
    *,
    prompt: str | None,
    response_format: str,
    output_format: str,
    compression: int | None,
):
    """One result -> ``(data entry, (width, height))``; runs off-loop.

    A None `prompt` omits `revised_prompt`.
    """
    def _encode(r):
        entry: dict = {}
        if prompt is not None:
            entry["revised_prompt"] = r.metadata.get("prompt", prompt)
        if response_format == "url":
            entry["url"] = to_data_url(r.image, fmt=output_format, compression=compression)
        else:
            entry["b64_json"] = base64.b64encode(
                to_bytes(r.image, fmt=output_format, compression=compression),
            ).decode()
        return entry, (r.width, r.height)

    return _encode


def _build_response(
    encoded: list,
    *,
    output_format: str,
    quality: str,
    background: str,
    ignored: list[str],
    prompt_prefix_applied: bool | None = None,
) -> dict:
    """Shared envelope for generations/edits/variations.

    `size` is what the first result actually measured, not what was
    requested. A None `prompt_prefix_applied` (the model declares no
    prompt_prefix) omits it.
    """
    metadata: dict = {"ignored": ignored}
    if prompt_prefix_applied is not None:
        metadata["prompt_prefix_applied"] = prompt_prefix_applied
    width, height = encoded[0][1]
    return {
        "created": int(time.time()),
        "data": [entry for entry, _size in encoded],
        "size": f"{width}x{height}",
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "metadata": metadata,
    }


async def _run_n(backend, n: int, produce, encode, *, inputs) -> list:
    """Produce and encode ``n`` results off the event loop."""
    stop = threading.Event()
    return await offload_call(
        lambda: produce_each(
            n, produce, encode, lock=backend._inference_lock, stop=stop,
        ),
        inputs=inputs,
        stop=stop,
    )


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
        effective_id = model.model_id
        caps = registry.capabilities(MODALITY, effective_id)

        # lora_scale only applies to LoRA adapter models; reject early
        # for others (mirrors the supports_img2img capability gate).
        if req.lora_scale is not None and not caps.get("lora_adapter"):
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
            if not caps.get("supports_img2img"):
                return error_response(
                    400,
                    "invalid_parameter",
                    f"model {effective_id!r} does not support img2img; "
                    f"use one of the diffusers-resolved models or sd-turbo",
                )
            try:
                init_image = await decode_image_input(req.image)
            except ImageInputError as e:
                return image_error_response(e, field="image")
        inputs = [init_image] if init_image is not None else []

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
            close_decoded_images(inputs)
            return error_response(400, e.code, e.message)
        width, height = opts["width"], opts["height"]
        prompt, prefix_applied = apply_prompt_prefix(req.prompt, caps)

        def _produce(seed_offset: int):
            kwargs: dict = {
                "width": width,
                "height": height,
                "negative_prompt": req.negative_prompt,
                "steps": opts["steps"],
                "guidance": req.guidance,
                "init_image": init_image,
                "strength": req.strength,
                "lora_scale": req.lora_scale,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            return model.generate(prompt, **kwargs)

        encoded = await _run_n(
            model, req.n, _produce,
            _entry_encoder(
                prompt=prompt,
                response_format=req.response_format,
                output_format=opts["output_format"],
                compression=req.output_compression,
            ),
            inputs=inputs,
        )
        return _build_response(
            encoded,
            output_format=opts["output_format"],
            quality=opts["quality"],
            background=opts["background"],
            ignored=opts["ignored"],
            prompt_prefix_applied=(
                prefix_applied if "prompt_prefix" in caps else None
            ),
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
        lora_scale: float | None = Form(None),
    ):
        try:
            check_length("prompt", prompt, 1, MAX_PROMPT_CHARS)
            _check_common_fields(
                n=n, response_format=response_format, steps=steps,
                guidance=guidance, lora_scale=lora_scale,
            )
        except FieldError as e:
            return field_error_response(e)

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
        effective_id = backend.model_id
        caps = registry.capabilities(MODALITY, effective_id)

        if lora_scale is not None and not caps.get("lora_adapter"):
            return error_response(
                400,
                "lora_not_supported",
                f"model {effective_id!r} is not a LoRA adapter model; "
                f"lora_scale only applies to LoRA models",
            )

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
            # backend.inpaint takes none of these (a LoRA inpaint uses the
            # configured default scale); report them as ignored.
            sent = {"steps": steps, "seed": seed, "guidance": guidance,
                    "negative_prompt": negative_prompt, "lora_scale": lora_scale}
            opts["ignored"] = sorted(
                set(opts["ignored"]) | {k for k, v in sent.items() if v is not None}
            )

        uploads = [
            ("image" if len(parts) == 1 else f"image[{i}]", part)
            for i, part in enumerate(parts)
        ]
        if mask is not None:
            uploads.append(("mask", mask))
        decoded_images, error = await read_and_decode_uploads(
            uploads, check_total_pixels=True,
        )
        if error is not None:
            return error
        mask_image = decoded_images[1] if mask is not None else None

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

        prompt, prefix_applied = apply_prompt_prefix(prompt, caps)

        def _produce(seed_offset: int):
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
                "lora_scale": lora_scale,
            }
            if seed is not None:
                kwargs["seed"] = seed + seed_offset
            return backend.edit(prompt, **kwargs)

        encoded = await _run_n(
            backend, n, _produce,
            _entry_encoder(
                prompt=prompt,
                response_format=response_format,
                output_format=opts["output_format"],
                compression=output_compression,
            ),
            inputs=decoded_images,
        )
        return _build_response(
            encoded,
            output_format=opts["output_format"],
            quality=opts["quality"],
            background=opts["background"],
            ignored=opts["ignored"],
            prompt_prefix_applied=(
                prefix_applied if "prompt_prefix" in caps else None
            ),
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
        try:
            _check_common_fields(n=n, response_format=response_format)
            w, h = parse_size(size, {})
        except FieldError as e:
            return field_error_response(e)
        except OptionError as e:
            return error_response(400, e.code, e.message)

        backend = registry.get_or_404(MODALITY, model)
        effective_id = backend.model_id
        if not registry.capabilities(MODALITY, effective_id).get("supports_variations"):
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support variations",
            )

        images, error = await read_and_decode_uploads([("image", image)])
        if error is not None:
            return error
        init_image = images[0]

        encoded = await _run_n(
            backend, n,
            lambda _i: backend.vary(width=w, height=h, init_image=init_image),
            # Variations envelope: no revised_prompt (no prompt).
            _entry_encoder(
                prompt=None, response_format=response_format,
                output_format="png", compression=None,
            ),
            inputs=images,
        )
        return _build_response(
            encoded,
            output_format="png",
            quality="auto",
            background="auto",
            ignored=[],
        )

    return router
