"""FastAPI router for /v1/images/upscale (multipart/form-data).

Mirrors the wire shape of /v1/images/edits and /v1/images/variations:
the source image arrives as a multipart file upload, all other fields
arrive as Form parameters. Output envelope mirrors /v1/images/generations.

Capability gating: the model's `supported_scales` capability must
include the requested `scale`; otherwise the route returns 400. An
env-tunable input-side cap (MUSE_UPSCALE_MAX_INPUT_SIDE, default 1024)
rejects oversized inputs before they reach the pipeline.
"""
from __future__ import annotations

import base64
import threading
import time

from fastapi import APIRouter, File, Form, UploadFile

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.fields import (
    FieldError,
    check_choice,
    check_length,
    check_range,
    field_error_response,
)
from muse.modalities._common.images import read_and_decode_upload
from muse.modalities._common.inference import offload_call, produce_each
from muse.modalities.image_upscale.codec import to_bytes, to_data_url

MODALITY = "image/upscale"
_SIDE_SETTING = "MUSE_UPSCALE_MAX_INPUT_SIDE"


def _max_input_side() -> int:
    """Read the per-request input-side cap via muse.core.config.

    Default 1024. Tunable via MUSE_UPSCALE_MAX_INPUT_SIDE so users with
    more VRAM can lift the cap without a code change. A resolved value
    that is non-positive falls back to the registry default.
    """
    return config.get_positive("limits.upscale_max_input_side")


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/upscale"])

    @router.post("/upscale")
    async def upscale(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        scale: int | None = Form(None),
        prompt: str = Form(""),
        negative_prompt: str | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        seed: int | None = Form(None),
        n: int = Form(1),
        response_format: str = Form("b64_json"),
    ):
        # Manual Form validation (Pydantic doesn't validate multipart).
        try:
            check_range("n", n, 1, 4)
            check_choice("response_format", response_format, ("b64_json", "url"))
            check_length("prompt", prompt, 0, 4000)
            check_length("negative_prompt", negative_prompt, 0, 4000)
            check_range("steps", steps, 1, 100)
            check_range("guidance", guidance, 0.0, 20.0)
        except FieldError as e:
            return field_error_response(e)

        backend = registry.get_or_404(MODALITY, model)
        effective_id = backend.model_id
        capabilities = registry.capabilities(MODALITY, effective_id)
        supported = list(capabilities.get("supported_scales") or [4])
        if scale is None:
            scale = int(capabilities.get("default_scale") or supported[0])
        if scale not in supported:
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} only supports scales: {supported}",
            )

        init_image, error = await read_and_decode_upload(
            image, max_side=_max_input_side(), side_setting=_SIDE_SETTING,
        )
        if error is not None:
            return error

        def _produce(seed_offset: int):
            kwargs: dict = {
                "scale": scale,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance": guidance,
            }
            if seed is not None:
                kwargs["seed"] = seed + seed_offset
            return backend.upscale(init_image, **kwargs)

        def _encode(r) -> dict:
            entry: dict = {"revised_prompt": prompt or None}
            if response_format == "url":
                entry["url"] = to_data_url(r.image, fmt="png")
            else:
                entry["b64_json"] = base64.b64encode(
                    to_bytes(r.image, fmt="png"),
                ).decode()
            return entry

        stop = threading.Event()
        data = await offload_call(
            lambda: produce_each(
                n, _produce, _encode, lock=backend._inference_lock, stop=stop,
            ),
            inputs=[init_image],
            stop=stop,
        )
        return {"created": int(time.time()), "data": data}

    return router
