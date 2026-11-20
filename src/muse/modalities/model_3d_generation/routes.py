"""FastAPI routes for /v1/3d/generations and /v1/3d/from-image.

Two routes share this MIME tag (`3d/generation`):

  POST /v1/3d/generations
    Text-to-3d. JSON body, validated via Pydantic. Capability flag
    `supports_text_to_3d` on the manifest gates the route.

  POST /v1/3d/from-image
    Image-to-3d. multipart/form-data. Capability flag
    `supports_image_to_3d` on the manifest gates the route. Per-request
    size cap via `MUSE_3D_INPUT_MAX_BYTES` (default 20 MiB; over it is
    413 `payload_too_large`, an undecodable upload 400 `invalid_image`).
    The image reaches the runtime as RGBA when it has transparency (TRELLIS
    and Hunyuan3D use the alpha cutout) and RGB otherwise (P, L, CMYK,
    I;16 ... are normalized); TripoSR converts to RGB itself.

Both routes run the backend call and the GLB envelope encoding off the
event loop under the per-backend `_inference_lock` (auto-attached at
registration time since v0.34.0). Backend exceptions reach the app's
global handler: a generic 500 `internal_error`, with the traceback
logged and never returned to the client.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.images import read_and_decode_upload
from muse.modalities._common.inference import offload_call
from muse.modalities.model_3d_generation.codec import encode_3d_response


MODALITY = "3d/generation"
_BYTES_SETTING = "MUSE_3D_INPUT_MAX_BYTES"


def _has_transparency(image) -> bool:
    return image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        "transparency" in image.info
    )


def _normalize_mode(image):
    """RGBA when the image has transparency, else RGB (may be ``image``)."""
    target = "RGBA" if _has_transparency(image) else "RGB"
    return image if image.mode == target else image.convert(target)


def _max_bytes() -> int:
    """Read the image-to-3d upload byte cap via muse.core.config (env:
    MUSE_3D_INPUT_MAX_BYTES) per call, so operators can change the cap
    without a server restart. Garbled values fall back to the registry
    default (20MB) with a logged warning. A resolved value that is None
    (empty env) or non-positive (parseable but nonsensical as a byte
    cap) also falls back to the registry default."""
    return config.get_positive("limits.model_3d_input_max_bytes")


class _Generation3DRequest(BaseModel):
    """JSON body for POST /v1/3d/generations.

    n is capped at 2 because each 3D asset is heavy (multi-GB VRAM,
    seconds-to-minutes per call). response_format is validated by the
    handler (codec also re-validates) so the 400 envelope reaches the
    client before we touch the backend.
    """
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=2)
    seed: int | None = None
    response_format: str = "b64_json"


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    # ---------------- /v1/3d/generations (text-to-3d, JSON) ----------------

    @router.post("/v1/3d/generations")
    async def text_to_3d_route(req: _Generation3DRequest):
        # response_format gate: 400 before invoking the backend.
        if req.response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                f"response_format must be 'b64_json' or 'url'; "
                f"got {req.response_format!r}",
            )

        # Resolve backend or 404.
        backend = registry.get_or_404(MODALITY, req.model)
        effective_id = backend.model_id

        # Capability gate: supports_text_to_3d=True required. Mismatch
        # returns 400 BEFORE invoking the backend, naming the alternate
        # route in the error message.
        capabilities = registry.capabilities(MODALITY, effective_id)
        if not capabilities.get("supports_text_to_3d"):
            return error_response(
                400, "unsupported_route",
                f"model {effective_id!r} does not support text-to-3d "
                f"generation; use POST /v1/3d/from-image with an "
                f"input image instead",
            )

        # Forward kwargs. Always include n; seed only when explicitly
        # provided (so backends that branch on `seed is None` for
        # randomness keep that branch).
        kwargs: dict = {"n": req.n}
        if req.seed is not None:
            kwargs["seed"] = req.seed

        body = await offload_call(
            lambda: encode_3d_response(
                backend.text_to_3d(req.prompt, **kwargs),
                model_id=effective_id,
                response_format=req.response_format,
            ),
            lock=backend._inference_lock,
        )
        return JSONResponse(content=body)

    # ---------------- /v1/3d/from-image (image-to-3d, multipart) -----------

    @router.post("/v1/3d/from-image")
    async def image_to_3d_route(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        n: int = Form(1),
        seed: int | None = Form(None),
        response_format: str = Form("b64_json"),
    ):
        cap = _max_bytes()

        # n + response_format gates: 400 before invoking the backend.
        if not (1 <= n <= 2):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 2]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                f"response_format must be 'b64_json' or 'url'; "
                f"got {response_format!r}",
            )

        # Resolve backend or 404.
        backend = registry.get_or_404(MODALITY, model)
        effective_id = backend.model_id

        # Capability gate: supports_image_to_3d=True required.
        capabilities = registry.capabilities(MODALITY, effective_id)
        if not capabilities.get("supports_image_to_3d"):
            return error_response(
                400, "unsupported_route",
                f"model {effective_id!r} does not support image-to-3d "
                f"generation; use POST /v1/3d/generations with a text "
                f"prompt instead",
            )

        pil_image, error = await read_and_decode_upload(
            image, max_bytes=cap, bytes_setting=_BYTES_SETTING,
        )
        if error is not None:
            return error

        kwargs: dict = {"n": n}
        if seed is not None:
            kwargs["seed"] = seed

        def _call() -> dict:
            source = _normalize_mode(pil_image)
            try:
                results = backend.image_to_3d(source, **kwargs)
            finally:
                if source is not pil_image:
                    source.close()
            return encode_3d_response(
                results, model_id=effective_id, response_format=response_format,
            )

        body = await offload_call(
            _call, lock=backend._inference_lock, inputs=[pil_image],
        )
        return JSONResponse(content=body)

    return router
