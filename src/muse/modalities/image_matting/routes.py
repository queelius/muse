"""FastAPI router for ``POST /v1/images/matting`` (multipart/form-data).

Uploads arrive via the shared bounded decoder (``muse.modalities._common.images``)
and inference + output encoding run off the event loop under the backend's
inference lock via ``muse.modalities._common.inference.offload_call``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.fields import (
    FieldError,
    check_choice,
    check_range,
    field_error_response,
)
from muse.modalities._common.images import (
    close_decoded_images,
    read_and_decode_upload,
)
from muse.modalities._common.inference import offload_call
from muse.modalities.image_matting.codec import encode_matting, png_bytes
from muse.modalities.image_matting.protocol import VALID_BACKGROUNDS, MattingOptions

logger = logging.getLogger(__name__)

MODALITY = "image/matting"


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=[MODALITY])

    @router.post("/matting")
    async def matte(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        background: str = Form("auto"),
        bg_threshold: int = Form(236),
        color_tolerance: int = Form(24),
        hard_alpha: bool = Form(False),
        alpha_threshold: int = Form(128),
        defringe: bool = Form(True),
        fringe_threshold: int = Form(212),
        clear_enclosed_white: bool = Form(False),
        crop: bool = Form(False),
        response_format: str = Form("json"),
    ):
        bg = (background or "").strip().lower()
        fmt = (response_format or "").strip().lower()
        try:
            check_choice("response_format", fmt, ("json", "png"))
            check_choice("background", bg, VALID_BACKGROUNDS)
            check_range("bg_threshold", bg_threshold, 0, 255)
            check_range("color_tolerance", color_tolerance, 0, 255)
            check_range("alpha_threshold", alpha_threshold, 0, 255)
            check_range("fringe_threshold", fringe_threshold, 0, 255)
        except FieldError as e:
            return field_error_response(e)

        backend = registry.get_or_404(MODALITY, model)

        source, error = await read_and_decode_upload(image, "image")
        if error is not None:
            return error

        opts = MattingOptions(
            background=bg,
            bg_threshold=bg_threshold,
            color_tolerance=color_tolerance,
            hard_alpha=hard_alpha,
            alpha_threshold=alpha_threshold,
            defringe=defringe,
            fringe_threshold=fringe_threshold,
            clear_enclosed_white=clear_enclosed_white,
            crop=crop,
        )

        def _call():
            result = backend.matte(source, opts)
            try:
                if fmt == "png":
                    return png_bytes(result.image)
                return encode_matting(result, backend.model_id)
            finally:
                close_decoded_images([result.image])

        try:
            out = await offload_call(
                _call,
                lock=backend._inference_lock,
                inputs=[source],
            )
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))
        except Exception:  # noqa: BLE001
            logger.exception("matting failed")
            return error_response(
                500, "internal_error", "matting failed; see server logs"
            )

        if fmt == "png":
            return Response(content=out, media_type="image/png")
        return JSONResponse(content=out)

    return router
