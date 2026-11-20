"""FastAPI routes for /v1/images/ocr.

Multipart in (image file + optional model/prompt/max_new_tokens),
JSON out (id, model, text, usage). Mirrors /v1/audio/transcriptions.

The upload goes through ``_common.images.read_and_decode_upload``
(MUSE_IMAGE_INPUT_MAX_BYTES cap; decode off the event loop). Inference
runs off-loop under the per-backend inference lock.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.images import read_and_decode_upload
from muse.modalities._common.inference import offload_call
from muse.modalities.image_ocr.codec import encode_ocr


MODALITY = "image/ocr"


# Hard ceiling on per-request decoder budget. Far above any sensible
# OCR (Nougat full-page is ~2-3K tokens). Beyond this is either a
# misuse or a budget-burner; reject with 400.
_HARD_MAX_NEW_TOKENS = 4096
_MAX_PROMPT_CHARS = 4000


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/images/ocr")
    async def images_ocr(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
    ):
        # Cap validation: positive integer, bounded.
        if max_new_tokens is not None:
            if max_new_tokens < 1 or max_new_tokens > _HARD_MAX_NEW_TOKENS:
                return error_response(
                    400, "invalid_parameter",
                    f"max_new_tokens must be in [1, {_HARD_MAX_NEW_TOKENS}]; "
                    f"got {max_new_tokens}",
                )
        if prompt is not None and len(prompt) > _MAX_PROMPT_CHARS:
            return error_response(
                400, "invalid_parameter",
                f"prompt must be 0 to {_MAX_PROMPT_CHARS} characters",
            )

        backend = registry.get_or_404(MODALITY, model)

        pil_image, error = await read_and_decode_upload(image)
        if error is not None:
            return error

        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens

        result = await offload_call(
            lambda: backend.ocr(pil_image, **kwargs),
            lock=backend._inference_lock,
            inputs=[pil_image],
        )

        body = encode_ocr(result)
        # Override result.model_id with the effective registry id in
        # case the runtime's model_id drifted (paranoia guard, mirrors
        # the moderation route's pattern).
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    return router
