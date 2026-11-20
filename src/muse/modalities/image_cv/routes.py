"""FastAPI routes for image/cv: depth, keypoints, object detection.

Three routes share the modality tag (image/cv) and registry namespace.
Capability flags on each model's manifest gate which route accepts it:

  POST /v1/images/depth        requires supports_depth=True
  POST /v1/images/keypoints    requires supports_keypoints=True
  POST /v1/images/detect       requires supports_detection=True

Mismatch returns 400 with code `wrong_primitive`.

All three routes:
  - Multipart in (``_common.images.read_and_decode_upload``;
    MUSE_IMAGE_INPUT_MAX_BYTES cap, 413 / 400 per the shared table).
  - Inference and envelope encoding run off the event loop under the
    per-backend inference lock.
  - JSON out via the codec layer's encode_*_envelope helpers.
  - OpenAI-shape error envelope on 400/404/413; backend failures reach
    the app's generic 500 handler.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.images import read_and_decode_upload
from muse.modalities._common.inference import offload_call
from muse.modalities.image_cv.codec import (
    encode_depth_envelope,
    encode_detections_envelope,
    encode_keypoints_envelope,
)


MODALITY = "image/cv"


def _capability_gate(
    registry: ModalityRegistry, backend, *, primitive: str,
) -> JSONResponse | None:
    """Return None if the backend supports the primitive; else a 400.

    primitive is one of "depth", "keypoints", "detection". The model's
    manifest must set the matching `supports_*` capability flag.
    """
    flag_name = {
        "depth": "supports_depth",
        "keypoints": "supports_keypoints",
        "detection": "supports_detection",
    }[primitive]
    capabilities = registry.capabilities(MODALITY, backend.model_id)
    if not capabilities.get(flag_name):
        return error_response(
            400, "wrong_primitive",
            f"model {backend.model_id!r} does not support the {primitive} "
            f"primitive (no {flag_name}=True in manifest); use a model "
            f"declaring that capability",
        )
    return None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    # ---------- POST /v1/images/depth ----------

    @router.post("/v1/images/depth")
    async def images_depth(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        response_format: str = Form("png16"),
    ):
        if response_format not in ("png16", "float32"):
            return error_response(
                400, "invalid_parameter",
                f"response_format must be 'png16' or 'float32'; "
                f"got {response_format!r}",
            )

        backend = registry.get_or_404(MODALITY, model)
        gate = _capability_gate(registry, backend, primitive="depth")
        if gate is not None:
            return gate

        pil_image, error = await read_and_decode_upload(image)
        if error is not None:
            return error

        body = await offload_call(
            lambda: encode_depth_envelope(
                backend.estimate_depth(pil_image),
                response_format=response_format,
            ),
            lock=backend._inference_lock,
            inputs=[pil_image],
        )
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    # ---------- POST /v1/images/keypoints ----------

    @router.post("/v1/images/keypoints")
    async def images_keypoints(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        threshold: float | None = Form(None),
    ):
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            return error_response(
                400, "invalid_parameter",
                f"threshold must be in [0, 1]; got {threshold}",
            )

        backend = registry.get_or_404(MODALITY, model)
        gate = _capability_gate(registry, backend, primitive="keypoints")
        if gate is not None:
            return gate

        pil_image, error = await read_and_decode_upload(image)
        if error is not None:
            return error

        kwargs: dict = {}
        if threshold is not None:
            kwargs["threshold"] = threshold

        body = await offload_call(
            lambda: encode_keypoints_envelope(
                backend.detect_keypoints(pil_image, **kwargs),
            ),
            lock=backend._inference_lock,
            inputs=[pil_image],
        )
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    # ---------- POST /v1/images/detect ----------

    @router.post("/v1/images/detect")
    async def images_detect(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        threshold: float | None = Form(None),
        max_detections: int | None = Form(None),
    ):
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            return error_response(
                400, "invalid_parameter",
                f"threshold must be in [0, 1]; got {threshold}",
            )
        if max_detections is not None and (max_detections < 1 or max_detections > 10000):
            return error_response(
                400, "invalid_parameter",
                f"max_detections must be in [1, 10000]; got {max_detections}",
            )

        backend = registry.get_or_404(MODALITY, model)
        gate = _capability_gate(registry, backend, primitive="detection")
        if gate is not None:
            return gate

        pil_image, error = await read_and_decode_upload(image)
        if error is not None:
            return error

        kwargs: dict = {}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if max_detections is not None:
            kwargs["max_detections"] = max_detections

        body = await offload_call(
            lambda: encode_detections_envelope(
                backend.detect_objects(pil_image, **kwargs),
            ),
            lock=backend._inference_lock,
            inputs=[pil_image],
        )
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    return router
