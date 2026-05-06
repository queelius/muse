"""FastAPI routes for /v1/3d/generations and /v1/3d/from-image.

Two routes share this MIME tag (`3d/generation`):

  POST /v1/3d/generations
    Text-to-3d. JSON body, validated via Pydantic. Capability flag
    `supports_text_to_3d` on the manifest gates the route.

  POST /v1/3d/from-image
    Image-to-3d. multipart/form-data. Capability flag
    `supports_image_to_3d` on the manifest gates the route. Per-request
    size cap via `MUSE_3D_INPUT_MAX_BYTES` (default 20 MiB; matches the
    pattern of `MUSE_IMAGE_INPUT_MAX_BYTES` and `MUSE_AUDIO_CLS_MAX_BYTES`).

Both routes use the per-backend `_inference_lock` (auto-attached at
registration time since v0.34.0) to serialize calls into one model
without blocking siblings on the same worker. Backend exceptions
surface as 500 `internal_error` with the original exception text;
tracebacks are logged via `logger.exception` and never returned to
the client.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.model_3d_generation.codec import encode_3d_response


MODALITY = "3d/generation"


logger = logging.getLogger(__name__)


_HARD_DEFAULT_MAX_BYTES = 20 * 1024 * 1024


def _max_bytes() -> int:
    """Read MUSE_3D_INPUT_MAX_BYTES per call so operators can change
    the cap without a server restart. Garbled or non-positive values
    fall back to the 20MB hardcoded default. Matches the v0.34.0
    MUSE_IMAGE_INPUT_MAX_BYTES + MUSE_AUDIO_CLS_MAX_BYTES pattern."""
    raw = os.environ.get("MUSE_3D_INPUT_MAX_BYTES")
    if raw is None:
        return _HARD_DEFAULT_MAX_BYTES
    try:
        n = int(raw)
        if n <= 0:
            raise ValueError("must be positive")
        return n
    except ValueError as e:
        logger.warning(
            "MUSE_3D_INPUT_MAX_BYTES=%r is not a positive integer (%s); "
            "falling back to %d-byte cap",
            raw, e, _HARD_DEFAULT_MAX_BYTES,
        )
        return _HARD_DEFAULT_MAX_BYTES


class _Generation3DRequest(BaseModel):
    """JSON body for POST /v1/3d/generations.

    n is capped at 2 because each 3D asset is heavy (multi-GB VRAM,
    seconds-to-minutes per call). response_format is validated by the
    handler (codec also re-validates) so the 400 envelope reaches the
    client before we touch the backend.
    """
    prompt: str = Field(..., min_length=1)
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
        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        # Resolve effective_id from the backend so error messages name
        # the real model (the registry default when req.model is None).
        effective_id = (
            getattr(backend, "model_id", None)
            or req.model
            or "<default>"
        )

        # Capability gate: supports_text_to_3d=True required. Mismatch
        # returns 400 BEFORE invoking the backend, naming the alternate
        # route in the error message.
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}
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

        def _call():
            with backend._inference_lock:
                return backend.text_to_3d(req.prompt, **kwargs)

        try:
            results = await asyncio.to_thread(_call)
        except Exception as e:  # noqa: BLE001
            logger.exception("text_to_3d failed")
            return error_response(500, "internal_error", str(e))

        body = encode_3d_response(
            results,
            model_id=effective_id,
            response_format=req.response_format,
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
        # Bounded read so a malicious giant upload doesn't OOM the
        # worker. Cap is read per-request (env override takes effect
        # immediately).
        cap = _max_bytes()
        raw = await image.read(cap + 1)
        if not raw:
            return error_response(
                400, "invalid_image", "uploaded image is empty",
            )
        if len(raw) > cap:
            return error_response(
                400, "invalid_image",
                f"uploaded image exceeds MUSE_3D_INPUT_MAX_BYTES={cap} bytes",
            )

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
        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        effective_id = (
            getattr(backend, "model_id", None)
            or model
            or "<default>"
        )

        # Capability gate: supports_image_to_3d=True required.
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}
        if not capabilities.get("supports_image_to_3d"):
            return error_response(
                400, "unsupported_route",
                f"model {effective_id!r} does not support image-to-3d "
                f"generation; use POST /v1/3d/generations with a text "
                f"prompt instead",
            )

        # Write the upload to a temp file so the backend can read it
        # by path (mirrors audio_classification's probe-with-temp-file
        # pattern + the v0.40.x cleanup discipline). delete=False
        # because the inference thread reads it after this with-block;
        # the finally block unlinks unconditionally.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        kwargs: dict = {"n": n}
        if seed is not None:
            kwargs["seed"] = seed

        def _call():
            with backend._inference_lock:
                return backend.image_to_3d(tmp_path, **kwargs)

        try:
            try:
                results = await asyncio.to_thread(_call)
            except Exception as e:  # noqa: BLE001
                logger.exception("image_to_3d failed")
                return error_response(500, "internal_error", str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        body = encode_3d_response(
            results,
            model_id=effective_id,
            response_format=response_format,
        )
        return JSONResponse(content=body)

    return router
