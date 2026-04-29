"""FastAPI router for /v1/images/segment (multipart/form-data).

Wire shape: multipart upload of an image plus form fields for the
prompt mode dispatch. Modes: auto, points, boxes, text. Each mode
maps to a capability flag on the model; mismatched modes return 400
before the runtime is invoked.

Capability gating table:
    auto   -> supports_automatic
    points -> supports_point_prompts
    boxes  -> supports_box_prompts
    text   -> supports_text_prompts

points and boxes arrive as JSON-encoded strings; the route parses
them with json.loads and validates the shape (list of pairs / list
of quads of integers). Bad JSON or shape mismatches yield 400.

The output envelope (see codec.encode_segmentation) carries an id,
the model id, the dispatch mode, the input image_size in PIL
convention (W, H), and a list of masks with index, score, bbox, area,
plus the encoded mask (PNG b64 or COCO RLE).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from threading import Lock

from fastapi import APIRouter, File, Form, UploadFile

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.image_generation.image_input import decode_image_file
from muse.modalities.image_segmentation.codec import encode_segmentation


logger = logging.getLogger(__name__)

MODALITY = "image/segmentation"
_inference_lock = Lock()


_VALID_MODES = ("auto", "points", "boxes", "text")
_VALID_MASK_FORMATS = ("png_b64", "rle")

_MODE_CAPABILITY = {
    "auto": "supports_automatic",
    "points": "supports_point_prompts",
    "boxes": "supports_box_prompts",
    "text": "supports_text_prompts",
}

_MODE_HUMAN = {
    "auto": "automatic",
    "points": "point-prompted",
    "boxes": "box-prompted",
    "text": "text-prompted",
}


def _max_input_side() -> int:
    raw = os.environ.get("MUSE_SEGMENTATION_MAX_INPUT_SIDE", "2048")
    try:
        v = int(raw)
        return v if v > 0 else 2048
    except ValueError:
        return 2048


def _parse_points_json(raw: str) -> list[list[int]] | None:
    """Parse a JSON-encoded list of [x, y] integer pairs.

    Returns None on bad shape or bad JSON; the caller decides the 400
    message.
    """
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    out: list[list[int]] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return None
        try:
            out.append([int(entry[0]), int(entry[1])])
        except (TypeError, ValueError):
            return None
    return out


def _parse_boxes_json(raw: str) -> list[list[int]] | None:
    """Parse a JSON-encoded list of [x1, y1, x2, y2] integer quads."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    out: list[list[int]] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            return None
        try:
            out.append([int(entry[0]), int(entry[1]),
                        int(entry[2]), int(entry[3])])
        except (TypeError, ValueError):
            return None
    return out


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/segmentation"])

    @router.post("/segment")
    async def segment(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        mode: str = Form("auto"),
        prompt: str | None = Form(None),
        points: str | None = Form(None),
        boxes: str | None = Form(None),
        mask_format: str = Form("png_b64"),
        max_masks: int = Form(16),
    ):
        # Manual Form validation (Pydantic doesn't validate multipart).
        if mode not in _VALID_MODES:
            return error_response(
                400, "invalid_parameter",
                f"mode must be one of {list(_VALID_MODES)}",
            )
        if mask_format not in _VALID_MASK_FORMATS:
            return error_response(
                400, "invalid_parameter",
                f"mask_format must be one of {list(_VALID_MASK_FORMATS)}",
            )
        if not (1 <= max_masks <= 256):
            return error_response(
                400, "invalid_parameter",
                "max_masks must be in [1, 256]",
            )
        if mode == "text" and (prompt is None or not prompt.strip()):
            return error_response(
                400, "invalid_parameter",
                "mode='text' requires a non-empty prompt",
            )
        if prompt is not None and len(prompt) > 4000:
            return error_response(
                400, "invalid_parameter",
                "prompt must be 0 to 4000 characters",
            )

        parsed_points: list[list[int]] | None = None
        parsed_boxes: list[list[int]] | None = None
        if mode == "points":
            if points is None:
                return error_response(
                    400, "invalid_parameter",
                    "mode='points' requires points (JSON list of [x, y] pairs)",
                )
            parsed_points = _parse_points_json(points)
            if parsed_points is None:
                return error_response(
                    400, "invalid_parameter",
                    "points must be a JSON list of [x, y] integer pairs",
                )
        if mode == "boxes":
            if boxes is None:
                return error_response(
                    400, "invalid_parameter",
                    "mode='boxes' requires boxes (JSON list of [x1, y1, x2, y2] quads)",
                )
            parsed_boxes = _parse_boxes_json(boxes)
            if parsed_boxes is None:
                return error_response(
                    400, "invalid_parameter",
                    "boxes must be a JSON list of [x1, y1, x2, y2] integer quads",
                )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities", {}) or {}

        # Capability gate (defense-in-depth before runtime invocation).
        cap_key = _MODE_CAPABILITY[mode]
        if not capabilities.get(cap_key, True):
            human = _MODE_HUMAN[mode]
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support {human} segmentation",
            )

        try:
            pil_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(
                400, "invalid_parameter", f"image decode failed: {e}",
            )

        max_side = _max_input_side()
        ow, oh = pil_image.size
        if ow > max_side or oh > max_side:
            return error_response(
                400, "invalid_parameter",
                f"image too large: {ow}x{oh} exceeds max input side "
                f"{max_side} (set MUSE_SEGMENTATION_MAX_INPUT_SIDE to raise)",
            )

        def _call() -> object:
            with _inference_lock:
                return backend.segment(
                    pil_image,
                    mode=mode,
                    prompt=prompt,
                    points=parsed_points,
                    boxes=parsed_boxes,
                    max_masks=max_masks,
                )

        try:
            result = await asyncio.to_thread(_call)
        except RuntimeError as e:
            # Runtime-level capability mismatches surface as RuntimeError;
            # treat them as 400 invalid_parameter (the gate above should
            # have caught these but we defend in depth).
            return error_response(
                400, "invalid_parameter", str(e),
            )
        except ValueError as e:
            return error_response(
                400, "invalid_parameter", str(e),
            )

        return encode_segmentation(
            result, model_id=effective_id, mask_format=mask_format,
        )

    return router
