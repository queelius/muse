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
plus the encoded mask (PNG b64 or muse's `p:` run-length format).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, UploadFile

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.images import read_and_decode_upload
from muse.modalities._common.inference import offload_call
from muse.modalities.image_segmentation.codec import encode_segmentation
from muse.modalities.image_segmentation.protocol import CapabilityError


MODALITY = "image/segmentation"
_SIDE_SETTING = "MUSE_SEGMENTATION_MAX_INPUT_SIDE"


class _ClientFault(Exception):
    """The backend rejected the prompt (CapabilityError or ValueError)."""


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
    """Read the per-request input-side cap via muse.core.config (env:
    MUSE_SEGMENTATION_MAX_INPUT_SIDE) per call, so operators can change
    the cap without a server restart. A resolved value that is
    non-positive (parseable but nonsensical as a side cap) falls back
    to the registry default."""
    return config.get_positive("limits.segmentation_max_input_side")


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

        backend = registry.get_or_404(MODALITY, model)
        effective_id = backend.model_id
        capabilities = registry.capabilities(MODALITY, effective_id)

        # Capability gate (defense-in-depth before runtime invocation).
        # A manifest that does not declare the mode's flag does not get it.
        cap_key = _MODE_CAPABILITY[mode]
        if not capabilities.get(cap_key, False):
            human = _MODE_HUMAN[mode]
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support {human} segmentation",
            )

        pil_image, error = await read_and_decode_upload(
            image, max_side=_max_input_side(), side_setting=_SIDE_SETTING,
        )
        if error is not None:
            return error

        def _call() -> dict:
            with backend._inference_lock:
                try:
                    result = backend.segment(
                        pil_image,
                        mode=mode,
                        prompt=prompt,
                        points=parsed_points,
                        boxes=parsed_boxes,
                        max_masks=max_masks,
                    )
                except (CapabilityError, ValueError) as e:
                    # Client faults: an unsupported mode (the runtime's
                    # defense-in-depth for a mismatch the gate above should
                    # already have caught) or a malformed prompt/points/boxes.
                    # Anything else is a server fault for the app's 500
                    # handler, which never leaks the exception text.
                    raise _ClientFault(str(e)) from e
            # Encode off the event loop too: up to 256 masks per request.
            return encode_segmentation(
                result, model_id=effective_id, mask_format=mask_format,
            )

        try:
            return await offload_call(_call, inputs=[pil_image])
        except _ClientFault as e:
            return error_response(400, "invalid_parameter", str(e))

    return router
