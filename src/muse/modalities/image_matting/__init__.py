"""Image matting modality (package ``image_matting``): clean RGBA foreground
extraction and background removal on opaque images.

Wire contract (multipart/form-data):
- ``POST /v1/images/matting``: ``image`` + background keying / defringe /
  alpha / crop fields -> JSON (``object: "image.matting"``) or raw ``image/png``.

Bundled models:
- ``sprite-matte`` (weightless CPU keying runtime: ``SpriteMatte``)
- ``birefnet-lite`` (neural matting runtime: ``BiRefNetRuntime``)
"""
from muse.modalities.image_matting.client import MattingClient
from muse.modalities.image_matting.protocol import (
    VALID_BACKGROUNDS,
    ImageMattingModel,
    MattingOptions,
    MattingResult,
)
from muse.modalities.image_matting.routes import build_router

MODALITY = "image/matting"

# AST-read by discovery; keep as a literal tuple of paths.
MODEL_OPTIONAL_PATHS = ("/v1/images/matting",)

DISCOVERY = {
    "summary": (
        "Foreground extraction and background removal returning clean RGBA "
        "PNGs with alpha transparency. Supports deterministic CPU keying "
        "(sprite-matte, no weights) for generated sprites/props on solid or "
        "white backdrops, and neural matting (birefnet-lite) for complex "
        "scenes, hair, foliage, and photography."
    ),
    "use_for": [
        "extracting transparent RGBA sprites and props from opaque generated images",
        "removing white, studio, or complex backgrounds with clean anti-aliased edges",
        "defringing background halos and tightly cropping cutouts to their alpha bounding box",
    ],
    "not_for": [
        "promptable multi-instance object masks (use image/segmentation)",
        "snapping images to a pixel-art grid or retro palette (use image/pixelart)",
        "generating or editing image content (use image/generation)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/images/matting",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_matte_image",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {
                "model": "<model>",
                "background": "auto",
                "defringe": True,
            },
        },
    ],
}


def _probe_call(model):
    """Matte a synthetic 64x64 sprite centered on a white background.
    PIL and numpy are imported lazily."""
    import numpy as np
    from PIL import Image

    arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    arr[16:48, 16:48] = (40, 120, 200)
    img = Image.fromarray(arr)
    result = model.matte(img, MattingOptions())
    assert result.image.mode == "RGBA", f"expected RGBA, got {result.image.mode}"
    assert result.width == 64 and result.height == 64
    return {
        "width": result.width,
        "height": result.height,
        "transparent_fraction": result.transparent_fraction,
        "opaque_fraction": result.opaque_fraction,
    }


PROBE_DEFAULTS = {
    "shape": "64x64 synthetic sprite centered on a white background",
    "call": _probe_call,
}

__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "MattingClient",
    "ImageMattingModel",
    "MattingOptions",
    "MattingResult",
    "VALID_BACKGROUNDS",
]
