"""Image upscale modality: image-to-image super-resolution.

Wire contract: POST /v1/images/upscale (multipart/form-data) with
{image, model?, scale?, prompt?, negative_prompt?, steps?, guidance?,
seed?, n?, response_format?} returns list of upscaled images in the
OpenAI-compatible envelope (b64_json bytes or data URL).

Bundled model: stable-diffusion-x4-upscaler. The HF resolver also
synthesizes manifests for any diffusers-shape upscaler (model_index.json
+ image-to-image tag + upscaler-name allowlist).

Models declaring `modality = "image/upscale"` in their MANIFEST and
satisfying the ImageUpscaleModel protocol plug into this modality.
"""
from muse.modalities.image_upscale.client import ImageUpscaleClient
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)
from muse.modalities.image_upscale.routes import build_router

MODALITY = "image/upscale"

DISCOVERY = {
    "summary": (
        "Super-resolution. Enlarges an image, adding plausible detail, as "
        "base64 PNG."
    ),
    "use_for": ["making a low-resolution image larger and sharper"],
    "not_for": [
        "generating new images (use image/generation)",
        "turning a raster into scalable vectors (use image/vectorization)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/images/upscale",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_upscale_image",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {"model": "<model>", "scale": 4},
        },
    ],
}


def _make_probe_image():
    """Generate a small synthetic test image for `muse models probe`.

    PIL is imported here lazily so the modality package loads without
    PIL on the host python (per the muse `--help` should not need ML
    deps contract).
    """
    from PIL import Image
    return Image.new("RGB", (128, 128), (128, 128, 128))


PROBE_DEFAULTS = {
    "shape": "128x128 -> 512x512 (4x), 20 steps",
    "call": lambda m: m.upscale(_make_probe_image(), scale=4),
}

__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageUpscaleClient",
    "ImageUpscaleModel",
    "UpscaleResult",
]
