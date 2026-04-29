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
    "PROBE_DEFAULTS",
    "build_router",
    "ImageUpscaleClient",
    "ImageUpscaleModel",
    "UpscaleResult",
]
