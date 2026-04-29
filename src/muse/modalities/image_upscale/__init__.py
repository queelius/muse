"""Image upscale modality: image-to-image super-resolution (skeleton).

Routes and HTTP client land in Task C. This file only exports the
data-shape primitives so Task A can ship without circular imports.
"""
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)

MODALITY = "image/upscale"

__all__ = ["MODALITY", "ImageUpscaleModel", "UpscaleResult"]
