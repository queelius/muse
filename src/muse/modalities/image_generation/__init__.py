"""Image generation modality: text-to-image.

Wire contract: POST /v1/images/generations with {prompt, model, n?, size?,
response_format? ('b64_json' | 'url'), negative_prompt?, steps?,
guidance?, seed?} returns list of generated images in OpenAI-compatible
shape (b64_json bytes or data URL).

Models declaring `modality = "image/generation"` in their MANIFEST and
satisfying the ImageModel protocol plug into this modality.
"""
from muse.modalities.image_generation.client import GenerationsClient
from muse.modalities.image_generation.protocol import ImageModel, ImageResult
from muse.modalities.image_generation.routes import build_router

MODALITY = "image/generation"

# Per-modality probe defaults read by `muse models probe`. Defaults to
# the model's own default size + steps; backends honor capabilities.
PROBE_DEFAULTS = {
    "shape": "default size, 1 image",
    "call": lambda m: m.generate("probe scene"),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "GenerationsClient",
    "ImageResult",
    "ImageModel",
]
