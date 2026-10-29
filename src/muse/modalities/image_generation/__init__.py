"""Image generation modality: text-to-image.

Wire contract: POST /v1/images/generations with {prompt, model, n?, size?,
response_format? ('b64_json' | 'url'), negative_prompt?, steps?,
guidance?, seed?} returns list of generated images in OpenAI-compatible
shape (b64_json bytes or data URL).

Models declaring `modality = "image/generation"` in their MANIFEST and
satisfying the ImageModel protocol plug into this modality.
"""
from muse.modalities.image_generation.client import (
    GenerationsClient,
    ImageEditsClient,
    ImageVariationsClient,
)
from muse.modalities.image_generation.protocol import ImageModel, ImageResult
from muse.modalities.image_generation.routes import build_router

MODALITY = "image/generation"

DISCOVERY = {
    "summary": (
        "Still image generation. Creates images from a prompt, edits with up "
        "to several reference images for consistent characters, inpaints with "
        "a mask, or makes variations, returned as base64 PNG, JPEG or WebP."
    ),
    "use_for": [
        "illustrations, concept art and backgrounds from a prompt",
        "new scenes of the same character from reference images",
        "inpainting a masked region of an image",
        "alternate takes on an existing image",
    ],
    "not_for": [
        "animation or video (use image/animation or video/generation)",
        "enlarging an image faithfully (use image/upscale)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/images/generations",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "openai:images.generations",
            "mcp_tool": "muse_generate_image",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {
                "model": "<model>",
                "prompt": "a red fox in snow, watercolor",
                "size": "512x512",
            },
        },
        {
            "method": "POST",
            "path": "/v1/images/edits",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": "openai:images.edits",
            "mcp_tool": "muse_edit_image",
            "file_fields": ["image[]", "mask"],
            "media_fields": ["data[].b64_json"],
            "example": {
                "model": "<model>",
                "prompt": "the same character, now reading in a library",
                "size": "1536x1024",
            },
        },
        {
            "method": "POST",
            "path": "/v1/images/variations",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": "openai:images.variations",
            "mcp_tool": "muse_vary_image",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {"model": "<model>"},
        },
    ],
}

# Per-modality probe defaults read by `muse models probe`. Defaults to
# the model's own default size + steps; backends honor capabilities. Each
# output must look like an image (not flat, not noise), else the probe
# fails and a pull disables the model (probe_check.py).
def _probe_call(m):
    from muse.modalities.image_generation.probe_check import check_not_degenerate

    result = m.generate("probe scene")
    check_not_degenerate(result.image, step="generate")
    if getattr(m, "supports_reference_images", False):
        edited = m.edit("the same scene at night", images=[result.image])
        check_not_degenerate(edited.image, step="reference edit")
    return result


PROBE_DEFAULTS = {
    "shape": (
        "default size, 1 image; plus 1 reference edit when supported; "
        "each checked for flat or noise-like output"
    ),
    "call": _probe_call,
}

__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "GenerationsClient",
    "ImageEditsClient",
    "ImageVariationsClient",
    "ImageResult",
    "ImageModel",
]
