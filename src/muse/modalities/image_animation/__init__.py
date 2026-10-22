"""image/animation modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - AnimationModel Protocol, AnimationResult dataclass
  - AnimationsClient (HTTP)

Wire contract: POST /v1/images/animations
"""
from muse.modalities.image_animation.client import AnimationsClient
from muse.modalities.image_animation.protocol import (
    AnimationModel,
    AnimationResult,
)
from muse.modalities.image_animation.routes import build_router

MODALITY = "image/animation"

DISCOVERY = {
    "summary": (
        "Short looping animations. Generates a few seconds of looping motion "
        "from a prompt, as animated WebP, GIF or MP4, optionally starting "
        "from an image."
    ),
    "use_for": [
        "looping motion graphics and animated stickers",
        "bringing a still image to life for a few seconds",
    ],
    "not_for": [
        "narrative clips longer than a few seconds (use video/generation)",
        "still images (use image/generation)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/images/animations",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_generate_animation",
            "file_fields": ["image"],
            "media_fields": ["data[].b64_json"],
            "example": {
                "model": "<model>",
                "prompt": "a lighthouse in a storm, waves rolling",
                "frames": 16,
            },
        },
    ],
}

# Per-modality probe defaults read by `muse models probe`. Defaults to
# the model's own default frames + size; backends honor capabilities.
PROBE_DEFAULTS = {
    "shape": "default frames @ default size",
    "call": lambda m: m.generate("probe motion"),
}

__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "AnimationModel",
    "AnimationResult",
    "AnimationsClient",
]
