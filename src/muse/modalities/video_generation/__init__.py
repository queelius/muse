"""video/generation modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - VideoGenerationModel Protocol, VideoResult dataclass
  - VideoGenerationClient (HTTP)

Wire contract: POST /v1/video/generations
"""
from muse.modalities.video_generation.client import VideoGenerationClient
from muse.modalities.video_generation.protocol import (
    VideoGenerationModel,
    VideoResult,
)
from muse.modalities.video_generation.routes import build_router

MODALITY = "video/generation"

DISCOVERY = {
    "summary": (
        "Video generation. Creates a narrative clip of several seconds from a "
        "prompt as MP4, WebM or frames. GPU only, slow, and each clip is "
        "heavy."
    ),
    "use_for": ["a short shot or scene from a text description", "b-roll and establishing shots"],
    "not_for": [
        "short looping motion (use image/animation)",
        "still images (use image/generation)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/video/generations",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_generate_video",
            "file_fields": [],
            "media_fields": ["data[].b64_json"],
            "example": {
                "model": "<model>",
                "prompt": "waves rolling onto a beach at dusk",
                "duration_seconds": 5,
            },
        },
    ],
}

# Per-modality probe defaults read by `muse models probe`. Use a very
# short duration + few steps so the measurement is bounded; real
# inference still reveals true VRAM peak. On a 12GB GPU the probe
# should complete in ~30-60s for Wan 1.3B.
# Read by muse.core.discovery via AST. With sequential CPU offload (the
# bundled Wan default on 8-12 GB cards) the 2-second probe clip alone takes
# more than the default 10-minute probe budget.
PROBE_TIMEOUT_SECONDS = 2700

PROBE_DEFAULTS = {
    "shape": "2-second clip at default size, steps=10",
    "call": lambda m: m.generate(
        "a flag waving in the wind",
        duration_seconds=2.0,
        steps=10,
    ),
}

__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "VideoGenerationClient",
    "VideoGenerationModel",
    "VideoResult",
]
