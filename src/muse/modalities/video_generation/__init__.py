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

# Per-modality probe defaults read by `muse models probe`. Use a very
# short duration + few steps so the measurement is bounded; real
# inference still reveals true VRAM peak. On a 12GB GPU the probe
# should complete in ~30-60s for Wan 1.3B.
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
    "PROBE_DEFAULTS",
    "build_router",
    "VideoGenerationClient",
    "VideoGenerationModel",
    "VideoResult",
]
