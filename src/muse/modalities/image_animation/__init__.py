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

# Per-modality probe defaults read by `muse models probe`. Defaults to
# the model's own default frames + size; backends honor capabilities.
PROBE_DEFAULTS = {
    "shape": "default frames @ default size",
    "call": lambda m: m.generate("probe motion"),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "AnimationModel",
    "AnimationResult",
    "AnimationsClient",
]
