"""Image animation modality: text-to-animation.

Wire contract (added in Task C): POST /v1/images/animations with
{prompt, model, frames?, fps?, size?, response_format? ('webp' | 'gif' |
'mp4' | 'frames_b64'), negative_prompt?, steps?, guidance?, seed?,
init_image?, strength?} returns the encoded animation bytes (or list of
base64 PNGs for frames_b64).

Models declaring `modality = "image/animation"` in their MANIFEST and
satisfying the AnimationModel protocol plug into this modality.

For now this package exports the protocol + result dataclass only. The
MODALITY constant and build_router come in Task C.
"""
from muse.modalities.image_animation.protocol import (
    AnimationModel,
    AnimationResult,
)

__all__ = [
    "AnimationModel",
    "AnimationResult",
]
