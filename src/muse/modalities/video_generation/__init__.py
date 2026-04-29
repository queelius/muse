"""video/generation modality (placeholder).

Final exports added in Task D once routes + client land.
"""
from muse.modalities.video_generation.protocol import (
    VideoGenerationModel,
    VideoResult,
)

MODALITY = "video/generation"

__all__ = [
    "MODALITY",
    "VideoGenerationModel",
    "VideoResult",
]
