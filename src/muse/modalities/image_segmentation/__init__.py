"""Image segmentation modality (placeholder; finalized in routes task).

Wire contract: POST /v1/images/segment (multipart/form-data) returning
a list of binary masks plus per-mask metadata. SAM-2 family is the
primary target; CLIPSeg-style text-prompted segmentation is opt-in via
the ``supports_text_prompts`` capability flag.

The full ``__init__.py`` exports (build_router, ImageSegmentationClient,
PROBE_DEFAULTS) land once routes + client modules exist. This skeleton
is enough to make protocol + codec importable.
"""
from muse.modalities.image_segmentation.protocol import (
    ImageSegmentationModel,
    MaskRecord,
    SegmentationResult,
)

MODALITY = "image/segmentation"

__all__ = [
    "MODALITY",
    "ImageSegmentationModel",
    "MaskRecord",
    "SegmentationResult",
]
