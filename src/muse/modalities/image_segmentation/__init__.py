"""Image segmentation modality: promptable segmentation.

Wire contract: POST /v1/images/segment (multipart/form-data) with
{image, model?, mode?, prompt?, points?, boxes?, mask_format?,
max_masks?} returns a list of binary masks plus per-mask metadata
(score, bbox, area). Mode dispatch is capability-gated: a request
for ``mode="text"`` against a model declaring
``supports_text_prompts: False`` returns 400 before the runtime
runs.

Bundled model: sam2-hiera-tiny. The HF resolver also synthesizes
manifests for any HF repo whose tags include mask-generation or
image-segmentation.

Models declaring ``modality = "image/segmentation"`` in their
MANIFEST and satisfying the ImageSegmentationModel protocol plug
into this modality.
"""
from muse.modalities.image_segmentation.client import ImageSegmentationClient
from muse.modalities.image_segmentation.protocol import (
    ImageSegmentationModel,
    MaskRecord,
    SegmentationResult,
)
from muse.modalities.image_segmentation.routes import build_router

MODALITY = "image/segmentation"


def _make_probe_image():
    """Generate a small synthetic test image for `muse models probe`.

    PIL is imported here lazily so the modality package loads without
    PIL on the host python (per the muse `--help` should not need ML
    deps contract).
    """
    from PIL import Image
    return Image.new("RGB", (256, 256), (128, 128, 128))


PROBE_DEFAULTS = {
    "shape": "256x256, automatic mode, max_masks=4",
    "call": lambda m: m.segment(_make_probe_image(), mode="auto", max_masks=4),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageSegmentationClient",
    "ImageSegmentationModel",
    "MaskRecord",
    "SegmentationResult",
]
