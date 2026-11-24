"""sprite-matte: deterministic CPU foreground matting for ``image/matting``
(no weights).

Serves ``POST /v1/images/matting`` using pure Pillow + numpy on the CPU:
border flood-fill background keying (auto, white, corners), optional enclosed
neutral-white pocket removal, 1px anti-aliased halo defringing, optional hard
alpha binarization, and tight alpha bounding-box cropping.

The ``Model`` class is the ``SpriteMatte`` runtime.
"""
from muse.modalities.image_matting.runtimes.sprite_matte import SpriteMatte as Model

MANIFEST = {
    "model_id": "sprite-matte",
    "modality": "image/matting",
    "weights": "none",
    "description": (
        "Deterministic CPU foreground matting: border flood-fill keying, "
        "1px halo defringing, enclosed white pocket removal, hard alpha, and "
        "tight alpha crop. No weights, MIT."
    ),
    "summary": "Fast deterministic CPU background removal for sprites and props on solid or white backdrops",
    "strengths": [
        "Instant CPU execution with zero model weights to download (~0.5 GB RAM)",
        "Auto, white, or corner border flood-fill with 1px halo defringing",
        "Optional enclosed white-pocket clearing, binary alpha, and tight alpha crop",
    ],
    "limits": [
        "Relies on uniform or near-white backgrounds; use birefnet-lite for busy photographic scenes",
        "Enclosed white removal may clear bright neutral interior details matching the threshold",
    ],
    "quality_tier": "good",
    "license": "MIT",
    "pip_extras": (
        "Pillow>=10.0",
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cpu",
        "memory_gb": 0.5,
        "supports_background_modes": ["auto", "white", "corners"],
        "supports_enclosed_white": True,
        "supports_defringe": True,
        "supports_hard_alpha": True,
        "supports_crop": True,
    },
}
