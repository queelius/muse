"""birefnet-lite: ZhengPeng7/BiRefNet_lite bundled model script for
``image/matting``.

BiRefNet_lite is a lightweight high-resolution neural dichotomous image
segmentation / foreground matting model (MIT license). Runs smoothly on GPU
(fp16) or CPU (fp32) via ``BiRefNetRuntime``.
"""
from muse.modalities.image_matting.runtimes.birefnet import BiRefNetRuntime as Model

MANIFEST = {
    "model_id": "birefnet-lite",
    "modality": "image/matting",
    "hf_repo": "ZhengPeng7/BiRefNet_lite",
    "revision": "aa62cd87eafb9cc43056d08ef3615a14628b831d",
    "description": (
        "BiRefNet_lite: lightweight neural foreground matting and background "
        "removal for sprites, props, portraits, and scenes. MIT."
    ),
    "summary": "Neural RGBA foreground extraction for complex scenes, soft edges, foliage, and photography",
    "strengths": [
        "Handles complex, textured, and non-uniform backgrounds where color keying fails",
        "Produces smooth anti-aliased alpha mattes at 1024x1024 internal resolution",
        "Lightweight (~1.5 GB VRAM/RAM) with MIT license",
    ],
    "limits": [
        "Requires downloading neural weights and runs slower on CPU than sprite-matte",
        "Very fine sub-pixel pixel-art edges may benefit from hard_alpha=true or sprite-matte",
    ],
    "quality_tier": "good",
    "license": "MIT",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "Pillow>=10.0",
        "numpy",
        "einops",
        "timm",
        "kornia",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "memory_gb": 1.5,
        "input_size": 1024,
        "supports_defringe": True,
        "supports_hard_alpha": True,
        "supports_crop": True,
    },
    "allow_patterns": [
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.md",
        "*.py",
    ],
}
