"""AnimateDiff motion v3 + SD 1.5 base, pulled as one pinned bundle.

The Model class is the generic `AnimateDiffRuntime`. `muse pull` lays the
two pinned artifacts out under `local_dir` (`motion_adapter/` and
`base_model/`); `load_backend` splats the capabilities below, including
`base_model` and the two subdir names, into the runtime constructor.
"""
from muse.modalities.image_animation.runtimes.animatediff import (
    AnimateDiffRuntime as Model,
)


_ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-3"
_ADAPTER_REVISION = "2e8139b1d1269fd8a21deb96ad19455e187692eb"
_BASE_REPO = "emilianJR/epiCRealism"
_BASE_REVISION = "6522cf856b8c8e14638a0aaa7bd89b1b098aed17"
_ADAPTER_SUBDIR = "motion_adapter"
_BASE_SUBDIR = "base_model"
_ADAPTER_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]
_ADAPTER_REQUIRED_PATTERNS = ["config.json", "*.safetensors"]
_BASE_ALLOW_PATTERNS = [
    "*.safetensors", "*.json", "*.txt", "*.model", "*.md",
    "feature_extractor/*", "scheduler/*",
    "safety_checker/*.safetensors", "safety_checker/*.json",
    "text_encoder/*.safetensors", "text_encoder/*.json",
    "tokenizer/*", "unet/*.safetensors", "unet/*.json",
    "vae/*.safetensors", "vae/*.json",
]
_BASE_REQUIRED_PATTERNS = [
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "safety_checker/*.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "unet/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
]


MANIFEST = {
    "model_id": "animatediff-motion-v3",
    "modality": "image/animation",
    "hf_repo": _ADAPTER_REPO,
    "revision": _ADAPTER_REVISION,
    "description": "AnimateDiff motion v3 + SD 1.5 base, 16 frames @ 8fps, 512x512",
    "summary": "Short looping text-to-animation clips (16 frames at 8 fps) on an SD 1.5 photoreal base",
    "strengths": [
        "Loops by default; animated WebP, GIF or MP4 output",
        "8-24 frames per clip",
        "Apache 2.0 motion adapter",
    ],
    "limits": [
        "SD 1.5-era quality at 512x512; weak at text and fine detail",
        "About 2 s per clip; not for narrative video (use video/generation)",
        "Needs a CUDA GPU (~10 GB); the epiCRealism base is CreativeML OpenRAIL-M",
    ],
    "quality_tier": "draft",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        "supports_text_to_animation": True,
        "supports_image_to_animation": False,
        "default_frames": 16,
        "default_fps": 8,
        "min_frames": 8,
        "max_frames": 24,
        "default_size": (512, 512),
        "default_steps": 25,
        "default_guidance": 7.5,
        "device": "cuda",
        "base_model": _BASE_REPO,
        "base_model_revision": _BASE_REVISION,
        "adapter_model_subdir": _ADAPTER_SUBDIR,
        "base_model_subdir": _BASE_SUBDIR,
        # SD 1.5 base + motion adapter at fp16, plus per-frame activations
        # for 16 frames at 512x512. Conservative peak estimate.
        "memory_gb": 10.0,
    },
    "hf_artifacts": [
        {
            "repo_id": _ADAPTER_REPO,
            "revision": _ADAPTER_REVISION,
            "subdir": _ADAPTER_SUBDIR,
            "allow_patterns": _ADAPTER_ALLOW_PATTERNS,
            "required_patterns": _ADAPTER_REQUIRED_PATTERNS,
        },
        {
            "repo_id": _BASE_REPO,
            "revision": _BASE_REVISION,
            "subdir": _BASE_SUBDIR,
            "allow_patterns": _BASE_ALLOW_PATTERNS,
            "required_patterns": _BASE_REQUIRED_PATTERNS,
        },
    ],
}
