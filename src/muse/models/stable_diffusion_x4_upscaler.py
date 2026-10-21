"""stabilityai/stable-diffusion-x4-upscaler bundled model script.

4x latent diffusion super-resolution. Apache 2.0. ~3GB on disk.
Uses diffusers.StableDiffusionUpscalePipeline. Roughly 30-60s per
512 -> 2048 image at 20 steps on a 12GB GPU.

The Model class is the generic `DiffusersUpscaleRuntime`; `load_backend`
splats the capabilities below (default_scale, supported_scales,
default_steps, default_guidance) into its constructor.
"""
from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
    DiffusersUpscaleRuntime as Model,
)


MANIFEST = {
    "model_id": "stable-diffusion-x4-upscaler",
    "modality": "image/upscale",
    "hf_repo": "stabilityai/stable-diffusion-x4-upscaler",
    "revision": "572c99286543a273bfd17fac263db5a77be12c4c",
    "description": (
        "SD x4 upscaler: 4x super-resolution via latent diffusion, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU is too slow at 20 steps for 256 -> 1024; mark cuda as the
        # advisory device. `auto` device selection still works if the
        # operator overrides it.
        "device": "cuda",
        "default_scale": 4,
        "supported_scales": [4],
        "default_steps": 20,
        "default_guidance": 9.0,
        # Conservative VRAM peak estimate (fp16, 256 -> 1024).
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "feature_extractor/*.json",
        "scheduler/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
        "unet/*.fp16.safetensors", "unet/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
    ],
}
