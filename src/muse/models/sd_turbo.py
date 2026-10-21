"""SD-Turbo: 1-step distilled Stable Diffusion, 512x512.

Model by Stability AI (SAI Community License, non-commercial + limited
commercial). Very fast (1 inference step by default) at modest quality;
good first backend to prove the image/generation modality end-to-end.

The Model class is the generic `DiffusersText2ImageModel` (the runtime
behind resolver-pulled diffusers text-to-image models), which serves
text-to-image, img2img, inpainting and variations. `load_backend`
splats the capabilities below (default_size, default_steps,
default_guidance) into its constructor. The fp16 variant is requested
because the snapshot holds only `.fp16.` weights (see allow_patterns).
"""
from muse.modalities.image_generation.runtimes.diffusers import (
    DiffusersText2ImageModel as Model,
)


MANIFEST = {
    "model_id": "sd-turbo",
    "modality": "image/generation",
    "hf_repo": "stabilityai/sd-turbo",
    "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
    "description": "Stable Diffusion Turbo: 1-step distilled, 512x512",
    "license": "SAI Community License",
    "pip_extras": (
        # torch: hard runtime dep. Pulled transitively by accelerate,
        # but declaring it keeps the contract self-contained.
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        # transformers: required by StableDiffusionPipeline at construct
        # time (CLIPTextModel text encoder). diffusers lists it as an
        # optional extra, so we declare it here or per-model venvs end
        # up missing it and the worker exits at load.
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        # Pinned "cuda", not "auto": allow_patterns below downloads ONLY
        # *.fp16.safetensors (no fp32 weights on disk), so an "auto"
        # resolution on a CPU-only host would load an fp16 pipeline that
        # errors at inference (no CPU kernel for half-precision matmul).
        # CPU diffusion is impractical anyway. Matches the
        # stable-audio-open / animatediff / x4-upscaler precedent.
        "device": "cuda",
        "default_size": (512, 512),
        # 1-step distilled: one denoise step, no classifier-free guidance.
        "default_steps": 1,
        "default_guidance": 0.0,
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
        "supports_img2img": True,
        "supports_inpainting": True,
        "supports_variations": True,
        # Conservative VRAM/RAM peak estimate (fp16, default 512x512).
        # Annotation feeds `muse models list`; `muse models probe sd-turbo`
        # measures the real number on this hardware.
        "memory_gb": 4.0,
        # SD-Turbo ships fp32 + fp16 + a standalone single-file checkpoint
        # (~47 GB total). The diffusers runtime only loads the fp16
        # subfolder weights (~4 GB). Mirror the resolver plugin's
        # allow_patterns so bundled-pull doesn't bloat the cache either.
        "allow_patterns": [
            "*.fp16.safetensors", "*.json", "*.txt", "*.md",
            "feature_extractor/*.json",
            "scheduler/*.json",
            "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
            "tokenizer/*",
            "unet/*.fp16.safetensors", "unet/*.json",
            "vae/*.fp16.safetensors", "vae/*.json",
        ],
    },
}
