"""Wan 2.1 T2V 1.3B: Apache 2.0, 5s clips at 832x480.

Default low-VRAM video generation bundle. The 1.3B transformer alone is
small, but the pipeline also bundles a UMT5-XXL text encoder (~11GB at
fp16); full-resident load is ~11-12GB and OOMs a 12GB card. The bundle
therefore defaults to sequential CPU offload (`capabilities.cpu_offload
= "sequential"`), which fits 8-12GB GPUs at the cost of speed. On a
bigger card, set `server.video_cpu_offload model` (whole-component
offload, faster) or `off` (full .to(cuda), fastest) to trade VRAM
headroom for speed.

The Model class is the generic `WanRuntime` (WanPipeline, with a
DiffusionPipeline fallback for older diffusers). `load_backend` splats
the capabilities below into its constructor: the generation defaults,
`cpu_offload` and `vae_tiling` (applied by the shared `place_pipeline`
helper).
"""
from muse.modalities.video_generation.runtimes.wan_runtime import (
    WanRuntime as Model,
)


MANIFEST = {
    "model_id": "wan2-1-t2v-1-3b",
    "modality": "video/generation",
    # The diffusers-format repo (model_index.json + transformer/ vae/
    # text_encoder/ scheduler/ tokenizer/ subfolders) that WanPipeline.
    # from_pretrained requires. The original "Wan-AI/Wan2.1-T2V-1.3B" repo
    # ships the native Wan layout (loose .pth files, no model_index.json)
    # which diffusers cannot load -- the "-Diffusers" suffix is the port.
    "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
    "description": (
        "Wan 2.1 T2V 1.3B: 5s videos at 832x480, Apache 2.0. The UMT5-XXL "
        "text encoder makes full-resident load ~11-12GB, so the bundle "
        "defaults to sequential CPU offload to fit 8-12GB GPUs (slower; "
        "set server.video_cpu_offload model|off on a bigger card for speed)"
    ),
    "summary": "Text-to-video clips of a few seconds at 832x480",
    "strengths": [
        "Clips up to 10 s at 832x480; MP4 or WebM output",
        "Fits 8-12 GB GPUs through sequential CPU offload",
        "Apache 2.0",
    ],
    "limits": [
        "Slow: sequential offload trades speed for VRAM (many minutes per clip)",
        "480p only and text-to-video only, so no character consistency from a reference image",
        "Needs a CUDA GPU",
    ],
    "quality_tier": "good",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.32.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",
        "imageio[ffmpeg]>=2.31.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cuda",
        "default_duration_seconds": 5.0,
        "default_fps": 5,
        "default_size": (832, 480),
        "min_duration_seconds": 1.0,
        "max_duration_seconds": 10.0,
        "default_steps": 30,
        "default_guidance": 5.0,
        "supports_image_to_video": False,
        "memory_gb": 5.0,
        "cpu_offload": "sequential",
        "vae_tiling": True,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "scheduler/*.json",
        "transformer/*.safetensors", "transformer/*.json",
        "vae/*.safetensors", "vae/*.json",
        "text_encoder/*.safetensors", "text_encoder/*.json",
        "tokenizer/*",
    ],
}
