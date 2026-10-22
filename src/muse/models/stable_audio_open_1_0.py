"""Stable Audio Open 1.0: 47s music + SFX, 44.1kHz stereo.

Model by Stability AI (Apache 2.0). Diffusion-based latent audio
generation via diffusers.StableAudioPipeline. ~3.4GB at fp16.

Defaults align with the official model card recommendations:
50 inference steps, guidance scale 7.0, 10s default duration,
44.1kHz stereo output. Native max ~47s.

The Model class is the generic `StableAudioRuntime`; `load_backend`
splats the capabilities below (duration defaults and bounds, steps,
guidance, sample rate) into its constructor.
"""
from muse.modalities.audio_generation.runtimes.stable_audio import (
    StableAudioRuntime as Model,
)


MANIFEST = {
    "model_id": "stable-audio-open-1.0",
    "modality": "audio/generation",
    "hf_repo": "stabilityai/stable-audio-open-1.0",
    "revision": "f21265c1e2710b3bd2386596943f0007f55f802e",
    "description": (
        "Stable Audio Open 1.0: 47s music + SFX, 44.1kHz stereo, "
        "Apache 2.0"
    ),
    "summary": "Sound effects, ambiences and short music beds from a text prompt",
    "strengths": [
        "Strongest at sound effects and field recordings",
        "Music and SFX routes, up to 47 s of 44.1 kHz stereo",
    ],
    "limits": [
        "No realistic vocals and weaker at music than at SFX (ace-step-v1-3.5b makes songs)",
        "Needs a CUDA GPU (~6 GB); English prompts work best",
        "Upstream weights use the Stability AI Community License, not Apache 2.0; commercial use is limited",
    ],
    "quality_tier": "good",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        # transformers required by StableAudioPipeline's text encoder
        # (T5). diffusers lists it as an optional extra; declare it
        # explicitly so per-model venvs aren't broken on transitive
        # availability.
        "transformers>=4.36.0",
        "accelerate",
        "soundfile",  # FLAC encoding in the codec layer.
        # numpy is pulled by torch but the runtime imports it directly
        # for waveform shaping (#110).
        "numpy",
        # StableAudioPipeline's default CosineDPMSolverMultistepScheduler
        # needs torchsde for its stochastic SDE solver; diffusers imports
        # it lazily at load time, so a venv without it fails at model load
        # (not at import). Declare it explicitly.
        "torchsde",
    ),
    # ffmpeg is optional; only needed for mp3/opus response_format.
    # Declaring it here lets `muse pull` ensure it's available, but
    # the server doesn't crash without it (codec returns 400 instead).
    "system_packages": ("ffmpeg",),
    "capabilities": {
        # cuda by default: Stable Audio is impractical on CPU.
        # Users without a GPU can override via curated capability
        # overlay.
        "device": "cuda",
        "supports_music": True,
        "supports_sfx": True,
        "default_duration": 10.0,
        "min_duration": 1.0,
        "max_duration": 47.0,
        "default_sample_rate": 44100,
        "default_steps": 50,
        "default_guidance": 7.0,
        # Conservative VRAM/RAM peak estimate (fp16, 10s clips).
        # Annotation feeds `muse models list`; `muse models probe`
        # measures the real number.
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "scheduler/*.json",
        "transformer/*.fp16.safetensors", "transformer/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
    ],
}
