"""Qwen3-TTS 0.6B Base: speech in a saved (cloned) voice.

Each voice is a 1-30 s reference clip plus its transcript, saved with
``POST /v1/audio/speech/voices`` and removed with ``DELETE
/v1/audio/speech/voices/{name}``; ``voice`` picks one per request. The
bundled read-only ``narrator`` voice is the default. Saved voices live under
``<paths.home>/voices/qwen3-tts-0.6b-base/``.

The Model class is the generic ``Qwen3TTSRuntime`` in clone mode.
"""
from muse.modalities.audio_speech.runtimes.qwen3_tts import Qwen3TTSRuntime as Model


MANIFEST = {
    "model_id": "qwen3-tts-0.6b-base",
    "modality": "audio/speech",
    "hf_repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
    "description": "Qwen3-TTS 0.6B Base: TTS that clones saved reference voices, 24kHz",
    "summary": "Speech in a saved voice cloned from a short clip, for consistent character voices",
    "strengths": [
        "Clones a voice from a 1-30 s clip and its transcript; saved voices persist",
        "Keeps a character's voice consistent across lines; bundled `narrator` default",
        "English and 9 other languages, detected automatically; Apache 2.0",
    ],
    "limits": [
        "Clone quality follows the reference: noisy or wrongly transcribed clips clone poorly",
        "No word timestamps, no speed control (`speed` is ignored), no streaming (one chunk)",
        "About 2.5 s per short line on a 12 GB GPU (RTF ~0.8); much slower on CPU",
    ],
    "quality_tier": "good",
    "license": "Apache-2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "numpy",
        "qwen-tts==0.1.1",
        # POST /v1/audio/speech/voices decodes the uploaded reference with
        # TorchCodec (muse.modalities._common.audio_decode) in this venv.
        "torchcodec>=0.12",
    ),
    # TorchCodec links ffmpeg's libav at import time.
    "system_packages": ("ffmpeg",),
    "capabilities": {
        "sample_rate": 24000,
        "device": "auto",
        # Spike: peak 2.5 GB VRAM (bf16, torch max allocated) on short lines.
        # 2026-11-25 with the CUDA-graphed code predictor: peak 2.3 GB.
        "memory_gb": 3.0,
        "mode": "clone",
        "supports_saved_voices": True,
        "voice_cloning": True,
    },
}
