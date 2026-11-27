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
        "No word timestamps, no speed control (`speed` is ignored); `stream: true` streams in chunks on CUDA only (one chunk on CPU)",
        "About 0.4 s per short line on a 12 GB GPU (RTF ~0.12); much slower on CPU",
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
        # 2026-11-25 with the CUDA-graphed talker + code predictor: peak 2.6 GB
        # (512-token KV cache; +57 MB per further 512 tokens of a long line).
        # 2026-11-26 with the Triton kernels: 2.33 GiB on a short line, 2.80 GiB
        # (~3.0 GB) on a 25 s paragraph, mostly the speech decoder's full chunk.
        # With int8_weights (CUDA with Triton): 2.0 GiB at load, 2.34 GiB
        # (~2.5 GB) on that paragraph; 2.8 leaves margin. Where the layers stay
        # float the peak is ~3.0 GB, which the pull-time probe records.
        "memory_gb": 2.8,
        # Talker and code-predictor layers stored as int8 only: ~40% less
        # frame time, 0.5 GiB less VRAM, speech measured unchanged (ASR,
        # UTMOS, blind listening); see docs/MODALITIES.md.
        "int8_weights": True,
        # Streaming chunk schedule in 12 Hz frames (80 ms each): a short
        # first chunk for a fast start, then doubling up to the steady size.
        "stream_first_frames": 2,
        "stream_chunk_frames": 16,
        "mode": "clone",
        "supports_saved_voices": True,
        "voice_cloning": True,
    },
}
