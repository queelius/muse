"""Qwen3-TTS 1.7B VoiceDesign: speech in a voice described in words.

The request's ``instructions`` describe the voice (age, gender, timbre,
accent, manner); without them the manifest's ``default_instructions``
narrator description is used. No named voices: save a designed line to
``qwen3-tts-0.6b-base`` with ``POST /v1/audio/speech/voices`` to reuse it.

The Model class is the generic ``Qwen3TTSRuntime`` in design mode.
"""
from muse.modalities.audio_speech.runtimes.qwen3_tts import (
    DEFAULT_INSTRUCTIONS,
    Qwen3TTSRuntime as Model,
)


MANIFEST = {
    "model_id": "qwen3-tts-1.7b-voicedesign",
    "modality": "audio/speech",
    "hf_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    "description": "Qwen3-TTS 1.7B VoiceDesign: TTS in a voice described by text, 24kHz",
    "summary": "Speech in a new voice described in plain words, for designing character voices",
    "strengths": [
        "Voice from a text description via `instructions` (age, timbre, accent, emotion)",
        "Expressive, natural prosody; 24 kHz",
        "English and 9 other languages, detected automatically; Apache 2.0",
    ],
    "limits": [
        "The same description gives a somewhat different voice each call; save a take "
        "to qwen3-tts-0.6b-base to keep a character consistent",
        "No word timestamps, no speed control (`speed` is ignored), `voice` is ignored (describe the voice in `instructions`); `stream: true` streams in chunks on CUDA only (one chunk on CPU)",
        "About 0.5 s per short line on a 12 GB GPU (RTF ~0.15); much slower on CPU",
    ],
    "quality_tier": "good",
    "license": "Apache-2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "numpy",
        "qwen-tts==0.1.1",
    ),
    "system_packages": (),
    "capabilities": {
        "sample_rate": 24000,
        "device": "auto",
        # Spike: peak 4.4 GB VRAM (bf16, torch max allocated) on short lines.
        # 2026-11-25 with the CUDA-graphed talker + code predictor: still 4.4 GB.
        # 2026-11-26 with the Triton kernels: 4.03 GiB on a short line, 4.70 GiB
        # on a 25 s paragraph. With int8_weights (CUDA with Triton): 3.94 GiB
        # (~4.2 GB) at load, before the layers are quantized, then 2.5 GiB
        # resident and 3.3 GiB on that paragraph; 4.5 leaves margin. Where the
        # layers stay float the peak is ~5.0 GB, which the pull-time probe records.
        "memory_gb": 4.5,
        # Talker and code-predictor layers stored as int8 only: ~40% less
        # frame time, 1.4 GiB less VRAM in use, speech measured unchanged
        # (ASR, UTMOS, blind listening); see docs/MODALITIES.md.
        "int8_weights": True,
        # Streaming chunk schedule in 12 Hz frames (80 ms each): a short
        # first chunk for a fast start, then larger ones.
        "stream_first_frames": 4,
        "stream_chunk_frames": 16,
        "mode": "design",
        "default_instructions": DEFAULT_INSTRUCTIONS,
        "supports_instructions": True,
        "voices": [],
    },
}
