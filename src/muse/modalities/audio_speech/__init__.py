"""Audio speech modality: text-to-speech.

Wire contract: POST /v1/audio/speech with {input, model, voice?, speed?,
response_format? ('wav' | 'opus'), stream?} returns WAV/Opus audio bytes
(or SSE-streamed base64 PCM chunks when stream=True).

Models declaring `modality = "audio/speech"` in their MANIFEST and
satisfying the TTSModel protocol plug into this modality.
"""
from muse.modalities.audio_speech.client import SpeechClient
from muse.modalities.audio_speech.protocol import (
    AudioChunk,
    AudioResult,
    TTSModel,
)
from muse.modalities.audio_speech.routes import build_router

MODALITY = "audio/speech"

__all__ = [
    "MODALITY",
    "build_router",
    "SpeechClient",
    "AudioChunk",
    "AudioResult",
    "TTSModel",
]
