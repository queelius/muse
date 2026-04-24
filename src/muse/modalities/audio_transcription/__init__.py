"""audio/transcription modality: automatic speech recognition.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - Word, Segment, TranscriptionResult dataclasses
  - TranscriptionModel Protocol

Wire contract (OpenAI-compat):
  - POST /v1/audio/transcriptions
  - POST /v1/audio/translations
"""
from muse.modalities.audio_transcription.protocol import (
    Word,
    Segment,
    TranscriptionResult,
    TranscriptionModel,
)
from muse.modalities.audio_transcription.routes import build_router
from muse.modalities.audio_transcription.client import TranscriptionClient


MODALITY = "audio/transcription"


__all__ = [
    "MODALITY",
    "build_router",
    "Word",
    "Segment",
    "TranscriptionResult",
    "TranscriptionModel",
    "TranscriptionClient",
]
