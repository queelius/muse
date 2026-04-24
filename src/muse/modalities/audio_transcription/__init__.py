"""audio/transcription modality: automatic speech recognition.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - Word, Segment, TranscriptionResult dataclasses
  - TranscriptionModel Protocol

Wire contract (OpenAI-compat):
  - POST /v1/audio/transcriptions
  - POST /v1/audio/translations

The `build_router` import is deferred (circular between routes.py and
this __init__), matching the pattern used by audio_speech.
"""
from __future__ import annotations

from muse.modalities.audio_transcription.protocol import (
    Word,
    Segment,
    TranscriptionResult,
    TranscriptionModel,
)


MODALITY = "audio/transcription"


def build_router(registry):
    """Lazy import of routes.build_router to keep __init__ deps-light.

    routes.py imports FastAPI at module top; __init__ must import cheaply
    so discovery works in the supervisor process (no ML deps installed).
    """
    from muse.modalities.audio_transcription.routes import (
        build_router as _build,
    )
    return _build(registry)


__all__ = [
    "MODALITY",
    "build_router",
    "Word",
    "Segment",
    "TranscriptionResult",
    "TranscriptionModel",
]
