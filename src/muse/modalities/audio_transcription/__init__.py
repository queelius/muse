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
from muse.modalities._common.probe_audio import probe_wav
from muse.modalities.audio_transcription.protocol import (
    Word,
    Segment,
    TranscriptionResult,
    TranscriptionModel,
)
from muse.modalities.audio_transcription.routes import build_router
from muse.modalities.audio_transcription.client import TranscriptionClient


MODALITY = "audio/transcription"


def _probe_call(model):
    """Transcribe 5s of 16kHz silence: the activation working set is what
    a memory probe wants, not output quality."""
    with probe_wav(5.0) as path:
        return model.transcribe(path)


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "5s 16k mono",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "Word",
    "Segment",
    "TranscriptionResult",
    "TranscriptionModel",
    "TranscriptionClient",
]
