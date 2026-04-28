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


def _probe_call(model):
    """Build 5s of silence as a wav and pass to model.transcribe.

    Backends accept either a path or bytes; we serialize to a tempfile
    so any backend signature works, then clean up. Silence is fine for
    a memory probe -- we want the activation working set, not output
    quality.
    """
    import io
    import os
    import tempfile
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 5)
    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(buf.read())
        path = f.name
    try:
        return model.transcribe(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


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
