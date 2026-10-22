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

DISCOVERY = {
    "summary": (
        "Speech recognition. Turns spoken audio into text, with optional "
        "segment and word timestamps and subtitle formats (srt, vtt). The "
        "translations route renders non-English speech as English text."
    ),
    "use_for": [
        "transcribing recorded speech or a narration track",
        "producing subtitles (srt, vtt) from audio",
        "translating foreign-language speech into English text",
    ],
    "not_for": [
        "timing a transcript you already trust (use audio/alignment; it is more exact)",
        "classifying sounds or emotion (use audio/classification)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": "openai:audio.transcriptions",
            "mcp_tool": "muse_transcribe",
            "file_fields": ["file"],
            "media_fields": [],
            "example": {"model": "<model>", "response_format": "json"},
        },
        {
            "method": "POST",
            "path": "/v1/audio/translations",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": "openai:audio.translations",
            "mcp_tool": "muse_translate_audio",
            "file_fields": ["file"],
            "media_fields": [],
            "example": {"model": "<model>", "response_format": "json"},
        },
    ],
}



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
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "Word",
    "Segment",
    "TranscriptionResult",
    "TranscriptionModel",
    "TranscriptionClient",
]
