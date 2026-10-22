"""Reference-text forced alignment with word-level timestamps.

``POST /v1/audio/alignments`` accepts one multipart audio file plus its
trusted transcript and returns timestamps for the transcript's words. The
``audio/alignment`` MIME-style modality deliberately performs no ASR: it
answers where known words occur, rather than deciding what was spoken.
"""
from __future__ import annotations

from muse.modalities._common.probe_audio import probe_wav
from muse.modalities.audio_alignment.client import AudioAlignmentClient
from muse.modalities.audio_alignment.protocol import (
    AlignmentWord,
    AudioAlignmentDecodeError,
    AudioAlignmentDurationExceededError,
    AudioAlignmentModel,
    AudioAlignmentResult,
    UnalignableTextError,
    UnsupportedAlignmentLanguageError,
)
from muse.modalities.audio_alignment.routes import build_router


MODALITY = "audio/alignment"
MODEL_OPTIONAL_PATHS = ("/v1/audio/alignments",)

DISCOVERY = {
    "summary": (
        "Forced alignment. Given an audio file and its trusted transcript, "
        "returns start and end times for every word. It locates the words "
        "you supply; it does not decide what was spoken."
    ),
    "use_for": [
        "word timing for captions and subtitles from a known script",
        "animation and cut cues keyed to when a word is spoken",
        "syncing a generated narration to its source text",
    ],
    "not_for": [
        "speech recognition when no transcript exists (use audio/transcription)",
        "judging audio quality (use audio/quality)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/alignments",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_align",
            "file_fields": ["file"],
            "media_fields": [],
            "example": {"model": "<model>", "text": "Hello from muse.", "language": "English"},
        },
    ],
}



def _probe_call(model):
    """Align one word against a short tone."""
    with probe_wav(1.0, tone_hz=220) as path:
        return model.align(path, "test", language="English")


PROBE_DEFAULTS = {
    "shape": "1 second of 16kHz mono audio plus one reference word",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "DISCOVERY",
    "MODEL_OPTIONAL_PATHS",
    "PROBE_DEFAULTS",
    "build_router",
    "AlignmentWord",
    "AudioAlignmentClient",
    "AudioAlignmentDecodeError",
    "AudioAlignmentDurationExceededError",
    "AudioAlignmentModel",
    "AudioAlignmentResult",
    "UnalignableTextError",
    "UnsupportedAlignmentLanguageError",
]
