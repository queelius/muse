"""Reference-text forced alignment with word-level timestamps.

``POST /v1/audio/alignments`` accepts one multipart audio file plus its
trusted transcript and returns timestamps for the transcript's words. The
``audio/alignment`` MIME-style modality deliberately performs no ASR: it
answers where known words occur, rather than deciding what was spoken.
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import wave

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


def _probe_call(model):
    """Align one word against a short tone and remove the temporary WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        frames = b"".join(
            struct.pack(
                "<h", int(2500 * math.sin(2 * math.pi * 220 * i / 16000))
            )
            for i in range(16000)
        )
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(frames)
        return model.align(path, "test", language="English")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


PROBE_DEFAULTS = {
    "shape": "1 second of 16kHz mono audio plus one reference word",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
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
