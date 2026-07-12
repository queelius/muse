"""Audio quality and speech-naturalness assessment.

``POST /v1/audio/quality`` accepts one multipart audio file and returns
named, scaled quality scores. The ``audio/quality`` MIME-style modality is
separate from ``audio/classification`` because MOS and aesthetic axes are
continuous measurements, not a class-probability distribution.
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import wave

from muse.modalities.audio_quality.client import AudioQualityClient
from muse.modalities.audio_quality.protocol import (
    AudioDurationExceededError,
    AudioQualityModel,
    AudioQualityResult,
    AudioQualityScore,
)
from muse.modalities.audio_quality.routes import build_router


MODALITY = "audio/quality"
MODEL_OPTIONAL_PATHS = ("/v1/audio/quality",)


def _probe_call(model):
    """Assess a short 16kHz tone and always remove the temporary WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        frames = b"".join(
            struct.pack("<h", int(2500 * math.sin(2 * math.pi * 220 * i / 16000)))
            for i in range(16000)
        )
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(frames)
        return model.assess(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


PROBE_DEFAULTS = {
    "shape": "1 second of 16kHz mono audio",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioQualityClient",
    "AudioDurationExceededError",
    "AudioQualityModel",
    "AudioQualityResult",
    "AudioQualityScore",
]
