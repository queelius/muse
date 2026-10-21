"""Audio quality and speech-naturalness assessment.

``POST /v1/audio/quality`` accepts one multipart audio file and returns
named, scaled quality scores. The ``audio/quality`` MIME-style modality is
separate from ``audio/classification`` because MOS and aesthetic axes are
continuous measurements, not a class-probability distribution.
"""
from __future__ import annotations

from muse.modalities._common.probe_audio import probe_wav
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
    """Assess a short 16kHz tone."""
    with probe_wav(1.0, tone_hz=220) as path:
        return model.assess(path)


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
