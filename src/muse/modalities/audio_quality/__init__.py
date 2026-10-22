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

DISCOVERY = {
    "summary": (
        "Single-ended audio quality assessment. Scores one audio file on "
        "named scales (speech naturalness, or production quality axes) and "
        "reports the weakest segment, so a bad stretch can be re-generated."
    ),
    "use_for": [
        "checking a generated narration sounds natural before using it",
        "picking the best of several generated takes",
        "finding the worst segment of a long track to redo",
    ],
    "not_for": [
        "comparing two clips head to head",
        "transcribing or timing speech (use audio/transcription or audio/alignment)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/quality",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_assess_audio_quality",
            "file_fields": ["file"],
            "media_fields": [],
            "example": {"model": "<model>"},
        },
    ],
}



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
    "DISCOVERY",
    "MODEL_OPTIONAL_PATHS",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioQualityClient",
    "AudioDurationExceededError",
    "AudioQualityModel",
    "AudioQualityResult",
    "AudioQualityScore",
]
