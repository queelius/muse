"""``audio/process`` modality: deterministic CPU audio mastering and looping.

Wire contract (multipart/form-data, CPU only, no weights):
  - ``POST /v1/audio/process``: ``file`` + trimming / fade / loop / peak
    normalization / mono / resample / format fields -> JSON
    (``object: "audio.process"`` with ``b64_audio`` and ``metrics``) or raw
    encoded audio bytes (``response_format="audio"``).

Bundled model: ``audio-tools`` (weightless; ``runtime.AudioTools``).
"""
from muse.modalities.audio_process.client import AudioProcessClient
from muse.modalities.audio_process.protocol import (
    AudioProcessModel,
    AudioProcessOptions,
    AudioProcessResult,
)
from muse.modalities.audio_process.routes import build_router

MODALITY = "audio/process"

# One weightless bundled model serves the route, so omitting `model` reaches
# it automatically via gateway resolution (AST-read; keep as a literal tuple).
MODEL_OPTIONAL_PATHS = ("/v1/audio/process",)

DISCOVERY = {
    "summary": (
        "Deterministic audio post-processing and mastering (CPU, no weights). "
        "Trims leading/trailing silence with click-free micro-fades, removes DC "
        "offset, creates seamless equal-power crossfade loops, peak-normalizes "
        "with a PCM16-safe ceiling, downmixes to mono, and resamples for "
        "game-ready SFX, ambient beds, and voice clips."
    ),
    "use_for": [
        "trimming silence and normalizing generated SFX or voice clips",
        "making ambient beds and music loops seamlessly loopable via equal-power crossfade",
        "resampling or mono-downmixing audio to target game-engine specs",
    ],
    "not_for": [
        "generating new audio from text (use audio/generation)",
        "text-to-speech narration (use audio/speech)",
        "transcribing speech to text (use audio/transcription)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/process",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_process_audio",
            "file_fields": ["file"],
            "media_fields": ["b64_audio"],
            "example": {
                "model": "<model>",
                "trim_silence": True,
                "target_peak_dbfs": -3.0,
                "output_format": "wav",
            },
        },
    ],
}


def _probe_call(model):
    """Process a synthetic padded 1s 440 Hz sine wave (numpy imported lazily)."""
    import numpy as np

    sr = 44100
    t = np.arange(sr // 2, dtype=np.float32) / sr
    tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    silence = np.zeros(sr // 4, dtype=np.float32)
    clip = np.concatenate([silence, tone, silence])

    result = model.process(
        clip,
        sr,
        AudioProcessOptions(trim_silence=True, target_peak_dbfs=-3.0),
    )
    assert result.metrics["frames"] > 0
    assert result.metrics["peak"] <= 0.999
    return result.metrics


PROBE_DEFAULTS = {
    "shape": "1s padded 440 Hz sine wave at 44.1 kHz (trim + normalize)",
    "call": _probe_call,
}

__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioProcessClient",
    "AudioProcessModel",
    "AudioProcessOptions",
    "AudioProcessResult",
]
