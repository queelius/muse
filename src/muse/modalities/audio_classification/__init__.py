"""audio/classification modality.

Wire shape mirrors /v1/text/classifications: per-input list of
{label, score} pairs, sorted by score desc, optionally top-k
truncated.

POST /v1/audio/classifications
  multipart in: file (audio), model (optional), top_k (optional)
  JSON out: {id, model, results: [[{label, score}, ...]]}

Bundled: ast-audioset (MIT AST, AudioSet 527-class). Curated:
emotion (wav2vec2-emotion-en, hubert-superb-er), speech commands
(ast-speech-commands), language ID (mms-lid-126).

Speaker diarization is structurally different (frame-level segments
out, not per-file labels) and lives in a future audio/diarization
modality. Speaker verification (paired-input similarity) similarly
deferred.
"""
from muse.modalities._common.probe_audio import probe_wav
from muse.modalities.audio_classification.client import (
    AudioClassificationsClient,
)
from muse.modalities.audio_classification.protocol import (
    AudioClassificationResult,
    AudioClassifierModel,
)
from muse.modalities.audio_classification.routes import build_router


MODALITY = "audio/classification"

DISCOVERY = {
    "summary": (
        "Audio classification. Scores an audio clip against a label set and "
        "returns the top labels with scores: sound events, emotion, language "
        "or speech commands, depending on the model."
    ),
    "use_for": [
        (
            "identifying what sounds occur in a clip (speech, music, dog bark, "
            "siren)"
        ),
        "detecting emotion or spoken language in a recording",
    ],
    "not_for": [
        "transcribing words (use audio/transcription)",
        "judging production quality (use audio/quality)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/classifications",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_classify_audio",
            "file_fields": ["file"],
            "media_fields": [],
            "example": {"model": "<model>", "top_k": 5},
        },
    ],
}


def _probe_call(model):
    """Classify 1 second of 16kHz silence (temp WAV always removed)."""
    with probe_wav(1.0) as path:
        return model.classify(path)


PROBE_DEFAULTS = {
    "shape": "1 second of 16kHz mono silence",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioClassificationResult",
    "AudioClassifierModel",
    "AudioClassificationsClient",
]
