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
from muse.modalities.audio_classification.client import (
    AudioClassificationsClient,
)
from muse.modalities.audio_classification.protocol import (
    AudioClassificationResult,
    AudioClassifierModel,
)
from muse.modalities.audio_classification.routes import build_router


MODALITY = "audio/classification"


def _probe_call(model):
    """Probe-default body: 1 second of silence at 16kHz, written to
    a temp WAV, then classified."""
    import tempfile
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)  # 1 second silence
    return model.classify(path)


PROBE_DEFAULTS = {
    "shape": "1 second of 16kHz mono silence",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioClassificationResult",
    "AudioClassifierModel",
    "AudioClassificationsClient",
]
