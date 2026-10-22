"""audio/embedding modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - AudioEmbeddingResult dataclass
  - AudioEmbeddingModel Protocol
  - AudioEmbeddingsClient (HTTP client; exported in Task D)
  - PROBE_DEFAULTS

Wire contract (multipart upload + OpenAI-shape response envelope
mirroring /v1/embeddings):
  - POST /v1/audio/embeddings

Each request carries one or more `file` parts whose bytes decode to
audio (wav/mp3/flac/ogg/...) via librosa. The runtime resamples to
the model's preferred rate (CLAP 48kHz, MERT 24kHz).

Cross-modal note: CLAP can also embed text. The current modality
exposes only audio embedding. Future work may add a text route at
`/v1/audio/embeddings/text` or thread a `text` form field through the
existing route guarded by a `supports_text_embeddings_too` capability
flag. Out of scope for v0.24.0.
"""
from muse.modalities.audio_embedding.client import AudioEmbeddingsClient
from muse.modalities.audio_embedding.protocol import (
    AudioEmbeddingModel,
    AudioEmbeddingResult,
)
from muse.modalities.audio_embedding.routes import build_router


MODALITY = "audio/embedding"

DISCOVERY = {
    "summary": (
        "Audio embeddings. Turns audio clips into vectors for similarity "
        "search, clustering and retrieval over music or sound libraries."
    ),
    "use_for": [
        "finding similar-sounding clips in a library",
        "clustering or indexing audio by content",
    ],
    "not_for": [
        "labeling sounds in words (use audio/classification)",
        "text embeddings (use embedding/text)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/embeddings",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_embed_audio",
            "file_fields": ["file"],
            "media_fields": [],
            "example": {"model": "<model>"},
        },
    ],
}


def _make_probe_audio() -> bytes:
    """Generate a 1-second 24kHz mono sine wave WAV.

    Deferred so the probe doesn't import numpy at module import time;
    `muse --help` and `muse pull` should not need numpy.
    """
    import io
    import wave

    import numpy as np

    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1s 24kHz sine wave",
    "call": lambda m: m.embed([_make_probe_audio()]),
}


__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioEmbeddingResult",
    "AudioEmbeddingModel",
    "AudioEmbeddingsClient",
]
