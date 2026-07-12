"""Wire encoding for ``POST /v1/audio/alignments``."""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.audio_alignment.protocol import AudioAlignmentResult


def encode_audio_alignment(
    result: AudioAlignmentResult,
    *,
    model_id: str,
) -> dict[str, Any]:
    """Convert a backend result to the stable JSON response envelope."""
    words: list[dict[str, Any]] = []
    for item in result.words:
        row: dict[str, Any] = {
            "word": item.word,
            "start": float(item.start),
            "end": float(item.end),
        }
        if item.confidence is not None:
            row["confidence"] = float(item.confidence)
        words.append(row)

    return {
        "id": f"audio-alignment-{uuid.uuid4().hex[:24]}",
        "object": "audio.alignment",
        "model": model_id,
        "text": result.text,
        "language": result.language,
        "duration_seconds": float(result.duration_seconds),
        "words": words,
        "metadata": dict(result.metadata),
    }
