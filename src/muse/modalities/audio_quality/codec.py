"""Wire encoding for ``POST /v1/audio/quality``."""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.audio_quality.protocol import AudioQualityResult


def encode_audio_quality(
    result: AudioQualityResult,
    *,
    model_id: str,
) -> dict[str, Any]:
    """Convert a backend result to the stable JSON response envelope."""
    scores: dict[str, dict[str, Any]] = {}
    for name, score in result.scores.items():
        row: dict[str, Any] = {
            "value": float(score.value),
            "direction": score.direction,
        }
        if score.minimum is not None:
            row["minimum"] = float(score.minimum)
        if score.maximum is not None:
            row["maximum"] = float(score.maximum)
        scores[name] = row

    return {
        "id": f"audio-quality-{uuid.uuid4().hex[:24]}",
        "object": "audio.quality",
        "model": model_id,
        "primary_score": result.primary_score,
        "scores": scores,
        "metadata": dict(result.metadata),
    }
