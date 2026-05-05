"""Encoding for /v1/audio/classifications responses."""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.audio_classification.protocol import (
    AudioClassificationResult,
)


def encode_audio_classifications(
    results: list[AudioClassificationResult],
    *,
    model_id: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Build the wire envelope for /v1/audio/classifications.

    Mirrors /v1/text/classifications: per-input list of {label, score}
    pairs, sorted by score desc, optionally truncated to top_k.
    """
    rows: list[list[dict[str, Any]]] = []
    for r in results:
        sorted_pairs = sorted(
            r.scores.items(), key=lambda kv: kv[1], reverse=True,
        )
        if top_k is not None and top_k > 0:
            sorted_pairs = sorted_pairs[:top_k]
        rows.append([
            {"label": k, "score": float(v)} for k, v in sorted_pairs
        ])
    return {
        "id": f"audio-cls-{uuid.uuid4().hex[:24]}",
        "model": model_id,
        "results": rows,
    }
