"""Encoding for /v1/summarize responses (Cohere-shape).

Pure functions: SummarizationResult -> envelope dict. Tested without
FastAPI.
"""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.text_summarization.protocol import SummarizationResult


def encode_summarization_response(
    result: SummarizationResult,
) -> dict[str, Any]:
    """Build the Cohere-shape summarize response.

    Returns a dict; the route layer wraps it in JSONResponse.
    `id` is a fresh sum-<24hex> per call so logs and traces can
    correlate request to response.

    `usage` rolls up prompt_tokens + completion_tokens for total_tokens
    so the caller doesn't have to do the addition.

    `meta` echoes the length/format the runtime used. Any extra
    runtime-side metadata (truncation_warning, language detected)
    passes through verbatim; user-supplied keys never overwrite the
    canonical length/format keys.
    """
    meta = dict(result.metadata)
    meta.setdefault("length", result.length)
    meta.setdefault("format", result.format)
    return {
        "id": f"sum-{uuid.uuid4().hex[:24]}",
        "model": result.model_id,
        "summary": result.summary,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "meta": meta,
    }
