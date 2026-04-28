"""Encoding for /v1/rerank responses (Cohere-shape).

Pure functions: list[RerankResult] + return_documents -> envelope dict.
Tested without FastAPI.
"""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.text_rerank.protocol import RerankResult


def encode_rerank_response(
    results: list[RerankResult],
    *,
    model_id: str,
    return_documents: bool,
) -> dict[str, Any]:
    """Build the Cohere-shape rerank response.

    Returns a dict; the route layer wraps it in JSONResponse.
    `id` is a fresh rrk-<24hex> per call so logs and traces can
    correlate request to response.

    `meta.billed_units.search_units` is a Cohere artifact (their pricing
    unit). muse always reports `1`; the field exists for SDK
    compatibility, not for billing.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "index": r.index,
            "relevance_score": r.relevance_score,
        }
        if return_documents:
            row["document"] = {"text": r.document_text}
        rows.append(row)
    return {
        "id": f"rrk-{uuid.uuid4().hex[:24]}",
        "model": model_id,
        "results": rows,
        "meta": {"billed_units": {"search_units": 1}},
    }
