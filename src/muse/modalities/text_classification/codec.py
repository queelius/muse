"""Encoding for /v1/moderations responses.

Pure functions: ClassificationResult + threshold to OpenAI envelope dict.
Tested without FastAPI.
"""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.text_classification.protocol import ClassificationResult


def encode_moderations(
    results: list[ClassificationResult],
    *,
    model_id: str,
    threshold: float,
    safe_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the OpenAI-shape moderations response.

    Returns a dict; the route layer wraps it in a JSONResponse.
    `id` is a fresh modr-<uuid4> per call so logs and traces can
    correlate request to response.

    `safe_labels` are categories that should never trigger `flagged=True`
    even if they win the argmax (single-label) or pass threshold
    (multi-label). KoalaAI/Text-Moderation's "OK" is the canonical case:
    the model assigns >0.99 confidence to OK on benign inputs, and
    flagging benign content because the model is confident it's safe
    is the opposite of what a moderation API should do.
    """
    out_results: list[dict[str, Any]] = []
    for r in results:
        cats, flagged = _flagged_categories(
            r.scores, threshold,
            multi_label=r.multi_label, safe_labels=safe_labels,
        )
        out_results.append({
            "flagged": flagged,
            "categories": cats,
            "category_scores": dict(r.scores),
        })
    return {
        "id": f"modr-{uuid.uuid4().hex[:24]}",
        "model": model_id,
        "results": out_results,
    }


def _resolve_threshold(
    request_threshold: float | None,
    manifest_capabilities: dict,
) -> float:
    """Pick the effective threshold for this request.

    Precedence: request > MANIFEST.capabilities.flag_threshold > 0.5.
    Non-numeric manifest values are silently ignored (default applies).
    """
    if request_threshold is not None:
        return float(request_threshold)
    cap = manifest_capabilities.get("flag_threshold")
    if isinstance(cap, (int, float)):
        return float(cap)
    return 0.5


def _resolve_safe_labels(
    manifest_capabilities: dict,
) -> tuple[str, ...]:
    """Pull the per-model safe-label list from MANIFEST capabilities.

    Returns an empty tuple when unset or malformed (must be a list/tuple
    of strings). Validation is permissive: a non-list value or any
    non-string element disables the feature silently rather than crashing
    the request.
    """
    cap = manifest_capabilities.get("safe_labels")
    if not isinstance(cap, (list, tuple)):
        return ()
    if not all(isinstance(x, str) for x in cap):
        return ()
    return tuple(cap)


def _flagged_categories(
    scores: dict[str, float],
    threshold: float,
    *,
    multi_label: bool,
    safe_labels: tuple[str, ...] = (),
) -> tuple[dict[str, bool], bool]:
    """Convert scores to (per-category booleans, overall flagged).

    Multi-label: each category True iff its score >= threshold AND it's
    not in safe_labels.
    Single-label: only the argmax can be True, and only if its score >=
    threshold AND it's not in safe_labels.

    Returns (categories_dict, any_flagged_bool). The overall `flagged`
    is just `any(categories.values())`.
    """
    safe = set(safe_labels)
    if multi_label:
        cats = {
            k: (v >= threshold and k not in safe)
            for k, v in scores.items()
        }
    else:
        if not scores:
            cats = {}
        else:
            top = max(scores, key=scores.get)
            cats = {
                k: (k == top and v >= threshold and k not in safe)
                for k, v in scores.items()
            }
    flagged = any(cats.values())
    return cats, flagged
