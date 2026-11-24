"""HTTP client for POST /v1/systemone and POST /v1/systemone/batch.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.

The body is built once, checked with the same strict parser the server
uses (so a missing model, unknown field or over-limit request fails
before any HTTP), and those exact bytes are sent. Returns the full
response envelope, never a selected answer.
"""
from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url
from muse.modalities.decision_evaluation.validation import (
    parse_batch_request,
    parse_request,
)


def encode_request(*, model: str, state: Any, questions: dict) -> bytes:
    """Serialize and strictly validate a request; raise DecisionInputError."""
    body = json.dumps(
        {"model": model, "state": state, "questions": questions},
        ensure_ascii=False, allow_nan=True,
    ).encode("utf-8")
    parse_request(body)
    return body


def encode_batch_request(*, model: str, items: Sequence[dict[str, Any]]) -> bytes:
    """Serialize and strictly validate a batch request; raise DecisionInputError."""
    body = json.dumps(
        {"model": model, "items": items},
        ensure_ascii=False, allow_nan=True,
    ).encode("utf-8")
    parse_batch_request(body)
    return body


class DecisionClient:
    """Minimal HTTP client for the decision/evaluation modality."""

    def __init__(self, server_url: str | None = None, *, timeout: float = 300.0) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def decide(self, *, model: str, state: str | dict | list, questions: dict) -> dict:
        body = encode_request(model=model, state=state, questions=questions)
        r = requests.post(
            f"{self.server_url}/v1/systemone",
            data=body,
            headers={"content-type": "application/json"},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()

    def evaluate_batch(self, *, model: str, items: Sequence[dict[str, Any]]) -> dict:
        body = encode_batch_request(model=model, items=items)
        r = requests.post(
            f"{self.server_url}/v1/systemone/batch",
            data=body,
            headers={"content-type": "application/json"},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
