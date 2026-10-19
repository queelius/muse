"""HTTP client for POST /v1/systemone.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.

The body is built once, checked with the same strict parser the server
uses (so a missing model, unknown field or over-limit request fails
before any HTTP), and those exact bytes are sent. Returns the full
response envelope (model, answers, usage), never a selected answer.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from muse.core import config
from muse.modalities.decision_evaluation.validation import parse_request


def encode_request(*, model: str, state: Any, questions: dict) -> bytes:
    """Serialize and strictly validate a request; raise DecisionInputError."""
    body = json.dumps(
        {"model": model, "state": state, "questions": questions},
        ensure_ascii=False, allow_nan=True,
    ).encode("utf-8")
    parse_request(body)
    return body


class DecisionClient:
    """Minimal HTTP client for the decision/evaluation modality."""

    def __init__(self, server_url: str | None = None, *, timeout: float = 300.0) -> None:
        url = server_url or config.get("client.server_url")
        self.server_url = url.rstrip("/")
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
