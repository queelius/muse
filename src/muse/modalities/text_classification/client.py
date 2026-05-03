"""HTTP clients for /v1/moderations and /v1/text/classifications.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class ModerationsClient:
    """Minimal HTTP client for the text/classification modality."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = (
            server_url
            or os.environ.get("MUSE_SERVER")
            or _DEFAULT_SERVER
        )
        self.server_url = url.rstrip("/")
        self._timeout = timeout

    def classify(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Send a moderation request.

        Returns:
          - dict (the single results[0]) when input is a scalar str
          - list[dict] (the full results array) when input is a list
        """
        body: dict[str, Any] = {"input": input}
        if model is not None:
            body["model"] = model
        if threshold is not None:
            body["threshold"] = threshold

        r = requests.post(
            f"{self.server_url}/v1/moderations",
            json=body, timeout=self._timeout,
        )
        r.raise_for_status()
        envelope = r.json()
        if isinstance(input, str):
            return envelope["results"][0]
        return envelope["results"]


class ClassificationsClient:
    """HTTP client for /v1/text/classifications.

    Returns the full per-input label distribution. Dispatches to the
    fine-tuned classifier path (no candidate_labels) or the zero-shot
    path (candidate_labels passed); the server-side capability gate
    decides which one the loaded model accepts and 400s on mismatch.
    """

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = (
            server_url
            or os.environ.get("MUSE_SERVER")
            or _DEFAULT_SERVER
        )
        self.server_url = url.rstrip("/")
        self._timeout = timeout

    def classify(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        candidate_labels: list[str] | None = None,
        top_k: int | None = None,
        multi_label: bool = False,
    ) -> list[list[dict[str, Any]]] | list[dict[str, Any]]:
        """Send a classification request.

        Returns:
          - list[dict] (one per-input list of {label, score} pairs)
            when input is a scalar str
          - list[list[dict]] when input is a list of strings

        candidate_labels=None routes to the fine-tuned classifier head;
        candidate_labels=[...] routes to zero-shot. The server checks
        the model's capability flags and 400s on mismatch.
        """
        body: dict[str, Any] = {"input": input, "multi_label": multi_label}
        if model is not None:
            body["model"] = model
        if candidate_labels is not None:
            body["candidate_labels"] = candidate_labels
        if top_k is not None:
            body["top_k"] = top_k

        r = requests.post(
            f"{self.server_url}/v1/text/classifications",
            json=body, timeout=self._timeout,
        )
        r.raise_for_status()
        envelope = r.json()
        if isinstance(input, str):
            return envelope["results"][0]
        return envelope["results"]
