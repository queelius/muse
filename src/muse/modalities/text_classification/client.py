"""HTTP client for /v1/moderations.

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
