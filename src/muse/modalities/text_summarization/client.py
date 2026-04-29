"""HTTP client for /v1/summarize.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class SummarizationClient:
    """Minimal HTTP client for the text/summarization modality.

    Cohere-compat: returns the full response envelope unchanged so the
    caller sees `id`, `model`, `summary`, `usage`, `meta` exactly as
    Cohere SDKs expect.
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

    def summarize(
        self,
        *,
        text: str,
        length: str | None = None,
        format: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Send a summarize request; return the full Cohere-shape envelope."""
        body: dict[str, Any] = {"text": text}
        if length is not None:
            body["length"] = length
        if format is not None:
            body["format"] = format
        if model is not None:
            body["model"] = model

        r = requests.post(
            f"{self.server_url}/v1/summarize",
            json=body, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
