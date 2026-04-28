"""HTTP client for /v1/rerank.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class RerankClient:
    """Minimal HTTP client for the text/rerank modality.

    Cohere-compat: returns the full response envelope unchanged so the
    caller sees `id`, `model`, `results`, `meta` exactly as Cohere SDKs
    expect.
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

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        model: str | None = None,
        return_documents: bool = False,
    ) -> dict[str, Any]:
        """Send a rerank request; return the full Cohere-shape envelope."""
        body: dict[str, Any] = {"query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        if model is not None:
            body["model"] = model
        if return_documents:
            body["return_documents"] = True

        r = requests.post(
            f"{self.server_url}/v1/rerank",
            json=body, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
