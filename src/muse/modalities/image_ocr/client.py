"""HTTP client for /v1/images/ocr.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url, upload_bytes


class OcrClient:
    """Minimal HTTP client for the image/ocr modality."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def ocr(
        self,
        image: bytes | str | Path | Any,
        *,
        model: str | None = None,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send an OCR request.

        `image` accepts bytes, a path-like, or a file-like object with
        .read(). PIL.Image is also accepted (encoded as PNG).

        Returns the full response dict: {id, model, text, usage}.
        """
        files = {"image": ("image.png", upload_bytes(image, kind="image", allow_pil=True), "image/png")}
        data: dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if prompt is not None:
            data["prompt"] = prompt
        if max_new_tokens is not None:
            data["max_new_tokens"] = str(max_new_tokens)

        r = requests.post(
            f"{self.server_url}/v1/images/ocr",
            files=files, data=data, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()


