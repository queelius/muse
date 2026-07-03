"""HTTP client for /v1/images/ocr.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import requests

from muse.core import config


class OcrClient:
    """Minimal HTTP client for the image/ocr modality."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = server_url or config.get("client.server_url")
        self.server_url = url.rstrip("/")
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
        files = {"image": ("image.png", _to_bytes(image), "image/png")}
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


def _to_bytes(image: Any) -> bytes:
    """Coerce bytes / path / file-like / PIL.Image to bytes."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if hasattr(image, "read"):
        return image.read()
    # PIL.Image fallback: serialize as PNG.
    if hasattr(image, "save"):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"unsupported image type: {type(image).__name__}")
