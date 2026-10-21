"""HTTP client for ``POST /v1/images/vectorize``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from muse.modalities._common.client import resolve_server_url, upload_bytes


class VectorizationClient:
    """Small requests-based client for raster-to-SVG conversion."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 600.0,
    ) -> None:
        self.server_url = resolve_server_url(server_url)
        self._timeout = timeout

    def vectorize(
        self,
        image: bytes | str | Path | Any,
        *,
        model: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_beams: int | None = None,
        seed: int | None = None,
        response_format: str = "json",
    ) -> dict[str, Any] | str:
        """Vectorize bytes, a path, file-like object, or PIL image.

        ``response_format="json"`` returns the full response envelope.
        ``response_format="svg"`` returns the raw SVG string.
        """
        files = {"image": ("image.png", upload_bytes(image, kind="image", allow_pil=True), "image/png")}
        fields = {
            "model": model,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "seed": seed,
            "response_format": response_format,
        }
        data = {key: str(value) for key, value in fields.items() if value is not None}
        response = requests.post(
            f"{self.server_url}/v1/images/vectorize",
            files=files,
            data=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
        if response_format == "svg":
            return response.text
        return response.json()


