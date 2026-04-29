"""HTTP client for /v1/images/embeddings.

Mirrors the shape of EmbeddingsClient: server_url constructor param
with MUSE_SERVER env fallback, synchronous POST, returns the essential
payload (list[list[float]] of vectors).

Adds a small helper for converting raw image bytes into a data URL so
callers don't need to encode by hand. Consumers who want the full
OpenAI-shape response (with usage, model, etc.) can POST directly or
use the openai-python SDK against muse via `extra_body` for image
inputs.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Iterable, Union

import requests

from muse.modalities.image_embedding.codec import base64_to_embedding


_DEFAULT_SERVER = "http://localhost:8000"


def _bytes_to_data_url(raw: bytes, *, mime: str = "image/png") -> str:
    """Encode raw image bytes as a `data:image/<mime>;base64,...` URL."""
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


class ImageEmbeddingsClient:
    """Thin HTTP client against muse's /v1/images/embeddings endpoint.

    The `embed(...)` helper accepts either:
      - a string (data: URL or http(s):// URL)
      - bytes (raw PNG/JPEG/WEBP, encoded into a data URL)
      - a list of either of the above (mixed is fine)

    Returns `list[list[float]]` regardless of wire format. The full
    envelope (with `usage`, `model`, etc.) is available via the
    lower-level `embed_envelope(...)` method.
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
        self.timeout = timeout

    def embed(
        self,
        input: Union[str, bytes, list[Union[str, bytes]]],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str = "float",
        mime: str = "image/png",
    ) -> list[list[float]]:
        """Embed image(s); return vectors as list[list[float]].

        Bytes inputs are auto-encoded as data URLs with `mime`. List
        inputs preserve order; mixed types (some str, some bytes) are
        fine.
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format must be 'float' or 'base64', got {encoding_format!r}"
            )

        payload = self._normalize_input(input, mime=mime)

        body: dict[str, Any] = {
            "input": payload,
            "encoding_format": encoding_format,
        }
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions

        r = requests.post(
            f"{self.server_url}/v1/images/embeddings",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        envelope = r.json()
        entries = envelope["data"]
        if encoding_format == "base64":
            return [base64_to_embedding(e["embedding"]) for e in entries]
        return [e["embedding"] for e in entries]

    def embed_envelope(
        self,
        input: Union[str, bytes, list[Union[str, bytes]]],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str = "float",
        mime: str = "image/png",
    ) -> dict[str, Any]:
        """Send the request and return the full OpenAI-shape envelope.

        Useful when callers need `usage`, `model`, or per-entry indices
        rather than just the float lists.
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format must be 'float' or 'base64', got {encoding_format!r}"
            )

        payload = self._normalize_input(input, mime=mime)
        body: dict[str, Any] = {
            "input": payload,
            "encoding_format": encoding_format,
        }
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions

        r = requests.post(
            f"{self.server_url}/v1/images/embeddings",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        return r.json()

    @staticmethod
    def _normalize_input(
        input: Union[str, bytes, list[Union[str, bytes]]],
        *,
        mime: str,
    ) -> Union[str, list[str]]:
        """Coerce the caller's input into a JSON-shaped payload.

        Single str or bytes -> single payload string.
        List of str/bytes -> list of payload strings (order preserved).
        Bytes get base64-encoded into a data URL with the given MIME.
        """
        def _one(item):
            if isinstance(item, bytes):
                return _bytes_to_data_url(item, mime=mime)
            return item

        if isinstance(input, (str, bytes)):
            return _one(input)
        if isinstance(input, list):
            return [_one(item) for item in input]
        raise TypeError(
            f"input must be str, bytes, or list of those; got {type(input).__name__}"
        )
