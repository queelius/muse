"""HTTP clients for the 3d/generation modality.

`Generation3DClient.from_text(prompt, ...)` posts JSON to
`/v1/3d/generations`; `.from_image(image, ...)` posts multipart to
`/v1/3d/from-image`. Both return the parsed envelope dict (the
caller decides whether to base64-decode the GLB blobs).
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


def _to_bytes(image: Any) -> bytes:
    """Coerce bytes / path / file-like to bytes.

    Mirrors AudioClassificationsClient._to_bytes and
    GenerationsClient's image-coercion shape.
    """
    if isinstance(image, bytes):
        return image
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if hasattr(image, "read"):
        return image.read()
    raise TypeError(f"unsupported image type: {type(image).__name__}")


class Generation3DClient:
    """HTTP client for the 3d/generation modality."""

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

    # ---------------- text-to-3d ----------------

    def from_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        seed: int | None = None,
        response_format: str = "b64_json",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Generate one or more 3D assets from a text prompt.

        Returns the parsed envelope: {id, created, model, data: [...]}.
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "response_format": response_format,
        }
        if model is not None:
            body["model"] = model
        if seed is not None:
            body["seed"] = seed
        r = requests.post(
            f"{self.server_url}/v1/3d/generations",
            json=body,
            timeout=timeout if timeout is not None else self._timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"server returned {r.status_code}: {r.text[:500]}"
            )
        return r.json()

    # ---------------- image-to-3d ----------------

    def from_image(
        self,
        image: bytes | str | Path | io.IOBase | Any,
        *,
        model: str | None = None,
        n: int = 1,
        seed: int | None = None,
        response_format: str = "b64_json",
        filename: str = "image.png",
        content_type: str = "image/png",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Generate one or more 3D assets from a single image.

        `image` accepts bytes, a path-like, or a file-like object with
        .read(). Returns the parsed envelope.
        """
        files = {"image": (filename, _to_bytes(image), content_type)}
        data: list[tuple[str, str]] = [
            ("n", str(n)),
            ("response_format", response_format),
        ]
        if model is not None:
            data.append(("model", model))
        if seed is not None:
            data.append(("seed", str(seed)))
        r = requests.post(
            f"{self.server_url}/v1/3d/from-image",
            files=files, data=data,
            timeout=timeout if timeout is not None else self._timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"server returned {r.status_code}: {r.text[:500]}"
            )
        return r.json()
