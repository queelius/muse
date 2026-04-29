"""HTTP clients for the image_generation modality.

- GenerationsClient: POST JSON to /v1/images/generations
- ImageEditsClient:  POST multipart to /v1/images/edits (inpainting)
- ImageVariationsClient: POST multipart to /v1/images/variations
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


def _resolve_base_url(base_url: str | None) -> str:
    base = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    return base.rstrip("/")


class GenerationsClient:
    """Thin HTTP client against the muse images.generations endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str = "512x512",
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
    ) -> list[bytes]:
        """Generate n PNG images. Returns raw PNG bytes per image."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }
        if model is not None:
            body["model"] = model
        if negative_prompt is not None:
            body["negative_prompt"] = negative_prompt
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed

        r = requests.post(
            f"{self.base_url}/v1/images/generations",
            json=body,
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        data = r.json()["data"]
        return [base64.b64decode(entry["b64_json"]) for entry in data]


class ImageEditsClient:
    """Thin HTTP client against the muse images.edits (inpainting) endpoint.

    Mirrors the OpenAI SDK's images.edit shape: image + mask + prompt
    via multipart/form-data. Default response_format is b64_json so
    edit() returns raw PNG bytes per image.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout

    def edit(
        self,
        prompt: str,
        *,
        image: bytes,
        mask: bytes,
        model: str | None = None,
        n: int = 1,
        size: str = "512x512",
        response_format: str = "b64_json",
    ) -> list[bytes]:
        """Inpaint the masked region of `image` per `prompt`.

        Returns a list of raw PNG bytes (one per requested variation).
        Mask convention: white pixels are regenerated; black are kept.
        """
        files = {
            "image": ("image.png", image, "image/png"),
            "mask": ("mask.png", mask, "image/png"),
        }
        data: list[tuple[str, str]] = [
            ("prompt", prompt),
            ("n", str(n)),
            ("size", size),
            ("response_format", response_format),
        ]
        if model is not None:
            data.append(("model", model))

        r = requests.post(
            f"{self.base_url}/v1/images/edits",
            files=files, data=data, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        entries = r.json()["data"]
        if response_format == "b64_json":
            return [base64.b64decode(e["b64_json"]) for e in entries]
        # response_format == "url": each entry has a data URL we decode
        # back to raw bytes for symmetry with the b64_json path.
        out: list[bytes] = []
        for e in entries:
            url = e["url"]
            comma = url.index(",")
            out.append(base64.b64decode(url[comma + 1:]))
        return out


class ImageVariationsClient:
    """Thin HTTP client against the muse images.variations endpoint.

    Mirrors the OpenAI SDK's images.create_variation shape: image only
    via multipart/form-data; no prompt.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout

    def vary(
        self,
        *,
        image: bytes,
        model: str | None = None,
        n: int = 1,
        size: str = "512x512",
        response_format: str = "b64_json",
    ) -> list[bytes]:
        """Generate n visually-similar variations of `image`.

        Returns a list of raw PNG bytes.
        """
        files = {"image": ("image.png", image, "image/png")}
        data: list[tuple[str, str]] = [
            ("n", str(n)),
            ("size", size),
            ("response_format", response_format),
        ]
        if model is not None:
            data.append(("model", model))

        r = requests.post(
            f"{self.base_url}/v1/images/variations",
            files=files, data=data, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        entries = r.json()["data"]
        if response_format == "b64_json":
            return [base64.b64decode(e["b64_json"]) for e in entries]
        out: list[bytes] = []
        for e in entries:
            url = e["url"]
            comma = url.index(",")
            out.append(base64.b64decode(url[comma + 1:]))
        return out
