"""HTTP client for /v1/images/upscale.

Mirrors the multipart shape used by ImageEditsClient and
ImageVariationsClient. Default response_format is b64_json so
upscale() returns raw PNG bytes.
"""
from __future__ import annotations

import base64
import os

import requests


def _resolve_base_url(base_url: str | None) -> str:
    base = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    return base.rstrip("/")


class ImageUpscaleClient:
    """Thin HTTP client against the muse images.upscale endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout

    def upscale(
        self,
        *,
        image: bytes,
        model: str | None = None,
        scale: int = 4,
        prompt: str = "",
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        n: int = 1,
        response_format: str = "b64_json",
    ) -> list[bytes]:
        """Upscale `image` by `scale`. Returns a list of raw PNG bytes."""
        files = {"image": ("image.png", image, "image/png")}
        data: list[tuple[str, str]] = [
            ("scale", str(scale)),
            ("prompt", prompt),
            ("n", str(n)),
            ("response_format", response_format),
        ]
        if model is not None:
            data.append(("model", model))
        if negative_prompt is not None:
            data.append(("negative_prompt", negative_prompt))
        if steps is not None:
            data.append(("steps", str(steps)))
        if guidance is not None:
            data.append(("guidance", str(guidance)))
        if seed is not None:
            data.append(("seed", str(seed)))

        r = requests.post(
            f"{self.base_url}/v1/images/upscale",
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
