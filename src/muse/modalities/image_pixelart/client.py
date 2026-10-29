"""HTTP client for /v1/images/pixelize and /v1/images/autotile."""
from __future__ import annotations

import requests

from muse.modalities._common.client import resolve_server_url


def _form(fields: dict) -> list[tuple[str, str]]:
    out = []
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = "true" if v else "false"
        out.append((k, str(v)))
    return out


class PixelartClient:
    """Thin HTTP client against the muse image/pixelart endpoints."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = resolve_server_url(base_url)
        self.timeout = timeout

    def _post(self, path: str, files: dict, fields: dict) -> requests.Response:
        r = requests.post(f"{self.base_url}{path}", files=files, data=_form(fields),
                          timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")
        return r

    def pixelize(self, image: bytes, *, model: str | None = None,
                 response_format: str = "json", **fields) -> dict | bytes:
        """Pixelize PNG/JPEG/WEBP bytes. Returns the JSON envelope, or raw
        PNG bytes with ``response_format="png"``. Extra keyword arguments are
        the route's form fields (``grid``, ``palette``, ``preview_scale``...)."""
        r = self._post("/v1/images/pixelize",
                       {"image": ("image.png", image, "image/png")},
                       {"model": model, "response_format": response_format, **fields})
        return r.content if response_format == "png" else r.json()

    def autotile(self, *, fill: bytes | None = None, base: bytes | None = None,
                 a2: bytes | None = None, model: str | None = None,
                 response_format: str = "json", **fields) -> dict | bytes:
        """Build an A2 block + blob-47 set from ``fill`` (+ ``base``) or an
        existing ``a2`` block. Returns the JSON envelope, or the zip bytes
        with ``response_format="zip"``."""
        files = {k: (f"{k}.png", v, "image/png")
                 for k, v in (("a2", a2), ("fill", fill), ("base", base)) if v is not None}
        r = self._post("/v1/images/autotile", files,
                       {"model": model, "response_format": response_format, **fields})
        return r.content if response_format == "zip" else r.json()
