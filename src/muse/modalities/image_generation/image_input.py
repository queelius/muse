"""Decode user-supplied images for img2img on /v1/images/generations.

Accepted shapes:
  - data:image/{png,jpeg,webp};base64,XYZ
  - https?://... (fetched via httpx, content-type validated, size-capped)

Returns PIL.Image. Failures raise ValueError so the route layer can
surface them as 400 responses with the OpenAI-shape error envelope.

PIL is already a pip_extra of the diffusers runtime. httpx is a
server-extras dep.
"""
from __future__ import annotations

import base64
import io
import re
from typing import Any

import httpx


_DATA_URL_RE = re.compile(
    r"^data:([a-zA-Z0-9.+/-]+);base64,(.*)$",
    re.DOTALL,
)
_ALLOWED_IMAGE_MIME = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/webp",
})
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_HTTP_TIMEOUT = 30.0


def decode_image_input(value: str, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> Any:
    """Parse a data URL or HTTP(S) URL into a PIL.Image.

    PIL is imported lazily so the module loads without it (the diffusers
    runtime that imports this helper has PIL as a pip_extra anyway, but
    discovery should not crash on it).
    """
    if value.startswith("data:"):
        return _decode_data_url(value, max_bytes=max_bytes)
    if value.startswith(("http://", "https://")):
        return _fetch_http_url(value, max_bytes=max_bytes)
    raise ValueError(
        f"image must be a data: URL or http(s):// URL; got {value[:30]!r}..."
    )


def _decode_data_url(value: str, *, max_bytes: int):
    m = _DATA_URL_RE.match(value)
    if not m:
        raise ValueError("malformed data URL")
    mime = m.group(1).lower()
    if mime not in _ALLOWED_IMAGE_MIME:
        raise ValueError(
            f"unsupported MIME {mime!r}; allowed: {sorted(_ALLOWED_IMAGE_MIME)}"
        )
    b64 = m.group(2)
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ValueError(
            f"image bytes exceeds max ({len(raw)} > {max_bytes})"
        )
    return _bytes_to_pil(raw)


def _fetch_http_url(value: str, *, max_bytes: int):
    try:
        resp = httpx.get(value, timeout=_HTTP_TIMEOUT, follow_redirects=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"fetch failed: {e}") from e
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype not in _ALLOWED_IMAGE_MIME:
        raise ValueError(
            f"content-type {ctype!r} not an allowed image MIME; "
            f"allowed: {sorted(_ALLOWED_IMAGE_MIME)}"
        )
    raw = resp.content
    if len(raw) > max_bytes:
        raise ValueError(f"image bytes exceeds max ({len(raw)} > {max_bytes})")
    return _bytes_to_pil(raw)


def _bytes_to_pil(raw: bytes):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode now so errors surface here, not later
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"image decode failed: {e}") from e
    return img
