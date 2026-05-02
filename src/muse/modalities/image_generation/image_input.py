"""Decode user-supplied images for img2img on /v1/images/generations.

Accepted shapes:
  - data:image/{png,jpeg,webp};base64,XYZ
  - https?://... (fetched via httpx, content-type validated, size-capped,
    SSRF-protected: hostnames that resolve to private/loopback/link-local
    IPs are refused unless MUSE_ALLOW_PRIVATE_FETCH=1)

Returns PIL.Image. Failures raise ValueError so the route layer can
surface them as 400 responses with the OpenAI-shape error envelope.

PIL is already a pip_extra of the diffusers runtime. httpx is a
server-extras dep.
"""
from __future__ import annotations

import base64
import io
import ipaddress
import os
import re
import socket
import urllib.parse
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


async def decode_image_input(value: str, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> Any:
    """Parse a data URL or HTTP(S) URL into a PIL.Image.

    Async because the http path uses `httpx.AsyncClient` and must not
    block the event loop. The data-URL path is in-memory and stays sync
    internally; awaiting an already-resolved coroutine is cheap.
    """
    if value.startswith("data:"):
        return _decode_data_url(value, max_bytes=max_bytes)
    if value.startswith(("http://", "https://")):
        return await _fetch_http_url(value, max_bytes=max_bytes)
    raise ValueError(
        f"image must be a data: URL or http(s):// URL; got {value[:30]!r}..."
    )


async def decode_image_file(file: Any, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> Any:
    """Decode a multipart UploadFile into a PIL.Image.

    Used by /v1/images/edits and /v1/images/variations, where the
    image (and mask) arrive as multipart/form-data file uploads rather
    than as data URLs in a JSON body. Mirrors the validation discipline
    of decode_image_input: empty file rejected, oversized rejected,
    undecodable bytes rejected. Failures raise ValueError so the route
    layer can surface them as 400 responses with the OpenAI-shape error
    envelope.

    The argument is typed Any so this helper doesn't bind us to FastAPI
    or Starlette at import time. Anything with an async .read() method
    works (UploadFile, plus any test double).

    Reads at most `max_bytes + 1` so a malicious giant upload doesn't
    fully buffer into worker memory before being rejected.
    """
    raw = await file.read(max_bytes + 1)
    if not raw:
        raise ValueError("empty image file")
    if len(raw) > max_bytes:
        raise ValueError(
            f"image bytes exceeds max ({len(raw)} > {max_bytes})"
        )
    return _bytes_to_pil(raw)


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
    # Reject before decoding: base64 inflates ~4/3, so any b64 string
    # larger than (max_bytes * 4 / 3) cannot fit under the cap. Adding 4
    # for padding slack avoids false-positives on exact-fit boundaries.
    max_b64_len = (max_bytes * 4) // 3 + 4
    if len(b64) > max_b64_len:
        raise ValueError(
            f"image bytes exceeds max ({len(b64)} b64 chars > {max_b64_len} limit)"
        )
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ValueError(
            f"image bytes exceeds max ({len(raw)} > {max_bytes})"
        )
    return _bytes_to_pil(raw)


async def _fetch_http_url(value: str, *, max_bytes: int):
    _validate_public_host(value)
    try:
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, follow_redirects=True,
        ) as client:
            resp = await client.get(value)
    except httpx.HTTPError as e:
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


def _validate_public_host(url: str) -> None:
    """Reject URLs whose host resolves to a non-public IP.

    Without this check, an unauthenticated `image` field can reach:
      - link-local: 169.254.169.254 (cloud instance metadata)
      - loopback: 127.0.0.1, ::1 (the worker's own admin/health port)
      - private LAN: 10.x, 172.16-31.x, 192.168.x (lateral scan)
      - multicast/reserved (uncommon but still untrusted)

    Operators on a trusted network who DO want to fetch from internal
    services can opt out by setting MUSE_ALLOW_PRIVATE_FETCH=1.

    Resolution is via `socket.gethostbyname` (IPv4 only). This is a
    documented limitation: an IPv6-only attacker URL would slip past
    if the hostname has only AAAA records. The fallback is acceptable
    because httpx will then fail to connect anyway, but a future
    hardening could use `socket.getaddrinfo` and reject if any returned
    address is non-public.
    """
    if os.environ.get("MUSE_ALLOW_PRIVATE_FETCH") == "1":
        return
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host:
        raise ValueError("URL has no hostname")
    try:
        ip_str = socket.gethostbyname(host)
    except OSError as e:
        raise ValueError(f"DNS resolution of {host!r} failed: {e}") from e
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError as e:
        raise ValueError(
            f"host {host!r} resolved to non-IP {ip_str!r}: {e}"
        ) from e
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(
            f"refusing to fetch URL whose host {host!r} resolves to "
            f"non-public IP {ip!s}; set MUSE_ALLOW_PRIVATE_FETCH=1 to override"
        )


def _bytes_to_pil(raw: bytes):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode now so errors surface here, not later
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"image decode failed: {e}") from e
    return img
