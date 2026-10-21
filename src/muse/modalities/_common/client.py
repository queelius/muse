"""Construction helpers shared by the per-modality HTTP clients."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from muse.core import config


def resolve_server_url(server_url: str | None) -> str:
    """Explicit URL, else `client.server_url` (MUSE_SERVER); no trailing slash."""
    return (server_url or config.get("client.server_url")).rstrip("/")


def upload_bytes(obj: Any, *, kind: str, allow_pil: bool = False) -> bytes:
    """Coerce bytes / path / file-like (and, with `allow_pil`, a PIL image
    saved as PNG) to the bytes of a multipart upload part."""
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (str, Path)):
        return Path(obj).read_bytes()
    if hasattr(obj, "read"):
        return obj.read()
    if allow_pil and hasattr(obj, "save"):
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"unsupported {kind} type: {type(obj).__name__}")
