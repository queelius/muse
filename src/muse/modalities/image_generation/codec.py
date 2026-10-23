"""Image encoding: accept PIL / numpy, output PNG / JPEG / base64 data URL."""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


class ImageFormatError(ValueError):
    """Raised when an image can't be normalized or encoded."""


def to_pil(image: Any):
    """Normalize supported inputs to a PIL.Image.

    Accepts:
      - PIL.Image — passthrough
      - np.ndarray uint8 HxWxC (C in {3,4}) — direct
      - np.ndarray float HxWxC — rescaled to uint8 assuming [0, 1]
      - np.ndarray HxW (uint8 or float) — grayscale
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return Image.fromarray(arr)
        raise ImageFormatError(f"numpy image has unsupported shape {arr.shape}")

    try:
        import torch
        if isinstance(image, torch.Tensor):
            return to_pil(image.detach().cpu().numpy())
    except ImportError:
        pass

    raise ImageFormatError(f"cannot convert {type(image).__name__} to PIL")


_FORMATS = {
    "png": ("PNG", "png"),
    "jpeg": ("JPEG", "jpeg"),
    "jpg": ("JPEG", "jpeg"),
    "webp": ("WEBP", "webp"),
}


def to_bytes(image: Any, *, fmt: str = "png", compression: int | None = None) -> bytes:
    """Encode to raw image bytes (PNG, JPEG or WebP).

    `compression` (0-100) maps to Pillow `quality=` for JPEG and WebP; PNG ignores it.
    """
    fmt = fmt.lower()
    if fmt not in _FORMATS:
        raise ImageFormatError(f"unsupported image format {fmt!r}")
    pil_fmt = _FORMATS[fmt][0]
    pil = to_pil(image)
    if pil_fmt == "JPEG" and pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    kwargs = {}
    if compression is not None and pil_fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = int(compression)
    buf = io.BytesIO()
    pil.save(buf, format=pil_fmt, **kwargs)
    return buf.getvalue()


def to_data_url(image: Any, *, fmt: str = "png", compression: int | None = None) -> str:
    """Encode as a data URL suitable for inline use."""
    payload = to_bytes(image, fmt=fmt, compression=compression)
    mime = _FORMATS[fmt.lower()][1]
    return f"data:image/{mime};base64,{base64.b64encode(payload).decode()}"
