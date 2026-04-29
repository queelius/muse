"""Encoding helpers for video/generation responses.

Pure functions: list[PIL.Image] + timing -> bytes (or list[base64 PNG]).

mp4 uses h264 via imageio[ffmpeg]; webm uses vp9 via imageio (with vp8
fallback if vp9 isn't bundled in the user's ffmpeg). frames_b64 is
per-frame base64 PNG.

The mp4 path is conceptually shared with image_animation but kept
independent here for clarity. Each modality owns its codec; coupling
through a shared module would tangle their evolution.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any


logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def _try_import_imageio():
    """Lazy import isolated for test patching."""
    try:
        import imageio
        return imageio
    except ImportError:
        return None


def _to_pil(frame: Any):
    """Normalize a frame to PIL.Image.

    Accepts PIL.Image, numpy ndarray (HWC uint8 or float in [0, 1]),
    or torch tensor (HWC). Returns a PIL.Image in RGB-compatible mode.
    """
    from PIL import Image
    if isinstance(frame, Image.Image):
        return frame
    import numpy as np
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def encode_mp4(frames: list[Any], fps: int) -> bytes:
    """Encode frames as h264 MP4 via imageio.

    Raises UnsupportedFormatError when imageio[ffmpeg] is not installed.
    """
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            "mp4 response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use frames_b64"
        )
    if not frames:
        raise ValueError("encode_mp4: frames list is empty")
    import numpy as np
    arrays = [np.array(_to_pil(f).convert("RGB")) for f in frames]
    buf = io.BytesIO()
    imageio.mimwrite(
        buf, arrays, fps=fps, codec="h264", format="mp4", quality=8,
    )
    return buf.getvalue()


def encode_webm(frames: list[Any], fps: int) -> bytes:
    """Encode frames as vp9/vp8 WebM via imageio.

    Tries vp9 first (better quality at the same bitrate). If vp9 isn't
    bundled in the user's ffmpeg build, falls back to vp8 with a
    warning. Raises UnsupportedFormatError if both fail.
    """
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            "webm response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use frames_b64"
        )
    if not frames:
        raise ValueError("encode_webm: frames list is empty")
    import numpy as np
    arrays = [np.array(_to_pil(f).convert("RGB")) for f in frames]
    buf = io.BytesIO()
    try:
        imageio.mimwrite(buf, arrays, fps=fps, codec="vp9", format="webm")
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning("vp9 encode failed (%s); falling back to vp8", e)
    buf = io.BytesIO()
    try:
        imageio.mimwrite(buf, arrays, fps=fps, codec="vp8", format="webm")
        return buf.getvalue()
    except Exception as e2:  # noqa: BLE001
        raise UnsupportedFormatError(
            f"webm encode failed (vp9 and vp8 both errored): {e2}"
        ) from e2


def encode_frames_b64(frames: list[Any]) -> list[str]:
    """Each frame as a standalone base64-encoded PNG."""
    out: list[str] = []
    for f in frames:
        buf = io.BytesIO()
        _to_pil(f).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out
