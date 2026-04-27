"""Encoding helpers for image/animation responses.

Pure functions: list[PIL.Image] + timing -> bytes (or list[base64-str]).

WebP and GIF use Pillow (always installed via the modality's pip_extras).
MP4 uses imageio[ffmpeg] (optional dep; raises UnsupportedFormatError
when missing so the route can return 400 with a clear message).
"""
from __future__ import annotations

import base64
import io
from typing import Any


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def encode_webp(
    frames: list[Any], fps: int, *, loop: bool = True, lossless: bool = False,
) -> bytes:
    """Encode frames as animated WebP.

    duration_ms_per_frame = round(1000 / fps).
    loop=True -> 0 (infinite). loop=False -> 1 (single play).
    """
    if not frames:
        raise ValueError("encode_webp: frames list is empty")
    duration = max(1, round(1000 / max(fps, 1)))
    loop_count = 0 if loop else 1
    buf = io.BytesIO()
    head = frames[0]
    head.save(
        buf, format="WEBP",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=loop_count,
        lossless=lossless,
        quality=85,
    )
    return buf.getvalue()


def encode_gif(frames: list[Any], fps: int, *, loop: bool = True) -> bytes:
    if not frames:
        raise ValueError("encode_gif: frames list is empty")
    duration = max(1, round(1000 / max(fps, 1)))
    loop_count = 0 if loop else 1
    buf = io.BytesIO()
    head = frames[0]
    head.save(
        buf, format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=loop_count,
        disposal=2,
    )
    return buf.getvalue()


def encode_mp4(frames: list[Any], fps: int) -> bytes:
    """Encode frames as h264 MP4 via imageio. Raises UnsupportedFormatError
    when imageio[ffmpeg] is not installed."""
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            "mp4 response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use webp/gif"
        )
    if not frames:
        raise ValueError("encode_mp4: frames list is empty")
    import numpy as np
    arrays = [np.array(f.convert("RGB")) for f in frames]
    buf = io.BytesIO()
    imageio.mimwrite(buf, arrays, fps=fps, codec="h264", format="mp4", quality=8)
    return buf.getvalue()


def encode_frames_b64(frames: list[Any]) -> list[str]:
    """Each frame as a standalone base64-encoded PNG."""
    out: list[str] = []
    for f in frames:
        buf = io.BytesIO()
        f.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _try_import_imageio():
    """Lazy import isolated for test patching."""
    try:
        import imageio
        return imageio
    except ImportError:
        return None
