"""Frame-sequence encoders shared by image/animation and video/generation.

Pure functions: frames + timing -> bytes (or list[base64 PNG]). Frames may
be PIL images, numpy arrays or torch tensors; they are normalised by
``muse.modalities.image_generation.codec.to_pil`` (floats are [0, 1]).

mp4 is h264 and webm is vp9 (vp8 fallback) via imageio[ffmpeg], an optional
dependency: ``require_imageio`` lets a route reject such a format before it
spends GPU time, and the encoders raise ``UnsupportedFormatError`` when it
is missing. numpy / PIL / imageio are imported lazily.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any


logger = logging.getLogger(__name__)

IMAGEIO_FORMATS = frozenset({"mp4", "webm"})


class UnsupportedFormatError(Exception):
    """Raised when a response_format requires deps that aren't installed."""


def _try_import_imageio():
    """Lazy import isolated for test patching."""
    try:
        import imageio
        return imageio
    except ImportError:
        return None


def _imageio_or_raise(fmt: str):
    imageio = _try_import_imageio()
    if imageio is None:
        raise UnsupportedFormatError(
            f"{fmt} response_format requires imageio[ffmpeg]; "
            "install via `pip install imageio[ffmpeg]` or use another "
            "response_format"
        )
    return imageio


def require_imageio(fmt: str) -> None:
    """Raise UnsupportedFormatError now if ``fmt`` needs a missing imageio."""
    if fmt in IMAGEIO_FORMATS:
        _imageio_or_raise(fmt)


def to_pil(frame: Any):
    from muse.modalities.image_generation.codec import to_pil as _to_pil
    return _to_pil(frame)


def frame_dimensions(frame: Any) -> tuple[int, int]:
    """Return (width, height) of a video frame, matching what the encoders
    produce.

    Normalizes through `to_pil`, so the dimensions can never disagree with
    the encoded output. This deliberately avoids the `getattr(frame, "size")`
    trap: a PIL.Image's `.size` is a `(w, h)` tuple, but a numpy array's
    `.size` is an int (element count) and a torch tensor's `.size` is a
    method.
    """
    width, height = to_pil(frame).size
    return int(width), int(height)


def _rgb_arrays(frames: list[Any]) -> list[Any]:
    import numpy as np
    return [np.array(to_pil(f).convert("RGB")) for f in frames]


def encode_mp4(frames: list[Any], fps: int) -> bytes:
    """Encode frames as h264 MP4 via imageio."""
    imageio = _imageio_or_raise("mp4")
    if not frames:
        raise ValueError("encode_mp4: frames list is empty")
    buf = io.BytesIO()
    imageio.mimwrite(
        buf, _rgb_arrays(frames), fps=fps, codec="h264", format="mp4",
        quality=8,
    )
    return buf.getvalue()


def encode_webm(frames: list[Any], fps: int) -> bytes:
    """Encode frames as vp9 WebM via imageio, falling back to vp8 (with a
    warning) when the ffmpeg build lacks vp9. Raises UnsupportedFormatError
    if both fail."""
    imageio = _imageio_or_raise("webm")
    if not frames:
        raise ValueError("encode_webm: frames list is empty")
    arrays = _rgb_arrays(frames)
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
        to_pil(f).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out
