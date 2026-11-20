"""Encoding helpers for image/animation responses.

Pure functions: list[PIL.Image] + timing -> bytes (or list[base64-str]).

WebP and GIF use Pillow (always installed via the modality's pip_extras).
MP4 and frames_b64 are the encoders shared with video/generation
(``muse.modalities._common.video_codec``); MP4 needs imageio[ffmpeg] and
raises UnsupportedFormatError when it is missing.
"""
from __future__ import annotations

import io
from typing import Any

from muse.modalities._common.video_codec import (  # noqa: F401 (re-exports)
    UnsupportedFormatError,
    encode_frames_b64,
    encode_mp4,
    require_imageio,
)


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
