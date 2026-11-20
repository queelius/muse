"""Encoding helpers for video/generation responses.

mp4 (h264), webm (vp9, vp8 fallback) and frames_b64 (per-frame base64
PNG). The encoders are shared with image/animation and live in
``muse.modalities._common.video_codec``; this module re-exports them so
the modality stays self-describing.
"""
from muse.modalities._common.video_codec import (
    UnsupportedFormatError,
    encode_frames_b64,
    encode_mp4,
    encode_webm,
    frame_dimensions,
    require_imageio,
)

__all__ = [
    "UnsupportedFormatError",
    "encode_frames_b64",
    "encode_mp4",
    "encode_webm",
    "frame_dimensions",
    "require_imageio",
]
