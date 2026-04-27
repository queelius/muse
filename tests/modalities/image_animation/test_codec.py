"""Tests for image_animation codec.

WebP/GIF use Pillow (always available). MP4 uses imageio (optional;
test verifies the lazy import + clean error message when unavailable).
"""
from unittest.mock import patch

import pytest
from PIL import Image

from muse.modalities.image_animation.codec import (
    encode_webp,
    encode_gif,
    encode_mp4,
    encode_frames_b64,
    UnsupportedFormatError,
)


def _frames(n=3, w=64, h=64):
    return [Image.new("RGB", (w, h), color=(i*40, 100, 200)) for i in range(n)]


def test_encode_webp_returns_bytes_with_riff_header():
    out = encode_webp(_frames(), fps=8, loop=True)
    assert isinstance(out, bytes)
    # WebP files start with "RIFF" (4 bytes) ... "WEBP" (at offset 8).
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WEBP"


def test_encode_webp_loop_count_is_honored():
    """loop=False (single play) and loop=True (infinite) produce different output."""
    looped = encode_webp(_frames(), fps=8, loop=True)
    once = encode_webp(_frames(), fps=8, loop=False)
    assert looped != once  # different loop counts encode to different bytes


def test_encode_gif_returns_bytes_with_gif_header():
    out = encode_gif(_frames(), fps=8, loop=True)
    assert out[:6] in (b"GIF87a", b"GIF89a")


def test_encode_frames_b64_returns_per_frame_base64():
    out = encode_frames_b64(_frames(n=3))
    assert isinstance(out, list)
    assert len(out) == 3
    import base64
    # Each entry must be base64-decodable into PNG bytes.
    for entry in out:
        png = base64.b64decode(entry)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_mp4_raises_when_imageio_unavailable():
    """mp4 requires imageio[ffmpeg]; if absent, raise a clean error."""
    with patch(
        "muse.modalities.image_animation.codec._try_import_imageio",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="imageio"):
            encode_mp4(_frames(), fps=8)


def test_encode_mp4_calls_imageio_when_available():
    """mp4 uses imageio.mimwrite to produce h264 bytes."""
    fake_imageio = type("ii", (), {})()
    captured = {}
    def fake_mimwrite(buf, frames, *, fps, codec, **_):
        captured["fps"] = fps
        captured["codec"] = codec
        captured["n_frames"] = len(frames)
        buf.write(b"fakeMP4DATA")
    fake_imageio.mimwrite = fake_mimwrite

    with patch(
        "muse.modalities.image_animation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        out = encode_mp4(_frames(n=3), fps=12)

    assert out == b"fakeMP4DATA"
    assert captured["fps"] == 12
    assert captured["codec"] == "h264"
    assert captured["n_frames"] == 3
