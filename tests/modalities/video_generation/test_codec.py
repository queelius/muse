"""Tests for video_generation codec.

mp4 + webm both use imageio (lazy import). frames_b64 uses Pillow only.
Heavy deps (imageio, ffmpeg) are mocked so tests run without the
optional install.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from muse.modalities.video_generation.codec import (
    UnsupportedFormatError,
    encode_frames_b64,
    encode_mp4,
    encode_webm,
)


def _frames(n=3, w=64, h=64):
    return [
        Image.new("RGB", (w, h), color=(i * 40, 100, 200)) for i in range(n)
    ]


def test_encode_mp4_raises_when_imageio_unavailable():
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="imageio"):
            encode_mp4(_frames(), fps=8)


def test_encode_mp4_calls_imageio_with_h264():
    fake_imageio = type("ii", (), {})()
    captured = {}

    def fake_mimwrite(buf, frames, *, fps, codec, format, **_):
        captured["fps"] = fps
        captured["codec"] = codec
        captured["format"] = format
        captured["n_frames"] = len(frames)
        buf.write(b"fakeMP4DATA")

    fake_imageio.mimwrite = fake_mimwrite

    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        out = encode_mp4(_frames(n=3), fps=12)

    assert out == b"fakeMP4DATA"
    assert captured["fps"] == 12
    assert captured["codec"] == "h264"
    assert captured["format"] == "mp4"
    assert captured["n_frames"] == 3


def test_encode_mp4_empty_frames_raises():
    fake_imageio = type("ii", (), {})()
    fake_imageio.mimwrite = lambda *a, **k: None
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        with pytest.raises(ValueError, match="empty"):
            encode_mp4([], fps=8)


def test_encode_webm_raises_when_imageio_unavailable():
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=None,
    ):
        with pytest.raises(UnsupportedFormatError, match="imageio"):
            encode_webm(_frames(), fps=8)


def test_encode_webm_calls_imageio_with_vp9():
    fake_imageio = type("ii", (), {})()
    captured = {}

    def fake_mimwrite(buf, frames, *, fps, codec, format, **_):
        captured["fps"] = fps
        captured["codec"] = codec
        captured["format"] = format
        buf.write(b"fakeWEBM")

    fake_imageio.mimwrite = fake_mimwrite

    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        out = encode_webm(_frames(n=2), fps=24)

    assert out == b"fakeWEBM"
    assert captured["codec"] == "vp9"
    assert captured["format"] == "webm"


def test_encode_webm_falls_back_to_vp8_on_vp9_failure():
    fake_imageio = type("ii", (), {})()
    calls = {"n": 0, "codecs": []}

    def fake_mimwrite(buf, frames, *, fps, codec, format, **_):
        calls["n"] += 1
        calls["codecs"].append(codec)
        if codec == "vp9":
            raise RuntimeError("ffmpeg has no vp9 support in this build")
        buf.write(b"vp8DATA")

    fake_imageio.mimwrite = fake_mimwrite

    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        out = encode_webm(_frames(), fps=24)

    assert out == b"vp8DATA"
    assert calls["codecs"] == ["vp9", "vp8"]


def test_encode_webm_raises_when_both_vp9_and_vp8_fail():
    fake_imageio = type("ii", (), {})()

    def fake_mimwrite(*_args, **_kwargs):
        raise RuntimeError("ffmpeg lacks both vp9 and vp8")

    fake_imageio.mimwrite = fake_mimwrite

    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        with pytest.raises(UnsupportedFormatError, match="vp9 and vp8"):
            encode_webm(_frames(), fps=24)


def test_encode_webm_empty_frames_raises():
    fake_imageio = type("ii", (), {})()
    fake_imageio.mimwrite = lambda *a, **k: None
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake_imageio,
    ):
        with pytest.raises(ValueError, match="empty"):
            encode_webm([], fps=8)


def test_encode_frames_b64_returns_per_frame_base64():
    out = encode_frames_b64(_frames(n=3))
    assert isinstance(out, list)
    assert len(out) == 3
    import base64
    for entry in out:
        png = base64.b64decode(entry)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_frames_b64_handles_numpy_input():
    """Backends may return numpy arrays; codec should normalize to PIL."""
    import numpy as np
    arrays = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
    out = encode_frames_b64(arrays)
    assert len(out) == 2
    import base64
    for entry in out:
        png = base64.b64decode(entry)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_frames_b64_handles_float_numpy_input():
    """Float arrays in [0, 1] are scaled into uint8 [0, 255]."""
    import numpy as np
    arrays = [np.full((8, 8, 3), 0.5, dtype=np.float32) for _ in range(2)]
    out = encode_frames_b64(arrays)
    assert len(out) == 2


# --- frame_dimensions: (width, height) for PIL / numpy / torch frames ---
# Regression for the "'int' object is not subscriptable" crash: the old
# generate() paths did getattr(frame, "size", (w, h))[0], but a numpy
# frame's .size is an int (element count), which crashed. Diffusers video
# pipelines return numpy frames by default, so wan/cog inference hit this
# the first time either model successfully loaded.

def test_frame_dimensions_pil_returns_w_h():
    from PIL import Image
    from muse.modalities.video_generation.codec import frame_dimensions
    img = Image.new("RGB", (64, 48))  # PIL size is (width, height)
    assert frame_dimensions(img) == (64, 48)


def test_frame_dimensions_numpy_hwc_returns_w_h():
    import numpy as np
    from muse.modalities.video_generation.codec import frame_dimensions
    arr = np.zeros((48, 64, 3), dtype=np.uint8)  # HWC: H=48, W=64
    assert frame_dimensions(arr) == (64, 48)


def test_frame_dimensions_numpy_size_int_trap_regression():
    import numpy as np
    from muse.modalities.video_generation.codec import frame_dimensions
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    # the trap the old code fell into: numpy .size is an int, not (w, h)
    assert isinstance(arr.size, int)
    # frame_dimensions must ignore .size and read the array shape instead
    assert frame_dimensions(arr) == (20, 10)


def test_frame_dimensions_float_frame_normalizes():
    import numpy as np
    from muse.modalities.video_generation.codec import frame_dimensions
    arr = np.zeros((32, 40, 3), dtype=np.float32)  # float in [0, 1]
    assert frame_dimensions(arr) == (40, 32)
