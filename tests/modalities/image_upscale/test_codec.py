"""Tests for the image/upscale codec module (re-exports)."""
import io

from PIL import Image

from muse.modalities.image_upscale.codec import to_bytes, to_data_url


def test_to_bytes_roundtrip_png():
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    raw = to_bytes(img, fmt="png")
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    # Re-decoding should yield the same pixel
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size == (16, 16)


def test_to_data_url_prefixes_png():
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    url = to_data_url(img, fmt="png")
    assert url.startswith("data:image/png;base64,")


def test_to_data_url_jpeg():
    img = Image.new("RGB", (8, 8), (255, 255, 255))
    url = to_data_url(img, fmt="jpeg")
    assert url.startswith("data:image/jpeg;base64,")


def test_codec_module_reexports_image_generation_helpers():
    """The codec module is a thin re-export of image/generation helpers."""
    from muse.modalities.image_generation.codec import (
        to_bytes as gen_to_bytes,
        to_data_url as gen_to_data_url,
    )
    assert to_bytes is gen_to_bytes
    assert to_data_url is gen_to_data_url
