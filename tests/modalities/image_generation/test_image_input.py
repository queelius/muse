"""Tests for image_input: parsing user-supplied images for img2img.

The helper accepts either:
  - a data URL: data:image/{png,jpeg,webp};base64,...
  - an http(s):// URL fetched via httpx (size-capped, content-type-checked)

Returns a PIL.Image. Decode failures raise ValueError so the route layer
can surface them as 400s.
"""
import base64
import io

import pytest
from unittest.mock import MagicMock, patch

from muse.modalities.image_generation.image_input import decode_image_input


def _png_bytes(width=64, height=64, color=(0, 128, 255)):
    """Build minimal PNG bytes via PIL."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_data_url_png():
    raw = _png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    img = decode_image_input(data_url)
    assert img.size == (64, 64)
    assert img.mode in ("RGB", "RGBA")


def test_decode_data_url_jpeg():
    from PIL import Image
    rgb = Image.new("RGB", (32, 32), (255, 0, 0))
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG")
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    img = decode_image_input(data_url)
    assert img.size == (32, 32)


def test_decode_http_url_fetches_via_httpx():
    raw = _png_bytes()
    fake_response = MagicMock()
    fake_response.content = raw
    fake_response.headers = {"content-type": "image/png"}
    fake_response.raise_for_status = MagicMock()
    with patch(
        "muse.modalities.image_generation.image_input.httpx.get",
        return_value=fake_response,
    ) as mock_get:
        img = decode_image_input("https://example.com/cat.png")
    mock_get.assert_called_once()
    assert img.size == (64, 64)


def test_decode_rejects_oversize_data_url():
    huge = b"\x00" * (11 * 1024 * 1024)  # 11MB
    data_url = f"data:image/png;base64,{base64.b64encode(huge).decode()}"
    with pytest.raises(ValueError, match="exceeds"):
        decode_image_input(data_url, max_bytes=10 * 1024 * 1024)


def test_decode_rejects_non_image_http_content_type():
    fake_response = MagicMock()
    fake_response.content = b"<html>nope</html>"
    fake_response.headers = {"content-type": "text/html"}
    fake_response.raise_for_status = MagicMock()
    with patch(
        "muse.modalities.image_generation.image_input.httpx.get",
        return_value=fake_response,
    ):
        with pytest.raises(ValueError, match="content-type"):
            decode_image_input("https://example.com/page.html")


def test_decode_rejects_unknown_data_url_mime():
    raw = b"some text"
    data_url = f"data:text/plain;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="MIME"):
        decode_image_input(data_url)


def test_decode_rejects_invalid_url_shape():
    with pytest.raises(ValueError, match="must be"):
        decode_image_input("ftp://example.com/img.png")


def test_decode_rejects_corrupt_image_bytes():
    raw = b"not really a png"
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="decode"):
        decode_image_input(data_url)
