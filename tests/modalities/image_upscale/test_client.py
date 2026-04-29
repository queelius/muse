"""Tests for ImageUpscaleClient (HTTP, multipart)."""
import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from muse.modalities.image_upscale.client import ImageUpscaleClient


def _png_bytes(width=64, height=64, color=(10, 20, 30)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _server_response(n=1):
    out = _png_bytes(256, 256)
    return {
        "created": 1730000000,
        "data": [
            {"b64_json": base64.b64encode(out).decode(), "revised_prompt": "x"}
            for _ in range(n)
        ],
    }


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    c = ImageUpscaleClient()
    assert c.base_url == "http://localhost:8000"


def test_muse_server_env_var(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://example.com:9000/")
    c = ImageUpscaleClient()
    assert c.base_url == "http://example.com:9000"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://ignored")
    c = ImageUpscaleClient(base_url="http://explicit.example/")
    assert c.base_url == "http://explicit.example"


def test_upscale_posts_multipart_and_returns_bytes():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        result = c.upscale(
            image=_png_bytes(), model="x", scale=4, prompt="sharp",
        )
    assert isinstance(result, list)
    assert isinstance(result[0], (bytes, bytearray))
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/images/upscale"
    assert "files" in kwargs
    assert "image" in kwargs["files"]
    assert "data" in kwargs


def test_upscale_data_carries_scale_and_prompt():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        c.upscale(
            image=_png_bytes(), model="m1", scale=4, prompt="hi", n=2,
        )
    data = dict(mock_post.call_args.kwargs["data"])
    assert data["scale"] == "4"
    assert data["prompt"] == "hi"
    assert data["model"] == "m1"
    assert data["n"] == "2"


def test_upscale_passes_optional_kwargs():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        c.upscale(
            image=_png_bytes(),
            model="m1",
            scale=4,
            negative_prompt="blur",
            steps=20,
            guidance=8.0,
            seed=7,
        )
    data = dict(mock_post.call_args.kwargs["data"])
    assert data["negative_prompt"] == "blur"
    assert data["steps"] == "20"
    assert data["guidance"] == "8.0"
    assert data["seed"] == "7"


def test_upscale_omits_optional_kwargs_when_none():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        c.upscale(image=_png_bytes(), model="m1", scale=4)
    data = dict(mock_post.call_args.kwargs["data"])
    assert "negative_prompt" not in data
    assert "steps" not in data
    assert "guidance" not in data
    assert "seed" not in data


def test_upscale_url_response_format_decodes_back_to_bytes():
    out = _png_bytes(256, 256)
    body = {
        "created": 1730000000,
        "data": [{
            "url": "data:image/png;base64," + base64.b64encode(out).decode(),
            "revised_prompt": None,
        }],
    }
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = body
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ):
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        result = c.upscale(
            image=_png_bytes(), model="x", scale=4,
            response_format="url",
        )
    assert result[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_upscale_non_200_raises():
    fake_resp = MagicMock(status_code=500, text="boom")
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ):
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        with pytest.raises(RuntimeError, match="500"):
            c.upscale(image=_png_bytes(), model="x", scale=4)


def test_upscale_omits_model_when_none():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_upscale.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        c.upscale(image=_png_bytes(), scale=4)
    data = dict(mock_post.call_args.kwargs["data"])
    assert "model" not in data
