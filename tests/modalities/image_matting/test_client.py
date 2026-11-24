"""Unit tests for ``MattingClient``."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_matting.client import MattingClient


def test_client_matte_json():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "object": "image.matting",
        "model": "sprite-matte",
        "data": [{"b64_json": "AAAA", "width": 32, "height": 32}],
    }
    with patch("muse.modalities.image_matting.client.requests.post", return_value=resp) as post:
        client = MattingClient(base_url="http://127.0.0.1:8000", timeout=42.0)
        out = client.matte(
            b"\x89PNG...",
            model="sprite-matte",
            background="white",
            hard_alpha=True,
            crop=False,
        )

    assert out["object"] == "image.matting"
    args, kwargs = post.call_args
    assert args[0] == "http://127.0.0.1:8000/v1/images/matting"
    assert kwargs["timeout"] == 42.0
    assert "image" in kwargs["files"]
    data_dict = dict(kwargs["data"])
    assert data_dict == {
        "model": "sprite-matte",
        "response_format": "json",
        "background": "white",
        "hard_alpha": "true",
        "crop": "false",
    }


def test_client_matte_png():
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"\x89PNG-RESULT"
    with patch("muse.modalities.image_matting.client.requests.post", return_value=resp):
        client = MattingClient(base_url="http://127.0.0.1:8000")
        out = client.matte(b"\x89PNG...", response_format="png")

    assert out == b"\x89PNG-RESULT"


def test_client_matte_error_raises():
    resp = MagicMock()
    resp.status_code = 400
    resp.text = '{"error": {"code": "invalid_parameter"}}'
    with patch("muse.modalities.image_matting.client.requests.post", return_value=resp):
        client = MattingClient(base_url="http://127.0.0.1:8000")
        with pytest.raises(RuntimeError, match="400"):
            client.matte(b"\x89PNG...")
