"""Tests for ImageSegmentationClient (HTTP, multipart, JSON points/boxes)."""
from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from muse.modalities.image_segmentation.client import ImageSegmentationClient


def _png_bytes(width=64, height=64, color=(10, 20, 30)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _server_response(masks_count=1):
    body = {
        "id": "seg-deadbeef",
        "model": "fake-seg",
        "mode": "auto",
        "image_size": [64, 64],
        "masks": [
            {
                "index": i,
                "score": 0.9 - 0.1 * i,
                "mask": base64.b64encode(_png_bytes(8, 8)).decode(),
                "bbox": [1, 2, 3, 4],
                "area": 12,
            }
            for i in range(masks_count)
        ],
    }
    return body


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    c = ImageSegmentationClient()
    assert c.base_url == "http://localhost:8000"


def test_muse_server_env_var(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://example.com:9000/")
    c = ImageSegmentationClient()
    assert c.base_url == "http://example.com:9000"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://ignored")
    c = ImageSegmentationClient(base_url="http://explicit.example/")
    assert c.base_url == "http://explicit.example"


def test_segment_posts_multipart_and_returns_envelope():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        result = c.segment(
            image=_png_bytes(), model="fake-seg", mode="auto",
        )
    assert isinstance(result, dict)
    assert result["model"] == "fake-seg"
    assert "masks" in result
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/images/segment"
    assert "files" in kwargs
    assert "image" in kwargs["files"]
    assert "data" in kwargs


def test_segment_data_carries_mode_and_mask_format():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        c.segment(
            image=_png_bytes(), model="fake-seg", mode="auto",
            mask_format="rle", max_masks=8,
        )
    data = dict(mock_post.call_args.kwargs["data"])
    assert data["mode"] == "auto"
    assert data["mask_format"] == "rle"
    assert data["max_masks"] == "8"
    assert data["model"] == "fake-seg"


def test_segment_serializes_points_as_json():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        c.segment(
            image=_png_bytes(), model="m1", mode="points",
            points=[[10, 20], [30, 40]],
        )
    data = dict(mock_post.call_args.kwargs["data"])
    assert "points" in data
    assert json.loads(data["points"]) == [[10, 20], [30, 40]]


def test_segment_serializes_boxes_as_json():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        c.segment(
            image=_png_bytes(), model="m1", mode="boxes",
            boxes=[[1, 2, 3, 4]],
        )
    data = dict(mock_post.call_args.kwargs["data"])
    assert "boxes" in data
    assert json.loads(data["boxes"]) == [[1, 2, 3, 4]]


def test_segment_omits_optional_fields_when_none():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        c.segment(image=_png_bytes(), model="m1", mode="auto")
    data = dict(mock_post.call_args.kwargs["data"])
    assert "prompt" not in data
    assert "points" not in data
    assert "boxes" not in data


def test_segment_omits_model_when_none():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        c.segment(image=_png_bytes(), mode="auto")
    data = dict(mock_post.call_args.kwargs["data"])
    assert "model" not in data


def test_segment_passes_prompt():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        c.segment(
            image=_png_bytes(), model="m1", mode="text", prompt="a cat",
        )
    data = dict(mock_post.call_args.kwargs["data"])
    assert data["prompt"] == "a cat"


def test_segment_non_200_raises():
    fake_resp = MagicMock(status_code=500, text="boom")
    with patch(
        "muse.modalities.image_segmentation.client.requests.post",
        return_value=fake_resp,
    ):
        c = ImageSegmentationClient(base_url="http://localhost:8000")
        with pytest.raises(RuntimeError, match="500"):
            c.segment(image=_png_bytes(), model="x", mode="auto")
