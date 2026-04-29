"""Tests for VideoGenerationClient HTTP client."""
from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.video_generation.client import VideoGenerationClient


def test_client_returns_mp4_bytes_by_default():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"FAKEMP4").decode()}],
        "model": "wan2-1-t2v-1-3b",
        "metadata": {"format": "mp4"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = VideoGenerationClient(server_url="http://x")
        out = c.generate("a cat in a field", model="wan2-1-t2v-1-3b")
    assert out == b"FAKEMP4"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["prompt"] == "a cat in a field"
    assert payload["model"] == "wan2-1-t2v-1-3b"
    assert payload["response_format"] == "mp4"
    assert payload["n"] == 1


def test_client_returns_webm_bytes():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"WEBMBYTES").decode()}],
        "model": "x", "metadata": {"format": "webm"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ):
        c = VideoGenerationClient(server_url="http://x")
        out = c.generate("x", response_format="webm")
    assert out == b"WEBMBYTES"


def test_client_response_format_frames_returns_list_of_pngs():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {"b64_json": base64.b64encode(b"png1").decode()},
            {"b64_json": base64.b64encode(b"png2").decode()},
        ],
        "model": "x", "metadata": {"format": "frames_b64"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ):
        c = VideoGenerationClient(server_url="http://x")
        out = c.generate("x", response_format="frames_b64")
    assert out == [b"png1", b"png2"]


def test_client_n_2_returns_list_of_videos():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {"b64_json": base64.b64encode(b"vid1").decode()},
            {"b64_json": base64.b64encode(b"vid2").decode()},
        ],
        "model": "x", "metadata": {"format": "mp4"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ):
        c = VideoGenerationClient(server_url="http://x")
        out = c.generate("x", n=2)
    assert out == [b"vid1", b"vid2"]


def test_client_passes_optional_fields():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"x").decode()}],
        "model": "x", "metadata": {"format": "mp4"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = VideoGenerationClient(server_url="http://x")
        c.generate(
            "x", model="m",
            duration_seconds=4.0, fps=10, size="720x480",
            seed=7, negative_prompt="blurry",
            steps=30, guidance=5.5,
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "m"
    assert payload["duration_seconds"] == 4.0
    assert payload["fps"] == 10
    assert payload["size"] == "720x480"
    assert payload["seed"] == 7
    assert payload["negative_prompt"] == "blurry"
    assert payload["steps"] == 30
    assert payload["guidance"] == 5.5


def test_client_omits_none_fields():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"x").decode()}],
        "model": "x", "metadata": {"format": "mp4"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = VideoGenerationClient(server_url="http://x")
        c.generate("x")
    payload = mock_post.call_args.kwargs["json"]
    assert "model" not in payload
    assert "duration_seconds" not in payload
    assert "seed" not in payload


def test_client_raises_on_non_200():
    fake_resp = MagicMock()
    fake_resp.status_code = 500
    fake_resp.text = '{"error": {"message": "internal"}}'
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ):
        c = VideoGenerationClient(server_url="http://x")
        with pytest.raises(RuntimeError, match="500"):
            c.generate("x")


def test_client_default_url_uses_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env-default:9999")
    c = VideoGenerationClient()
    assert c.server_url == "http://env-default:9999"


def test_client_default_url_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    c = VideoGenerationClient()
    assert c.server_url == "http://localhost:8000"


def test_client_strips_trailing_slash():
    c = VideoGenerationClient(server_url="http://x:8000/")
    assert c.server_url == "http://x:8000"


def test_client_targets_correct_endpoint():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"x").decode()}],
        "model": "x", "metadata": {"format": "mp4"},
    }
    with patch(
        "muse.modalities.video_generation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = VideoGenerationClient(server_url="http://server")
        c.generate("x")
    assert mock_post.call_args.args[0] == "http://server/v1/video/generations"
