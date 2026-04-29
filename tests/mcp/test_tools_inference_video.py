"""Tests for the inference video tool."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    client = MuseClient(server_url="http://test")
    return MCPServer(client=client, filter_kind="inference")


def _parse(blocks):
    return json.loads(blocks[0]["text"])


class TestRegistry:
    def test_video_tool_present(self, server):
        names = {t.name for t in server.tools}
        assert "muse_generate_video" in names


class TestGenerateVideo:
    def test_returns_envelope(self, server):
        server.client.generate_video = MagicMock(return_value={
            "model": "wan2_1_t2v_1_3b",
            "duration_seconds": 4.0,
            "fps": 24,
            "size": "640x480",
            "data": [{"b64": "AAAA"}],
        })
        blocks = server.call_handler("muse_generate_video", {
            "prompt": "a sunset",
        })
        body = _parse(blocks)
        assert body["model"] == "wan2_1_t2v_1_3b"
        assert body["duration_seconds"] == 4.0
        assert body["fps"] == 24
        assert body["data"][0]["b64"] == "AAAA"

    def test_default_format_mp4(self, server):
        server.client.generate_video = MagicMock(return_value={"data": []})
        blocks = server.call_handler("muse_generate_video", {"prompt": "x"})
        body = _parse(blocks)
        assert body["format"] == "mp4"
        call = server.client.generate_video.call_args
        assert call.kwargs["response_format"] == "mp4"

    def test_passes_frames_b64(self, server):
        server.client.generate_video = MagicMock(return_value={
            "model": "cogvideox", "data": [],
        })
        server.call_handler("muse_generate_video", {
            "prompt": "x", "response_format": "frames_b64",
        })
        call = server.client.generate_video.call_args
        assert call.kwargs["response_format"] == "frames_b64"
