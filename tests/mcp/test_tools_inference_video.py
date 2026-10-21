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
        # The real /v1/video/generations shape: clip metadata is nested
        # under `metadata`, not at the top level.
        envelope = {
            "data": [{"b64_json": "AAAA"}],
            "model": "wan2_1_t2v_1_3b",
            "metadata": {
                "frames": 96,
                "fps": 24,
                "duration_seconds": 4.0,
                "format": "mp4",
                "size": [640, 480],
            },
        }
        server.client.generate_video = MagicMock(return_value=envelope)
        blocks = server.call_handler("muse_generate_video", {
            "prompt": "a sunset",
        })
        assert _parse(blocks) == envelope

    def test_default_format_mp4(self, server):
        server.client.generate_video = MagicMock(return_value={"data": []})
        server.call_handler("muse_generate_video", {"prompt": "x"})
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
