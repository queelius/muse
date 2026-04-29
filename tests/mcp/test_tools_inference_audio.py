"""Tests for the 5 inference audio tools.

Cover audio-output packing (speak/music/sfx) and audio binary input
resolution (transcribe, embed_audio).
"""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


SAMPLE_WAV = b"RIFF" + b"\x00" * 32
SAMPLE_WAV_B64 = base64.b64encode(SAMPLE_WAV).decode("ascii")


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    client = MuseClient(server_url="http://test")
    return MCPServer(client=client, filter_kind="inference")


def _split(blocks):
    audios = [b for b in blocks if b["type"] == "audio"]
    images = [b for b in blocks if b["type"] == "image"]
    texts = [b for b in blocks if b["type"] == "text"]
    return audios, images, texts


def _parse_text(block):
    return json.loads(block["text"])


class TestRegistry:
    def test_audio_tools_present(self, server):
        names = {t.name for t in server.tools}
        for expected in (
            "muse_speak",
            "muse_transcribe",
            "muse_generate_music",
            "muse_generate_sfx",
            "muse_embed_audio",
        ):
            assert expected in names

    def test_transcribe_has_audio_fields(self, server):
        t = next(t for t in server.tools if t.name == "muse_transcribe")
        props = t.inputSchema["properties"]
        for f in ("audio_b64", "audio_url", "audio_path"):
            assert f in props


class TestSpeak:
    def test_returns_audio_block_plus_summary(self, server):
        server.client.speak = MagicMock(return_value=SAMPLE_WAV)
        blocks = server.call_handler("muse_speak", {
            "input": "hello", "model": "kokoro-82m",
        })
        audios, _, texts = _split(blocks)
        assert len(audios) == 1
        assert base64.b64decode(audios[0]["data"]) == SAMPLE_WAV
        assert audios[0]["mimeType"] == "audio/wav"
        summary = _parse_text(texts[0])
        assert summary["model"] == "kokoro-82m"
        assert summary["size_bytes"] == len(SAMPLE_WAV)


class TestTranscribe:
    def test_resolves_audio_binary(self, server):
        server.client.transcribe = MagicMock(
            return_value={"text": "hello world"},
        )
        blocks = server.call_handler("muse_transcribe", {
            "audio_b64": SAMPLE_WAV_B64, "model": "whisper-tiny",
        })
        body = _parse_text(blocks[0])
        assert body["text"] == "hello world"
        call = server.client.transcribe.call_args
        assert call.kwargs["audio"] == SAMPLE_WAV
        assert "audio_b64" not in call.kwargs

    def test_handles_text_response(self, server):
        # When response_format='text' the server may return raw bytes
        server.client.transcribe = MagicMock(return_value=b"hello world")
        blocks = server.call_handler("muse_transcribe", {
            "audio_b64": SAMPLE_WAV_B64, "response_format": "text",
        })
        body = _parse_text(blocks[0])
        assert body["text"] == "hello world"

    def test_missing_audio_returns_error(self, server):
        server.client.transcribe = MagicMock()
        blocks = server.call_handler("muse_transcribe", {})
        body = _parse_text(blocks[0])
        assert "missing audio input" in body["error"]


class TestMusicAndSfx:
    def test_music_returns_audio_block(self, server):
        server.client.generate_music = MagicMock(return_value=SAMPLE_WAV)
        blocks = server.call_handler("muse_generate_music", {
            "prompt": "ambient pad", "duration": 5.0,
        })
        audios, _, texts = _split(blocks)
        assert len(audios) == 1
        assert base64.b64decode(audios[0]["data"]) == SAMPLE_WAV
        summary = _parse_text(texts[0])
        assert summary["duration"] == 5.0
        call = server.client.generate_music.call_args
        assert call.kwargs["prompt"] == "ambient pad"

    def test_sfx_returns_audio_block(self, server):
        server.client.generate_sfx = MagicMock(return_value=SAMPLE_WAV)
        blocks = server.call_handler("muse_generate_sfx", {
            "prompt": "thunder",
        })
        audios, _, texts = _split(blocks)
        assert len(audios) == 1


class TestEmbedAudio:
    def test_resolves_audio_binary(self, server):
        server.client.embed_audio = MagicMock(
            return_value={"data": [{"embedding": [0.1, 0.2]}]},
        )
        blocks = server.call_handler("muse_embed_audio", {
            "audio_b64": SAMPLE_WAV_B64, "model": "mert-v1",
        })
        body = _parse_text(blocks[0])
        assert body["data"][0]["embedding"] == [0.1, 0.2]
        call = server.client.embed_audio.call_args
        assert call.kwargs["audio"] == SAMPLE_WAV
        assert call.kwargs["model"] == "mert-v1"
