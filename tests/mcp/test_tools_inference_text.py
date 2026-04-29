"""Tests for the 5 inference text tools.

Each handler invokes its corresponding MuseClient method, mocked.
"""
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
    server = MCPServer(client=client, filter_kind="inference")
    return server


def _parse(blocks):
    assert blocks
    assert blocks[0]["type"] == "text"
    return json.loads(blocks[0]["text"])


class TestRegistry:
    def test_text_tools_present(self, server):
        names = {t.name for t in server.tools}
        for expected in (
            "muse_chat",
            "muse_summarize",
            "muse_rerank",
            "muse_classify",
            "muse_embed_text",
        ):
            assert expected in names

    def test_chat_schema_requires_messages(self, server):
        chat = next(t for t in server.tools if t.name == "muse_chat")
        assert "messages" in chat.inputSchema["required"]


class TestChat:
    def test_passes_body(self, server):
        server.client.chat = MagicMock(
            return_value={"choices": [{"message": {"content": "hi"}}]},
        )
        out = _parse(server.call_handler("muse_chat", {
            "model": "qwen", "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.7,
        }))
        assert out["choices"][0]["message"]["content"] == "hi"
        call = server.client.chat.call_args
        assert call.kwargs["model"] == "qwen"
        assert call.kwargs["temperature"] == 0.7

    def test_drops_none_values(self, server):
        server.client.chat = MagicMock(return_value={"x": 1})
        server.call_handler("muse_chat", {
            "messages": [], "temperature": None, "max_tokens": None,
        })
        call = server.client.chat.call_args
        assert "temperature" not in call.kwargs
        assert "max_tokens" not in call.kwargs


class TestSummarize:
    def test_passes_body(self, server):
        server.client.summarize = MagicMock(
            return_value={"summary": "tldr", "model": "bart"},
        )
        out = _parse(server.call_handler("muse_summarize", {
            "text": "long passage", "length": "short",
        }))
        assert out["summary"] == "tldr"
        call = server.client.summarize.call_args
        assert call.kwargs["text"] == "long passage"
        assert call.kwargs["length"] == "short"


class TestRerank:
    def test_passes_query_and_docs(self, server):
        server.client.rerank = MagicMock(
            return_value={"results": [{"index": 0, "relevance_score": 0.9}]},
        )
        out = _parse(server.call_handler("muse_rerank", {
            "query": "q", "documents": ["a", "b", "c"], "top_n": 2,
        }))
        assert out["results"][0]["index"] == 0
        call = server.client.rerank.call_args
        assert call.kwargs["query"] == "q"
        assert call.kwargs["documents"] == ["a", "b", "c"]
        assert call.kwargs["top_n"] == 2


class TestClassify:
    def test_passes_input(self, server):
        server.client.classify = MagicMock(
            return_value={"results": [{"flagged": False}]},
        )
        out = _parse(server.call_handler("muse_classify", {
            "input": "hello world",
        }))
        assert out["results"][0]["flagged"] is False
        call = server.client.classify.call_args
        assert call.kwargs["input"] == "hello world"

    def test_passes_threshold(self, server):
        server.client.classify = MagicMock(return_value={"results": []})
        server.call_handler("muse_classify", {
            "input": ["a", "b"], "threshold": 0.5,
        })
        call = server.client.classify.call_args
        assert call.kwargs["threshold"] == 0.5


class TestEmbedText:
    def test_returns_envelope_with_vectors(self, server):
        server.client.embed_text = MagicMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        )
        out = _parse(server.call_handler("muse_embed_text", {
            "input": "hello", "model": "minilm",
        }))
        assert out["data"][0]["embedding"] == [0.1, 0.2, 0.3]
