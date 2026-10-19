"""MCP model validation at the real SDK dispatch/HTTP boundary (no network)."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from mcp import types

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


MODEL_REQUIRED_TOOLS = [
    "muse_chat", "muse_summarize", "muse_rerank", "muse_classify",
    "muse_embed_text", "muse_generate_image", "muse_edit_image",
    "muse_vary_image", "muse_upscale_image", "muse_segment_image",
    "muse_vectorize_image", "muse_generate_animation", "muse_embed_image",
    "muse_speak", "muse_transcribe", "muse_generate_music",
    "muse_generate_sfx", "muse_embed_audio", "muse_generate_video",
    "muse_decide",
]


def _dispatch(server, name, arguments):
    handler = server._server.request_handlers[types.CallToolRequest]
    result = asyncio.run(handler(types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )))
    return result.root if hasattr(result, "root") else result


@pytest.mark.parametrize("name", MODEL_REQUIRED_TOOLS)
def test_missing_model_rejected_before_http(name, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [], "choices": [], "text": "ok"})

    original_client = httpx.Client
    monkeypatch.setattr(
        "muse.mcp.client.httpx.Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    server = MCPServer(client=MuseClient(server_url="http://gateway"), filter_kind="inference")
    result = _dispatch(server, name, {
        "messages": [], "text": "hello", "query": "hello", "documents": ["hi"],
        "input": "hello", "prompt": "hello", "image_b64": "eA==",
        "mask_b64": "eA==", "audio_b64": "eA==",
    })
    assert result.isError is True
    assert "model" in result.content[0].text
    assert requests == []


def test_translation_without_model_still_reaches_http(monkeypatch):
    def respond(request):
        assert request.url.path == "/v1/translate"
        assert b'"model"' not in request.content
        return httpx.Response(200, json={"translatedText": "hola"})

    original_client = httpx.Client
    monkeypatch.setattr(
        "muse.mcp.client.httpx.Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    server = MCPServer(client=MuseClient(server_url="http://gateway"), filter_kind="inference")
    result = _dispatch(server, "muse_translate", {"q": "hello", "source": "en", "target": "es"})
    assert result.isError is False
    assert "hola" in result.content[0].text
