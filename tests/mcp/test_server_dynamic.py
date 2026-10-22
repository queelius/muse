"""MCPServer with tools generated from the server's discovery documents."""
from __future__ import annotations

import asyncio
import io
import json
import time
import wave

import httpx
import jsonschema
import pytest
from mcp import types

from muse.mcp import server as server_mod
from muse.mcp.client import MuseClient
from muse.mcp.generated import OUTPUT_SCHEMA
from muse.mcp.server import MCPServer
from muse.mcp.tools import ADMIN_TOOLS

ADMIN_NAMES = {
    "muse_list_models", "muse_get_model_info", "muse_search_models",
    "muse_pull_model", "muse_remove_model", "muse_enable_model",
    "muse_disable_model", "muse_probe_model", "muse_get_memory_status",
    "muse_get_workers", "muse_get_jobs",
}


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 800)
    return buf.getvalue()


def _unreachable_client() -> MuseClient:
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    return MuseClient(server_url="http://down.test",
                      transport=httpx.MockTransport(refuse))


def _list(server):
    handler = server._server.request_handlers[types.ListToolsRequest]
    result = asyncio.run(handler(types.ListToolsRequest(method="tools/list")))
    return result.root.tools


def _call(server, name, arguments):
    handler = server._server.request_handlers[types.CallToolRequest]
    result = asyncio.run(handler(types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )))
    return result.root


def test_list_tools_is_describe_generated_and_admin(fake_http, media_env):
    srv = MCPServer(client=fake_http(), filter_kind="all")
    names = [t.name for t in _list(srv)]
    assert names[0] == "muse_describe"
    assert {"muse_speak", "muse_align", "muse_ocr"} <= set(names)
    assert ADMIN_NAMES <= set(names)
    assert len(names) == len(set(names))


def test_filter_inference_has_no_admin(fake_http, media_env):
    names = {t.name for t in _list(MCPServer(client=fake_http(), filter_kind="inference"))}
    assert "muse_describe" in names and "muse_speak" in names
    assert not ADMIN_NAMES & names


def test_filter_admin_never_fetches(fake_http, media_env):
    srv = MCPServer(client=fake_http(), filter_kind="admin")
    assert {t.name for t in _list(srv)} == ADMIN_NAMES
    assert fake_http.requests == []


def test_unknown_filter_raises(fake_http):
    with pytest.raises(ValueError, match="unknown filter_kind"):
        MCPServer(client=fake_http(), filter_kind="nope")


def test_unreachable_server_lists_only_describe_and_admin(media_env):
    srv = MCPServer(client=_unreachable_client(), filter_kind="all")
    names = {t.name for t in _list(srv)}
    assert names == {"muse_describe"} | ADMIN_NAMES
    result = _call(srv, "muse_describe", {})
    assert result.isError is True
    assert "connection refused" in result.content[0].text


def test_capabilities_cached_for_30_seconds(fake_http, media_env, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(server_mod.time, "monotonic", lambda: now[0])
    srv = MCPServer(client=fake_http(), filter_kind="inference")
    _list(srv)
    _list(srv)
    fetches = lambda: sum(r.url.path == "/v1/capabilities" for r in fake_http.requests)  # noqa: E731
    assert fetches() == 1
    now[0] += 31
    _list(srv)
    assert fetches() == 2


def test_existing_admin_tools_unchanged(fake_http):
    srv = MCPServer(client=fake_http(), filter_kind="admin")
    assert srv.tools == [e.tool for e in ADMIN_TOOLS]
    assert {e.tool.name for e in ADMIN_TOOLS} == ADMIN_NAMES


def test_generated_call_returns_structured_content_and_link(fake_http, media_env):
    client = fake_http(lambda r: httpx.Response(
        200, content=_wav(),
        headers={"content-type": "audio/wav", "x-muse-model": "kokoro-82m",
                 "x-muse-model-revision": "c" * 40}))
    srv = MCPServer(client=client, filter_kind="inference")
    result = _call(srv, "muse_speak", {"model": "kokoro-82m", "input": "hi"})
    assert result.isError is False
    jsonschema.validate(result.structuredContent, OUTPUT_SCHEMA)
    assert result.structuredContent["revision"] == "c" * 40
    kinds = [b.type for b in result.content]
    assert kinds[0] == "text" and "resource_link" in kinds and "audio" not in kinds


def test_missing_model_rejected_before_http(fake_http, media_env):
    srv = MCPServer(client=fake_http(lambda r: httpx.Response(200, json={})),
                    filter_kind="inference")
    _list(srv)
    result = _call(srv, "muse_speak", {"input": "hi"})
    assert result.isError is True and "model" in result.content[0].text
    result = _call(srv, "muse_speak", {"model": "not-served", "input": "hi"})
    assert result.isError is True
    assert not [r for r in fake_http.requests if r.url.path == "/v1/audio/speech"]


def test_http_error_reports_muse_error_body(fake_http, media_env):
    client = fake_http(lambda r: httpx.Response(
        404, json={"error": {"code": "model_not_found", "message": "no such model"}}))
    srv = MCPServer(client=client, filter_kind="inference")
    result = _call(srv, "muse_chat", {"model": "qwen3.5-4b-q4", "messages": [
        {"role": "user", "content": "hi"}]})
    assert result.isError is True
    assert json.loads(result.content[0].text) == {
        "error": "muse returned HTTP 404 (model_not_found): no such model",
        "tool": "muse_chat",
    }


def test_describe_returns_document_with_filters(fake_http, media_env):
    srv = MCPServer(client=fake_http(), filter_kind="inference")
    result = _call(srv, "muse_describe", {"modality": "audio/speech"})
    assert result.isError is False
    assert result.structuredContent["object"] == "muse.capabilities"
    req = [r for r in fake_http.requests if r.url.path == "/v1/capabilities"][-1]
    assert req.url.params["modality"] == "audio/speech"


def test_describe_large_document_is_written_to_file(fake_http, media_env, caps_doc):
    caps_doc["models"] = caps_doc["models"] * 3
    srv = MCPServer(client=fake_http(caps=caps_doc), filter_kind="inference")
    result = _call(srv, "muse_describe", {})
    out = result.structuredContent
    assert json.load(open(out["written_to"]))["object"] == "muse.capabilities"
    assert len(result.content[0].text) < 20_000
    assert out["models"][0]["id"]


def test_resources_list_and_read(fake_http, media_env):
    srv = MCPServer(client=fake_http(), filter_kind="all")
    rh = srv._server.request_handlers
    listed = asyncio.run(rh[types.ListResourcesRequest](
        types.ListResourcesRequest(method="resources/list"))).root.resources
    uris = {str(r.uri) for r in listed}
    assert "muse://capabilities" in uris and "muse://models/kokoro-82m" in uris
    templates = asyncio.run(rh[types.ListResourceTemplatesRequest](
        types.ListResourceTemplatesRequest(method="resources/templates/list"),
    )).root.resourceTemplates
    assert templates[0].uriTemplate == "muse://models/{id}"

    def read(uri):
        return asyncio.run(rh[types.ReadResourceRequest](types.ReadResourceRequest(
            method="resources/read", params=types.ReadResourceRequestParams(uri=uri),
        ))).root.contents[0]

    doc = read("muse://capabilities")
    assert doc.mimeType == "application/json"
    assert json.loads(doc.text)["object"] == "muse.capabilities"
    read("muse://models/kokoro-82m")
    req = [r for r in fake_http.requests if r.url.path == "/v1/capabilities"][-1]
    assert req.url.params["model"] == "kokoro-82m"


def test_call_tool_does_not_block_event_loop(fake_http, media_env, monkeypatch):
    srv = MCPServer(client=fake_http(), filter_kind="inference")
    _list(srv)

    def slow(client, gen, args):  # noqa: ARG001
        time.sleep(0.3)
        return [{"type": "text", "text": "done"},
                {"type": "structured", "data": {"model": None, "revision": None,
                 "artifacts": None, "files": [], "response": None}}]

    monkeypatch.setattr(server_mod, "call_generated", slow)
    handler = srv._server.request_handlers[types.CallToolRequest]

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        await handler(types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="muse_speak", arguments={"model": "kokoro-82m", "input": "hi"}),
        ))
        task.cancel()
        return ticks

    assert asyncio.run(scenario()) >= 5


def test_unknown_tool_returns_error_block(fake_http, media_env):
    srv = MCPServer(client=fake_http(), filter_kind="all")
    [block] = srv.call_handler("not_a_real_tool", {})
    assert block["error"] is True and "unknown tool" in block["text"]
