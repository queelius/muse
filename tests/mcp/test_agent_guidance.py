"""What a connecting agent is told: instructions, titles, annotations, and
which tools need a `model`."""
from __future__ import annotations

import asyncio

import pytest
from mcp import types

from muse.mcp.generated import describe_tool, generate_tools
from muse.mcp.server import INSTRUCTIONS, MCPServer
from muse.mcp.tools import ADMIN_TOOLS


def _tools(caps_doc, openapi_doc):
    return {g.name: g.tool for g in generate_tools(caps_doc, openapi_doc)}


# ---- model optional ----

@pytest.mark.parametrize("name", ["muse_pixelize_image", "muse_autotile",
                                  "muse_translate", "muse_list_languages"])
def test_model_is_optional_on_default_model_routes(caps_doc, openapi_doc, name):
    tool = _tools(caps_doc, openapi_doc)[name]
    assert "model" not in tool.inputSchema["required"]
    desc = tool.inputSchema["properties"]["model"]["description"]
    assert "Optional" in desc and "first enabled" in desc


def test_model_still_required_elsewhere(caps_doc, openapi_doc):
    tools = _tools(caps_doc, openapi_doc)
    assert "model" in tools["muse_generate_image"].inputSchema["required"]
    assert "model" in tools["muse_speak"].inputSchema["required"]


def test_optional_model_call_omits_model(caps_doc, openapi_doc, fake_http, media_env):
    import httpx

    from muse.mcp.generated import call_generated

    seen = []

    def route(request):
        seen.append(request)
        return httpx.Response(200, json=[{"code": "en", "name": "English", "targets": []}])

    gen = next(g for g in generate_tools(caps_doc, openapi_doc)
               if g.name == "muse_list_languages")
    call_generated(fake_http(route), gen, {})
    assert "model" not in str(seen[-1].url)


def _gen_named(caps_doc, openapi_doc, name):
    return next(g for g in generate_tools(caps_doc, openapi_doc) if g.name == name)


def test_optional_model_json_post_omits_model(caps_doc, openapi_doc, fake_http,
                                              media_env):
    import json

    import httpx

    from muse.mcp.generated import call_generated

    seen = []

    def route(request):
        seen.append(request)
        return httpx.Response(200, json={"translatedText": "hola"})

    gen = _gen_named(caps_doc, openapi_doc, "muse_translate")
    call_generated(fake_http(route), gen, {"q": "hi", "source": "en", "target": "es"})
    assert "model" not in json.loads(seen[-1].content)


def test_optional_model_multipart_post_omits_model(caps_doc, openapi_doc, fake_http,
                                                   media_env):
    import httpx

    from muse.mcp.generated import call_generated

    (media_env / "in.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    seen = []

    def route(request):
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    gen = _gen_named(caps_doc, openapi_doc, "muse_pixelize_image")
    call_generated(fake_http(route), gen, {"image_path": "in.png"})
    assert b'name="model"' not in seen[-1].content
    assert b'name="image"' in seen[-1].content


def test_optional_model_route_sends_an_explicit_model(caps_doc, openapi_doc, fake_http,
                                                      media_env):
    import httpx

    from muse.mcp.generated import call_generated

    (media_env / "in.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    seen = []

    def route(request):
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    gen = _gen_named(caps_doc, openapi_doc, "muse_pixelize_image")
    call_generated(fake_http(route), gen, {"image_path": "in.png", "model": "pt-2"})
    assert b'name="model"\r\n\r\npt-2' in seen[-1].content


# ---- titles and annotations ----

def test_every_tool_has_a_title_and_annotations(caps_doc, openapi_doc):
    tools = [describe_tool(), *_tools(caps_doc, openapi_doc).values(),
             *(e.tool for e in ADMIN_TOOLS)]
    for t in tools:
        assert t.title, t.name
        assert isinstance(t.annotations, types.ToolAnnotations), t.name
        assert t.annotations.title is None  # Tool.title is the one display name


def test_generated_titles_read_as_actions(caps_doc, openapi_doc):
    tools = _tools(caps_doc, openapi_doc)
    assert tools["muse_pixelize_image"].title == "Pixelize image"
    assert tools["muse_speak"].title == "Speak"


def test_generated_annotations(caps_doc, openapi_doc):
    from muse.mcp.generated import generate_tools

    tools = _tools(caps_doc, openapi_doc)
    methods = {g.name: g.route["method"].upper()
               for g in generate_tools(caps_doc, openapi_doc)}
    for name, t in tools.items():
        # Any generated tool can write a local file (media, a non-JSON body,
        # or a spilled large result), so none is read-only; only a DELETE
        # route's tool removes muse state, so only those are destructive.
        assert t.annotations.readOnlyHint is False, name
        assert t.annotations.destructiveHint is (methods[name] == "DELETE"), name
        props = t.inputSchema["properties"]
        takes_url = any(k.endswith(("_url", "_urls")) for k in props)
        assert t.annotations.openWorldHint is takes_url, name
    assert tools["muse_pixelize_image"].annotations.openWorldHint is True
    assert tools["muse_embed_text"].annotations.openWorldHint is False
    assert tools["muse_list_languages"].annotations.idempotentHint is True
    assert tools["muse_delete_voice"].annotations.destructiveHint is True


def test_model_argument_is_non_empty(caps_doc, openapi_doc):
    for name, t in _tools(caps_doc, openapi_doc).items():
        assert t.inputSchema["properties"]["model"]["minLength"] == 1, name


def test_describe_is_read_only():
    ann = describe_tool().annotations
    assert ann.readOnlyHint is True and ann.idempotentHint is True
    assert ann.openWorldHint is False


def test_admin_annotations_match_their_effect():
    ann = {e.tool.name: e.tool.annotations for e in ADMIN_TOOLS}
    for name in ("muse_list_models", "muse_get_model_info", "muse_get_memory_status",
                 "muse_get_workers", "muse_get_jobs", "muse_search_models"):
        assert ann[name].readOnlyHint is True, name
    # A re-pull replaces weights and a failed probe disables the model.
    for name in ("muse_remove_model", "muse_disable_model", "muse_pull_model"):
        assert ann[name].readOnlyHint is False and ann[name].destructiveHint is True
    for name in ("muse_enable_model", "muse_probe_model"):
        assert ann[name].readOnlyHint is False and ann[name].destructiveHint is False
    assert ann["muse_search_models"].openWorldHint is True
    assert ann["muse_pull_model"].openWorldHint is True
    assert ann["muse_remove_model"].openWorldHint is False


# ---- instructions ----

def test_instructions_are_sent_at_initialize(fake_http, media_env):
    from mcp.shared.memory import create_connected_server_and_client_session

    srv = MCPServer(client=fake_http(), filter_kind="all")
    got = {}

    async def scenario():
        async with create_connected_server_and_client_session(srv._server) as client:
            got["result"] = await client.initialize()

    asyncio.run(scenario())
    assert got["result"].instructions == INSTRUCTIONS


def test_instructions_carry_the_workflow():
    text = INSTRUCTIONS
    for needle in ("muse_describe", "modality", "loaded", "cold", "file",
                   "out_path", "_path", "_paths", "_urls", "job_id", "model",
                   "MUSE_MCP_ALLOWED_PATH_PREFIXES", "output dir",
                   "muse_list_models", "muse_search_models"):
        assert needle in text, needle
    assert "cwd `/` is never" in text
    assert len(text) < 2500  # cheap: sent once per session


def test_admin_only_instructions_skip_inference(fake_http):
    srv = MCPServer(client=fake_http(), filter_kind="admin")
    assert "muse_describe" not in srv._server.instructions
    assert "job_id" in srv._server.instructions
