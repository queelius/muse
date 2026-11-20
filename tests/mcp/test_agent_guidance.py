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
    tools = _tools(caps_doc, openapi_doc)
    for t in tools.values():
        assert t.annotations.destructiveHint is False
        assert t.annotations.openWorldHint is False
    # Media tools write files; JSON-only and GET tools change nothing.
    assert tools["muse_generate_image"].annotations.readOnlyHint is False
    assert tools["muse_list_voices"].annotations.readOnlyHint is True
    assert tools["muse_embed_text"].annotations.readOnlyHint is True
    assert tools["muse_list_languages"].annotations.idempotentHint is True


def test_describe_is_read_only():
    ann = describe_tool().annotations
    assert ann.readOnlyHint is True and ann.idempotentHint is True
    assert ann.openWorldHint is False


def test_admin_annotations_match_their_effect():
    ann = {e.tool.name: e.tool.annotations for e in ADMIN_TOOLS}
    for name in ("muse_list_models", "muse_get_model_info", "muse_get_memory_status",
                 "muse_get_workers", "muse_get_jobs", "muse_search_models"):
        assert ann[name].readOnlyHint is True, name
    for name in ("muse_remove_model", "muse_disable_model"):
        assert ann[name].readOnlyHint is False and ann[name].destructiveHint is True
    for name in ("muse_pull_model", "muse_enable_model", "muse_probe_model"):
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
                   "out_path", "_path", "job_id", "model"):
        assert needle in text, needle
    assert len(text) < 2500  # cheap: sent once per session


def test_admin_only_instructions_skip_inference(fake_http):
    srv = MCPServer(client=fake_http(), filter_kind="admin")
    assert "muse_describe" not in srv._server.instructions
    assert "job_id" in srv._server.instructions
