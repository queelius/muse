"""The muse_decide MCP tool."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mcp import types

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer
from muse.mcp.tools import INFERENCE_TOOLS


ENVELOPE = {
    "model": "decision-0.8b-q8",
    "answers": {
        "team": {"type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.9, "technical": 0.1}, "confidence": 0.8},
    },
    "usage": {"input_tokens": 128, "output_tokens": 0},
}
ARGS = {
    "model": "decision-0.8b-q8",
    "state": {"message": "I was charged twice."},
    "questions": {"team": {"type": "choice", "instructions": "Which team?",
                           "criteria": {"billing": "Charges", "technical": None}}},
}


@pytest.fixture
def transport(monkeypatch):
    calls = []
    responses = {"next": httpx.Response(200, json=ENVELOPE)}

    def respond(request):
        calls.append(request)
        return responses["next"]

    original = httpx.Client
    monkeypatch.setattr(
        "muse.mcp.client.httpx.Client",
        lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
    )
    return calls, responses


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    return MCPServer(client=MuseClient(server_url="http://gateway"), filter_kind="inference")


def _dispatch(server, arguments):
    handler = server._server.request_handlers[types.CallToolRequest]
    result = asyncio.run(handler(types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="muse_decide", arguments=arguments),
    )))
    return result.root if hasattr(result, "root") else result


def test_registered_exactly_once_with_model_required():
    entries = [e for e in INFERENCE_TOOLS if e.tool.name == "muse_decide"]
    assert len(entries) == 1
    schema = entries[0].tool.inputSchema
    assert schema["required"] == ["model", "state", "questions"]
    assert schema["additionalProperties"] is False


def test_returns_the_full_envelope_from_the_same_body(server, transport):
    calls, _ = transport
    result = _dispatch(server, ARGS)
    assert result.isError is False
    assert json.loads(result.content[0].text) == ENVELOPE
    (request,) = calls
    assert request.url.path == "/v1/systemone"
    sent = json.loads(request.content)
    assert sent == ARGS


def test_unicode_ids_survive_the_round_trip(server, transport):
    calls, responses = transport
    args = {**ARGS, "questions": {"équipe": ARGS["questions"]["team"]}}
    responses["next"] = httpx.Response(200, json={**ENVELOPE, "answers": {"équipe": ENVELOPE["answers"]["team"]}})
    out = json.loads(_dispatch(server, args).content[0].text)
    assert list(out["answers"]) == ["équipe"]
    assert "équipe".encode("utf-8") in calls[0].content


def test_missing_model_fails_before_http(server, transport):
    calls, _ = transport
    result = _dispatch(server, {k: v for k, v in ARGS.items() if k != "model"})
    assert result.isError is True and calls == []


def test_unsupported_fields_are_rejected_before_http(server, transport):
    calls, _ = transport
    result = _dispatch(server, {**ARGS, "temperature": 0.1})
    assert result.isError is True
    assert calls == []


def test_invalid_question_is_rejected_locally(server, transport):
    calls, _ = transport
    result = _dispatch(server, {**ARGS, "questions": {"q": {"type": "choice", "instructions": "i",
                                                         "criteria": {"only": None}}}})
    assert calls == []
    assert "invalid_criteria" in result.content[0].text


@pytest.mark.parametrize("status, code", [
    (400, "input_too_long"), (403, "forbidden"), (404, "model_not_found"),
    (413, "body_too_large"), (503, "queue_full"),
])
def test_http_errors_are_errors_not_answers(status, code, server, transport):
    _, responses = transport
    responses["next"] = httpx.Response(status, json={"error": {"code": code, "message": "nope"}})
    text = _dispatch(server, ARGS).content[0].text
    assert f"HTTP {status}" in text and code in text
    assert "answers" not in text and "probabilities" not in text


def test_transport_errors_do_not_leak_details(server, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("connect failed to /home/secret/socket with token=abc")

    original = httpx.Client
    monkeypatch.setattr(
        "muse.mcp.client.httpx.Client",
        lambda **kw: original(transport=httpx.MockTransport(boom), **kw),
    )
    text = _dispatch(server, ARGS).content[0].text
    assert "ConnectError" in text
    assert "/home/secret" not in text and "token=abc" not in text


def test_upstream_messages_are_bounded(server, transport):
    _, responses = transport
    responses["next"] = httpx.Response(400, json={"error": {"code": "x", "message": "y" * 5000}})
    assert len(_dispatch(server, ARGS).content[0].text) < 1000
