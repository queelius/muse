"""Test fixtures + pytest hooks for the muse mcp test suite.

Skips the entire mcp test suite when the ``mcp`` Python package is not
installed. The rest of muse's tests run regardless.

The discovery fixtures are live-shaped: the capabilities document is built
by the same ``build_document`` the gateway serves (with every real
``DISCOVERY`` block), and the OpenAPI document is the gateway's real merged
schema. Both are JSON round-tripped so tests see exactly the wire shape.
"""
from __future__ import annotations

import functools
import json

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Mark every mcp test as skipped if the SDK isn't importable."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(
            reason="mcp Python package not installed; "
                   "install via `pip install 'museq[mcp]'`",
        )
        for item in items:
            if "/tests/mcp/" in str(item.fspath) or "\\tests\\mcp\\" in str(item.fspath):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """MCP tests never reach a real server: any name lookup fails at once.

    Servers built against a placeholder URL (``http://test``) then see an
    unreachable muse instead of a slow DNS failure.
    """
    import socket

    def refuse(*_a, **_k):
        raise socket.gaierror(socket.EAI_NONAME, "network disabled in mcp tests")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)


# Two speech models so enum/description tests see more than one id.
_NAMED = {
    "audio/speech": ["kokoro-82m", "soprano-80m"],
    "audio/alignment": ["qwen3-forced-aligner-0.6b"],
    "audio/transcription": ["whisper-base"],
    "audio/quality": ["utmos"],
    "chat/completion": ["qwen3.5-4b-q4"],
    "image/generation": ["sd-turbo"],
    "image/segmentation": ["sam2-hiera-tiny"],
    "embedding/text": ["all-minilm-l6-v2"],
}


@functools.lru_cache(maxsize=1)
def _caps_json() -> str:
    from muse.core.capabilities import build_document, model_facts, server_block
    from muse.core.discovery import modality_discovery
    from muse.core.provenance import ModelRevision

    discovery = modality_discovery()
    models = []
    for tag in sorted(discovery):
        ids = _NAMED.get(tag) or [tag.replace("/", "-") + "-model"]
        for i, mid in enumerate(ids):
            manifest = {
                "modality": tag,
                "hf_repo": f"org/{mid}",
                "summary": f"{mid} does {tag} well.",
                "quality_tier": "good" if i == 0 else "draft",
                "capabilities": {"memory_gb": 1.0},
            }
            facts = model_facts(
                mid, manifest,
                revision=ModelRevision(mid, "a" * 40, (), None),
                memory_gb=1.0,
            )
            models.append((facts, None))
    doc = build_document(
        server=server_block(version="0.0-test", node="node-a",
                            base_url="http://muse.test"),
        discovery=discovery,
        models=models,
    )
    return json.dumps(doc)


@functools.lru_cache(maxsize=1)
def _openapi_json() -> str:
    from fastapi import FastAPI

    from muse.cli_impl.agent_routes import merge_openapi

    return json.dumps(merge_openapi(FastAPI().openapi()))


@pytest.fixture
def caps_doc() -> dict:
    return json.loads(_caps_json())


@pytest.fixture
def openapi_doc() -> dict:
    return json.loads(_openapi_json())


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    """cwd = tmp_path (the default allowed prefix); outputs under ./out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUSE_MCP_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", raising=False)
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    return tmp_path


def _sliced(doc: dict, params) -> dict:
    """Honour ?modality= / ?model= the way the gateway does (roughly)."""
    modality, model = params.get("modality"), params.get("model")
    if model:
        doc = {**doc, "models": [m for m in doc["models"] if m["id"] == model]}
        modality = modality or (doc["models"][0]["modality"] if doc["models"] else None)
    if modality:
        doc = {**doc,
               "modalities": [m for m in doc["modalities"] if m["modality"] == modality],
               "models": [m for m in doc["models"] if m["modality"] == modality]}
    return doc


@pytest.fixture
def fake_http(caps_doc, openapi_doc):
    """``make(handler) -> MuseClient`` over an httpx.MockTransport.

    The transport serves the two discovery documents itself and passes
    every other request to ``handler(request) -> httpx.Response``. Each
    request is appended to ``make.requests``.
    """
    import httpx

    from muse.mcp.client import MuseClient

    def make(handler=None, *, caps=None, openapi=None):
        def route(request: httpx.Request) -> httpx.Response:
            make.requests.append(request)
            if request.url.path == "/v1/capabilities":
                return httpx.Response(200, json=_sliced(
                    caps or caps_doc, request.url.params))
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=openapi or openapi_doc)
            if handler is None:
                return httpx.Response(404, json={"error": {"code": "nope"}})
            return handler(request)

        return MuseClient(server_url="http://muse.test", admin_token="t",
                          transport=httpx.MockTransport(route))

    make.requests = []
    return make
