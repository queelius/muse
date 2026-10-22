"""Revision headers stamped by the gateway on every forwarded response."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muse.cli_impl.gateway import _forward
from muse.core.provenance import ModelRevision, revision_headers

REV = "a" * 40


class _Raw:
    def __init__(self, chunks):
        self.chunks = chunks

    def aiter_raw(self, *, chunk_size=None):
        async def _c():
            for c in self.chunks:
                yield c
        return _c()


def _extra(model="kokoro-82m"):
    return revision_headers(ModelRevision(model, REV, (), None))


async def _run(kind, extra_headers):
    chunks = {
        "buffered": [b"hello"],
        "large_streamed": [b"a" * 600_000, b"b" * 600_000, b"tail"],
        "sse": [b"data: x\n\n"],
    }[kind]
    ctype = "text/event-stream" if kind == "sse" else "application/octet-stream"
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {
        "content-type": ctype,
        "x-muse-model-revision": "worker-value",
        "x-muse-model": "worker-model",
    }
    resp.aiter_raw = _Raw(chunks).aiter_raw
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock()
    client.stream.return_value = ctx
    client.aclose = AsyncMock()
    request = MagicMock()
    request.body = AsyncMock(return_value=b"")
    request.headers = {}
    request.method = "POST"
    request.query_params = {}
    with patch("muse.cli_impl.gateway.httpx.AsyncClient", return_value=client):
        kw = {} if extra_headers is None else {"extra_headers": extra_headers}
        response = await _forward(request, "http://127.0.0.1:9/x", 1.0, **kw)
    if hasattr(response, "body_iterator"):
        _ = [c async for c in response.body_iterator]
    return response


def _hdrs(response):
    out = {}
    for k, v in response.raw_headers:
        out.setdefault(k.lower(), []).append(v)
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["buffered", "large_streamed", "sse"])
async def test_forwarded_response_carries_revision_headers(kind):
    h = _hdrs(await _run(kind, _extra()))
    assert h[b"x-muse-model"] == [b"kokoro-82m"]
    # gateway value replaces the worker's same-named header
    assert h[b"x-muse-model-revision"] == [REV.encode()]


@pytest.mark.asyncio
async def test_without_extra_headers_node_headers_pass_through():
    h = _hdrs(await _run("buffered", None))
    assert h[b"x-muse-model-revision"] == [b"worker-value"]
    assert h[b"x-muse-model"] == [b"worker-model"]


# --- through-gateway wiring -------------------------------------------------
import asyncio
import threading

from fastapi.testclient import TestClient

from muse.cli_impl import gateway
from muse.cli_impl.gateway import build_gateway
from tests.cli_impl.test_gateway_lazy import (
    _isolate_catalog,  # noqa: F401  (autouse fixture)
    _make_state_with_director,
    _patch_get_manifest,
    _wire_async_client_json,
)


def test_gateway_route_stamps_revision_headers(monkeypatch):
    monkeypatch.setattr(
        gateway, "lookup_revision",
        lambda m: ModelRevision(m, REV, (), None),
    )
    state = _make_state_with_director(acquire_port=9001)
    client = TestClient(build_gateway(state=state))
    with _patch_get_manifest(), \
         patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        _wire_async_client_json(mock_cls)
        r = client.post(
            "/v1/audio/speech", json={"input": "hi", "model": "fake-model"},
        )
    assert r.status_code == 200
    assert r.headers["x-muse-model"] == "fake-model"
    assert r.headers["x-muse-model-revision"] == REV


@pytest.mark.asyncio
async def test_cancel_during_revision_lookup_holds_no_refcount_or_slot(monkeypatch):
    started = threading.Event()
    gate_ev = threading.Event()

    def slow_lookup(m):
        started.set()
        gate_ev.wait(2)
        return ModelRevision(m, REV, (), None)

    monkeypatch.setattr(gateway, "lookup_revision", slow_lookup)
    state = _make_state_with_director()
    with _patch_get_manifest():
        task = asyncio.create_task(
            gateway._route_via_director(
                MagicMock(), "v1/audio/speech", "fake-model", state, 1.0,
            )
        )
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate_ev.set()
    # Nothing was acquired before the lookup, so nothing can leak.
    state.director.acquire.assert_not_called()
    state.director.release.assert_not_called()
