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
