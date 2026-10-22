"""Tests for the federation coordinator app (muse.cli_impl.federation).

Uses a FakeReg (plain object exposing .snapshot()) so tests never touch
the real NodeRegistry's background poll loop. `fed._forward` is
monkeypatched at the module level so no real network call happens; this
also exercises the failover path (first candidate raises a connect
error, second candidate succeeds).
"""

import httpx
import pytest
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

import muse.cli_impl.federation as fed
from muse.federation.nodes import NodeSpec
from muse.federation.state import ModelAvail, NodeState


def _snap():
    a = NodeState(NodeSpec("http://a:8000", "a"), True, {"m1": ModelAvail(True)}, 0, 0.0)
    b = NodeState(NodeSpec("http://b:8000", "b"), True, {"m2": ModelAvail(True)}, 0, 0.0)
    return [a, b]


class FakeReg:
    def snapshot(self):
        return _snap()


@pytest.mark.parametrize("model", [[], {}, 1, True, "", "   ", "x" * 513, "bad\nname"])
def test_invalid_model_returns_structured_400(model):
    app = fed.build_coordinator(FakeReg(), timeout=5)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions", json={"model": model, "messages": []},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/health"])
def test_coordinator_rejects_oversized_body_before_dispatch(monkeypatch, path):
    monkeypatch.setenv("MUSE_MAX_REQUEST_BODY_MB", "1")
    app = fed.build_coordinator(FakeReg(), timeout=5)
    response = TestClient(app).request("GET", path, content=b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_coordinator_rejects_chunked_body_without_content_length(monkeypatch):
    monkeypatch.setenv("MUSE_MAX_REQUEST_BODY_MB", "1")

    async def forward_in_memory(request, target_url, timeout):
        return JSONResponse({"unexpected_forward": True})

    monkeypatch.setattr(fed, "_forward", forward_in_memory)
    app = fed.build_coordinator(FakeReg(), timeout=5)

    async def chunks():
        yield b'{"model":"m1","padding":"'
        yield b"x" * (1024 * 1024)
        yield b'"}'

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://coordinator",
    ) as client:
        response = await client.post(
            "/v1/chat/completions", content=chunks(),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_routes_to_node_with_model(monkeypatch):
    seen = {}

    async def fake_forward(request, target_url, timeout):
        seen["url"] = target_url
        return JSONResponse({"ok": True})

    monkeypatch.setattr(fed, "_forward", fake_forward)
    app = fed.build_coordinator(FakeReg(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "m2", "messages": []})
    assert r.status_code == 200 and seen["url"] == "http://b:8000/v1/chat/completions"


def test_404_when_no_node_has_model(monkeypatch):
    app = fed.build_coordinator(FakeReg(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_available"


def test_v1_models_union():
    c = TestClient(fed.build_coordinator(FakeReg(), timeout=5))
    ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
    assert ids == {"m1", "m2"}


def test_health_aggregate():
    assert TestClient(fed.build_coordinator(FakeReg(), timeout=5)).get("/health").json()["status"] == "ok"


def test_model_required_when_no_model_field():
    app = fed.build_coordinator(FakeReg(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_required"


def test_federation_nodes_endpoint():
    app = fed.build_coordinator(FakeReg(), timeout=5)
    c = TestClient(app)
    r = c.get("/v1/federation/nodes")
    assert r.status_code == 200
    names = {n["name"] for n in r.json()["nodes"]}
    assert names == {"a", "b"}


def test_failover_to_second_node(monkeypatch):
    """Two nodes serve model `m`; the first raises a connect error, the
    second succeeds. The coordinator must retry against the second node
    and return its response, not 502."""

    def _snap_both():
        a = NodeState(NodeSpec("http://a:8000", "a"), True, {"m": ModelAvail(True)}, 0, 0.0)
        b = NodeState(NodeSpec("http://b:8000", "b"), True, {"m": ModelAvail(True)}, 0, 0.0)
        return [a, b]

    class FakeRegBoth:
        def snapshot(self):
            return _snap_both()

    seen_urls = []

    async def fake_forward(request, target_url, timeout):
        seen_urls.append(target_url)
        if target_url == "http://a:8000/v1/chat/completions":
            raise httpx.ConnectError("connection refused")
        return JSONResponse({"ok": True, "url": target_url})

    monkeypatch.setattr(fed, "_forward", fake_forward)
    app = fed.build_coordinator(FakeRegBoth(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 200
    assert r.json()["url"] == "http://b:8000/v1/chat/completions"
    assert seen_urls == [
        "http://a:8000/v1/chat/completions",
        "http://b:8000/v1/chat/completions",
    ]


def test_default_rr_counter_rotates_across_tied_same_model_nodes(monkeypatch):
    """Regression guard for the deployed (no explicit rr_counter) path.

    `run_coordinator` never passes an rr_counter, so `build_coordinator`
    must default it to a real dict rather than leaving it None -- else
    `select_node` falls back to its deterministic first-by-url pick and
    same-model traffic across a multi-replica deployment never rotates.
    Two nodes both serve the SAME model, both with in_flight=None (no
    telemetry token), so they tie on every select_node criterion up to
    the round-robin tie-break. Firing the same request twice must hit
    two DIFFERENT node urls.
    """

    def _snap_tied():
        a = NodeState(NodeSpec("http://a:8000", "a"), True, {"m": ModelAvail(True)}, None, 0.0)
        b = NodeState(NodeSpec("http://b:8000", "b"), True, {"m": ModelAvail(True)}, None, 0.0)
        return [a, b]

    class FakeRegTied:
        def snapshot(self):
            return _snap_tied()

    seen_urls = []

    async def fake_forward(request, target_url, timeout):
        seen_urls.append(target_url)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(fed, "_forward", fake_forward)
    # No rr_counter passed: exercises the deployed default path.
    app = fed.build_coordinator(FakeRegTied(), timeout=5)
    c = TestClient(app)

    r1 = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
    r2 = c.post("/v1/chat/completions", json={"model": "m", "messages": []})

    assert r1.status_code == 200 and r2.status_code == 200
    assert len(seen_urls) == 2
    assert seen_urls[0] != seen_urls[1]


def test_failover_502_when_all_nodes_fail(monkeypatch):
    def _snap_both():
        a = NodeState(NodeSpec("http://a:8000", "a"), True, {"m": ModelAvail(True)}, 0, 0.0)
        b = NodeState(NodeSpec("http://b:8000", "b"), True, {"m": ModelAvail(True)}, 0, 0.0)
        return [a, b]

    class FakeRegBoth:
        def snapshot(self):
            return _snap_both()

    async def fake_forward(request, target_url, timeout):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(fed, "_forward", fake_forward)
    app = fed.build_coordinator(FakeRegBoth(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 502 and r.json()["error"]["code"] == "no_node_available"


def test_read_timeout_maps_504_without_failover(monkeypatch):
    class FakeReg:
        def snapshot(self):
            return [
                NodeState(
                    NodeSpec("http://a:8000", "a"), True,
                    {"m": ModelAvail(True)}, 0, 0.0,
                ),
                NodeState(
                    NodeSpec("http://b:8000", "b"), True,
                    {"m": ModelAvail(True)}, 0, 0.0,
                ),
            ]

    seen = []

    async def fake_forward(request, target_url, timeout):
        seen.append(target_url)
        raise httpx.ReadTimeout("accepted but stalled")

    monkeypatch.setattr(fed, "_forward", fake_forward)
    app = fed.build_coordinator(FakeReg(), timeout=5)
    r = TestClient(app).post(
        "/v1/chat/completions", json={"model": "m", "messages": []},
    )

    assert r.status_code == 504
    assert r.json()["error"]["code"] == "node_timeout"
    assert len(seen) == 1


def test_nonconnect_transport_failure_maps_502_without_failover(monkeypatch):
    class FakeReg:
        def snapshot(self):
            return [
                NodeState(
                    NodeSpec("http://a:8000", "a"), True,
                    {"m": ModelAvail(True)}, 0, 0.0,
                ),
            ]

    async def fake_forward(request, target_url, timeout):
        raise httpx.RemoteProtocolError("truncated response")

    monkeypatch.setattr(fed, "_forward", fake_forward)
    app = fed.build_coordinator(FakeReg(), timeout=5)
    r = TestClient(app).post(
        "/v1/chat/completions", json={"model": "m", "messages": []},
    )

    assert r.status_code == 502
    assert r.json()["error"]["code"] == "node_forward_failed"


# ---- discovery -----------------------------------------------------------

def _node_doc(mid):
    return {
        "object": "muse.capabilities", "server": {},
        "modalities": [{"modality": "audio/speech", "summary": "tts", "routes": [], "models": [mid]}],
        "models": [{"id": mid, "modality": "audio/speech", "revision": "1" * 40,
                    "loaded": True, "queue_depth": 0}],
    }


@pytest.fixture
def coordinator_client(monkeypatch):
    class Reg:
        def snapshot(self):
            return [
                NodeState(NodeSpec("http://a:8000", "a", token="secret"), True, {}, 0, 0.0),
                NodeState(NodeSpec("http://b:8000", "b"), True, {}, 0, 0.0),
                NodeState(NodeSpec("http://c:8000", "c"), True, {}, 0, 0.0),
                NodeState(NodeSpec("http://d:8000", "d"), False, {}, 0, 0.0),
            ]

    seen = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            seen.append((url, kw))
            if url.startswith("http://a:"):
                if url.endswith("/openapi.json"):
                    return httpx.Response(200, json={"openapi": "3.1.0", "paths": {"/v1/x": {}}, "info": {}})
                return httpx.Response(200, json=_node_doc("m"))
            if url.startswith("http://b:"):
                return httpx.Response(200, content=b"not json")
            raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(fed.httpx, "AsyncClient", FakeClient)
    c = TestClient(fed.build_coordinator(Reg(), timeout=5))
    c.seen = seen
    return c


def test_bad_node_degrades_not_fails(coordinator_client):
    r = coordinator_client.get("/v1/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert set(body["unreachable_nodes"]) == {"b", "c", "d"}
    assert body["models"][0]["nodes"][0]["name"] == "a"
    assert body["server"]["mcp"]["env"]["MUSE_SERVER"] == "http://testserver"
    assert [n["name"] for n in body["server"]["nodes"]] == ["a", "b", "c", "d"]
    assert "secret" not in r.text
    assert all("Authorization" not in (kw.get("headers") or {}) for _, kw in coordinator_client.seen)


def test_llms_txt_and_pointers(coordinator_client):
    r = coordinator_client.get("/llms.txt")
    assert r.status_code == 200 and "audio/speech" in r.text
    assert coordinator_client.get("/").json()["capabilities"] == "/v1/capabilities"
    assert coordinator_client.get("/.well-known/muse.json").status_code == 200


def test_openapi_is_first_node_plus_own_paths(coordinator_client):
    r = coordinator_client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/v1/x" in paths and "/v1/federation/nodes" in paths and "/v1/capabilities" in paths


def test_openapi_without_reachable_node_still_serves(monkeypatch):
    class Reg:
        def snapshot(self):
            return []
    r = TestClient(fed.build_coordinator(Reg(), timeout=5)).get("/openapi.json")
    assert r.status_code == 200 and "/v1/capabilities" in r.json()["paths"]


def test_capabilities_filters_apply_to_merged_doc(coordinator_client):
    r = coordinator_client.get("/v1/capabilities", params={"model": "m"})
    assert r.status_code == 200
    body = r.json()
    assert [m["id"] for m in body["models"]] == ["m"]
    assert [m["modality"] for m in body["modalities"]] == ["audio/speech"]
    assert "unreachable_nodes" in body

    r = coordinator_client.get("/v1/capabilities", params={"modality": "audio/speech"})
    assert r.status_code == 200 and [m["id"] for m in r.json()["models"]] == ["m"]


@pytest.mark.parametrize("params", [
    {"modality": "video/nope"}, {"model": "nope"},
    {"modality": "image/generation", "model": "m"},
])
def test_capabilities_unknown_filter_is_openai_404(coordinator_client, params):
    r = coordinator_client.get("/v1/capabilities", params=params)
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "not_found" and isinstance(err["message"], str)
