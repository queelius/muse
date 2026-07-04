"""Slow e2e test: real coordinator over two real muse-node stand-ins.

Approach chosen: (a) two REAL uvicorn servers on real ephemeral ports.
This is the most honest e2e available: the coordinator's `_forward`
(reused from `muse.cli_impl.gateway`) always opens a real
`httpx.AsyncClient` connection to `node.spec.url` -- there is no
injection point to swap in an `httpx.ASGITransport` without
monkeypatching `_forward` itself, and `test_federation_app.py` already
covers that monkeypatched-`_forward` path. Standing up two tiny FastAPI
"node" apps on real ports lets this test exercise the REAL `NodeRegistry`
(real httpx polls of /v1/models + /health) AND the REAL `_forward` (real
httpx request/response over a real socket), including a genuine
`httpx.ConnectError` when a node's process is stopped. It's slower than
an ASGITransport-only test (two background threads, real socket binds)
but it proves the actual wire path end-to-end, which is why this test is
marked `@pytest.mark.slow` rather than living in the fast lane.

Topology:
  - node "a" serves model-a (loaded) and model-shared (loaded).
  - node "b" serves model-b (loaded) and model-shared (present but NOT
    loaded -- i.e. catalog-enabled-but-unloaded on b).

Because model-shared is loaded on a but only enabled-on b,
`select_node`'s loaded-preference rule deterministically prefers node a
for model-shared (regardless of in-flight tie-break / url sort order),
which makes the failover scenario below reproducible without relying on
port-number ordering.

Assertions:
  1. model-a routes to node a.
  2. model-b routes to node b.
  3. After node a's real server process is stopped: a request for
     model-shared (present on both) fails over from a (preferred, now
     down) to b, proving the coordinator's one-shot failover retry
     against a real connection failure.
  4. After node a is stopped: a request for model-a (present ONLY on a)
     has no failover candidate and returns 502 no_node_available.
"""

from __future__ import annotations

import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from muse.cli_impl.federation import build_coordinator
from muse.core.venv import find_free_port
from muse.federation.nodes import NodeSpec
from muse.federation.registry import NodeRegistry

pytestmark = pytest.mark.slow


def _make_node_app(node_name: str, models: list[dict]) -> FastAPI:
    """Tiny fake muse-node app: /health, /v1/models, and an echo
    /v1/chat/completions that reports which node answered."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):  # noqa: ARG001
        return {"node": node_name}

    return app


class _ThreadedUvicorn:
    """Runs one FastAPI app on a real ephemeral port in a background thread."""

    def __init__(self, app: FastAPI, port: int) -> None:
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout: float = 10.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("fake node server did not start in time")

    def stop(self, timeout: float = 10.0) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=timeout)


@pytest.mark.timeout(60)
async def test_coordinator_routes_and_fails_over_across_two_real_nodes():
    port_a = find_free_port(20100, 20199)
    port_b = find_free_port(20200, 20299)

    app_a = _make_node_app(
        "a",
        [
            {"id": "model-a", "loaded": True},
            {"id": "model-shared", "loaded": True},
        ],
    )
    app_b = _make_node_app(
        "b",
        [
            {"id": "model-b", "loaded": True},
            # present (catalog-enabled) but NOT loaded on b, so
            # select_node's loaded-preference rule always prefers a
            # for model-shared while a is up.
            {"id": "model-shared", "loaded": False},
        ],
    )

    server_a = _ThreadedUvicorn(app_a, port_a)
    server_b = _ThreadedUvicorn(app_b, port_b)
    server_a.start()
    server_b.start()

    try:
        node_a = NodeSpec(url=f"http://127.0.0.1:{port_a}", name="a")
        node_b = NodeSpec(url=f"http://127.0.0.1:{port_b}", name="b")
        registry = NodeRegistry([node_a, node_b], refresh_interval=3600.0)

        # Real poll: real httpx GETs against the two real node servers.
        await registry.refresh_once()
        states = registry.snapshot()
        assert {s.spec.name for s in states if s.reachable} == {"a", "b"}, states

        app = build_coordinator(registry, timeout=5.0)
        # Bare TestClient (no `with` block) never triggers the ASGI
        # lifespan, so registry.start()'s background refresh task never
        # launches; the coordinator only ever reads the one snapshot
        # captured above via refresh_once(), which is what makes the
        # failover scenario below deterministic (the snapshot still
        # shows node a as reachable+preferred even after its real
        # process is stopped).
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["node"] == "a"

        r = client.post(
            "/v1/chat/completions",
            json={"model": "model-b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["node"] == "b"

        # Stop node a's real server (real socket close). The cached
        # snapshot is unchanged, so the coordinator still believes a is
        # reachable and prefers it for model-shared.
        server_a.stop()

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-shared",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["node"] == "b"

        # model-a exists ONLY on node a. With a down and no other
        # candidate, the coordinator returns 502 no_node_available
        # rather than silently succeeding against the wrong node.
        r = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "no_node_available"
    finally:
        server_a.stop()
        server_b.stop()
