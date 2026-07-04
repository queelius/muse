"""FastAPI coordinator app for muse federation (v1: model-locality routing).

`build_coordinator` assembles a small FastAPI app that fronts a fixed set
of remote muse `serve` nodes (each itself a gateway). Unlike the
single-host gateway (`muse.cli_impl.gateway`), the coordinator does not
own workers or a LoadDirector; it only reads a `NodeRegistry` snapshot
(polled on a background interval) and forwards each request to whichever
node currently serves the requested model.

Three explicit routes are registered BEFORE the catch-all proxy so they
are not shadowed by it (FastAPI dispatches by first-match registration
order, same discipline as `muse.cli_impl.gateway.build_gateway`):

  - GET /health: aggregate reachability across the node snapshot.
  - GET /v1/models: OpenAI-shape union of model ids across reachable
    nodes, plus muse-specific `loaded` / `nodes` extras.
  - GET /v1/federation/nodes: operator-facing per-node detail (loaded
    model list, in_flight, last-poll age).

The catch-all extracts `model` from the request, asks `select_node` for
the best node, and forwards. `_forward` is imported from the existing
gateway module and re-exported as a MODULE-LEVEL name here so tests can
`monkeypatch.setattr(fed, "_forward", fake_forward)` -- re-binding the
import target has no effect on the gateway module's own callers, and
this module always calls its OWN `_forward` name (not
`gateway._forward`), so the patch takes effect.

Failover: a `_forward` call that fails at connect/timeout (never a
mid-stream failure -- `_forward` only raises those exceptions synchronously
before it returns a Response; once it returns a StreamingResponse the
relay runs independently and this function is not on the hook for it)
retries ONCE against a different candidate for the same model. If no
other candidate exists, or the retry also fails, the coordinator returns
502 `no_node_available`.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from muse.cli_impl.gateway import _forward, _openai_error, extract_model_from_request
from muse.federation.router import select_node
from muse.federation.state import NodeState

# Re-bound as a module-level name (not read through `gateway._forward`)
# specifically so `monkeypatch.setattr(fed, "_forward", fake_forward)`
# takes effect: this module's code paths all call the bare `_forward`
# name below, which Python resolves against THIS module's globals.
_forward = _forward

# Connect/timeout failures that make a node worth failing over from.
# Deliberately NOT httpx.HTTPError broadly: a worker that connects and
# then returns a 4xx/5xx response is not a transport failure, it is a
# real answer that should be relayed to the client, not retried against
# a different node.
_FAILOVER_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)


def build_coordinator(
    registry: Any,
    *,
    timeout: float,
    rr_counter: dict | None = None,
) -> FastAPI:
    """Build the federation coordinator FastAPI app.

    `registry` is a `muse.federation.registry.NodeRegistry` (or anything
    exposing `.snapshot() -> list[NodeState]`; tests pass a plain fake).
    `timeout` bounds each forwarded request. `rr_counter` is an optional
    shared dict passed through to `select_node` for round-robin
    tie-breaking across requests; `None` disables rotation (deterministic
    first-by-url).
    """
    app = FastAPI(title="Muse Federation Coordinator")
    app.state.registry = registry
    app.state.timeout = timeout
    app.state.rr_counter = rr_counter

    @app.get("/health")
    async def health():
        states: list[NodeState] = registry.snapshot()
        nodes = [
            {
                "name": s.spec.name,
                "url": s.spec.url,
                "reachable": s.reachable,
                "model_count": len(s.models),
                "in_flight": s.in_flight,
            }
            for s in states
        ]
        status = "ok" if any(s.reachable for s in states) else "degraded"
        return {"status": status, "nodes": nodes}

    @app.get("/v1/models")
    async def list_models():
        states: list[NodeState] = registry.snapshot()
        # model_id -> (loaded_anywhere, [node names that have it])
        union: dict[str, tuple[bool, list[str]]] = {}
        for s in states:
            if not s.reachable:
                continue
            for model_id, avail in s.models.items():
                loaded_anywhere, names = union.get(model_id, (False, []))
                names = names + [s.spec.name]
                union[model_id] = (loaded_anywhere or avail.loaded, names)

        data = [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "muse",
                "loaded": loaded_anywhere,
                "nodes": names,
            }
            for model_id, (loaded_anywhere, names) in sorted(union.items())
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/federation/nodes")
    async def federation_nodes():
        states: list[NodeState] = registry.snapshot()
        now = time.monotonic()
        nodes = [
            {
                "name": s.spec.name,
                "url": s.spec.url,
                "reachable": s.reachable,
                "loaded": sorted(
                    model_id for model_id, avail in s.models.items() if avail.loaded
                ),
                "in_flight": s.in_flight,
                "last_poll_age_seconds": max(0.0, now - s.last_poll_ts),
            }
            for s in states
        ]
        return {"nodes": nodes}

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def proxy(request: Request, full_path: str) -> JSONResponse:
        model_id = await extract_model_from_request(request)
        if model_id is None:
            return _openai_error(
                400, "model_required",
                "request is missing a `model` field (required for "
                "federation routing)",
            )

        states: list[NodeState] = registry.snapshot()
        node = select_node(model_id, states, rr_counter=app.state.rr_counter)
        if node is None:
            return _openai_error(
                404, "model_not_available",
                f"no node serves model {model_id!r}",
            )

        target_url = f"{node.spec.url}/{full_path}"
        try:
            return await _forward(request, target_url, app.state.timeout)
        except _FAILOVER_EXCEPTIONS:
            # Retry ONCE against another candidate for the same model,
            # excluding the failed node's url. Re-running select_node
            # against the filtered snapshot re-derives loaded-preference
            # and in-flight tie-break rather than blindly grabbing
            # "the other one" (there may be more than two candidates).
            remaining = [s for s in states if s.spec.url != node.spec.url]
            fallback = select_node(model_id, remaining, rr_counter=app.state.rr_counter)
            if fallback is None:
                return _openai_error(
                    502, "no_node_available",
                    f"node {node.spec.url!r} failed and no other node "
                    f"serves model {model_id!r}",
                    error_type="server_error",
                )
            fallback_url = f"{fallback.spec.url}/{full_path}"
            try:
                return await _forward(request, fallback_url, app.state.timeout)
            except _FAILOVER_EXCEPTIONS:
                return _openai_error(
                    502, "no_node_available",
                    f"both {node.spec.url!r} and {fallback.spec.url!r} "
                    f"failed for model {model_id!r}",
                    error_type="server_error",
                )

    return app
