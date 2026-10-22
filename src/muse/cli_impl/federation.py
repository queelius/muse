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
the best node, and forwards via the gateway's `_forward`, imported as a
module-level name here so tests can `monkeypatch.setattr(fed,
"_forward", fake_forward)` (this module calls its own global, never
`gateway._forward`).

Every proxied request requires an explicit model, including translation:
the coordinator has no cluster-wide default-model policy. LibreTranslate
clients that omit `model` must connect to a node gateway directly.

Failover: a `_forward` call that fails at CONNECT time (the node is
unreachable/down -- never a mid-stream failure, and never a request the
node already accepted; `_forward` only raises those exceptions
synchronously before it returns a Response; once it returns a
StreamingResponse the relay runs independently and this function is not
on the hook for it) retries ONCE against a different candidate for the
same model. If no other candidate exists, or the retry also fails, the
coordinator returns 502 `no_node_available`. A `httpx.ReadTimeout` (the
node accepted the request and is processing or stuck) is deliberately
NOT a failover trigger: retrying would re-execute a full, possibly
non-idempotent generation on a second node. A ReadTimeout is mapped to a
504 response without retrying.

`run_coordinator` is this module's other export: it resolves the node
list (CLI `--node` entries merged with an optional `federation.yaml`),
builds a `NodeRegistry` + the coordinator app, and blocks serving it via
uvicorn. It is the implementation behind the `muse federate` CLI command
in `muse.cli`.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from muse.cli_impl.gateway import (
    RequestBodyLimitMiddleware,
    _forward,
    _valid_model_id,
    extract_model_from_request,
)
from muse.core import config
from muse.core.errors import error_response
from muse.cli_impl.serve_util import run_uvicorn
from muse.federation.capabilities import merge_capabilities
from muse.federation.nodes import load_nodes
from muse.federation.registry import NodeRegistry
from muse.federation.router import select_node
from muse.federation.state import NodeState

# Connect-phase failures that make a node worth failing over from.
# Deliberately NOT httpx.HTTPError broadly: a worker that connects and
# then returns a 4xx/5xx response is not a transport failure, it is a
# real answer that should be relayed to the client, not retried against
# a different node. Also deliberately NOT httpx.ReadTimeout: a read
# timeout means the node ACCEPTED the request and is processing (or
# stuck) -- failing over would re-execute a full (possibly
# non-idempotent) generation on a second node. Only connect-phase
# failures (the node is unreachable/down) trigger failover; a request
# the node has already accepted is never retried elsewhere and any
# ReadTimeout maps to 504 without failover.
_FAILOVER_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout)


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
    tie-breaking across requests. When the caller passes `None` (the
    default -- this is what `run_coordinator` does for the deployed
    coordinator), a fresh `{}` is created and stored on `app.state`
    instead, so a real deployment actually rotates same-model traffic
    across tied candidates rather than pinning every request to the
    alphabetically-first node. Pass an explicit dict to inject a shared
    counter (or to observe/reset it) from a test.

    Registry lifecycle is wired via a FastAPI lifespan context manager:
    the coordinator warms the snapshot with one `await
    registry.refresh_once()` BEFORE serving (best-effort: a slow or
    failing node at boot is caught and logged rather than crashing
    startup, since the background loop will retry it), then
    `registry.start()` runs (launching the background refresh task on
    the server's OWN event loop, which is what `NodeRegistry.start`
    needs -- `asyncio.ensure_future` requires a running loop), and
    `registry.aclose()` runs on shutdown. This only fires when the ASGI
    lifespan protocol actually runs (real `run_uvicorn` serving, or a
    `with TestClient(app) as c:` block); a bare `TestClient(app).get(...)`
    (as the existing route tests in this module use) never triggers
    lifespan, so those tests' plain-`.snapshot()`-only fakes (which don't
    implement `refresh_once`/`start`/`aclose`) are unaffected by this
    wiring.
    """
    rr_counter = {} if rr_counter is None else rr_counter

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            await registry.refresh_once()
        except Exception:
            logging.getLogger(__name__).warning(
                "federation: initial refresh_once failed at boot; serving "
                "with an empty snapshot until the background loop's first "
                "poll lands",
                exc_info=True,
            )
        registry.start()
        try:
            yield
        finally:
            await registry.aclose()

    app = FastAPI(
        title="Muse Federation Coordinator", lifespan=_lifespan,
        openapi_url=None,  # served below: first node's schema + our paths
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        limit=int(config.get("server.max_request_body_mb")) * 1024 * 1024,
    )
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

    # ---- agent discovery ------------------------------------------------
    # Coordinator-specific routes rather than agent_routes.install_agent_routes:
    # that helper replaces app.openapi with the LOCAL modality merge, which
    # is wrong here (the schema comes from a node). Node tokens are never
    # sent: discovery documents are public on a node.

    async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict | None:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            body = r.json()
            return body if isinstance(body, dict) else None
        except Exception:  # noqa: BLE001 - any failure == unreachable
            return None

    async def _fetch_nodes(path: str, states: list[NodeState]):
        t = float(config.get("server.aggregation_timeout_seconds"))
        async with httpx.AsyncClient(timeout=t) as client:
            return await asyncio.gather(
                *(_fetch_json(client, f"{s.spec.url}{path}") for s in states
                  if s.reachable)
            )

    async def _merged_capabilities(request: Request) -> dict:
        from muse import __version__
        from muse.core.capabilities import server_block

        states: list[NodeState] = registry.snapshot()
        live = [s for s in states if s.reachable]
        docs = await _fetch_nodes("/v1/capabilities", states)
        by_name = {s.spec.name: d for s, d in zip(live, docs)}
        node_docs = [(s.spec.name, by_name.get(s.spec.name)) for s in states]
        base_url = str(request.base_url).rstrip("/")
        server = server_block(
            version=__version__, node="federation-coordinator", base_url=base_url,
        )
        server["nodes"] = [
            {"name": s.spec.name, "url": s.spec.url} for s in states
        ]
        return merge_capabilities(node_docs, server=server)

    @app.get("/v1/capabilities")
    async def capabilities(
        request: Request, modality: str | None = None, model: str | None = None,
    ):
        # Same filter semantics as a node's gateway, applied to the merge.
        from muse.core.capabilities import UnknownFilter, filter_document

        doc = await _merged_capabilities(request)
        try:
            return filter_document(doc, modality=modality, model=model)
        except UnknownFilter as exc:
            return error_response(404, "not_found", str(exc))

    @app.get("/", include_in_schema=False)
    @app.get("/.well-known/muse.json", include_in_schema=False)
    async def pointer():
        from muse.cli_impl.agent_routes import POINTER

        return dict(POINTER)

    @app.get("/llms.txt", include_in_schema=False)
    async def llms_txt(request: Request):
        from muse.core.capabilities import render_llms_txt

        return PlainTextResponse(
            render_llms_txt(await _merged_capabilities(request)),
            media_type="text/markdown; charset=utf-8",
        )

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_json():
        from fastapi.openapi.utils import get_openapi

        own = get_openapi(
            title=app.title, version="1", routes=app.routes,
        )
        states = [s for s in registry.snapshot() if s.reachable]
        base = None
        for s, doc in zip(states, await _fetch_nodes("/openapi.json", states)):
            if doc is not None:
                base = doc
                break
        if base is None:
            return own
        base.setdefault("paths", {})
        for path, item in own.get("paths", {}).items():
            base["paths"][path] = item
        schemas = base.setdefault("components", {}).setdefault("schemas", {})
        for name, schema in own.get("components", {}).get("schemas", {}).items():
            schemas.setdefault(name, schema)
        return base

    @app.api_route("/{full_path:path}", methods=["GET", "POST"], include_in_schema=False)
    async def proxy(request: Request, full_path: str) -> JSONResponse:
        model_id = await extract_model_from_request(request)
        if model_id is None:
            return error_response(
                400, "model_required",
                "request is missing a `model` field (required for "
                "federation routing)",
            )

        if not _valid_model_id(model_id):
            return error_response(
                400, "invalid_model",
                "`model` must be a non-empty string of at most "
                "512 characters without control characters",
            )

        states: list[NodeState] = registry.snapshot()
        node = select_node(model_id, states, rr_counter=app.state.rr_counter)
        if node is None:
            return error_response(
                404, "model_not_available",
                f"no node serves model {model_id!r}",
            )

        # At most two attempts: the selected node, then ONE failover
        # candidate when the first fails at connect time. Re-running
        # select_node against the snapshot minus the failed node
        # re-derives loaded-preference and in-flight tie-break rather than
        # blindly grabbing "the other one" (there may be more than two).
        failed: NodeState | None = None
        while True:
            try:
                return await _forward(
                    request, f"{node.spec.url}/{full_path}", app.state.timeout,
                )
            except _FAILOVER_EXCEPTIONS:
                if failed is not None:
                    return error_response(
                        502, "no_node_available",
                        f"both {failed.spec.url!r} and {node.spec.url!r} "
                        f"failed for model {model_id!r}",
                    )
                failed = node
                remaining = [s for s in states if s.spec.url != failed.spec.url]
                node = select_node(
                    model_id, remaining, rr_counter=app.state.rr_counter,
                )
                if node is None:
                    return error_response(
                        502, "no_node_available",
                        f"node {failed.spec.url!r} failed and no other node "
                        f"serves model {model_id!r}",
                    )
            except httpx.TimeoutException:
                return error_response(
                    504, "node_timeout",
                    f"node {node.spec.url!r} timed out while serving "
                    f"model {model_id!r}",
                )
            except httpx.TransportError:
                return error_response(
                    502, "node_forward_failed",
                    f"node {node.spec.url!r} failed while serving "
                    f"model {model_id!r}",
                )

    return app


def run_coordinator(
    *,
    host: str,
    port: int,
    cli_nodes: list[str] | None,
    config_path: str | None,
) -> int:
    """Resolve federation nodes, build the coordinator app, and serve it.

    Node-list resolution order:
      1. `config_path` (the CLI `--config` flag) if given.
      2. Else `federation.config_file` (`MUSE_FEDERATION_CONFIG` / config.yaml).
      3. Else `<catalog_dir>/federation.yaml`, ONLY if that file actually
         exists (so a fresh install with no federation.yaml does not
         error trying to read a missing file -- `load_nodes` treats
         `config_path=None` as "no yaml source" and merges CLI nodes only).

    `load_nodes` then merges CLI `--node` entries with the resolved yaml
    file (CLI wins on url collision, per `muse.federation.nodes`).

    Returns 2 (and starts nothing) when the merged node list is empty --
    a coordinator with zero nodes would boot healthy and 404 every
    request, which is a worse failure mode than refusing to start.
    Otherwise builds the `NodeRegistry` + coordinator app (whose lifespan
    hook starts/stops the registry's background refresh loop, see
    `build_coordinator`) and blocks on uvicorn; returns 0 after a clean
    shutdown.
    """
    from muse.core import config

    resolved_config_path = config_path or config.get("federation.config_file")
    if resolved_config_path is None:
        catalog_dir = Path(config.get("paths.catalog_dir")).expanduser()
        candidate = catalog_dir / "federation.yaml"
        if candidate.exists():
            resolved_config_path = str(candidate)

    nodes = load_nodes(cli_nodes=cli_nodes, config_path=resolved_config_path)
    if not nodes:
        print(
            "no federation nodes: pass --node URL or create a federation.yaml",
            file=sys.stderr,
        )
        return 2

    registry = NodeRegistry(
        nodes,
        refresh_interval=config.get("federation.refresh_interval_seconds"),
        poll_timeout=config.get("federation.poll_timeout_seconds"),
    )
    app = build_coordinator(
        registry, timeout=config.get("federation.forward_timeout_seconds")
    )
    run_uvicorn(app, host=host, port=port)
    return 0
