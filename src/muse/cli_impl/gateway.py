"""FastAPI gateway: proxy requests by model-id to the right worker.

The gateway is the user-facing process (port 8000 by default). Workers
live on internal ports (9001+). The gateway:
  1. Reads catalog + venv map at startup, builds a model-id -> worker-url table
  2. Extracts `model` from each request (body for POST, query for GET)
  3. Forwards the request to the hosting worker, streaming the response (D4)
  4. Aggregates /v1/models and /health across all workers (D3)

Proxy routing is modality-agnostic: any request with a `model` field
routes to the worker hosting that model, regardless of URL path. This
means future modalities (/v1/embeddings, /v1/audio/transcriptions, ...)
work without gateway changes.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerRoute:
    """One entry in the gateway's routing table.

    A worker may host multiple models; each gets its own WorkerRoute
    pointing at the same worker_url.
    """
    model_id: str
    worker_url: str


async def extract_model_from_request(request: Any) -> str | None:
    """Extract the `model` field from a request.

    - POST with JSON body: body["model"]
    - GET: query_params["model"]
    - Anything else: None

    Returns None (not raises) on missing/invalid. The caller decides
    what "no model specified" means (400, or fall back to default).
    """
    if request.method == "GET":
        return request.query_params.get("model")

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return None
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
            if not isinstance(body, dict):
                return None
            return body.get("model")
        except (json.JSONDecodeError, ValueError):
            return None

    return None


def _openai_error(status: int, code: str, message: str) -> JSONResponse:
    """OpenAI-compatible error envelope."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "type": "invalid_request_error"}},
    )


def build_gateway(routes: list[WorkerRoute], timeout: float = 300.0) -> FastAPI:
    """Build the gateway FastAPI app.

    `routes` is the model-id -> worker-url table. This task (D2) implements
    buffered forwarding; streaming support lands in D4.
    """
    app = FastAPI(title="Muse Gateway")
    app.state.routes = {r.model_id: r for r in routes}
    app.state.timeout = timeout

    @app.get("/_gateway-info")
    def info():
        return {
            "routes": [
                {"model_id": r.model_id, "worker_url": r.worker_url}
                for r in app.state.routes.values()
            ],
        }

    @app.get("/v1/models")
    async def list_models():
        worker_urls = {r.worker_url for r in app.state.routes.values()}
        aggregated: list[dict] = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            async def _one(url: str) -> list[dict]:
                try:
                    r = await client.get(f"{url}/v1/models")
                    if r.status_code != 200:
                        return []
                    return r.json().get("data", [])
                except httpx.HTTPError as e:
                    logger.warning("worker %s unreachable: %s", url, e)
                    return []
            results = await asyncio.gather(*[_one(u) for u in worker_urls])
        for items in results:
            aggregated.extend(items)
        return {"object": "list", "data": aggregated}

    @app.get("/health")
    async def health():
        worker_urls = {r.worker_url for r in app.state.routes.values()}
        modalities: set[str] = set()
        models: set[str] = set()
        any_down = False
        async with httpx.AsyncClient(timeout=5.0) as client:
            async def _one(url: str) -> dict | None:
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code != 200:
                        return None
                    return r.json()
                except httpx.HTTPError:
                    return None
            results = await asyncio.gather(*[_one(u) for u in worker_urls])
        for body in results:
            if body is None:
                any_down = True
                continue
            modalities.update(body.get("modalities", []))
            models.update(body.get("models", []))
        return {
            "status": "degraded" if any_down else "ok",
            "modalities": sorted(modalities),
            "models": sorted(models),
        }

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, full_path: str):
        # NOTE: aggregated endpoints (/v1/models, /health) land in D3 as
        # explicit routes registered BEFORE this catch-all, so they win
        # via FastAPI's registration order.

        model_id = await extract_model_from_request(request)
        if model_id is None:
            return _openai_error(
                400, "model_required",
                "request is missing a `model` field (required for gateway routing)",
            )

        route = app.state.routes.get(model_id)
        if route is None:
            return _openai_error(
                404, "model_not_found",
                f"model {model_id!r} is not registered with any worker; "
                f"known: {sorted(app.state.routes)}",
            )

        target_url = f"{route.worker_url}/{full_path}"
        return await _forward(request, target_url, app.state.timeout)

    return app


async def _forward(request: Request, target_url: str, timeout: float) -> Response:
    """Forward a request to target_url.

    Detects streaming content-types (text/event-stream) and relays chunks
    via StreamingResponse. Non-streaming responses are read fully and
    returned in one go.

    The httpx client and stream context are held open for the duration of
    a streaming response so chunks dispatch as they arrive from the worker
    (not after full synthesis completes). Same producer-consumer shape as
    the audio.speech router's internal streaming.
    """
    body = await request.body()
    excluded = {"host", "content-length", "transfer-encoding", "connection"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = client.stream(
        method=request.method,
        url=target_url,
        headers=fwd_headers,
        content=body,
        params=dict(request.query_params),
    )
    response = await stream_ctx.__aenter__()

    content_type = response.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type

    resp_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in excluded
    }

    if is_stream:
        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await client.aclose()

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    # Non-streaming: read once, close stream + client, return buffered.
    try:
        content = await response.aread()
    finally:
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()

    return Response(
        content=content,
        status_code=response.status_code,
        headers=resp_headers,
    )
