"""FastAPI gateway: proxy requests by model-id to the right worker.

The gateway is the user-facing process (port 8000 by default). Workers
live on internal ports (9001+). The gateway:
  1. Reads catalog + venv map at startup, builds a model-id -> worker-url table
  2. Extracts `model` from each request (body for POST, query for GET)
  3. Forwards the request to the hosting worker, streaming the response
  4. Aggregates /v1/models and /health across all workers

This file in Task D1 implements ONLY the skeleton: WorkerRoute dataclass,
extract_model_from_request, and build_gateway with a diagnostic endpoint.
Proxy logic (D2), aggregation (D3), and streaming (D4) follow.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

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


def build_gateway(routes: list[WorkerRoute]) -> FastAPI:
    """Build a FastAPI app that proxies requests based on the route table.

    Full implementation lands in subsequent tasks. For now this returns
    an app with only a /_gateway-info diagnostic endpoint so tests can
    assert the route table is preserved.
    """
    app = FastAPI(title="Muse Gateway")
    app.state.routes = {r.model_id: r for r in routes}

    @app.get("/_gateway-info")
    def info():
        return {
            "routes": [
                {"model_id": r.model_id, "worker_url": r.worker_url}
                for r in app.state.routes.values()
            ],
        }

    return app
