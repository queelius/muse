"""Discovery entry points on the gateway: `/`, `/.well-known/muse.json`,
`/llms.txt`, and a complete `/openapi.json`.

Every inference route lives behind the gateway's catch-all proxy, so the
gateway's own OpenAPI document lists only admin and telemetry paths. The
merged document built here adds every modality router's operations (each
router mounted on a throwaway app against an empty registry), so an agent
can read real request schemas.
"""
from __future__ import annotations

import copy
import logging
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from muse.core.discovery import (
    discover_modalities,
    modality_dirs,
    modality_discovery,
)
from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)

POINTER = {
    "object": "muse.server",
    "capabilities": "/v1/capabilities",
    "openapi": "/openapi.json",
    "llms_txt": "/llms.txt",
    "models": "/v1/models",
}

_METHODS = ("get", "post", "put", "delete", "patch")


def _rewrite_refs(node, renames: dict[str, str]):
    """Rewrite `#/components/schemas/<old>` refs in place."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            name = ref.rsplit("/", 1)[1]
            if name in renames:
                node["$ref"] = "#/components/schemas/" + renames[name]
        for v in node.values():
            _rewrite_refs(v, renames)
    elif isinstance(node, list):
        for v in node:
            _rewrite_refs(v, renames)


def _slug(tag: str) -> str:
    return "".join(p.capitalize() for p in tag.replace("/", "_").split("_"))


def _modality_schema(tag: str, build_router) -> dict | None:
    try:
        app = FastAPI()
        app.include_router(build_router(ModalityRegistry()))
        return copy.deepcopy(app.openapi())
    except Exception as exc:  # noqa: BLE001
        logger.warning("skipping %s in OpenAPI: router failed to build: %s", tag, exc)
        return None


def merge_openapi(base: dict) -> dict:
    """`base` (the gateway's own schema) plus every modality's operations.

    The gateway wins on a (path, method) clash. A component schema whose
    name is already taken by a different shape is renamed with the modality
    prefix (and that modality's refs rewritten); identical shapes are shared.
    """
    merged = copy.deepcopy(base)
    paths = merged.setdefault("paths", {})
    schemas = merged.setdefault("components", {}).setdefault("schemas", {})
    discovery = modality_discovery()

    try:
        routers = discover_modalities(modality_dirs())
    except Exception as exc:  # noqa: BLE001
        logger.warning("modality discovery failed for OpenAPI: %s", exc)
        routers = {}

    for tag, build_router in sorted(routers.items()):
        sub = _modality_schema(tag, build_router)
        if sub is None:
            continue
        sub_schemas = sub.get("components", {}).get("schemas", {})
        renames: dict[str, str] = {}
        for name, schema in sub_schemas.items():
            if name in schemas and schemas[name] != schema:
                new = f"{_slug(tag)}{name}"
                renames[name] = new
        _rewrite_refs(sub, renames)
        for name, schema in sub_schemas.items():
            schemas.setdefault(renames.get(name, name), schema)

        summaries = {
            (r["method"].lower(), r["path"]): d["summary"]
            for d in [discovery.get(tag)] if d
            for r in d["routes"]
        }
        for path, item in sub.get("paths", {}).items():
            for method in _METHODS:
                op = item.get(method)
                if op is None:
                    continue
                target = paths.setdefault(path, {})
                if method in target:
                    continue
                op = copy.deepcopy(op)
                if (method, path) in summaries:
                    op["description"] = summaries[(method, path)]
                op["tags"] = [tag]
                target[method] = op
    return merged


def install_agent_routes(
    app: FastAPI,
    *,
    capabilities: Callable[[Request], Awaitable[dict]],
) -> None:
    """Mount the discovery entry points; call BEFORE the catch-all proxy."""

    @app.get("/", include_in_schema=False)
    @app.get("/.well-known/muse.json", include_in_schema=False)
    async def pointer():
        return dict(POINTER)

    @app.get("/llms.txt", include_in_schema=False)
    async def llms_txt(request: Request):
        from muse.core.capabilities import render_llms_txt

        doc = await capabilities(request)
        return PlainTextResponse(
            render_llms_txt(doc), media_type="text/markdown; charset=utf-8",
        )

    gateway_openapi = app.openapi

    def openapi() -> dict:
        if app.openapi_schema is None:
            app.openapi_schema = merge_openapi(gateway_openapi())
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
