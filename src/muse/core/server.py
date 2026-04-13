"""FastAPI application factory.

Modality routers are mounted via `create_app(registry, routers=...)`.
Each modality supplies its own APIRouter; core adds /health and /v1/models
(aggregated across modalities).
"""
from __future__ import annotations

import logging
from typing import Mapping

from fastapi import APIRouter, FastAPI

from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)


def create_app(
    *,
    registry: ModalityRegistry,
    routers: Mapping[str, APIRouter],
    title: str = "Muse",
) -> FastAPI:
    """Build a FastAPI app with shared /health + /v1/models endpoints.

    `routers` maps modality-name → APIRouter. Each router is mounted
    with its own internal paths (e.g. /v1/audio/speech).
    """
    app = FastAPI(title=title)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "modalities": registry.modalities(),
            "models": [info.model_id for info in registry.list_all()],
        }

    @app.get("/v1/models")
    def list_models():
        data = []
        for info in registry.list_all():
            entry = {"id": info.model_id, "modality": info.modality, "object": "model"}
            if info.extra:
                entry.update(info.extra)
            data.append(entry)
        return {"object": "list", "data": data}

    for name, router in routers.items():
        logger.info("mounting modality router %s", name)
        app.include_router(router)

    app.state.registry = registry
    return app
