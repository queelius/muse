"""FastAPI routes for /v1/3d/generations and /v1/3d/from-image.

Skeleton only; bodies are filled in by Task B (routes + capability
gating). Both endpoints currently raise NotImplementedError so they
fail loudly if the supervisor mounts them before Task B lands.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


MODALITY = "3d/generation"


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/3d/generations")
    async def text_to_3d_route(req: dict):
        raise NotImplementedError("Task B implements")

    @router.post("/v1/3d/from-image")
    async def image_to_3d_route():
        raise NotImplementedError("Task B implements")

    return router
