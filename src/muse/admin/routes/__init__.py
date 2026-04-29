"""Admin router assembly.

`build_admin_router()` returns one APIRouter mounted at /v1/admin that
includes every per-resource sub-router. The auth dependency is added
once at the parent level so each child router does not need to re-
declare it on every endpoint.

Sub-routers are split by resource (models, workers, memory, jobs) for
file-size hygiene; the wire layout is flat, all under /v1/admin/*.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from muse.admin.auth import verify_admin_token
from muse.admin.routes.jobs import build_jobs_router
from muse.admin.routes.memory import build_memory_router
from muse.admin.routes.models import build_models_router
from muse.admin.routes.workers import build_workers_router


def build_admin_router() -> APIRouter:
    """Assemble the /v1/admin/* router with auth applied once at the top."""
    router = APIRouter(
        prefix="/v1/admin",
        dependencies=[Depends(verify_admin_token)],
    )
    router.include_router(build_models_router())
    router.include_router(build_workers_router())
    router.include_router(build_memory_router())
    router.include_router(build_jobs_router())
    return router
