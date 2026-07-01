"""Async-job inspection routes.

  GET /v1/admin/jobs/{job_id}    one job (404 once reaped or unknown)
  GET /v1/admin/jobs             recent jobs, newest first

The JobStore is an in-memory singleton; restarting `muse serve` drops
all in-flight job records. A pull subprocess that survives a restart
will land in the catalog (the subprocess writes catalog.json directly),
but callers polling its job_id afterward get a 404.
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.admin.jobs import get_default_store
from muse.core.errors import error_response


def build_jobs_router() -> APIRouter:
    router = APIRouter()

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str):
        store = get_default_store()
        job = store.get(job_id)
        if job is None:
            return error_response(
                404, "job_not_found",
                f"job {job_id!r} unknown or already reaped",
            )
        return job.to_dict()

    @router.get("/jobs")
    def list_jobs():
        store = get_default_store()
        jobs = [j.to_dict() for j in store.list_recent()]
        return {"jobs": jobs}

    return router
