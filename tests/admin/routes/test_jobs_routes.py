"""Tests for /v1/admin/jobs and /v1/admin/jobs/{job_id}."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.jobs import get_default_store, reset_default_store
from muse.admin.routes.jobs import build_jobs_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(
        build_jobs_router(),
        prefix="/v1/admin",
        dependencies=[Depends(verify_admin_token)],
    )
    return app


@pytest.fixture
def client(app, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "test-token")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _reset_store():
    reset_default_store()
    yield
    reset_default_store()


class TestGetJob:
    def test_unknown_returns_404(self, client, headers):
        r = client.get("/v1/admin/jobs/nonexistent", headers=headers)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "job_not_found"

    def test_known_returns_dict(self, client, headers):
        store = get_default_store()
        job = store.create("enable", "kokoro-82m")
        r = client.get(f"/v1/admin/jobs/{job.job_id}", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job.job_id
        assert body["op"] == "enable"
        assert body["model_id"] == "kokoro-82m"


class TestListJobs:
    def test_empty(self, client, headers):
        r = client.get("/v1/admin/jobs", headers=headers)
        assert r.status_code == 200
        assert r.json() == {"jobs": []}

    def test_lists_recent(self, client, headers):
        store = get_default_store()
        a = store.create("enable", "model-a")
        b = store.create("pull", "model-b")
        r = client.get("/v1/admin/jobs", headers=headers)
        assert r.status_code == 200
        body = r.json()
        ids = [j["job_id"] for j in body["jobs"]]
        # Newest first per JobStore.list_recent contract
        assert ids == [b.job_id, a.job_id]
