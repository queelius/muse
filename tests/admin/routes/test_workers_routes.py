"""Tests for /v1/admin/workers and /v1/admin/workers/{port}/restart."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.routes.workers import build_workers_router
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    set_supervisor_state,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(
        build_workers_router(),
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
def _state_reset():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


class TestListWorkers:
    def test_empty_state_returns_empty_list(self, client, headers):
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        r = client.get("/v1/admin/workers", headers=headers)
        assert r.status_code == 200
        assert r.json() == {"workers": []}

    def test_lists_one_worker(self, client, headers):
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        spec.status = "running"
        spec.last_spawn_at = 0.0
        spec.process = MagicMock(pid=999)
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.get("/v1/admin/workers", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert len(body["workers"]) == 1
        w = body["workers"][0]
        assert w["port"] == 9001
        assert w["models"] == ["kokoro-82m"]
        assert w["pid"] == 999
        assert w["status"] == "running"


class TestRestartWorker:
    def test_unknown_port_returns_404(self, client, headers):
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        r = client.post("/v1/admin/workers/99999/restart", headers=headers)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "worker_not_found"

    def test_dead_process_less_worker_is_revived(self, client, headers):
        """R11: a worker that exhausted its restart budget has no process
        and status "dead"; the monitor skips dead specs. The route revives
        it (status unhealthy, budget reset) so the monitor respawns it."""
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.status = "dead"
        spec.restart_count = 10
        spec.failure_count = 3
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["signal"] is None and body["revived"] is True
        assert body["restart_count_reset"] is True
        assert spec.status == "unhealthy"
        assert spec.restart_count == 0 and spec.failure_count == 0

    def test_process_less_worker_without_dead_status_is_revived(self, client, headers):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        # process is None by default
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        assert spec.status == "unhealthy"

    def test_process_less_worker_mid_transition_returns_409(self, client, headers):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.job_id = "owning-op"
        spec.status = "dead"
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "worker_busy"
        assert spec.status == "dead"

    def test_exited_process_is_revived_not_409(self, client, headers):
        """A process that already exited cannot be signalled; the spec is
        revived instead so the monitor's next tick respawns it."""
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = MagicMock(pid=4242)
        spec.process.poll.return_value = 1
        spec.status = "dead"
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        assert r.json()["revived"] is True
        assert spec.status == "unhealthy"
        spec.process.terminate.assert_not_called()

    def test_signalled_dead_worker_status_is_revived(self, client, headers):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = MagicMock(pid=4242)
        spec.process.poll.return_value = None
        spec.status = "dead"
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        assert r.json()["signal"] == "SIGTERM"
        assert spec.status == "unhealthy"

    def test_terminate_called_and_200_returned(self, client, headers):
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        spec.process = MagicMock(pid=4242)
        spec.process.poll.return_value = None
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        assert r.json()["signal"] == "SIGTERM"
        spec.process.terminate.assert_called_once()

    def test_terminate_failure_returns_500_server_error(self, client, headers):
        """A SIGTERM that raises is a 500 terminate_failed, and its OpenAI
        type is derived from the status (server_error for 5xx), not the
        hardcoded invalid_request_error the old local builder emitted."""
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        spec.process = MagicMock(pid=4242)
        spec.process.poll.return_value = None
        spec.process.terminate.side_effect = OSError("no such process")
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "terminate_failed"
        assert body["error"]["type"] == "server_error"

    def test_resets_restart_and_failure_counters(self, client, headers):
        """v0.34.0 finding #12: an operator's explicit restart must
        re-arm the restart budget. Without this, a worker 9-deep into
        the 10-restart cap would die permanently on the next failure
        even though the operator just intervened."""
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        spec.process = MagicMock(pid=4242)
        spec.process.poll.return_value = None
        spec.restart_count = 9
        spec.failure_count = 3
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))

        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["restart_count_reset"] is True
        assert spec.restart_count == 0
        assert spec.failure_count == 0
        spec.process.terminate.assert_called_once()


def test_non_integer_port_is_an_openai_400(monkeypatch):
    from muse.admin.errors import install_admin_error_handler

    app = FastAPI()
    app.include_router(
        build_workers_router(), prefix="/v1/admin",
        dependencies=[Depends(verify_admin_token)],
    )
    install_admin_error_handler(app)
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "test-token")
    r = TestClient(app).post(
        "/v1/admin/workers/abc/restart",
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
