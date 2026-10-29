"""Readiness must reject the same stale artifacts as the worker loader."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl import supervisor
from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.load_director import LoadDirector
from muse.cli_impl.supervisor import SupervisorState, WorkerSpec
from muse.core import catalog


MODEL = "readiness-test"
REVISION = "1" * 40


class CpuProbe:
    def cpu_free_gb(self):
        return 16.0

    def gpu_free_gb(self):
        return None


@pytest.fixture
def entry(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    receipt = [{"repo_id": "test/readiness", "revision": REVISION, "subdir": "."}]
    data = {
        "enabled": True,
        "hf_repo": "test/readiness",
        "source": "hf://test/readiness",
        "local_dir": str(weights),
        "python_path": sys.executable,
        "revision": REVISION,
        "artifact_provenance": receipt,
        "manifest": {
            "model_id": MODEL,
            "modality": "embedding/text",
            "hf_repo": "test/readiness",
            "backend_path": "muse.fake:Model",
            "revision": REVISION,
            "artifact_provenance": copy.deepcopy(receipt),
            "capabilities": {"device": "cpu", "memory_gb": 0.01},
        },
    }
    catalog._write_catalog({MODEL: data})
    monkeypatch.setattr(supervisor, "_MemoryProbeAdapter", CpuProbe)
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "readiness-test-token")
    supervisor.clear_supervisor_state()
    yield data
    supervisor.clear_supervisor_state()


def _state():
    state = SupervisorState(device="cpu")
    state.director = LoadDirector(
        enable_fn=lambda model_id: pytest.fail("blocked model reached worker startup"),
        disable_fn=lambda model_id: pytest.fail("blocked model evicted a worker"),
        memory_probe=CpuProbe(),
    )
    supervisor.set_supervisor_state(state)
    return state


@pytest.mark.parametrize("mismatch", ["revision", "receipt", "remote_code"])
def test_boot_readiness_reports_loader_provenance_failure(entry, mismatch):
    if mismatch == "revision":
        entry["revision"] = "2" * 40
    elif mismatch == "receipt":
        entry["artifact_provenance"][0]["revision"] = "2" * 40
    else:
        entry["manifest"]["capabilities"].update(
            trust_remote_code=True, code_revision="3" * 40,
        )
    catalog._write_catalog({MODEL: entry})
    state = _state()
    supervisor.validate_catalog_at_boot(state)

    assert MODEL in state.unservable_reasons
    assert f"muse pull {MODEL}" in state.unservable_reasons[MODEL]
    # The same artifact policy is enforced before importing the ML backend.
    with pytest.raises(RuntimeError) as failure:
        catalog.load_backend(MODEL)
    assert state.unservable_reasons[MODEL] == str(failure.value)


@pytest.mark.parametrize("missing", ["weights", "interpreter", "interpreter_record", "non_executable"])
def test_missing_environment_is_visible_and_repair_clears_reason(entry, tmp_path, missing):
    original = copy.deepcopy(entry)
    if missing == "weights":
        entry["local_dir"] = str(tmp_path / "missing-weights")
    elif missing == "interpreter":
        entry["python_path"] = str(tmp_path / "missing-python")
    elif missing == "interpreter_record":
        entry.pop("python_path")
    else:
        interpreter = tmp_path / "not-executable"
        interpreter.write_text("not a runnable interpreter")
        interpreter.chmod(0o600)
        entry["python_path"] = str(interpreter)
    catalog._write_catalog({MODEL: entry})
    state = _state()
    supervisor.validate_catalog_at_boot(state)

    client = TestClient(build_gateway(state=state))
    response = client.get("/v1/models")
    assert response.status_code == 200
    model, = response.json()["data"]
    assert model["id"] == MODEL
    assert model["loaded"] is False
    assert model["unservable_reason"]
    assert f"muse pull {MODEL}" in model["unservable_reason"]

    catalog._write_catalog({MODEL: original})
    model, = client.get("/v1/models").json()["data"]
    assert model["unservable_reason"] is None
    assert MODEL not in state.unservable_reasons


def test_listing_refreshes_provenance_without_an_inference_request(entry):
    state = _state()
    supervisor.validate_catalog_at_boot(state)
    client = TestClient(build_gateway(state=state))
    assert client.get("/v1/models").json()["data"][0]["unservable_reason"] is None

    entry["revision"] = "2" * 40
    catalog._write_catalog({MODEL: entry})
    model, = client.get("/v1/models").json()["data"]
    assert model["unservable_reason"] and "provenance" in model["unservable_reason"]

    entry["revision"] = REVISION
    catalog._write_catalog({MODEL: entry})
    model, = client.get("/v1/models").json()["data"]
    assert model["unservable_reason"] is None


def test_cold_request_refuses_stale_artifacts_before_director_admission(entry):
    entry["revision"] = "2" * 40
    catalog._write_catalog({MODEL: entry})
    state = _state()
    response = TestClient(build_gateway(state=state)).post(
        "/v1/embeddings", json={"model": MODEL, "input": "hello"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unservable"
    assert "provenance" in response.json()["error"]["message"]
    assert not state.director.loaded


def test_admin_status_reports_current_blocking_reason(entry):
    entry["revision"] = "2" * 40
    catalog._write_catalog({MODEL: entry})
    state = _state()
    client = TestClient(build_gateway(state=state))
    response = client.get(
        f"/v1/admin/models/{MODEL}/status",
        headers={"Authorization": "Bearer readiness-test-token"},
    )
    assert response.status_code == 200
    status = response.json()
    assert status["enabled"] is True
    assert status["loaded"] is False
    assert status["unservable_reason"] and "provenance" in status["unservable_reason"]


def test_admin_readiness_uses_the_same_headroom_as_the_gateway(entry):
    entry["manifest"]["capabilities"]["memory_gb"] = 15.0
    catalog._write_catalog({MODEL: entry})
    state = _state()
    state.director.cpu_headroom_gb = 0.0
    client = TestClient(build_gateway(state=state))
    listed, = client.get("/v1/models").json()["data"]
    assert listed["unservable_reason"] is None  # 15 GB fits in 16 - 0 GB.
    status = client.get(
        f"/v1/admin/models/{MODEL}/status",
        headers={"Authorization": "Bearer readiness-test-token"},
    ).json()
    assert status["unservable_reason"] is None


@pytest.mark.parametrize("component", ["weights", "interpreter"])
def test_inaccessible_environment_is_a_model_blocker(entry, tmp_path, monkeypatch, component):
    if component == "interpreter":
        interpreter = tmp_path / "python"
        interpreter.symlink_to(sys.executable)
        entry["python_path"] = str(interpreter)
        catalog._write_catalog({MODEL: entry})
    target = entry["local_dir" if component == "weights" else "python_path"]
    original_stat = Path.stat

    def inaccessible(path, *args, **kwargs):
        if str(path) == target:
            raise PermissionError("fixture path is inaccessible")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible)
    state = _state()
    supervisor.validate_catalog_at_boot(state)
    assert MODEL in state.unservable_reasons
    assert f"muse pull {MODEL}" in state.unservable_reasons[MODEL]
    model, = TestClient(build_gateway(state=state)).get("/v1/models").json()["data"]
    assert model["unservable_reason"]


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs POSIX permissions enforced for a non-root user",
)
@pytest.mark.parametrize("component", ["weights", "interpreter"])
def test_permission_denied_environment_is_not_reported_missing(entry, tmp_path, component):
    """A real EACCES is "inaccessible", not "missing", on every Python.

    Python 3.14's ``Path.is_dir()``/``is_file()`` are ``os.path`` predicates
    that swallow EACCES as False, which used to turn this into the misleading
    "model files are missing" diagnosis there (and only there).
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    if component == "weights":
        target = locked / "weights"
        target.mkdir()
        entry["local_dir"] = str(target)
    else:
        target = locked / "python"
        target.symlink_to(sys.executable)
        entry["python_path"] = str(target)
    locked.chmod(0)
    try:
        reason = catalog.model_load_blocker(MODEL, entry)
    finally:
        locked.chmod(0o700)
    assert reason is not None
    assert "inaccessible" in reason
    assert f"muse pull {MODEL}" in reason


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks")
@pytest.mark.parametrize("component", ["weights", "interpreter"])
def test_symlink_loop_environment_is_reported_inaccessible(entry, tmp_path, component):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    entry["local_dir" if component == "weights" else "python_path"] = str(loop)
    reason = catalog.model_load_blocker(MODEL, entry)
    assert reason is not None and "inaccessible" in reason


@pytest.mark.parametrize("component", ["weights", "interpreter"])
def test_absent_environment_is_reported_missing(entry, tmp_path, component):
    if component == "weights":
        entry["local_dir"] = str(tmp_path / "absent" / "weights")
        expected = "model files are missing"
    else:
        entry["python_path"] = str(tmp_path / "absent" / "python")
        expected = "worker interpreter is missing"
    reason = catalog.model_load_blocker(MODEL, entry)
    assert reason is not None and expected in reason


@pytest.mark.parametrize("invalid", ["backend_path", "capabilities"])
def test_invalid_manifest_does_not_break_listing_of_other_models(entry, invalid):
    valid = copy.deepcopy(entry)
    valid["manifest"]["model_id"] = "valid-readiness-test"
    if invalid == "backend_path":
        entry["manifest"]["backend_path"] = "not-a-backend"
    else:
        entry["manifest"]["capabilities"] = ["not-a-capabilities-map"]
    # Simulate a legacy/operator-written invalid manifest. Internal writes
    # correctly reject it, while reads retain the other valid model.
    catalog._catalog_path().write_text(json.dumps({MODEL: entry, "valid-readiness-test": valid}))
    state = _state()
    response = TestClient(build_gateway(state=state), raise_server_exceptions=False).get("/v1/models")
    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["data"]}
    assert models[MODEL]["unservable_reason"]
    assert models["valid-readiness-test"]["unservable_reason"] is None


def test_admin_warmup_refuses_stale_provenance_before_admission(entry):
    from muse.admin.operations import OperationError, warmup_model

    entry["revision"] = "2" * 40
    catalog._write_catalog({MODEL: entry})
    state = _state()
    with pytest.raises(OperationError) as failure:
        warmup_model(MODEL, state=state)
    assert failure.value.code == "model_unservable"
    assert failure.value.status == 503


def test_admin_enable_refuses_stale_provenance_before_admission(entry):
    from muse.admin.jobs import JobStore
    from muse.admin.operations import enable_model

    entry["revision"] = "2" * 40
    catalog._write_catalog({MODEL: entry})
    state = _state()
    calls = []
    state.director.enable_fn = lambda model_id: calls.append(model_id) or 9001
    store = JobStore()
    job = store.create("enable", MODEL)
    enable_model(MODEL, state=state, store=store, job=job)
    assert job.state == "failed"
    assert "provenance" in job.error
    assert calls == []


@pytest.mark.parametrize(
    ("worker_status", "process_state", "loaded"),
    [
        ("running", "absent", False),
        ("dead", "absent", False),
        ("running", "exited", False),
        ("pending", "alive", False),
        ("running", "alive", True),
    ],
)
def test_admin_loaded_requires_a_live_ready_worker(entry, worker_status, process_state, loaded):
    state = _state()
    spec = WorkerSpec(models=[MODEL], python_path=sys.executable, port=9001, status=worker_status)
    if process_state != "absent":
        code = 1 if process_state == "exited" else None
        spec.process = SimpleNamespace(pid=424242, returncode=code, poll=lambda: code)
    state.workers.append(spec)
    client = TestClient(build_gateway(state=state))
    status = client.get(
        f"/v1/admin/models/{MODEL}/status",
        headers={"Authorization": "Bearer readiness-test-token"},
    ).json()

    assert status["loaded"] is loaded
    # Retain ownership diagnostics even when the record is not a live model.
    assert status["worker_status"] == worker_status
    assert status["worker_port"] == 9001
