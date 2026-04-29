"""Tests for admin operations: enable / disable / remove / probe / pull.

All tests mock subprocess.Popen + spawn_worker + wait_for_ready so no
actual workers are spawned. SupervisorState instances are local to each
test (never the singleton) so tests don't bleed into each other.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.admin.jobs import JobStore
from muse.admin.operations import (
    OperationError,
    disable_model,
    enable_model,
    find_worker_for_model,
    launch_async,
    probe_model,
    pull_model,
    remove_model,
)
from muse.cli_impl.supervisor import SupervisorState, WorkerSpec


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data: dict) -> None:
    from muse.core.catalog import _catalog_path, _reset_known_models_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    _reset_known_models_cache()


@pytest.fixture
def state():
    return SupervisorState(workers=[], device="cpu")


@pytest.fixture
def store():
    return JobStore()


class TestFindWorkerForModel:
    def test_finds_model_in_worker(self, state):
        spec = WorkerSpec(
            models=["soprano-80m"], python_path="/p", port=9001,
        )
        state.workers.append(spec)
        assert find_worker_for_model(state, "soprano-80m") is spec

    def test_returns_none_when_unhosted(self, state):
        assert find_worker_for_model(state, "unknown") is None


class TestEnableModel:
    def test_unknown_model_marks_failed(self, tmp_catalog, state, store):
        _seed_catalog({})
        job = store.create("enable", "ghost")
        enable_model("ghost", state=state, store=store, job=job)
        assert job.state == "failed"
        assert "unknown model" in job.error

    def test_already_loaded_marks_done_no_spawn(self, tmp_catalog, state, store, monkeypatch):
        # Seed with a bundled-style model so known_models() picks it up.
        from muse.core import catalog as catalog_mod
        # Inject a fake known_models entry.
        monkeypatch.setattr(catalog_mod, "_known_models_cache", None)
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "hexgrad/Kokoro-82M",
                "local_dir": "/tmp/kokoro",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        # Pre-load it into a worker.
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        state.workers.append(spec)
        job = store.create("enable", "kokoro-82m")
        with patch("muse.admin.operations.spawn_worker") as mock_spawn:
            enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "done"
        assert job.result["spawned_new"] is False
        assert job.result["worker_port"] == 9001
        mock_spawn.assert_not_called()

    def test_spawns_new_worker_when_no_venv_match(self, tmp_catalog, state, store):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "hexgrad/Kokoro-82M",
                "local_dir": "/tmp/kokoro",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })
        job = store.create("enable", "kokoro-82m")
        with patch("muse.admin.operations.spawn_worker") as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready") as mock_wait, \
             patch("muse.admin.operations.find_free_port", return_value=9123):
            enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "done"
        assert job.result["spawned_new"] is True
        assert job.result["worker_port"] == 9123
        mock_spawn.assert_called_once()
        mock_wait.assert_called_once_with(port=9123, timeout=120.0)
        assert len(state.workers) == 1
        assert state.workers[0].port == 9123

    def test_joins_existing_venv_group(self, tmp_catalog, state, store):
        # Two bundled models share a venv path; one is already running,
        # the second enable should restart-in-place.
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": False,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/shared/bin/python",
            port=9001,
        )
        state.workers.append(spec)
        job = store.create("enable", "soprano-80m")
        with patch("muse.admin.operations._restart_worker_inplace") as mock_restart:
            enable_model("soprano-80m", state=state, store=store, job=job)
        assert job.state == "done", f"expected done, got {job.state} (error={job.error})"
        assert job.result["spawned_new"] is False
        assert "soprano-80m" in spec.models
        assert "kokoro-82m" in spec.models
        mock_restart.assert_called_once_with(spec, device="cpu")

    def test_unpulled_model_marks_failed(self, tmp_catalog, state, store):
        # Bundled known but not yet in catalog.json.
        _seed_catalog({})
        job = store.create("enable", "kokoro-82m")
        # known_models scans bundled scripts; kokoro-82m is bundled.
        enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "failed"
        assert "not pulled" in job.error


class TestDisableModel:
    def test_unknown_raises_operation_error(self, tmp_catalog, state):
        _seed_catalog({})
        with pytest.raises(OperationError) as exc:
            disable_model("ghost", state=state)
        assert exc.value.status == 404

    def test_unloaded_returns_unloaded_record(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        out = disable_model("kokoro-82m", state=state)
        assert out["model_id"] == "kokoro-82m"
        assert out["loaded"] is False
        assert out["worker_terminated"] is False

    def test_terminates_worker_when_only_model(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        state.workers.append(spec)
        with patch("muse.admin.operations._shutdown_workers") as mock_sd:
            out = disable_model("kokoro-82m", state=state)
        assert out["worker_terminated"] is True
        assert out["worker_port"] == 9001
        assert state.workers == []
        mock_sd.assert_called_once_with([spec])

    def test_restarts_worker_when_other_models_remain(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m", "soprano-80m"],
            python_path="/venv/shared/bin/python", port=9001,
        )
        state.workers.append(spec)
        with patch("muse.admin.operations._restart_worker_inplace") as mock_restart:
            out = disable_model("soprano-80m", state=state)
        assert out["worker_terminated"] is False
        assert "kokoro-82m" in out["remaining_models_in_worker"]
        assert "soprano-80m" not in spec.models
        mock_restart.assert_called_once_with(spec, device="cpu")


class TestRemoveModel:
    def test_unknown_model_raises_404(self, tmp_catalog, state):
        _seed_catalog({})
        with pytest.raises(OperationError) as exc:
            remove_model("ghost", state=state, purge=False)
        assert exc.value.status == 404

    def test_loaded_model_raises_409(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        state.workers.append(spec)
        with pytest.raises(OperationError) as exc:
            remove_model("kokoro-82m", state=state, purge=False)
        assert exc.value.status == 409

    def test_unloaded_model_is_removed(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        out = remove_model("kokoro-82m", state=state, purge=False)
        assert out == {"model_id": "kokoro-82m", "removed": True, "purged": False}


class TestProbeAndPull:
    def test_probe_runs_subprocess(self, tmp_catalog, store):
        job = store.create("probe", "kokoro-82m")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="ok", stderr="",
            )
            probe_model(
                "kokoro-82m", no_inference=True, device=None,
                store=store, job=job,
            )
        assert job.state == "done"
        assert job.result["op"] == "probe"
        assert "--no-inference" in mock_run.call_args.args[0]

    def test_probe_failure_marks_failed(self, tmp_catalog, store):
        job = store.create("probe", "kokoro-82m")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="boom",
            )
            probe_model(
                "kokoro-82m", no_inference=False, device="cpu",
                store=store, job=job,
            )
        assert job.state == "failed"
        assert "boom" in job.error

    def test_pull_runs_subprocess(self, tmp_catalog, store):
        job = store.create("pull", "qwen3-9b-q4")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="pulled", stderr="",
            )
            pull_model("qwen3-9b-q4", store=store, job=job)
        assert job.state == "done"
        cmd = mock_run.call_args.args[0]
        assert "pull" in cmd
        assert "qwen3-9b-q4" in cmd

    def test_pull_timeout_marks_failed(self, tmp_catalog, store):
        import subprocess as sp
        job = store.create("pull", "qwen3-9b-q4")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.side_effect = sp.TimeoutExpired(cmd="x", timeout=1)
            pull_model("qwen3-9b-q4", store=store, job=job)
        assert job.state == "failed"
        assert "timed out" in job.error


class TestLaunchAsync:
    def test_creates_job_and_thread(self, store):
        ran = {}

        def op(*, job, store, **_kwargs):  # noqa: ARG001
            ran["job_id"] = job.job_id

        job = launch_async(
            op, op_name="enable", model_id="m", store=store,
        )
        job.thread.join(timeout=2.0)
        assert ran["job_id"] == job.job_id
        assert job.thread is not None

    def test_thread_is_daemon(self, store):
        def op(*, job, store, **_kwargs):  # noqa: ARG001
            pass

        job = launch_async(op, op_name="enable", model_id="m", store=store)
        job.thread.join(timeout=2.0)
        assert job.thread.daemon is True
