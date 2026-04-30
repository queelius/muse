"""Tests for the supervisor: catalog -> worker specs."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    get_supervisor_state,
    plan_workers,
    set_supervisor_state,
)


@pytest.fixture(autouse=True)
def _reset_supervisor_state():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data):
    """Write catalog.json directly."""
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


class TestPlanWorkers:
    def test_empty_catalog_yields_no_workers(self, tmp_catalog):
        _seed_catalog({})
        specs = plan_workers()
        assert specs == []

    def test_one_model_yields_one_worker(self, tmp_catalog):
        _seed_catalog({
            "soprano-80m": {
                "pulled_at": "2026-04-13T00:00:00Z",
                "hf_repo": "ekwek/Soprano-1.1-80M",
                "local_dir": "/fake/local",
                "venv_path": "/home/user/.muse/venvs/soprano-80m",
                "python_path": "/home/user/.muse/venvs/soprano-80m/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.models == ["soprano-80m"]
        assert spec.python_path == "/home/user/.muse/venvs/soprano-80m/bin/python"
        assert isinstance(spec.port, int)
        assert 9001 <= spec.port <= 9999

    def test_models_in_same_venv_share_a_worker(self, tmp_catalog):
        """If two models share venv_path (rare in M1 but supported), one worker."""
        shared_venv = "/home/user/.muse/venvs/shared"
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": shared_venv,
                "python_path": f"{shared_venv}/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": shared_venv,
                "python_path": f"{shared_venv}/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 1
        assert set(specs[0].models) == {"model-a", "model-b"}

    def test_different_venvs_yield_different_workers(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 2
        assert specs[0].port != specs[1].port

    def test_skips_pre_worker_entries_without_python_path(self, tmp_catalog, caplog):
        """Old catalog entries (no python_path field) are skipped with warning."""
        _seed_catalog({
            "legacy-model": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                # No venv_path / python_path - pre-worker entry
            },
            "new-model": {
                "pulled_at": "...", "hf_repo": "y", "local_dir": "/y",
                "venv_path": "/venvs/y",
                "python_path": "/venvs/y/bin/python",
            },
        })
        import logging
        caplog.set_level(logging.WARNING)
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "legacy-model" not in all_models
        assert "new-model" in all_models
        # Warning should mention the legacy model id or re-pulling
        assert "legacy-model" in caplog.text or "re-pull" in caplog.text.lower() or "re-run" in caplog.text.lower()

    def test_skips_disabled_models(self, tmp_catalog):
        """Disabled models are filtered out of plan_workers results."""
        _seed_catalog({
            "enabled-model": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
            "disabled-model": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
                "enabled": False,
            },
        })
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "enabled-model" in all_models
        assert "disabled-model" not in all_models

    def test_legacy_entries_without_enabled_field_treated_as_enabled(self, tmp_catalog):
        """Pre-flag entries (no `enabled` key) must still be planned."""
        _seed_catalog({
            "legacy-model": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                # no 'enabled' key
            },
        })
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "legacy-model" in all_models


class TestSpawnWorker:
    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_invokes_venv_python_with_worker_subcommand(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=12345)
        spec = WorkerSpec(
            models=["soprano-80m"],
            python_path="/venvs/soprano-80m/bin/python",
            port=9001,
        )
        from muse.cli_impl.supervisor import spawn_worker
        spawn_worker(spec, device="cpu")
        mock_popen.assert_called_once()
        args = mock_popen.call_args.args[0]
        assert args[0] == "/venvs/soprano-80m/bin/python"
        assert args[1:4] == ["-m", "muse.cli", "_worker"]
        assert "--port" in args and "9001" in args
        assert "--model" in args and "soprano-80m" in args
        assert "--device" in args and "cpu" in args
        assert spec.process is mock_popen.return_value

    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_passes_all_models_in_group(self, mock_popen):
        spec = WorkerSpec(
            models=["model-a", "model-b"],
            python_path="/venvs/shared/bin/python",
            port=9001,
        )
        from muse.cli_impl.supervisor import spawn_worker
        spawn_worker(spec, device="cuda")
        args = mock_popen.call_args.args[0]
        # Each model passed via separate --model
        model_values = [args[i+1] for i, v in enumerate(args) if v == "--model"]
        assert set(model_values) == {"model-a", "model-b"}


class TestWaitForReady:
    def test_returns_when_health_responds_200(self):
        from muse.cli_impl.supervisor import wait_for_ready

        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            # Should return cleanly
            wait_for_ready(port=9001, timeout=5.0, poll_interval=0.01)

    def test_raises_timeouterror_when_worker_never_responds(self):
        from muse.cli_impl.supervisor import wait_for_ready
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            import httpx
            mock_get.side_effect = httpx.ConnectError("nope", request=None)
            with pytest.raises(TimeoutError, match="did not become ready"):
                wait_for_ready(port=9001, timeout=0.1, poll_interval=0.01)

    def test_polls_multiple_times_before_success(self):
        from muse.cli_impl.supervisor import wait_for_ready
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            # First two calls fail, third succeeds
            mock_get.side_effect = [
                httpx.ConnectError("not yet", request=None),
                httpx.ConnectError("not yet", request=None),
                MagicMock(status_code=200),
            ]
            wait_for_ready(port=9001, timeout=5.0, poll_interval=0.001)
            assert mock_get.call_count == 3


class TestRunSupervisor:
    def test_supervisor_spawns_all_workers_and_waits_for_first(self, tmp_catalog):
        """All workers spawn, but only the FIRST is waited on synchronously.

        Remaining workers promote on the boot thread (mocked out here so
        the test doesn't actually poll httpx).
        """
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor._promote_workers") as mock_promote, \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            # _wait_for_first_ready returns the first spec passed in.
            mock_first.side_effect = lambda specs, **kw: specs[0]
            # Simulate graceful shutdown by raising KeyboardInterrupt from uvicorn.run
            mock_uvicorn.run.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # Two workers planned: both spawned, but only ONE _wait call.
            assert mock_spawn.call_count == 2
            mock_first.assert_called_once()
            mock_uvicorn.run.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_supervisor_tears_down_workers_if_gateway_fails(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            mock_first.side_effect = lambda specs, **kw: specs[0]
            mock_uvicorn.run.side_effect = RuntimeError("uvicorn died")

            with pytest.raises(RuntimeError, match="uvicorn died"):
                run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            mock_shutdown.assert_called_once()


class TestWorkerSpecExtensions:
    def test_worker_spec_has_device_field_with_default(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.device == "auto"

    def test_worker_spec_accepts_explicit_device(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cuda")
        assert spec.device == "cuda"

    def test_worker_spec_default_status_is_pending(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.status == "pending"

    def test_worker_spec_default_restart_and_failure_counts(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.restart_count == 0
        assert spec.failure_count == 0

    def test_worker_spec_has_last_spawn_at_default(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.last_spawn_at == 0.0


class TestAttemptRestart:
    def test_respawns_after_process_death(self):
        """If process exited, terminate (no-op if dead) + respawn."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
        )
        spec.process = MagicMock(poll=MagicMock(return_value=1))  # already exited
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_called_once_with(spec, device="cpu")
        mock_wait.assert_called_once()
        assert spec.restart_count == 1
        assert spec.failure_count == 0
        assert spec.status == "running"

    def test_terminates_still_running_process_before_respawn(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        old_process = MagicMock(poll=MagicMock(return_value=None))  # still alive
        spec.process = old_process
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        old_process.terminate.assert_called_once()

    def test_marks_dead_after_max_restarts(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
            restart_count=10,  # already at budget
        )
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_not_called()
        assert spec.status == "dead"

    def test_spawn_failure_keeps_status_unhealthy(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            mock_wait.side_effect = TimeoutError("never ready")
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        assert spec.restart_count == 1  # counter still increments
        assert spec.status != "running"  # spawn tried, but didn't become ready

    def test_respects_stop_event_during_backoff(self):
        """If stop_event is set during backoff wait, skip the restart."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()
        stop_event.set()  # shutdown already requested

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=1)

        # With stop_event set, we don't respawn
        mock_spawn.assert_not_called()


class TestMonitorLoop:
    def test_monitor_calls_restart_after_threshold_failures(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))  # alive
        stop_event = threading.Event()

        # First 3 checks fail, then we stop
        health_calls = {"count": 0}
        def health_side_effect(**kwargs):
            health_calls["count"] += 1
            if health_calls["count"] >= 4:
                stop_event.set()
            return False

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect), \
             patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # After 3 consecutive unhealthy polls, restart should be invoked at least once
        assert mock_restart.called

    def test_monitor_resets_failure_count_on_success(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))
        spec.failure_count = 2  # close to threshold
        stop_event = threading.Event()

        call_count = {"n": 0}
        def health_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                stop_event.set()
            return True  # healthy

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect), \
             patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        assert spec.failure_count == 0
        assert spec.status == "running"
        mock_restart.assert_not_called()

    def test_monitor_stops_when_event_set(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))
        stop_event = threading.Event()
        stop_event.set()  # already stopped

        with patch("muse.cli_impl.supervisor.check_worker_health", return_value=True) as mock_health:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # Loop should exit immediately without any health checks
        mock_health.assert_not_called()

    def test_monitor_skips_dead_workers(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        alive_spec = WorkerSpec(models=["a"], python_path="/p", port=9001, device="cpu")
        alive_spec.process = MagicMock(poll=MagicMock(return_value=None))
        dead_spec = WorkerSpec(models=["b"], python_path="/p", port=9002, device="cpu")
        dead_spec.status = "dead"

        stop_event = threading.Event()
        checked_ports = []

        def health_side_effect(**kwargs):
            checked_ports.append(kwargs["port"])
            if len(checked_ports) >= 1:
                stop_event.set()
            return True

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect):
            _monitor_workers(
                [alive_spec, dead_spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # Only the alive worker should have been polled
        assert 9001 in checked_ports
        assert 9002 not in checked_ports


class TestCheckWorkerHealth:
    def test_returns_true_on_200(self):
        from muse.cli_impl.supervisor import check_worker_health
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert check_worker_health(port=9001) is True

    def test_returns_false_on_non_200(self):
        from muse.cli_impl.supervisor import check_worker_health
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            assert check_worker_health(port=9001) is False

    def test_returns_false_on_connection_error(self):
        from muse.cli_impl.supervisor import check_worker_health
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("down", request=None)
            assert check_worker_health(port=9001) is False

    def test_returns_false_on_timeout(self):
        from muse.cli_impl.supervisor import check_worker_health
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("slow", request=None)
            assert check_worker_health(port=9001) is False


class TestRunSupervisorMonitor:
    def test_run_supervisor_starts_monitor_thread(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread") as mock_thread_cls:
            mock_first.side_effect = lambda specs, **kw: specs[0]
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # At least one daemon thread (the monitor) was created and started.
            # When there are remaining workers, a second daemon thread (the
            # boot promoter) is also created; with one model only the monitor.
            assert mock_thread_cls.call_count >= 1
            for call in mock_thread_cls.call_args_list:
                assert call.kwargs.get("daemon") is True
                assert call.kwargs.get("target") is not None
            mock_thread.start.assert_called()

    def test_run_supervisor_sets_stop_event_on_exit(self, tmp_catalog):
        """On shutdown path, the monitor must be told to stop."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor
        import threading

        captured_events = []
        real_event_cls = threading.Event
        def capture_event(*a, **kw):
            e = real_event_cls(*a, **kw)
            captured_events.append(e)
            return e

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Event", side_effect=capture_event), \
             patch("muse.cli_impl.supervisor.threading.Thread"):
            mock_first.side_effect = lambda specs, **kw: specs[0]
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # The shutdown Event was set
        assert captured_events, "no threading.Event was created"
        assert any(e.is_set() for e in captured_events)


class TestSupervisorState:
    def test_default_state_has_empty_workers(self):
        s = SupervisorState()
        assert s.workers == []
        assert s.device == "auto"
        assert s.started_at >= 0.0

    def test_default_state_has_rlock(self):
        import threading
        s = SupervisorState()
        # An RLock can be acquired twice from the same thread without deadlock
        with s.lock:
            with s.lock:
                pass

    def test_get_supervisor_state_returns_sentinel_when_unset(self):
        clear_supervisor_state()
        s = get_supervisor_state()
        assert isinstance(s, SupervisorState)
        assert s.workers == []

    def test_set_and_get_singleton_round_trip(self):
        s = SupervisorState(device="cuda")
        set_supervisor_state(s)
        assert get_supervisor_state() is s

    def test_clear_supervisor_state_drops_singleton(self):
        s = SupervisorState(device="mps")
        set_supervisor_state(s)
        clear_supervisor_state()
        # Subsequent get yields a fresh sentinel, not the cleared one
        out = get_supervisor_state()
        assert out is not s


class TestRunSupervisorRegistersState:
    def test_run_supervisor_registers_state_during_run(self, tmp_catalog):
        """run_supervisor should set the singleton before uvicorn.run."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_state = {"value": None}

        def capture_state(*args, **kwargs):
            seen_state["value"] = get_supervisor_state()
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_first.side_effect = lambda specs, **kw: specs[0]
            mock_uvicorn.run.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # State was non-None during the run
        assert seen_state["value"] is not None
        assert seen_state["value"].device == "cpu"
        assert len(seen_state["value"].workers) == 1
        # And it's been cleared on exit
        cleared = get_supervisor_state()
        assert cleared.workers == []

    def test_run_supervisor_clears_state_on_exception(self, tmp_catalog):
        """Crash path: state must still get cleared in the finally block."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_first.side_effect = lambda specs, **kw: specs[0]
            mock_uvicorn.run.side_effect = RuntimeError("uvicorn boom")
            with pytest.raises(RuntimeError):
                run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        cleared = get_supervisor_state()
        assert cleared.workers == []


class TestWaitForFirstReady:
    def test_returns_first_ready_spec(self):
        """Round-robin polling: a fast spec buried behind a slow one wins."""
        from muse.cli_impl.supervisor import _wait_for_first_ready
        import httpx

        slow = WorkerSpec(models=["slow"], python_path="/p", port=9001)
        fast = WorkerSpec(models=["fast"], python_path="/p", port=9002)

        def side_effect(url, **kw):
            if "9002" in url:
                return MagicMock(status_code=200)
            raise httpx.ConnectError("not yet", request=None)

        with patch("muse.cli_impl.supervisor.httpx.get", side_effect=side_effect):
            result = _wait_for_first_ready([slow, fast], timeout=2.0,
                                           poll_interval=0.01)
        assert result is fast

    def test_returns_immediately_on_first_ready(self):
        """When the first spec is the one that responds, no extra polls."""
        from muse.cli_impl.supervisor import _wait_for_first_ready

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = _wait_for_first_ready([spec], timeout=2.0,
                                           poll_interval=0.01)
        assert result is spec

    def test_raises_when_no_worker_ready(self):
        from muse.cli_impl.supervisor import _wait_for_first_ready
        import httpx

        a = WorkerSpec(models=["a"], python_path="/p", port=9001)
        b = WorkerSpec(models=["b"], python_path="/p", port=9002)
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("nope", request=None)
            with pytest.raises(TimeoutError, match="no worker became ready"):
                _wait_for_first_ready([a, b], timeout=0.05,
                                      poll_interval=0.01)

    def test_raises_on_empty_specs(self):
        from muse.cli_impl.supervisor import _wait_for_first_ready
        with pytest.raises(ValueError, match="no specs"):
            _wait_for_first_ready([], timeout=1.0)


class TestPromoteWorkers:
    def test_promotes_each_to_running(self):
        from muse.cli_impl.supervisor import _promote_workers

        a = WorkerSpec(models=["a"], python_path="/p", port=9001,
                       status="pending")
        b = WorkerSpec(models=["b"], python_path="/p", port=9002,
                       status="pending")
        state = SupervisorState(workers=[a, b])

        with patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            _promote_workers([a, b], state, timeout=1.0)

        assert a.status == "running"
        assert b.status == "running"
        assert mock_wait.call_count == 2

    def test_marks_unhealthy_on_timeout(self):
        from muse.cli_impl.supervisor import _promote_workers

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001,
                          status="pending")
        state = SupervisorState(workers=[spec])
        with patch("muse.cli_impl.supervisor.wait_for_ready",
                   side_effect=TimeoutError("never ready")):
            _promote_workers([spec], state, timeout=0.1)
        assert spec.status == "unhealthy"

    def test_continues_past_failed_promotions(self):
        """One spec failing should not block promotion of the others."""
        from muse.cli_impl.supervisor import _promote_workers

        a = WorkerSpec(models=["a"], python_path="/p", port=9001,
                       status="pending")
        b = WorkerSpec(models=["b"], python_path="/p", port=9002,
                       status="pending")
        state = SupervisorState(workers=[a, b])

        # First call (for a) raises; second (for b) returns
        call_log: list[int] = []

        def side_effect(*args, **kw):
            port = kw.get("port") or args[0]
            call_log.append(port)
            if port == 9001:
                raise TimeoutError("a never ready")

        with patch("muse.cli_impl.supervisor.wait_for_ready",
                   side_effect=side_effect):
            _promote_workers([a, b], state, timeout=0.1)

        assert a.status == "unhealthy"
        assert b.status == "running"
        assert 9001 in call_log and 9002 in call_log


class TestRunSupervisorFirstReadyOrdering:
    def test_gateway_starts_after_first_ready_and_before_remaining(self, tmp_catalog):
        """The gateway must boot once the first spec is healthy; remaining
        specs promote in the background after that."""
        _seed_catalog({
            "fast": {
                "pulled_at": "...", "hf_repo": "f", "local_dir": "/f",
                "venv_path": "/venvs/fast",
                "python_path": "/venvs/fast/bin/python",
                "enabled": True,
            },
            "slow": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venvs/slow",
                "python_path": "/venvs/slow/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        events: list[str] = []

        def first_ready_side(specs, **kw):
            events.append("first_ready_returned")
            return specs[0]

        def gateway_side(*a, **kw):
            events.append("gateway_started")
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready",
                   side_effect=first_ready_side), \
             patch("muse.cli_impl.supervisor._promote_workers"), \
             patch("muse.cli_impl.supervisor._monitor_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = gateway_side

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        assert "first_ready_returned" in events
        assert "gateway_started" in events
        # First-ready must precede gateway-start
        assert events.index("first_ready_returned") < events.index("gateway_started")

    def test_first_ready_failure_propagates(self, tmp_catalog):
        """If no worker comes up, run_supervisor raises and gateway is never started."""
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": "/venvs/x",
                "python_path": "/venvs/x/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready",
                   side_effect=TimeoutError("nobody ready")), \
             patch("muse.cli_impl.supervisor._monitor_workers"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            with pytest.raises(TimeoutError, match="nobody ready"):
                run_supervisor(host="0.0.0.0", port=8000, device="cpu")
            mock_uvicorn.run.assert_not_called()

    def test_late_workers_promote_in_background_thread(self, tmp_catalog):
        """The remaining-specs list is handed to _promote_workers."""
        _seed_catalog({
            "fast": {
                "pulled_at": "...", "hf_repo": "f", "local_dir": "/f",
                "venv_path": "/venvs/fast",
                "python_path": "/venvs/fast/bin/python",
                "enabled": True,
            },
            "slow1": {
                "pulled_at": "...", "hf_repo": "s1", "local_dir": "/s1",
                "venv_path": "/venvs/slow1",
                "python_path": "/venvs/slow1/bin/python",
                "enabled": True,
            },
            "slow2": {
                "pulled_at": "...", "hf_repo": "s2", "local_dir": "/s2",
                "venv_path": "/venvs/slow2",
                "python_path": "/venvs/slow2/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_thread_targets: list = []

        def fake_thread_init(*args, **kw):
            target = kw.get("target")
            if target is not None:
                seen_thread_targets.append(target.__name__)
            t = MagicMock()
            t.start = MagicMock()
            return t

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor.threading.Thread",
                   side_effect=fake_thread_init), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_first.side_effect = lambda specs, **kw: specs[0]
            mock_uvicorn.run.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # _promote_workers AND _monitor_workers were both wired as thread
        # targets when remaining specs exist.
        assert "_promote_workers" in seen_thread_targets
        assert "_monitor_workers" in seen_thread_targets


class TestGatewayStateRoutes:
    def test_routes_only_running_workers(self):
        """Gateway derives routes from state.workers, filters out non-running."""
        from muse.cli_impl.gateway import build_gateway

        running = WorkerSpec(models=["a"], python_path="/p", port=9001,
                             status="running")
        pending = WorkerSpec(models=["b"], python_path="/p", port=9002,
                             status="pending")
        state = SupervisorState(workers=[running, pending])

        app = build_gateway(state=state)
        routes = app.state.routes_now()
        assert "a" in routes
        assert "b" not in routes

    def test_routes_update_when_pending_promotes(self):
        """A pending spec that becomes running shows up in routes_now."""
        from muse.cli_impl.gateway import build_gateway

        slow = WorkerSpec(models=["slow"], python_path="/p", port=9001,
                          status="pending")
        state = SupervisorState(workers=[slow])

        app = build_gateway(state=state)
        assert "slow" not in app.state.routes_now()

        with state.lock:
            slow.status = "running"
        assert "slow" in app.state.routes_now()

    def test_static_routes_still_supported(self):
        """When state= isn't passed, build_gateway uses the legacy routes list."""
        from muse.cli_impl.gateway import build_gateway, WorkerRoute

        routes = [WorkerRoute(model_id="m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes=routes)
        cur = app.state.routes_now()
        assert "m" in cur
        assert cur["m"].worker_url == "http://127.0.0.1:9001"
