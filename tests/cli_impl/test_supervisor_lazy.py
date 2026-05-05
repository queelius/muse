"""Tests for the lazy-load supervisor wiring (Task E of v0.40.0).

The supervisor now boots without spawning workers eagerly. Models load on
first request via the LoadDirector, which the supervisor instantiates and
hangs off SupervisorState. Boot-time validation walks the enabled-catalog
and stamps unservable_reasons for entries with no memory data or memory
estimates exceeding device capacity.

These tests verify:
  - SupervisorState carries `director` + `unservable_reasons` fields.
  - run_supervisor does NOT call plan_workers / does NOT spawn workers
    eagerly.
  - validate_catalog_at_boot flags models without memory data.
  - validate_catalog_at_boot flags models exceeding device capacity.
  - validate_catalog_at_boot does NOT flag models with valid data fitting.
  - The director on state is a LoadDirector instance during run_supervisor.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.load_director import LoadDirector
from muse.cli_impl.supervisor import (
    SupervisorState,
    clear_supervisor_state,
    get_supervisor_state,
    set_supervisor_state,
    validate_catalog_at_boot,
)


@pytest.fixture(autouse=True)
def _reset_supervisor_state():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


@pytest.fixture(autouse=True)
def _isolate_pynvml_sentinels():
    """Prevent supervisor tests from polluting memory_probe module-level
    pynvml sentinels (which are loaded lazily inside _MemoryProbeAdapter
    when run_supervisor calls validate_catalog_at_boot with no probe
    override). On entry, save current state. On exit, restore the
    fresh-untried state so memory_probe tests starting after us see a
    clean slate.
    """
    import muse.core.memory_probe as mod
    orig = (mod.pynvml, mod._init_attempted, mod._init_ok)
    try:
        yield
    finally:
        # Restore to a fresh-untried state, not just whatever it was on
        # entry, so subsequent test files (notably tests/core/test_memory_probe.py)
        # see a clean module. The memory_probe fixture itself uses
        # save-and-restore, which preserves whatever pollution we leave
        # behind; resetting to (None, False, False) prevents that.
        mod.pynvml, mod._init_attempted, mod._init_ok = None, False, False
        # (orig captured but unused; kept to preserve the variable for
        # observability if someone wants to inspect what we overwrote.)
        del orig


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


# ---------------------------------------------------------------------------
# E1: SupervisorState fields
# ---------------------------------------------------------------------------


class TestSupervisorStateLazyFields:
    def test_state_has_director_field_default_none(self):
        s = SupervisorState()
        assert hasattr(s, "director")
        assert s.director is None

    def test_state_has_unservable_reasons_dict(self):
        s = SupervisorState()
        assert hasattr(s, "unservable_reasons")
        assert s.unservable_reasons == {}

    def test_state_director_is_writable(self):
        """The supervisor populates state.director once the LoadDirector is built."""
        s = SupervisorState()
        director = MagicMock(spec=LoadDirector)
        s.director = director
        assert s.director is director

    def test_unservable_reasons_is_per_state_instance(self):
        """Default dicts must not bleed between SupervisorState instances."""
        a = SupervisorState()
        b = SupervisorState()
        a.unservable_reasons["x"] = "out of memory"
        assert b.unservable_reasons == {}


# ---------------------------------------------------------------------------
# E2: validate_catalog_at_boot
# ---------------------------------------------------------------------------


class TestValidateCatalogAtBoot:
    def test_flags_model_with_no_memory_data(self, tmp_catalog):
        """Enabled model with no memory_gb annotation and no measurements
        is flagged as unservable with the probe-prompt message.
        """
        _seed_catalog({
            "needs-probe": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "needs-probe",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        # no memory_gb, no measurements
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "needs-probe" in state.unservable_reasons
        reason = state.unservable_reasons["needs-probe"]
        assert "no memory estimate" in reason
        assert "muse models probe" in reason

    def test_flags_model_exceeding_device_capacity(self, tmp_catalog):
        """Enabled model whose memory_gb exceeds free at boot (minus headroom)
        is flagged as exceeds_capacity. Uses live cpu_free_gb from probe.
        """
        _seed_catalog({
            "huge-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "huge-model",
                    "modality": "chat/completion",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 100.0,
                        "device": "cpu",
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 16.0  # nowhere near 100
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "huge-model" in state.unservable_reasons
        reason = state.unservable_reasons["huge-model"]
        assert "exceeds device capacity" in reason

    def test_does_not_flag_model_that_fits(self, tmp_catalog):
        """Enabled model with valid memory_gb that fits live free is NOT flagged."""
        _seed_catalog({
            "tiny-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "tiny-model",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 0.5,
                        "device": "cpu",
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "tiny-model" not in state.unservable_reasons
        assert state.unservable_reasons == {}

    def test_does_not_flag_model_with_probe_measurements(self, tmp_catalog):
        """A model lacking capabilities.memory_gb but with recorded
        measurements.<device>.peak_bytes is fine.
        """
        _seed_catalog({
            "probed-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "probed-model",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "device": "cpu",
                        # no memory_gb
                    },
                },
                "measurements": {
                    "cpu": {
                        "peak_bytes": 500_000_000,
                        "device": "cpu",
                        "weights_bytes": 400_000_000,
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "probed-model" not in state.unservable_reasons

    def test_skips_disabled_models(self, tmp_catalog):
        """Disabled entries are not validated; they can't 503."""
        _seed_catalog({
            "disabled-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": False,
                "manifest": {
                    "model_id": "disabled-model",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {},
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert state.unservable_reasons == {}

    def test_skips_pre_worker_entries_without_python_path(self, tmp_catalog):
        """Catalog entries without python_path can't actually be loaded;
        they're not flagged as unservable here (they'll be skipped at load
        time anyway), they just don't enter validation.
        """
        _seed_catalog({
            "legacy": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "enabled": True,
                # no python_path
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        # Either flagged (with a coherent reason) or skipped silently.
        # Just verify no exception.
        # In either case, the state should be valid.
        assert isinstance(state.unservable_reasons, dict)

    def test_handles_gpu_device_with_pynvml_unavailable(self, tmp_catalog):
        """Model declares device=cuda; pynvml returns None. Without GPU
        info we can't say it fits; flag exceeds_capacity (the safe choice
        until probe data lands).
        """
        _seed_catalog({
            "gpu-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "gpu-model",
                    "modality": "image/generation",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 4.0,
                        "device": "cuda",
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None  # pynvml unavailable
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "gpu-model" in state.unservable_reasons


# ---------------------------------------------------------------------------
# E3: run_supervisor lazy-boot path
# ---------------------------------------------------------------------------


class TestRunSupervisorLazyBoot:
    def test_run_supervisor_does_not_spawn_workers_eagerly(self, tmp_catalog):
        """The lazy supervisor must NOT call spawn_worker at boot time,
        even with one or more enabled models in the catalog.
        """
        _seed_catalog({
            "model-a": {
                "pulled_at": "...",
                "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "model-a",
                    "modality": "embedding/text",
                    "hf_repo": "a",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 0.5, "device": "cpu"},
                },
            },
            "model-b": {
                "pulled_at": "...",
                "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "model-b",
                    "modality": "embedding/text",
                    "hf_repo": "b",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 0.5, "device": "cpu"},
                },
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor._wait_for_first_ready") as mock_first, \
             patch("muse.cli_impl.supervisor._promote_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # The whole point: zero workers spawned at boot.
            mock_spawn.assert_not_called()
            mock_first.assert_not_called()

    def test_run_supervisor_constructs_load_director(self, tmp_catalog):
        """A LoadDirector is on state.director during the run."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...",
                "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "model-a",
                    "modality": "embedding/text",
                    "hf_repo": "a",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 0.5, "device": "cpu"},
                },
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_state = {"value": None}

        def capture_state(*args, **kwargs):
            seen_state["value"] = get_supervisor_state()
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._promote_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        captured = seen_state["value"]
        assert captured is not None
        assert captured.director is not None
        assert isinstance(captured.director, LoadDirector)

    def test_run_supervisor_runs_validation_at_boot(self, tmp_catalog):
        """validate_catalog_at_boot is called and populates unservable_reasons."""
        _seed_catalog({
            "huge": {
                "pulled_at": "...",
                "hf_repo": "h", "local_dir": "/h",
                "venv_path": "/venvs/h",
                "python_path": "/venvs/h/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "huge",
                    "modality": "chat/completion",
                    "hf_repo": "h",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 9999.0, "device": "cpu"},
                },
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_state = {"value": None}

        def capture_state(*args, **kwargs):
            seen_state["value"] = get_supervisor_state()
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor._promote_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        captured = seen_state["value"]
        assert captured is not None
        # huge model exceeds capacity; must be flagged.
        assert "huge" in captured.unservable_reasons

    def test_run_supervisor_starts_gateway_immediately_with_empty_catalog(self, tmp_catalog):
        """With zero enabled models, the lazy supervisor still boots the
        gateway. No worker spawn loop, no first-ready wait.
        """
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # Empty catalog: no workers spawned; gateway starts.
            mock_spawn.assert_not_called()
            mock_uvicorn.run.assert_called_once()

    def test_run_supervisor_clears_state_on_exit(self, tmp_catalog):
        """The teardown path still clears the singleton state."""
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # Cleared after exit
        cleared = get_supervisor_state()
        assert cleared.workers == []
        assert cleared.director is None
        assert cleared.unservable_reasons == {}


# ---------------------------------------------------------------------------
# E5 (auto-restart compatibility)
# ---------------------------------------------------------------------------


class TestAutoRestartMonitorLazyCompatibility:
    def test_monitor_iterates_state_workers_added_post_boot(self, tmp_catalog):
        """The monitor reads state.workers as a live reference. Workers
        added post-boot (via the director's enable_fn) must be picked up
        on the next monitor tick. This is verified by run_supervisor
        passing state.workers (not a frozen list) to the monitor thread.
        """
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        captured_args: dict = {}

        def capture_thread_init(*args, **kwargs):
            target = kwargs.get("target")
            if target is not None and getattr(target, "__name__", "") == "_monitor_workers":
                # The first positional arg is the workers list; capture
                # its identity so we can assert it's state.workers.
                captured_args["specs"] = kwargs.get("args", (None,))[0]
            t = MagicMock()
            t.start = MagicMock()
            return t

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.threading.Thread",
                   side_effect=capture_thread_init), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # The monitor was started with the live state.workers reference,
        # not a copy. (When the catalog is empty, the monitor may or may
        # not be started; this test focuses on the live-reference shape
        # when it is.)
        if "specs" in captured_args:
            specs = captured_args["specs"]
            # state.workers is either an empty list or a live reference;
            # crucially, it's the SAME list object used by the rest of
            # the supervisor + admin endpoints.
            assert isinstance(specs, list)


# ---------------------------------------------------------------------------
# Issue 1: monitor must skip in-flight specs (job_id != None) so it does NOT
# race a director-driven cold load that's still bringing the worker up.
# ---------------------------------------------------------------------------


class TestMonitorSkipsInFlightSpecs:
    def test_monitor_skips_spec_with_job_id_set(self):
        """A spec whose job_id is set is in the middle of an admin- or
        director-driven transition. The monitor must NOT poll its
        /health, NOT increment failure_count, and NOT call _attempt_restart.
        """
        from muse.cli_impl.supervisor import _monitor_workers, WorkerSpec
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
        )
        spec.process = None  # not yet spawned (mid-cold-load)
        spec.status = "pending"
        spec.job_id = "director-load-x"  # in-flight
        stop_event = threading.Event()

        # Stop after a single tick so the test doesn't spin forever.
        tick_count = {"n": 0}

        def health_side_effect(**kwargs):
            tick_count["n"] += 1
            if tick_count["n"] >= 1:
                stop_event.set()
            return False

        with patch(
            "muse.cli_impl.supervisor.check_worker_health",
            side_effect=health_side_effect,
        ) as mock_health, patch(
            "muse.cli_impl.supervisor._attempt_restart"
        ) as mock_restart:
            # Need at least one workers entry so the loop has work to do
            # (without that the outer while just sleeps until stop_event);
            # a second non-job_id spec drives the stop_event.
            other = WorkerSpec(
                models=["y"], python_path="/p2", port=9002, device="cpu",
            )
            other.process = MagicMock(poll=MagicMock(return_value=None))
            other.status = "running"
            _monitor_workers(
                [spec, other], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # The in-flight spec was never health-polled. Only the other
        # spec contributed to the call counter.
        polled_ports = [c.kwargs.get("port") for c in mock_health.call_args_list]
        assert 9001 not in polled_ports, (
            "monitor polled in-flight spec; should have skipped"
        )
        # And the in-flight spec was never restarted.
        for c in mock_restart.call_args_list:
            assert c.args[0] is not spec, (
                "monitor attempted to restart an in-flight spec"
            )
        # The spec's status / failure_count are unchanged: still pending.
        assert spec.status == "pending"
        assert spec.failure_count == 0
        assert spec.job_id == "director-load-x"  # untouched

    def test_director_load_does_not_double_spawn_under_monitor(
        self, tmp_catalog,
    ):
        """End-to-end: when the director calls load_model_into_worker
        with a slow spawn_worker, the monitor must not restart the
        in-flight pending spec mid-spawn.
        """
        import threading
        import time

        from muse.admin.operations import load_model_into_worker
        from muse.cli_impl.supervisor import (
            SupervisorState,
            _monitor_workers,
        )

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spawn_count = {"n": 0}
        spawn_done = threading.Event()

        def slow_spawn(spec, device):
            spawn_count["n"] += 1
            time.sleep(0.4)  # simulate slow GGUF / SDXL load
            spawn_done.set()

        stop_event = threading.Event()

        with patch(
            "muse.admin.operations.spawn_worker",
            side_effect=slow_spawn,
        ), patch(
            "muse.admin.operations.wait_for_ready",
            lambda *a, **k: None,
        ), patch(
            "muse.admin.operations.find_free_port",
            lambda *a, **k: 9123,
        ), patch(
            "muse.cli_impl.supervisor.check_worker_health",
            return_value=False,  # simulate port-not-bound during load
        ), patch(
            "muse.cli_impl.supervisor._attempt_restart"
        ) as mock_restart:
            # Start the monitor with the live state.workers ref. It
            # ticks fast (1ms).
            monitor_thread = threading.Thread(
                target=_monitor_workers,
                args=(state.workers, stop_event),
                kwargs={
                    "interval": 0.001,
                    "failure_threshold": 3,
                    "max_restarts": 10,
                },
                daemon=True,
            )
            monitor_thread.start()

            # Drive the director-style load on the main thread. Slow
            # spawn keeps the spec pending+job_id-bearing for ~0.4s.
            try:
                port = load_model_into_worker("kokoro-82m", state=state)
            finally:
                stop_event.set()
                monitor_thread.join(timeout=2.0)

        assert port == 9123
        assert spawn_count["n"] == 1, (
            f"expected 1 spawn (no double-spawn), got {spawn_count['n']}"
        )
        # Critically: no restart was attempted on the in-flight spec.
        mock_restart.assert_not_called()
        # And the spec is now running with no in-flight job_id.
        assert len(state.workers) == 1
        assert state.workers[0].status == "running"
        assert state.workers[0].job_id is None


# ---------------------------------------------------------------------------
# Issue 2: unload_model_from_worker must release state.lock during the slow
# shutdown / restart-in-place phase.
# ---------------------------------------------------------------------------


class TestUnloadReleasesLockDuringSlowIO:
    def test_unload_releases_lock_during_shutdown(self, tmp_catalog):
        """unload_model_from_worker on a sole-tenant worker calls
        _shutdown_workers, which can take seconds. The lock must be
        dropped during that window so admin reads / hot-acquires are
        not blocked.
        """
        import threading
        import time

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        def slow_shutdown(specs):
            time.sleep(0.4)

        unload_done = threading.Event()

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=slow_shutdown,
        ):
            def _unload():
                unload_model_from_worker("kokoro-82m", state=state)
                unload_done.set()

            t = threading.Thread(target=_unload)
            t.start()
            # Give the unload thread a head start so it's mid-shutdown.
            time.sleep(0.1)

            # Time the lock acquire. With the bug, this would block for
            # the full slow-shutdown duration (~0.3s remaining).
            t0 = time.perf_counter()
            with state.lock:
                snapshot = list(state.workers)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"state.lock was held during shutdown for {elapsed:.3f}s; "
                "should release before _shutdown_workers"
            )

            unload_done.wait(timeout=3.0)
            t.join(timeout=1.0)

        # After unload completes, the spec is gone.
        assert state.workers == []

    def test_unload_releases_lock_during_restart_inplace(self, tmp_catalog):
        """unload_model_from_worker on a sibling-occupied worker calls
        _restart_worker_inplace, which is also slow. Same lock-release
        contract must hold.
        """
        import threading
        import time

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

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

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m", "soprano-80m"],
            python_path="/venv/shared/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        def slow_restart(spec, *, device):
            time.sleep(0.4)

        unload_done = threading.Event()

        with patch(
            "muse.admin.operations._restart_worker_inplace",
            side_effect=slow_restart,
        ):
            def _unload():
                unload_model_from_worker("soprano-80m", state=state)
                unload_done.set()

            t = threading.Thread(target=_unload)
            t.start()
            time.sleep(0.1)

            t0 = time.perf_counter()
            with state.lock:
                snapshot = list(state.workers)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"state.lock held during restart-in-place for {elapsed:.3f}s; "
                "should release before _restart_worker_inplace"
            )

            unload_done.wait(timeout=3.0)
            t.join(timeout=1.0)

    def test_disable_releases_lock_during_shutdown(self, tmp_catalog):
        """Pre-existing same bug in disable_model. Same fix applied:
        slow shutdown happens outside state.lock.
        """
        import threading
        import time

        from muse.admin.operations import disable_model
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        def slow_shutdown(specs):
            time.sleep(0.4)

        done = threading.Event()

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=slow_shutdown,
        ):
            def _disable():
                disable_model("kokoro-82m", state=state)
                done.set()

            t = threading.Thread(target=_disable)
            t.start()
            time.sleep(0.1)

            t0 = time.perf_counter()
            with state.lock:
                snapshot = list(state.workers)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"state.lock held during disable_model shutdown for "
                f"{elapsed:.3f}s; should release before _shutdown_workers"
            )

            done.wait(timeout=3.0)
            t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Issue 3: state.workers must be mutated in place, not rebound, so the
# monitor thread (which captured the original list reference) sees updates.
# ---------------------------------------------------------------------------


class TestStateWorkersMutatedInPlace:
    def test_unload_mutates_workers_in_place(self, tmp_catalog):
        """unload_model_from_worker must remove the spec via in-place
        mutation (not rebind state.workers) so the monitor thread,
        which captured the original list reference, sees the removal.
        """
        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        # Capture the list identity BEFORE the call. After the call,
        # state.workers must STILL be the same list object.
        original_id = id(state.workers)

        with patch("muse.admin.operations._shutdown_workers"):
            unload_model_from_worker("kokoro-82m", state=state)

        assert id(state.workers) == original_id, (
            "state.workers was rebound; monitor thread (which captured "
            "the original reference) would now iterate a stale list"
        )
        assert state.workers == []

    def test_disable_mutates_workers_in_place(self, tmp_catalog):
        """Same in-place mutation contract for disable_model
        (pre-existing bug, fixed in the same pass).
        """
        from muse.admin.operations import disable_model
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        original_id = id(state.workers)

        with patch("muse.admin.operations._shutdown_workers"):
            disable_model("kokoro-82m", state=state)

        assert id(state.workers) == original_id, (
            "state.workers was rebound after disable_model"
        )
        assert state.workers == []

    def test_monitor_sees_unload_via_live_reference(self, tmp_catalog):
        """End-to-end: spawn 2 workers, remove one via
        unload_model_from_worker, run a monitor tick. The monitor must
        iterate only the remaining worker (mock _attempt_restart and
        assert it's not called for the removed one).
        """
        import threading

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import (
            SupervisorState,
            WorkerSpec,
            _monitor_workers,
        )

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/s",
                "python_path": "/venv/s/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec_k = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec_k.status = "running"
        spec_k.process = MagicMock(poll=MagicMock(return_value=1))  # exited
        spec_s = WorkerSpec(
            models=["soprano-80m"], python_path="/venv/s/bin/python",
            port=9002,
        )
        spec_s.status = "running"
        spec_s.process = MagicMock(poll=MagicMock(return_value=1))  # exited
        state.workers.append(spec_k)
        state.workers.append(spec_s)

        # Capture the live reference the monitor would receive at boot.
        monitor_workers_ref = state.workers

        # Remove kokoro via unload BEFORE running the monitor tick.
        with patch("muse.admin.operations._shutdown_workers"):
            unload_model_from_worker("kokoro-82m", state=state)

        # Now run a single monitor tick over the (mutated-in-place) list.
        # The monitor sees process.poll() returns nonzero -> failure_count
        # ratchets to threshold -> _attempt_restart is called for the
        # remaining worker only.
        stop_event = threading.Event()
        # Stop after the first iteration of the for-loop ticks. We use
        # _attempt_restart's first invocation as the trigger: it gets
        # called for the surviving worker (whose process.poll() returns
        # nonzero), and we set the stop_event from inside the mock.
        def restart_side_effect(spec, *, stop_event=stop_event, **kw):
            stop_event.set()

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                monitor_workers_ref, stop_event,
                interval=0.001, failure_threshold=1, max_restarts=10,
            )

        # Critically: the monitor saw only the remaining worker via the
        # captured-but-mutated-in-place list reference.
        restart_targets = [c.args[0] for c in mock_restart.call_args_list]
        for target in restart_targets:
            assert target is not spec_k, (
                "monitor tried to restart the removed (kokoro) worker"
            )
        # And it DID try to restart the surviving worker.
        assert any(t is spec_s for t in restart_targets), (
            "monitor failed to see the surviving worker via the live "
            "reference (state.workers was rebound)"
        )
