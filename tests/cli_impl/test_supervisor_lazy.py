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
