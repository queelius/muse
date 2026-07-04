"""Tests for Task 11: supervisor telemetry lifecycle + per-worker log piping.

Covers the testable units factored out of run_supervisor/spawn_worker:
  - `_pump_worker_logs`: the daemon reader loop that pipes a worker's
    stdout lines into a LogHub (and re-emits them to the aggregate log).
  - `_init_telemetry`: the boot-time wiring that creates a TelemetryStore,
    initializes the recorder, builds a LogHub, and starts a Sampler plus
    a retention-prune daemon -- all attached to SupervisorState.
  - `_attempt_restart` / `_monitor_workers`: the auto-restart respawn path
    must keep forwarding a live `state.log_hub` to `spawn_worker`, same as
    every other spawn site (muse.admin.operations), so a respawned
    worker's stdout keeps flowing into the dashboard log tail instead of
    going silent after a crash-restart.

None of these tests drive uvicorn or `run_supervisor` itself; all call the
factored-out functions directly, per the brief.
"""
from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    _attempt_restart,
    _init_telemetry,
    _monitor_workers,
    _pump_worker_logs,
)
from muse.core import config
from muse.observability.logs import LogHub
from muse.observability.recorder import get_recorder, reset_recorder
from muse.observability.store import TelemetryStore


class TestPumpWorkerLogs:
    def test_lines_land_in_hub(self):
        proc = types.SimpleNamespace(stdout=iter(["hello\n", "world\n"]))
        hub = LogHub()

        _pump_worker_logs(proc, "m", hub)

        assert hub.snapshot("m") == ["hello\n", "world\n"]

    def test_reader_exits_on_eof_without_raising(self):
        # An empty stdout iterator hits StopIteration immediately; the
        # loop must return cleanly (this is what a daemon thread target
        # does when the worker process exits).
        proc = types.SimpleNamespace(stdout=iter([]))
        hub = LogHub()

        _pump_worker_logs(proc, "m", hub)  # must not raise

        assert hub.snapshot("m") == []

    def test_exception_from_stdout_iteration_is_swallowed(self):
        class _BoomIter:
            def __iter__(self):
                return self

            def __next__(self):
                raise RuntimeError("boom")

        proc = types.SimpleNamespace(stdout=_BoomIter())
        hub = LogHub()

        _pump_worker_logs(proc, "m", hub)  # must not raise


class TestInitTelemetry:
    @pytest.fixture(autouse=True)
    def _cleanup_recorder(self):
        yield
        reset_recorder()

    def test_wires_store_hub_sampler_and_recorder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()

        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})

        try:
            _init_telemetry(state)

            assert isinstance(state.telemetry_store, TelemetryStore)
            assert isinstance(state.log_hub, LogHub)
            assert (tmp_path / "telemetry.db").exists()

            # The recorder is now the real (non-noop) recorder: recording
            # and flushing an event must not raise, and dropped stays 0
            # for a single event on a fresh queue.
            recorder = get_recorder()
            recorder.record("request", model_id="m", latency_ms=1.0)
            recorder.flush()
            assert recorder.dropped == 0

            # The sampler shares state.stop_event (mirrors IdleSweeper): once
            # the shared event is set and the sampler is stopped, its thread
            # must actually exit -- this is the shutdown path run_supervisor's
            # finally block now performs (sampler.stop() + store.close()).
            assert state.telemetry_sampler is not None
            assert state.telemetry_sampler._stop is state.stop_event
        finally:
            state.stop_event.set()
            sampler_thread = state.telemetry_sampler._thread
            state.telemetry_sampler.stop()
            # stop() joins the thread and clears the handle to None; check
            # the captured reference (not the now-None state.telemetry_sampler
            # ._thread) to confirm the thread actually exited.
            assert sampler_thread is not None and not sampler_thread.is_alive()
            assert state.telemetry_sampler._thread is None
            if state.telemetry_store is not None:
                state.telemetry_store.close()
            reset_recorder()
            config.reset_config()

    def test_prune_loop_uses_state_stop_event(self, tmp_path, monkeypatch):
        # Regression guard: _init_telemetry must not crash when
        # state.stop_event is already a real threading.Event (as run_supervisor
        # sets it before calling _init_telemetry).
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()

        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})

        try:
            _init_telemetry(state)
            assert not state.stop_event.is_set()
        finally:
            state.stop_event.set()
            if state.telemetry_sampler is not None:
                state.telemetry_sampler.stop()
            if state.telemetry_store is not None:
                state.telemetry_store.close()
            reset_recorder()
            config.reset_config()


class TestRestartForwardsLogHub:
    """Finding 1 (final whole-branch review, observability dashboard).

    Every OTHER spawn_worker call site forwards
    `log_hub=getattr(state, "log_hub", None)` (see muse.admin.operations'
    enable_model / _restart_worker_inplace). Before this fix, the
    auto-restart monitor's respawn path (`_attempt_restart` ->
    `spawn_worker(spec, device=spec.device)`) passed no `log_hub` at all,
    so a crashed-and-respawned worker's stdout stopped flowing into the
    LogHub and its dashboard log tail went silent permanently.

    `_monitor_workers` is started (from `run_supervisor`) before
    `_init_telemetry` populates `state.log_hub`, so it must accept the
    live `state` object (not a value snapshotted at thread-start) and
    read `state.log_hub` at the moment it decides to restart -- these
    tests assert that live threading end-to-end.
    """

    def test_attempt_restart_forwards_log_hub_to_spawn_worker(self):
        hub = LogHub()
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))  # already exited
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(
                spec, stop_event=stop_event, max_restarts=10, backoff_base=0,
                log_hub=hub,
            )

        mock_spawn.assert_called_once_with(spec, device="cpu", log_hub=hub)

    def test_attempt_restart_defaults_log_hub_to_none(self):
        """Regression guard: callers that don't pass log_hub (e.g. direct
        unit tests elsewhere in the suite) keep today's telemetry-disabled
        behavior -- spawn_worker still gets an explicit log_hub=None."""
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_called_once_with(spec, device="cpu", log_hub=None)

    def test_monitor_workers_forwards_state_log_hub_to_attempt_restart(self):
        """_monitor_workers, given a `state` carrying a LogHub, must thread
        `log_hub=state.log_hub` into `_attempt_restart` at restart time."""
        hub = LogHub()
        state = SupervisorState()
        state.log_hub = hub

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))  # already exited
        stop_event = threading.Event()

        def _restart_side_effect(*args, **kwargs):
            stop_event.set()  # stop the loop after the first restart attempt

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=1, max_restarts=10,
                state=state,
            )

        mock_restart.assert_called_once()
        _, kwargs = mock_restart.call_args
        assert kwargs["log_hub"] is hub

    def test_monitor_workers_without_state_forwards_none(self):
        """Callers that don't pass `state` (the many existing
        _monitor_workers tests elsewhere in the suite that call with just
        (specs, stop_event)) keep today's behavior: no log_hub forwarded."""
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        def _restart_side_effect(*args, **kwargs):
            stop_event.set()

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=1, max_restarts=10,
            )

        mock_restart.assert_called_once()
        _, kwargs = mock_restart.call_args
        assert kwargs["log_hub"] is None

    def test_monitor_workers_reads_log_hub_live_not_at_thread_start(self):
        """The real bug: run_supervisor starts the monitor thread BEFORE
        `_init_telemetry` populates `state.log_hub`. If `_monitor_workers`
        captured `state.log_hub` once (e.g. at thread-creation time via
        `kwargs={"log_hub": state.log_hub}`), a later-populated hub would
        never reach a subsequent restart. Simulate that ordering: log_hub
        starts None, gets populated by a side effect mid-tick (standing in
        for _init_telemetry finishing shortly after the monitor thread
        starts), and the eventual restart must still see the populated hub
        because `state` (not a snapshotted value) is what's threaded
        through.
        """
        hub = LogHub()
        state = SupervisorState()
        state.log_hub = None  # not yet populated, like at monitor-thread-start

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))  # still alive
        stop_event = threading.Event()

        call_count = {"n": 0}

        def _health_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Simulate _init_telemetry populating state.log_hub between
                # the first (non-restarting) tick and the second.
                state.log_hub = hub
            return False  # always unhealthy

        def _restart_side_effect(*args, **kwargs):
            stop_event.set()

        with patch(
            "muse.cli_impl.supervisor.check_worker_health",
            side_effect=_health_side_effect,
        ), patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=2, max_restarts=10,
                state=state,
            )

        mock_restart.assert_called_once()
        _, kwargs = mock_restart.call_args
        assert kwargs["log_hub"] is hub
