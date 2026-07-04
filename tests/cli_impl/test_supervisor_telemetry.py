"""Tests for Task 11: supervisor telemetry lifecycle + per-worker log piping.

Covers the two testable units factored out of run_supervisor/spawn_worker:
  - `_pump_worker_logs`: the daemon reader loop that pipes a worker's
    stdout lines into a LogHub (and re-emits them to the aggregate log).
  - `_init_telemetry`: the boot-time wiring that creates a TelemetryStore,
    initializes the recorder, builds a LogHub, and starts a Sampler plus
    a retention-prune daemon -- all attached to SupervisorState.

Neither test drives uvicorn or `run_supervisor` itself; both call the
factored-out functions directly, per the brief.
"""
from __future__ import annotations

import types

import pytest

from muse.cli_impl.supervisor import SupervisorState, _init_telemetry, _pump_worker_logs
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
