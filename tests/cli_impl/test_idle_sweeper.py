"""Tests for IdleSweeper: per-model idle-timeout eviction (v0.40.1).

Task A of the v0.40.1 plan. The sweeper is a CLIENT of LoadDirector's
public surface (`director.lock`, `director.loaded`, `director.disable_fn`,
`director.memory_probe`, `director.recent_decisions`); no source-side
changes to LoadDirector itself.

These tests drive `tick()` directly so we don't depend on the real
thread loop; the loop is exercised separately in
TestThreadLoop / TestStopEventExitsThreadPromptly.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from muse.cli_impl.idle_sweeper import IdleSweeper
from muse.cli_impl.load_director import (
    DecisionLogEntry,
    LoadDirector,
    LoadEntry,
)


# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------

def _make_probe(gpu_free: float = 32.0, cpu_free: float = 64.0) -> MagicMock:
    """Default probe: deterministic free values."""
    probe = MagicMock()
    probe.gpu_free_gb.return_value = gpu_free
    probe.cpu_free_gb.return_value = cpu_free
    return probe


def _manifest(
    *,
    memory_gb: float = 0.5,
    device: str = "cpu",
    idle_timeout_seconds: float | None = 10.0,
) -> dict:
    caps: dict = {"memory_gb": memory_gb, "device": device}
    if idle_timeout_seconds is not None:
        caps["idle_timeout_seconds"] = idle_timeout_seconds
    return {
        "model_id": "fake-model",
        "modality": "audio/speech",
        "capabilities": caps,
    }


def _director(*, enable_fn=None, disable_fn=None, probe=None) -> LoadDirector:
    return LoadDirector(
        enable_fn=enable_fn or MagicMock(return_value=9001),
        disable_fn=disable_fn or MagicMock(),
        memory_probe=probe or _make_probe(),
    )


def _preload(
    director: LoadDirector,
    model_id: str,
    *,
    memory_gb: float = 0.5,
    refcount: int = 0,
    last_touched_at: float | None = None,
    worker_port: int = 9001,
) -> LoadEntry:
    """Inject a LoadEntry directly so we don't depend on acquire() success."""
    now = time.monotonic()
    entry = LoadEntry(
        model_id=model_id,
        worker_port=worker_port,
        memory_gb=memory_gb,
        refcount=refcount,
        last_touched_at=last_touched_at if last_touched_at is not None else now,
        loaded_at=now,
    )
    with director.lock:
        director.loaded[model_id] = entry
    return entry


def _catalog_lookup_factory(catalog: dict[str, dict]):
    """Build a catalog_lookup callable that raises KeyError on missing ids
    (matching the sweeper's silent-skip contract)."""
    def lookup(model_id: str) -> dict:
        return catalog[model_id]
    return lookup


# ----------------------------------------------------------------------
# A3.1: idle candidate (refcount=0, last_touched > timeout)
# ----------------------------------------------------------------------

class TestEvictsIdleCandidateWithRefcountZero:
    def test_evicts_idle_candidate_with_refcount_zero(self):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        # Preload with last_touched_at well in the past (15s ago) and an
        # idle_timeout_seconds of 10. Eligible.
        _preload(
            director,
            "fake-model",
            last_touched_at=time.monotonic() - 15.0,
        )
        catalog = {"fake-model": _manifest(idle_timeout_seconds=10.0)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        assert evicted == ["fake-model"]
        disable_fn.assert_called_once_with("fake-model")
        with director.lock:
            assert "fake-model" not in director.loaded


# ----------------------------------------------------------------------
# A3.2: skip when no idle_timeout_seconds set
# ----------------------------------------------------------------------

class TestSkipsModelWithNoIdleTimeout:
    def test_skips_model_with_no_idle_timeout(self):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        # Last touched hours ago; would be evicted if a timeout existed.
        _preload(
            director,
            "fake-model",
            last_touched_at=time.monotonic() - 3600.0,
        )
        # Manifest has NO idle_timeout_seconds key.
        catalog = {"fake-model": _manifest(idle_timeout_seconds=None)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        assert evicted == []
        disable_fn.assert_not_called()
        with director.lock:
            assert "fake-model" in director.loaded


# ----------------------------------------------------------------------
# A3.3: defensive against zero / negative timeouts (operator typo)
# ----------------------------------------------------------------------

class TestSkipsModelWithZeroOrNegativeTimeout:
    @pytest.mark.parametrize("bad_timeout", [0, 0.0, -1, -42.5])
    def test_skips_model_with_zero_or_negative_timeout(self, bad_timeout):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        _preload(
            director,
            "fake-model",
            last_touched_at=time.monotonic() - 1000.0,
        )
        catalog = {"fake-model": _manifest(idle_timeout_seconds=bad_timeout)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        assert evicted == []
        disable_fn.assert_not_called()


# ----------------------------------------------------------------------
# A3.4: refcount > 0 always wins
# ----------------------------------------------------------------------

class TestSkipsModelWithRefcountPositive:
    def test_skips_model_with_refcount_positive(self):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        _preload(
            director,
            "fake-model",
            refcount=1,
            last_touched_at=time.monotonic() - 10000.0,
        )
        catalog = {"fake-model": _manifest(idle_timeout_seconds=1.0)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        assert evicted == []
        disable_fn.assert_not_called()
        with director.lock:
            assert director.loaded["fake-model"].refcount == 1


# ----------------------------------------------------------------------
# A3.5: not yet idle (last_touched < timeout ago)
# ----------------------------------------------------------------------

class TestSkipsModelRecentlyTouched:
    def test_skips_model_recently_touched(self):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        # Touched 1s ago, timeout 60s -> not idle yet.
        _preload(
            director,
            "fake-model",
            last_touched_at=time.monotonic() - 1.0,
        )
        catalog = {"fake-model": _manifest(idle_timeout_seconds=60.0)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        assert evicted == []
        disable_fn.assert_not_called()


# ----------------------------------------------------------------------
# A3.6: disable_fn raises -> re-insert load entry
# ----------------------------------------------------------------------

class TestDisableFnFailureReinsertsLoadEntry:
    def test_disable_fn_failure_reinserts_load_entry(self, caplog):
        disable_fn = MagicMock(side_effect=RuntimeError("boom"))
        director = _director(disable_fn=disable_fn)

        original_last_touched = time.monotonic() - 30.0
        original_refcount = 0
        _preload(
            director,
            "fake-model",
            last_touched_at=original_last_touched,
            refcount=original_refcount,
        )
        catalog = {"fake-model": _manifest(idle_timeout_seconds=5.0)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        # Nothing reported as evicted; the entry was re-inserted on failure.
        assert evicted == []
        disable_fn.assert_called_once_with("fake-model")

        with director.lock:
            assert "fake-model" in director.loaded
            restored = director.loaded["fake-model"]
            # Must preserve the original last_touched_at + refcount so the
            # entry isn't re-evicted on the next tick or treated specially
            # by LRU ordering.
            assert restored.last_touched_at == original_last_touched
            assert restored.refcount == original_refcount


# ----------------------------------------------------------------------
# A3.7: disable_fn raises on first candidate -> second still processed
# ----------------------------------------------------------------------

class TestDisableFnFailureContinuesToNextCandidate:
    def test_disable_fn_failure_continues_to_next_candidate(self):
        # First call (for "model-a") raises; second (for "model-b") succeeds.
        disable_fn = MagicMock(side_effect=[RuntimeError("a-boom"), None])
        director = _director(disable_fn=disable_fn)

        # Both idle, both eligible.
        _preload(
            director,
            "model-a",
            last_touched_at=time.monotonic() - 100.0,
        )
        _preload(
            director,
            "model-b",
            last_touched_at=time.monotonic() - 200.0,
        )

        catalog = {
            "model-a": _manifest(idle_timeout_seconds=5.0),
            "model-b": _manifest(idle_timeout_seconds=5.0),
        }

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        # Only model-b actually evicted.
        assert evicted == ["model-b"]
        with director.lock:
            assert "model-a" in director.loaded   # re-inserted
            assert "model-b" not in director.loaded
        # disable_fn called once per candidate.
        assert disable_fn.call_count == 2


# ----------------------------------------------------------------------
# A3.8: DecisionLogEntry recorded with reason="idle_timeout:Ns"
# ----------------------------------------------------------------------

class TestDecisionLogEntryRecorded:
    def test_decision_log_entry_recorded(self):
        director = _director()

        _preload(
            director,
            "fake-model",
            memory_gb=2.5,
            last_touched_at=time.monotonic() - 60.0,
        )
        catalog = {"fake-model": _manifest(memory_gb=2.5, idle_timeout_seconds=30.0)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        sweeper.tick()

        with director.lock:
            decisions = list(director.recent_decisions)

        assert len(decisions) == 1
        d = decisions[0]
        assert isinstance(d, DecisionLogEntry)
        assert d.action == "evict"
        assert d.model_id == "fake-model"
        assert d.memory_gb == 2.5
        assert d.evicted == ["fake-model"]
        assert d.reason == "idle_timeout:30s"
        assert d.free_before_gb is not None
        assert d.free_after_gb is not None

    def test_decision_log_entry_reason_uses_int_seconds(self):
        """A non-integer idle_timeout_seconds must still render with int()
        seconds in the reason string per the spec example."""
        director = _director()

        _preload(
            director,
            "fake-model",
            last_touched_at=time.monotonic() - 50.0,
        )
        catalog = {"fake-model": _manifest(idle_timeout_seconds=15.7)}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        sweeper.tick()

        with director.lock:
            d = director.recent_decisions[-1]
        assert d.reason == "idle_timeout:15s"


# ----------------------------------------------------------------------
# A3.9: catalog lookup raises -> silently skipped
# ----------------------------------------------------------------------

class TestCatalogLookupRaisesSilentlySkips:
    def test_catalog_lookup_raises_silently_skips(self):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        _preload(
            director,
            "ghost-model",
            last_touched_at=time.monotonic() - 1000.0,
        )

        def lookup(model_id: str) -> dict:
            raise KeyError(model_id)

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lookup,
        )

        # Tick must not raise.
        evicted = sweeper.tick()

        assert evicted == []
        disable_fn.assert_not_called()
        with director.lock:
            assert "ghost-model" in director.loaded


# ----------------------------------------------------------------------
# A3.10: multiple idle candidates evicted in one tick
# ----------------------------------------------------------------------

class TestMultipleCandidatesEvictedInOneTick:
    def test_multiple_candidates_evicted_in_one_tick(self):
        disable_fn = MagicMock()
        director = _director(disable_fn=disable_fn)

        for i in range(3):
            _preload(
                director,
                f"model-{i}",
                last_touched_at=time.monotonic() - (50.0 + i),
            )

        catalog = {
            f"model-{i}": _manifest(idle_timeout_seconds=10.0)
            for i in range(3)
        }

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
        )

        evicted = sweeper.tick()

        assert sorted(evicted) == ["model-0", "model-1", "model-2"]
        assert disable_fn.call_count == 3
        with director.lock:
            assert director.loaded == {}


# ----------------------------------------------------------------------
# A3.11: thread loop calls tick repeatedly
# ----------------------------------------------------------------------

class TestThreadLoop:
    def test_thread_loop_calls_tick_repeatedly(self):
        director = _director()
        catalog: dict[str, dict] = {}

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory(catalog),
            interval_seconds=0.05,
        )

        call_count = {"n": 0}

        original_tick = sweeper.tick

        def counted_tick():
            call_count["n"] += 1
            return original_tick()

        sweeper.tick = counted_tick  # type: ignore[method-assign]

        thread = sweeper.start()
        try:
            # Sleep long enough for several iterations at 0.05s interval.
            time.sleep(0.25)
        finally:
            sweeper.stop()
            thread.join(timeout=2.0)

        assert not thread.is_alive()
        # We expect at least 2 ticks (one immediate + one after a wait()).
        assert call_count["n"] >= 2

    def test_tick_exception_does_not_kill_loop(self, caplog):
        director = _director()

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lambda mid: {"capabilities": {}},
            interval_seconds=0.05,
        )

        results = {"calls": 0}

        def flaky_tick():
            results["calls"] += 1
            if results["calls"] == 1:
                raise RuntimeError("first-tick-boom")
            return []

        sweeper.tick = flaky_tick  # type: ignore[method-assign]

        thread = sweeper.start()
        try:
            time.sleep(0.25)
        finally:
            sweeper.stop()
            thread.join(timeout=2.0)

        # Loop survived the first-tick exception and called tick again.
        assert results["calls"] >= 2
        assert not thread.is_alive()


# ----------------------------------------------------------------------
# A3.12: stop event exits thread promptly
# ----------------------------------------------------------------------

class TestStopEventExitsThreadPromptly:
    def test_stop_event_exits_thread_promptly(self):
        director = _director()

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lambda mid: {"capabilities": {}},
            interval_seconds=10.0,  # would block 10s if wait() didn't honor stop
        )

        thread = sweeper.start()
        # Brief delay so the loop is parked in wait().
        time.sleep(0.05)
        start_time = time.monotonic()
        sweeper.stop()
        thread.join(timeout=2.0)
        elapsed = time.monotonic() - start_time

        assert not thread.is_alive()
        # Should exit well before the 10s interval would naturally fire.
        assert elapsed < 1.0

    def test_external_stop_event_can_be_shared(self):
        """Constructor accepts an external stop_event so the supervisor
        can share its existing state.stop_event with the sweeper."""
        director = _director()
        external_event = threading.Event()

        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lambda mid: {"capabilities": {}},
            interval_seconds=10.0,
            stop_event=external_event,
        )

        thread = sweeper.start()
        time.sleep(0.05)
        # Setting the external event must stop the thread.
        external_event.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()


class TestFreeForDeviceResolvesAuto:
    """The sweeper's free-memory reading for the decision log must agree
    with the LoadDirector's pool resolution. After the v0.48.0 director fix
    (auto -> VRAM pool on a GPU host), the sweeper's local copy drifted: it
    treated 'auto' as CPU, so an auto model's free_before/after_gb in the
    idle-eviction decision log would report host RAM instead of VRAM. The
    sweeper now delegates to the director, so there is one resolution."""

    def test_free_for_device_auto_reads_gpu_pool_when_gpu_present(self):
        probe = _make_probe(gpu_free=7.0, cpu_free=512.0)
        director = _director(probe=probe)
        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory({}),
        )
        assert sweeper._free_for_device("auto") == 7.0

    def test_free_for_device_auto_reads_cpu_pool_when_no_gpu(self):
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 512.0
        director = _director(probe=probe)
        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=_catalog_lookup_factory({}),
        )
        assert sweeper._free_for_device("auto") == 512.0
