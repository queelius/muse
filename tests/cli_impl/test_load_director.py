"""Tests for LoadDirector: hot/cold acquire, refcount, singleton-load coordination.

Task B of the v0.40.0 lazy-load plan. Eviction is Task C and is exercised here
only as a placeholder watchdog: a cold acquire that does NOT fit must raise
NotImplementedError until Task C lands.

Concurrency tests use threading.Barrier to make two sync threads enter the
decision phase as close to simultaneously as the GIL allows. Both threads
must see the same worker_port and the injected enable_fn must be called
exactly once.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from muse.cli_impl.load_director import (
    DecisionLogEntry,
    LoadDirector,
    LoadEntry,
)


def _make_probe(gpu_free: float = 32.0, cpu_free: float = 64.0) -> MagicMock:
    """Default probe: lots of room. The 'fits' branch is exercised."""
    probe = MagicMock()
    probe.gpu_free_gb.return_value = gpu_free
    probe.cpu_free_gb.return_value = cpu_free
    return probe


def _manifest(memory_gb: float = 0.5, device: str = "cpu") -> dict:
    return {
        "model_id": "fake-model",
        "modality": "audio/speech",
        "capabilities": {
            "memory_gb": memory_gb,
            "device": device,
        },
    }


# ----------------------------------------------------------------------
# B3: hot path
# ----------------------------------------------------------------------

class TestHotAcquire:
    def test_returns_existing_port_increments_refcount(self):
        enable_fn = MagicMock(return_value=9001)
        disable_fn = MagicMock()
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=_make_probe(),
        )

        # Cold acquire (loads the model)
        port1 = director.acquire("fake-model", manifest=_manifest())
        assert port1 == 9001
        assert enable_fn.call_count == 1

        # Hot acquire returns the same port without calling enable_fn again
        port2 = director.acquire("fake-model", manifest=_manifest())
        assert port2 == 9001
        assert enable_fn.call_count == 1

        # Refcount incremented twice: once per acquire
        snapshot = director.status()
        assert snapshot["fake-model"]["refcount"] == 2

    def test_hot_acquire_updates_last_touched(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        first_touched = director.status()["fake-model"]["last_touched_at"]

        # Sleep enough that monotonic advances past time-resolution noise
        time.sleep(0.01)

        director.acquire("fake-model", manifest=_manifest())
        second_touched = director.status()["fake-model"]["last_touched_at"]

        assert second_touched > first_touched

    def test_hot_acquire_does_not_invoke_enable_fn(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        enable_fn.reset_mock()

        for _ in range(5):
            director.acquire("fake-model", manifest=_manifest())

        enable_fn.assert_not_called()


# ----------------------------------------------------------------------
# B4: cold path that fits
# ----------------------------------------------------------------------

class TestColdAcquireFits:
    def test_calls_enable_fn_once_returns_port(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        port = director.acquire("fake-model", manifest=_manifest())

        assert port == 9001
        enable_fn.assert_called_once_with("fake-model")

    def test_populates_load_entry_with_manifest_memory(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5))

        snapshot = director.status()
        assert "fake-model" in snapshot
        assert snapshot["fake-model"]["worker_port"] == 9001
        assert snapshot["fake-model"]["loaded"] is True
        assert snapshot["fake-model"]["refcount"] == 1

    def test_load_entry_has_loaded_at_and_last_touched_at(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        before = time.monotonic()
        director.acquire("fake-model", manifest=_manifest())
        after = time.monotonic()

        snapshot = director.status()
        loaded_at = snapshot["fake-model"]["last_touched_at"]
        # last_touched_at is monotonic seconds, recorded inside acquire
        assert before <= loaded_at <= after

    def test_default_budgets_and_headrooms_used_when_unset(self):
        # Sanity check: probe returning generous numbers + manifest
        # demanding 0.5 GB triggers the fits branch with all defaults.
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        port = director.acquire("fake-model", manifest=_manifest())
        assert port == 9001


# ----------------------------------------------------------------------
# B4 (cont.): cold path that does NOT fit (placeholder for Task C)
# ----------------------------------------------------------------------

class TestColdAcquireDoesNotFitPlaceholder:
    """When the model needs more than (free - headroom), Task C will run an
    eviction loop. Until that lands, LoadDirector raises NotImplementedError
    so the lack of eviction is visible (no silent success).
    """

    def test_raises_not_implemented_for_now(self):
        # 100 GB model demanded, only 32 GB GPU free, 64 GB CPU free.
        # Even with default headrooms, this can't fit.
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=32.0, cpu_free=64.0),
        )

        with pytest.raises(NotImplementedError, match="Task C"):
            director.acquire("huge-model", manifest=_manifest(memory_gb=100.0, device="cpu"))

    def test_raises_for_gpu_oversize_too(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=4.0, cpu_free=64.0),
        )

        with pytest.raises(NotImplementedError, match="Task C"):
            director.acquire("huge-gpu-model", manifest=_manifest(memory_gb=20.0, device="cuda"))

    def test_does_not_call_enable_fn_when_no_fit(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=4.0, cpu_free=8.0),
        )

        with pytest.raises(NotImplementedError):
            director.acquire("huge", manifest=_manifest(memory_gb=100.0, device="cpu"))

        enable_fn.assert_not_called()

    def test_no_in_flight_event_left_behind_on_no_fit(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=4.0, cpu_free=8.0),
        )

        with pytest.raises(NotImplementedError):
            director.acquire("huge", manifest=_manifest(memory_gb=100.0, device="cpu"))

        # After the failed decision, in_flight_loads must be empty so a
        # later acquire (e.g. once Task C lands and budget changes) is
        # not blocked by a stale Event.
        assert director.in_flight_loads == {}


# ----------------------------------------------------------------------
# release: refcount decrement, no auto-evict
# ----------------------------------------------------------------------

class TestRelease:
    def test_decrements_refcount(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        director.acquire("fake-model", manifest=_manifest())
        assert director.status()["fake-model"]["refcount"] == 2

        director.release("fake-model")
        assert director.status()["fake-model"]["refcount"] == 1

        director.release("fake-model")
        assert director.status()["fake-model"]["refcount"] == 0

    def test_does_not_call_disable_fn_on_release(self):
        # v0.40.0 Task B: release does not auto-evict. Eviction is on-demand
        # only (Task C, triggered from a *cold* acquire that needs room).
        disable_fn = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=disable_fn,
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        director.release("fake-model")

        disable_fn.assert_not_called()

    def test_release_updates_last_touched(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        first = director.status()["fake-model"]["last_touched_at"]

        time.sleep(0.01)
        director.release("fake-model")
        second = director.status()["fake-model"]["last_touched_at"]
        assert second > first

    def test_release_unknown_model_is_noop(self):
        # Defensive: a release for a model that's no longer in the loaded
        # set (e.g. evicted between acquire and release on another thread)
        # should not crash.
        director = LoadDirector(
            enable_fn=MagicMock(),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        # Should not raise.
        director.release("never-loaded")


# ----------------------------------------------------------------------
# B6: singleton load coordination across threads
# ----------------------------------------------------------------------

class TestConcurrentAcquireSameModel:
    def test_two_threads_share_one_load(self):
        # Two threads call acquire(X) where X is unloaded. They MUST share
        # one enable_fn invocation; the loser awaits the winner's Event
        # and reads the freshly populated loaded entry.
        barrier = threading.Barrier(2)
        slow_load_started = threading.Event()
        slow_load_release = threading.Event()

        def slow_enable(model_id: str) -> int:
            slow_load_started.set()
            # Hold inside the load phase (lock NOT held) long enough for
            # the second thread to enter the decision phase, see the
            # in-flight event, and start awaiting.
            slow_load_release.wait(timeout=5.0)
            return 9001

        director = LoadDirector(
            enable_fn=slow_enable,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        results: list[int] = []
        errors: list[BaseException] = []

        def worker():
            try:
                barrier.wait(timeout=5.0)
                p = director.acquire("fake-model", manifest=_manifest())
                results.append(p)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()

        # Wait for the first thread to actually begin enable_fn so we
        # know it has surrendered the lock and the second thread is
        # blocked on the Event (not still racing in the decision phase).
        assert slow_load_started.wait(timeout=5.0)
        # Tiny pause so the second thread has time to reach the wait.
        time.sleep(0.05)
        slow_load_release.set()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not errors, errors
        assert results == [9001, 9001], f"both threads must see the same port; got {results}"

    def test_enable_fn_called_exactly_once_under_contention(self):
        barrier = threading.Barrier(2)
        slow_release = threading.Event()
        call_count = {"n": 0}
        count_lock = threading.Lock()

        def slow_enable(model_id: str) -> int:
            with count_lock:
                call_count["n"] += 1
            slow_release.wait(timeout=5.0)
            return 9001

        director = LoadDirector(
            enable_fn=slow_enable,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        def worker():
            barrier.wait(timeout=5.0)
            director.acquire("fake-model", manifest=_manifest())

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()

        # Give both threads a chance to reach the decision phase. The
        # winner reaches the load phase and increments call_count; the
        # loser blocks on the Event. Releasing now lets the winner
        # complete; the loser then re-enters the decision phase and
        # sees a hot entry, so it does NOT call enable_fn.
        time.sleep(0.1)
        slow_release.set()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert call_count["n"] == 1, f"enable_fn was called {call_count['n']} times; expected 1"

        # Refcount must be 2: each acquire bumps it.
        assert director.status()["fake-model"]["refcount"] == 2


# ----------------------------------------------------------------------
# DecisionLogEntry recorded on every load
# ----------------------------------------------------------------------

class TestDecisionLog:
    def test_decision_recorded_on_load(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5))

        # recent_decisions exposes the same deque the admin endpoint reads
        decisions = list(director.recent_decisions)
        assert len(decisions) == 1

        d = decisions[0]
        assert isinstance(d, DecisionLogEntry)
        assert d.model_id == "fake-model"
        assert d.action == "load"
        assert d.memory_gb == 0.5
        assert d.evicted == []
        assert d.reason  # free-form, but must be non-empty

    def test_decision_has_free_before_and_after(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=32.0, cpu_free=64.0),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5, device="cpu"))

        d = list(director.recent_decisions)[0]
        # CPU model: free_before should reflect cpu_free_gb
        assert d.free_before_gb == 64.0
        # free_after_gb is captured after the load returns; with the
        # MagicMock probe it stays at 64.0
        assert d.free_after_gb == 64.0

    def test_recent_decisions_capped_at_20(self):
        enable_fn = MagicMock(side_effect=lambda mid: 9000 + hash(mid) % 1000)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        for i in range(25):
            director.acquire(f"model-{i}", manifest=_manifest())

        # deque(maxlen=20) caps the recent log at 20 entries
        assert len(director.recent_decisions) == 20

    def test_no_decision_logged_on_hot_acquire(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())  # cold; logs
        director.acquire("fake-model", manifest=_manifest())  # hot; no log

        assert len(director.recent_decisions) == 1


# ----------------------------------------------------------------------
# Exception during enable_fn: cleanup + waiter wake
# ----------------------------------------------------------------------

class TestEnableFnException:
    def test_exception_propagates_in_flight_cleared(self):
        boom = RuntimeError("worker spawn died")

        def failing_enable(model_id: str) -> int:
            raise boom

        director = LoadDirector(
            enable_fn=failing_enable,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        with pytest.raises(RuntimeError, match="worker spawn died"):
            director.acquire("fake-model", manifest=_manifest())

        # in_flight_loads must be empty so the next acquire isn't
        # blocked indefinitely waiting on a dead Event.
        assert director.in_flight_loads == {}
        # And no LoadEntry was inserted on failure.
        assert "fake-model" not in director.status()

    def test_concurrent_waiter_wakes_and_reattempts(self):
        # Two threads racing on a cold acquire: the winner's enable_fn
        # raises. The loser was awaiting the winner's Event; on wake
        # it must re-enter the decision phase. With enable_fn now
        # patched to succeed, the loser should win the second round.
        attempt_count = {"n": 0}
        attempt_lock = threading.Lock()
        first_started = threading.Event()
        first_release = threading.Event()

        def enable_fn(model_id: str) -> int:
            with attempt_lock:
                attempt_count["n"] += 1
                this_attempt = attempt_count["n"]
            if this_attempt == 1:
                # The first attempt waits for our signal, then raises.
                # This holds the lock-free "load phase" open long enough
                # for the second thread to await on the in-flight Event.
                first_started.set()
                first_release.wait(timeout=5.0)
                raise RuntimeError("first attempt fails")
            # Subsequent attempts succeed.
            return 9001

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        results: list[object] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def worker():
            try:
                barrier.wait(timeout=5.0)
                results.append(director.acquire("fake-model", manifest=_manifest()))
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()

        assert first_started.wait(timeout=5.0)
        # Brief pause so the second thread reaches the Event-await.
        time.sleep(0.05)
        first_release.set()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        # One thread (the winner) raised. The other thread saw an empty
        # entry on wakeup, re-decided, and successfully loaded.
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert results == [9001]
        # enable_fn was called twice: once for the failed attempt, once
        # for the retry by the woken waiter.
        assert attempt_count["n"] == 2


# ----------------------------------------------------------------------
# LoadEntry shape sanity (the dataclass is part of the public API)
# ----------------------------------------------------------------------

class TestLoadEntryShape:
    def test_fields_present(self):
        e = LoadEntry(
            model_id="x",
            worker_port=9001,
            memory_gb=0.5,
            refcount=0,
            last_touched_at=time.monotonic(),
            loaded_at=time.monotonic(),
        )
        assert e.model_id == "x"
        assert e.worker_port == 9001
        assert e.memory_gb == 0.5
        assert e.refcount == 0
        assert e.last_touched_at > 0
        assert e.loaded_at > 0


class TestDecisionLogEntryShape:
    def test_fields_present(self):
        d = DecisionLogEntry(
            timestamp=time.time(),
            model_id="x",
            action="load",
            memory_gb=0.5,
            free_before_gb=10.0,
            free_after_gb=9.5,
            reason="fit",
            evicted=[],
        )
        assert d.model_id == "x"
        assert d.action == "load"
        assert d.evicted == []
