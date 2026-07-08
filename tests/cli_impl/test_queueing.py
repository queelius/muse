"""Unit tests for the gateway queueing primitives (spec 2026-07-08)."""
from __future__ import annotations

import asyncio
import time

import pytest

from muse.cli_impl.queueing import (
    CapacityNotifier, ConcurrencyGate, QueueFull, QueueTimeout,
)


def _deadline(seconds: float) -> float:
    return time.monotonic() + seconds


class TestConcurrencyGate:
    async def test_unlimited_cap_is_noop(self):
        gate = ConcurrencyGate()
        async with gate.slot("m", None, deadline=_deadline(1)):
            assert gate.depth("m") == 0  # unlimited: not even tracked
        async with gate.slot("m", 0, deadline=_deadline(1)):
            pass  # 0 also means unlimited

    async def test_cap_serializes_and_wakes_fifo(self):
        gate = ConcurrencyGate()
        order: list[int] = []
        release_first = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                order.append(0)
                await release_first.wait()

        async def waiter(i: int):
            async with gate.slot("m", 1, deadline=_deadline(5)):
                order.append(i)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)  # holder owns the slot
        w1 = asyncio.create_task(waiter(1))
        await asyncio.sleep(0.01)
        w2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.01)
        assert gate.depth("m") == 2  # two parked waiters
        release_first.set()
        await asyncio.gather(h, w1, w2)
        assert order == [0, 1, 2]  # FIFO
        assert gate.depth("m") == 0

    async def test_timeout_raises_queue_timeout(self):
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                started.set()
                await asyncio.sleep(0.5)

        h = asyncio.create_task(holder())
        await started.wait()
        with pytest.raises(QueueTimeout) as exc:
            async with gate.slot("m", 1, deadline=_deadline(0.05)):
                pass
        assert exc.value.model_id == "m"
        h.cancel()

    async def test_depth_bound_raises_queue_full(self, monkeypatch):
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                started.set()
                await asyncio.sleep(0.5)

        h = asyncio.create_task(holder())
        await started.wait()
        w = asyncio.create_task(gate.slot("m", 1, deadline=_deadline(5)).__aenter__())
        await asyncio.sleep(0.01)  # w is parked; depth == 1 == bound
        with pytest.raises(QueueFull):
            async with gate.slot("m", 1, deadline=_deadline(5)):
                pass
        h.cancel(); w.cancel()

    async def test_slot_released_on_exception(self):
        gate = ConcurrencyGate()
        with pytest.raises(RuntimeError):
            async with gate.slot("m", 1, deadline=_deadline(1)):
                raise RuntimeError("boom")
        # slot free again: immediate re-acquire succeeds
        async with gate.slot("m", 1, deadline=_deadline(1)):
            pass

    async def test_burst_arrival_enforces_queue_full(self, monkeypatch):
        # Regression test for a burst-arrival race: asyncio.wait_for wraps
        # sem.acquire() in its own sub-Task, so EVERY slot() call runs its
        # synchronous entry-check prefix before any acquire sub-Task has a
        # chance to run. A check based on sem.locked() is therefore stale
        # for concurrent arrivals with no sleeps between them; the entry
        # check must be a single synchronous compare-and-increment.
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
        gate = ConcurrencyGate()
        ok: list[int] = []
        full: list[int] = []

        async def worker(i: int):
            try:
                async with gate.slot("m", 1, deadline=_deadline(5)):
                    ok.append(i)
                    await asyncio.sleep(0.05)  # hold the slot briefly
            except QueueFull:
                full.append(i)
            except QueueTimeout:
                pass

        # No sleeps between task creation: all 10 workers are scheduled
        # in the same burst.
        await asyncio.gather(*[worker(i) for i in range(10)])
        assert len(ok) <= 2  # 1 holding + 1 queued (cap=1, max_depth=1)
        assert len(full) >= 8

    async def test_burst_depth_counts_only_excess(self):
        # depth() must report entered-minus-cap ("actually queued"), not
        # a raw attempt/entry count, even when a burst of concurrent
        # entries transiently inflates the entered counter before their
        # acquires settle.
        gate = ConcurrencyGate()
        samples: list[int] = []

        async def worker(i: int):
            async with gate.slot("m", 5, deadline=_deadline(5)):
                samples.append(gate.depth("m"))
                await asyncio.sleep(0.02)

        # 20 concurrent acquires against cap=5: entered can transiently
        # reach 20 (every task runs its synchronous entry-check prefix
        # before any acquire sub-Task completes -- see
        # test_burst_arrival_enforces_queue_full), so depth must never
        # exceed entered - cap == 15, and must settle back to 0.
        await asyncio.gather(*[worker(i) for i in range(20)])
        assert all(d <= 15 for d in samples)
        assert gate.depth("m") == 0

        # No contention (concurrency == cap): depth stays 0 throughout,
        # not just after the burst settles, because excess never goes
        # positive when entered never exceeds cap.
        samples.clear()

        async def worker_no_contention(i: int):
            async with gate.slot("m2", 5, deadline=_deadline(5)):
                samples.append(gate.depth("m2"))
                await asyncio.sleep(0.01)

        await asyncio.gather(*[worker_no_contention(i) for i in range(5)])
        assert all(d == 0 for d in samples)
        assert gate.depth("m2") == 0


class TestCapacityNotifier:
    async def test_snapshot_then_notify_wakes(self):
        n = CapacityNotifier()
        ev = n.snapshot()
        assert not ev.is_set()
        n.notify()  # threadsafe path, same loop here
        await asyncio.wait_for(ev.wait(), timeout=1)

    async def test_notify_before_any_snapshot_is_noop(self):
        n = CapacityNotifier()
        n.notify()  # must not raise (loop not yet captured)

    async def test_generation_semantics_no_missed_wakeup(self):
        n = CapacityNotifier()
        ev1 = n.snapshot()
        n.notify()
        await asyncio.wait_for(ev1.wait(), timeout=1)
        ev2 = n.snapshot()
        assert ev2 is not ev1 and not ev2.is_set()  # fresh generation

    async def test_notify_from_thread(self):
        n = CapacityNotifier()
        ev = n.snapshot()
        await asyncio.to_thread(n.notify)
        await asyncio.wait_for(ev.wait(), timeout=1)


class TestConfigRows:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("MUSE_CONFIG", "/nonexistent-config.yaml")
        from muse.core import config
        assert config.get("server.default_max_concurrency") == 0
        assert config.get("server.queue_timeout_seconds") == 300.0
        assert config.get("server.max_queue_depth") == 0
