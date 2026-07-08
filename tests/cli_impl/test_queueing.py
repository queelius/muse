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
