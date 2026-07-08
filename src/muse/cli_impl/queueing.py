"""Gateway-side request queueing primitives (spec 2026-07-08).

Both primitives keep all waiting ON the asyncio event loop -- never in
ThreadPoolExecutor threads (the #318/#319 invariant). The LoadDirector
stays synchronous; it only *fires* the CapacityNotifier from its worker
threads via call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from muse.core import config


class QueueTimeout(Exception):
    """Waited past the deadline for a concurrency slot."""

    def __init__(self, model_id: str):
        super().__init__(f"queue timeout for {model_id!r}")
        self.model_id = model_id


class QueueFull(Exception):
    """Per-model waiter count exceeds server.max_queue_depth."""

    def __init__(self, model_id: str, depth: int):
        super().__init__(f"queue full for {model_id!r} (depth {depth})")
        self.model_id = model_id
        self.depth = depth


class ConcurrencyGate:
    """Per-model concurrency slots with FIFO waiters.

    A model with cap None/<=0 is UNLIMITED: no semaphore is created and
    slot() is a no-op context (today's behavior, zero overhead).
    CPython asyncio.Semaphore wakes waiters FIFO, so fairness is free.
    Semaphores are created lazily and sized once per process lifetime;
    cap changes apply on supervisor restart (documented in the spec).
    """

    def __init__(self) -> None:
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._waiting: dict[str, int] = {}

    def depth(self, model_id: str) -> int:
        return self._waiting.get(model_id, 0)

    def depths(self) -> dict[str, int]:
        return {m: n for m, n in self._waiting.items() if n > 0}

    def _sem(self, model_id: str, cap: int) -> asyncio.Semaphore:
        sem = self._sems.get(model_id)
        if sem is None:
            sem = asyncio.Semaphore(cap)
            self._sems[model_id] = sem
        return sem

    @asynccontextmanager
    async def slot(self, model_id: str, cap: int | None, *, deadline: float):
        if not cap or cap <= 0:
            yield  # unlimited: no gating at all
            return
        sem = self._sem(model_id, cap)
        if sem.locked():
            # Will park: enforce the flood bound BEFORE parking.
            max_depth = config.get("server.max_queue_depth") or 0
            depth = self._waiting.get(model_id, 0)
            if max_depth > 0 and depth >= max_depth:
                raise QueueFull(model_id, depth)
        self._waiting[model_id] = self._waiting.get(model_id, 0) + 1
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueueTimeout(model_id)
            try:
                await asyncio.wait_for(sem.acquire(), timeout=remaining)
            except asyncio.TimeoutError:
                raise QueueTimeout(model_id) from None
        finally:
            self._waiting[model_id] -= 1
            if self._waiting[model_id] <= 0:
                self._waiting.pop(model_id, None)
        try:
            yield
        finally:
            sem.release()


class CapacityNotifier:
    """Generation-event capacity broadcast, threadsafe on the fire side.

    Waiter protocol (missed-wakeup-free):
      1. ev = notifier.snapshot()   BEFORE the acquire attempt
      2. attempt director.acquire; on retryable capacity failure:
      3. await ev.wait() (bounded), then loop back to 1.

    notify() may be called from any thread (the director's release /
    eviction paths). It replaces the current event with a fresh one and
    sets the old, so late snapshots never see a stale set() and early
    waiters are always woken. Before the first snapshot() there is no
    captured loop; notify() is a silent no-op (nothing is waiting yet).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def snapshot(self) -> asyncio.Event:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        if self._event is None or self._event.is_set():
            self._event = asyncio.Event()
        return self._event

    def notify(self) -> None:
        loop = self._loop
        if loop is None:
            return

        def _fire() -> None:
            ev = self._event
            if ev is not None and not ev.is_set():
                ev.set()

        try:
            loop.call_soon_threadsafe(_fire)
        except RuntimeError:
            pass  # loop closed during shutdown; nothing to wake
