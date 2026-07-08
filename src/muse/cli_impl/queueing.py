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
        self._caps: dict[str, int] = {}
        self._entered: dict[str, int] = {}
        # Captured on the first acquire_slot / slot entry so release_slot can
        # schedule its (loop-affine) decrement + semaphore release from a
        # non-loop thread via call_soon_threadsafe. asyncio primitives are NOT
        # threadsafe; the gateway relay / finally paths may call release_slot
        # off the event loop.
        self._loop: asyncio.AbstractEventLoop | None = None

    def depth(self, model_id: str) -> int:
        entered = self._entered.get(model_id, 0)
        cap = self._caps.get(model_id, 0)
        return max(0, entered - cap)

    def depths(self) -> dict[str, int]:
        result = {}
        for model_id in self._entered:
            d = self.depth(model_id)
            if d > 0:
                result[model_id] = d
        return result

    def _sem(self, model_id: str, cap: int) -> asyncio.Semaphore:
        sem = self._sems.get(model_id)
        if sem is None:
            sem = asyncio.Semaphore(cap)
            self._sems[model_id] = sem
            self._caps[model_id] = cap
        return sem

    async def _acquire_permit(
        self, sem: asyncio.Semaphore, model_id: str, deadline: float,
    ) -> None:
        """Acquire one permit from sem, honoring the wait deadline.

        A free permit is granted immediately regardless of the deadline: a
        zero/expired wait budget means "do not WAIT", not "do not TRY" --
        only a genuinely unavailable permit fails fast with QueueTimeout.
        This mirrors the capacity-wait path (_acquire_with_capacity_wait in
        gateway.py), which also attempts before it ever consults a deadline.

        sem.locked() reports live availability (free value, and no queued
        FIFO waiters ahead of us). On CPython, Semaphore.acquire() has a
        fast path -- `if not self.locked(): self._value -= 1; return True`
        -- with no `await` in between, so it cannot yield to another task
        on the same event loop. There is therefore no window between the
        locked() check and the grant for a concurrent same-tick caller to
        steal the permit (verified against 3.10's asyncio/locks.py source
        plus an empirical burst test; see task-3-report.md).
        """
        if not sem.locked():
            await sem.acquire()
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QueueTimeout(model_id)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=remaining)
        except asyncio.TimeoutError:
            raise QueueTimeout(model_id) from None

    @asynccontextmanager
    async def slot(self, model_id: str, cap: int | None, *, deadline: float):
        if not cap or cap <= 0:
            yield  # unlimited: no gating at all
            return
        sem = self._sem(model_id, cap)
        # Synchronous entry check: no await between the read and the
        # increment, so this is accurate even under a concurrent burst
        # of arrivals on one event loop (no interleaving is possible
        # between here and the increment below).
        entered = self._entered.get(model_id, 0)
        excess = entered - cap
        max_depth = config.get("server.max_queue_depth") or 0
        if max_depth > 0 and excess >= max_depth:
            raise QueueFull(model_id, excess)
        self._entered[model_id] = entered + 1
        try:
            await self._acquire_permit(sem, model_id, deadline)
            try:
                yield
            finally:
                sem.release()
        finally:
            self._entered[model_id] -= 1
            if self._entered[model_id] <= 0:
                self._entered.pop(model_id, None)

    # ------------------------------------------------------------------
    # Split acquire/release variant (used by the gateway request path).
    #
    # slot() is a self-contained async ctx manager; it is ideal when the
    # held region is a lexical block. The gateway instead holds the slot
    # across a NON-lexical span: acquire happens in _route_via_director,
    # but release fires later from _forward_with_release's buffered/stream
    # finally (possibly OFF the event loop). acquire_slot/release_slot expose
    # the same entry semantics as slot()'s __aenter__ without the held-region
    # context, and release_slot is threadsafe. slot() is untouched.
    # ------------------------------------------------------------------

    async def acquire_slot(
        self, model_id: str, cap: int | None, *, deadline: float,
    ) -> None:
        """Acquire one concurrency slot (no context manager).

        Same entry semantics as slot()'s __aenter__: QueueFull if the
        per-model queue is already at server.max_queue_depth, QueueTimeout
        if the deadline passes before a slot frees. cap None/<=0 is a no-op
        (unlimited): nothing is reserved and release_slot is a matching no-op.

        On success the entered counter is incremented and the semaphore is
        HELD; the caller MUST pair the call with exactly one release_slot.
        On a failed acquisition (QueueTimeout) the entered increment is undone
        before the exception surfaces, so the counter stays balanced;
        QueueFull raises before the increment. Any other exception (e.g. a
        CancelledError while awaiting the semaphore) also undoes the increment.
        """
        # Capture the loop for threadsafe release, always (even unlimited):
        # cheap and idempotent, and keeps release_slot correct if a later
        # capped request reuses this gate.
        self._loop = asyncio.get_running_loop()
        if not cap or cap <= 0:
            return  # unlimited: no gating, nothing to release
        sem = self._sem(model_id, cap)
        # Synchronous compare-and-increment (no await between the read and the
        # increment), so a concurrent burst on one loop cannot slip past the
        # depth bound -- mirrors slot()'s entry check exactly.
        entered = self._entered.get(model_id, 0)
        excess = entered - cap
        max_depth = config.get("server.max_queue_depth") or 0
        if max_depth > 0 and excess >= max_depth:
            raise QueueFull(model_id, excess)
        self._entered[model_id] = entered + 1
        try:
            await self._acquire_permit(sem, model_id, deadline)
        except BaseException:
            # Failed acquisition (timeout / cancellation): undo the entered
            # increment so the counter stays balanced. The semaphore was NOT
            # acquired on any of these paths, so nothing to release.
            self._decr_entered(model_id)
            raise
        # Success: entered stays incremented and the semaphore is held until
        # release_slot decrements + releases.

    def release_slot(self, model_id: str) -> None:
        """Release one held concurrency slot. Threadsafe + over-release-safe.

        Decrements the entered counter and releases the semaphore for the
        matching acquire_slot. Safe to call from a non-loop thread: the
        loop-affine mutation is scheduled on the captured loop via
        call_soon_threadsafe. A no-op for a model with no semaphore
        (unlimited cap, or nothing ever acquired). Guarded against
        over-release: an extra call while nothing is held is dropped rather
        than driving the entered counter negative or over-releasing the
        semaphore. Exactly-once pairing is the caller's contract (the gateway
        guards with a slot_released flag); this guard is belt-and-suspenders.
        """
        if self._sems.get(model_id) is None:
            return  # unlimited / never acquired: nothing to release
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and (loop is None or running is loop):
            self._do_release(model_id)
        elif loop is not None:
            loop.call_soon_threadsafe(self._do_release, model_id)
        else:
            # No loop was ever captured (should not happen after acquire_slot):
            # best-effort direct release.
            self._do_release(model_id)

    def _do_release(self, model_id: str) -> None:
        """Loop-affine release body: drop entered + release the semaphore.

        Only acts when a slot is genuinely held (entered > 0); an extra call
        is dropped so the semaphore is never over-released and entered never
        goes negative.
        """
        sem = self._sems.get(model_id)
        if sem is None:
            return
        if self._entered.get(model_id, 0) <= 0:
            return  # over-release guard: nothing held
        self._decr_entered(model_id)
        sem.release()

    def _decr_entered(self, model_id: str) -> None:
        """Decrement the entered counter, clamped at 0, popping empties."""
        n = self._entered.get(model_id, 0)
        if n <= 0:
            self._entered.pop(model_id, None)
            return
        self._entered[model_id] = n - 1
        if self._entered[model_id] <= 0:
            self._entered.pop(model_id, None)


class CapacityNotifier:
    """Generation-event capacity broadcast, threadsafe on the fire side.

    Waiter protocol (missed-wakeup-free):
      1. ev = notifier.snapshot()   BEFORE the acquire attempt
      2. attempt director.acquire; on retryable capacity failure:
      3. await ev.wait() (bounded), then loop back to 1.

    notify() may be called from any thread (the director's release /
    eviction paths). It only sets() the CURRENT event; it does not itself
    swap in a fresh one. The generation rollover happens lazily inside
    snapshot(): the next caller to snapshot() sees an already-set event
    and replaces it with a fresh, unset one before handing it out. This
    keeps notify() a cheap threadsafe fire-and-forget and still guarantees
    every waiter that snapshotted before a notify() gets woken, while a
    waiter that snapshots AFTER a notify() gets a new, unset event rather
    than one that is already (stale-)set. Before the first snapshot()
    there is no captured loop; notify() is a silent no-op (nothing is
    waiting yet).
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
