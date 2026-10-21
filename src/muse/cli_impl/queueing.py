"""Gateway-side request queueing primitives (spec 2026-07-08).

Both primitives keep all waiting ON the asyncio event loop -- never in
ThreadPoolExecutor threads (the #318/#319 invariant). The LoadDirector
stays synchronous; it only *fires* the CapacityNotifier from its worker
threads via call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import threading
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
    acquiring is a no-op (zero overhead).
    CPython asyncio.Semaphore wakes waiters FIFO, so fairness is free.
    Semaphores are created lazily. A LOWER cap takes effect live: the
    surplus permits are retired (free ones at once, held ones as their
    holders release), so no new permit is granted until the active count
    is under the new cap. A HIGHER cap applies on supervisor restart
    (documented in the spec).

    max_queue_depth (per call) is a ceiling on waiters beyond the cap,
    combined with server.max_queue_depth by taking the smaller positive
    value; a global 0 (unbounded) does not remove a per-call ceiling.
    """

    def __init__(self) -> None:
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._caps: dict[str, int] = {}
        self._entered: dict[str, int] = {}
        # Permits still to retire after a live cap decrease (see class doc).
        self._retire: dict[str, int] = {}
        # Captured on every acquire_slot so release_slot can
        # schedule its (loop-affine) decrement + semaphore release from a
        # non-loop thread via call_soon_threadsafe. asyncio primitives are NOT
        # threadsafe; the gateway relay / finally paths may call release_slot
        # off the event loop.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Guards every read AND mutation of _entered / _caps. depth()/depths()
        # are called from SYNC admin/telemetry route handlers, which Starlette
        # dispatches via run_in_threadpool -- a different OS thread than the
        # event loop thread that mutates _entered on every acquire/release.
        # Without this lock, depths() iterating _entered can race a
        # concurrent key-add/key-pop on the loop thread and raise
        # "dictionary changed size during iteration". A plain (non-reentrant)
        # Lock is enough: every critical section below is a small, await-free
        # dict read/write, so there is no risk of blocking the event loop or
        # of nested acquisition. asyncio.Semaphore operations stay OUTSIDE
        # the lock -- they are loop-internal already and don't need it.
        self._mutex = threading.Lock()

    def depth(self, model_id: str) -> int:
        with self._mutex:
            entered = self._entered.get(model_id, 0)
            cap = self._caps.get(model_id, 0)
            return max(0, entered - cap)

    def depths(self) -> dict[str, int]:
        with self._mutex:
            result = {}
            for model_id, entered in self._entered.items():
                cap = self._caps.get(model_id, 0)
                d = max(0, entered - cap)
                if d > 0:
                    result[model_id] = d
            return result

    def _sem(self, model_id: str, cap: int) -> asyncio.Semaphore:
        sem = self._sems.get(model_id)
        if sem is None:
            sem = asyncio.Semaphore(cap)
            self._sems[model_id] = sem
            with self._mutex:
                self._caps[model_id] = cap
            return sem
        with self._mutex:
            current = self._caps.get(model_id, cap)
            if cap < current:
                self._caps[model_id] = cap
                self._retire[model_id] = self._retire.get(model_id, 0) + current - cap
        return sem

    async def _retire_free_permits(self, model_id: str, sem: asyncio.Semaphore) -> None:
        """Withdraw free permits owed to a cap decrease.

        A free permit exists only when no waiter is queued (FIFO), so
        taking it cannot starve anyone, and acquire() on an unlocked
        semaphore returns without suspending (see _acquire_permit), so no
        other task can interleave with this loop.
        """
        while True:
            with self._mutex:
                owed = self._retire.get(model_id, 0)
            if owed <= 0 or sem.locked():
                return
            await sem.acquire()
            with self._mutex:
                self._set_retire_locked(model_id, self._retire.get(model_id, 0) - 1)

    def _set_retire_locked(self, model_id: str, owed: int) -> None:
        if owed > 0:
            self._retire[model_id] = owed
        else:
            self._retire.pop(model_id, None)

    def _check_and_enter_locked(
        self, model_id: str, cap: int, max_queue_depth: int | None = None,
    ) -> None:
        """Synchronous check-then-increment for one caller's entry.

        Caller MUST already hold self._mutex. No await anywhere in this
        method: the read of `entered` and the write to `self._entered` stay
        adjacent with nothing able to interleave, preserving the burst-
        arrival accuracy the check relies on (see _acquire_permit's
        docstring) while also being race-free against a cross-thread
        depth()/depths() reader.
        """
        entered = self._entered.get(model_id, 0)
        excess = entered - cap
        limits = [
            n for n in (config.get("server.max_queue_depth") or 0, max_queue_depth or 0)
            if n > 0
        ]
        if limits and excess >= min(limits):
            raise QueueFull(model_id, excess)
        self._entered[model_id] = entered + 1

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
        plus an empirical burst test).
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
    async def slot(
        self, model_id: str, cap: int | None, *, deadline: float,
        max_queue_depth: int | None = None,
    ):
        """acquire_slot/release_slot as a lexical async context."""
        held = await self.acquire_slot(
            model_id, cap, deadline=deadline, max_queue_depth=max_queue_depth,
        )
        try:
            yield
        finally:
            if held:
                self.release_slot(model_id)

    # The gateway holds a slot across a NON-lexical span: acquire happens in
    # _route_via_director, but release fires later from
    # _forward's buffered/stream finally (possibly OFF the event
    # loop), so it uses acquire_slot/release_slot directly.

    async def acquire_slot(
        self, model_id: str, cap: int | None, *, deadline: float,
        max_queue_depth: int | None = None,
    ) -> bool:
        """Acquire one concurrency slot (no context manager).

        QueueFull if the
        per-model queue is already at server.max_queue_depth, QueueTimeout
        if the deadline passes before a slot frees. cap None/<=0 is a no-op
        (unlimited): nothing is reserved and False is returned.

        Returns True when a slot was actually taken (entered counter
        incremented, semaphore HELD): the caller MUST pair a True return
        with exactly one release_slot, and MUST NOT release on False (#332:
        the cap is re-read live per-request from
        server.default_max_concurrency, so an uncapped request against a
        model whose semaphore survives from an earlier capped era would
        otherwise over-release a concurrent capped holder's slot).

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
            return False  # unlimited: no gating, nothing to release
        sem = self._sem(model_id, cap)
        await self._retire_free_permits(model_id, sem)
        # Synchronous compare-and-increment (no await between the read and the
        # increment), so a concurrent burst on one loop cannot slip past the
        # depth bound.
        with self._mutex:
            self._check_and_enter_locked(model_id, cap, max_queue_depth)
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
        return True

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
            # Mirror CapacityNotifier.notify's shutdown guard: a loop closed
            # mid-shutdown must not turn a release into a RuntimeError --
            # the slot dies with the loop.
            try:
                loop.call_soon_threadsafe(self._do_release, model_id)
            except RuntimeError:
                pass  # loop closed during shutdown; nothing to release into
        else:
            # No loop was ever captured (should not happen after acquire_slot):
            # best-effort direct release.
            self._do_release(model_id)

    def _do_release(self, model_id: str) -> None:
        """Loop-affine release body: drop entered + release the semaphore.

        Only acts when a slot is genuinely held (entered > 0); an extra call
        is dropped so the semaphore is never over-released and entered never
        goes negative. The guard check and the decrement happen inside one
        locked section so a concurrent depth()/depths() reader can never
        observe (or race) a half-applied release; sem.release() itself runs
        outside the lock (asyncio.Semaphore is loop-internal already, and
        this method itself only ever runs on the loop thread, so there is no
        cross-thread race on the release call).
        """
        sem = self._sems.get(model_id)
        if sem is None:
            return
        with self._mutex:
            if self._entered.get(model_id, 0) <= 0:
                return  # over-release guard: nothing held
            self._decr_entered_locked(model_id)
        self._release_permit(model_id, sem)

    def _release_permit(self, model_id: str, sem: asyncio.Semaphore) -> None:
        """Return a held permit, or retire it if a cap decrease is owed one."""
        with self._mutex:
            owed = self._retire.get(model_id, 0)
            if owed > 0:
                self._set_retire_locked(model_id, owed - 1)
                return
        sem.release()

    def _decr_entered_locked(self, model_id: str) -> None:
        """Decrement _entered, clamped at 0, popping empties.

        Caller MUST already hold self._mutex.
        """
        n = self._entered.get(model_id, 0)
        if n <= 0:
            self._entered.pop(model_id, None)
            return
        self._entered[model_id] = n - 1
        if self._entered[model_id] <= 0:
            self._entered.pop(model_id, None)

    def _decr_entered(self, model_id: str) -> None:
        """Decrement the entered counter, clamped at 0, popping empties."""
        with self._mutex:
            self._decr_entered_locked(model_id)


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
