# Request Queueing Implementation Plan (v0.55.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-model concurrency caps (busy models queue excess requests FIFO) plus bounded capacity-wait (a cold model that cannot fit because loaded models are in-use waits for a release instead of 503-ing), per `docs/superpowers/specs/2026-07-08-request-queueing-design.md`.

**Architecture:** All waiting happens ON the gateway's asyncio event loop (asyncio.Semaphore / asyncio.Event); the synchronous LoadDirector gains only a `retryable` flag on its capacity error and a fire-and-forget capacity-freed notifier. A new `muse/cli_impl/queueing.py` module owns both primitives.

**Tech Stack:** Python 3.10+, asyncio, FastAPI/Starlette, pytest + pytest-asyncio (already configured, `asyncio_mode = "auto"`).

## Global Constraints

- ASCII only in all code/comments/commits; NO em-dash (a hook rejects it).
- Waiters must NEVER park in ThreadPoolExecutor threads; all waiting uses asyncio primitives on the loop (the #318/#319 invariant).
- Defaults preserve today's behavior exactly: no `capabilities.max_concurrency` and default config = no gating; `server.queue_timeout_seconds: 0` = today's immediate capacity-503.
- Every release (director refcount AND gate slot) fires on every path including `BaseException` (cancellation).
- Errors use the OpenAI envelope via the gateway's existing `_openai_error(status, code, message, error_type=...)`; new codes: `queue_timeout`, `queue_full`.
- New settings are rows in `muse.core.config.SETTINGS` (no parallel lookup tables); lenient reads on the request path.
- Commit trailer on every commit:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01X2M12d5tRULNFUHFW2Hijx`
- Work on branch `feature/request-queueing` off current main. Fast lane (`python -m pytest tests/ -q -m "not slow"`, baseline ~3800 passed) must stay green; run it with `MUSE_CONFIG=$(mktemp) MUSE_CATALOG_DIR=$(mktemp -d)` to dodge this box's local admin.token / live server (and ignore the 4 `test_config_cli.py` failures those overrides themselves cause).

## File Structure

| File | Responsibility |
|---|---|
| Create `src/muse/cli_impl/queueing.py` | `ConcurrencyGate` (per-model asyncio.Semaphore + waiter counts + depth bound) and `CapacityNotifier` (generation-event, threadsafe fire) |
| Modify `src/muse/admin/operations.py:108` | `OperationError` gains `retryable: bool = False` |
| Modify `src/muse/cli_impl/load_director.py` | `capacity_listener` slot; notify on release-to-zero + post-eviction; tag capacity 503 retryable |
| Modify `src/muse/cli_impl/supervisor.py:81-150` | `SupervisorState` gains `concurrency_gate` + `capacity_notifier` fields; wire listener at director construction |
| Modify `src/muse/cli_impl/gateway.py:539-720` | deadline, gate acquire, capacity-wait retry loop, gate release threading, `queued_ms` telemetry |
| Modify `src/muse/core/config.py` | 3 new Setting rows |
| Modify `src/muse/observability/events.py` + `store.py` | `queued_ms` column + generic missing-column migration |
| Modify `src/muse/admin/routes/memory.py` + `src/muse/observability/dashboard.py` | per-model `queue_depth` |
| Tests | `tests/cli_impl/test_queueing.py` (new), `tests/cli_impl/test_queueing_gateway.py` (new), additions to `tests/cli_impl/test_load_director.py`, `tests/admin/routes/test_memory_routes.py`, `tests/observability/test_store.py` |

---

### Task 1: queueing primitives (`ConcurrencyGate` + `CapacityNotifier`)

**Files:**
- Create: `src/muse/cli_impl/queueing.py`
- Create: `tests/cli_impl/test_queueing.py`
- Modify: `src/muse/core/config.py` (3 Setting rows, in the "server" group next to `server.shutdown_grace_seconds`)
- Modify: `docs/CONFIG.md` (3 rows in the server table)

**Interfaces:**
- Produces: `ConcurrencyGate` with `async slot(model_id: str, cap: int | None) -> AsyncIterator[None]` (async context manager; `cap` None/<=0 means unlimited = no-op), `depth(model_id: str) -> int`, `depths() -> dict[str, int]`, and exceptions `QueueFull`, `QueueTimeout` (both `Exception` subclasses carrying `model_id`). The gate reads `server.max_queue_depth` and the caller passes a per-call `deadline: float` (monotonic) via `slot(..., deadline=...)`.
- Produces: `CapacityNotifier` with `snapshot() -> asyncio.Event` (arm BEFORE an acquire attempt; lazily captures the running loop) and `notify() -> None` (threadsafe, callable from director threads; no-op before first snapshot).
- Produces: config keys `server.default_max_concurrency` (int, 0), `server.queue_timeout_seconds` (float, 300.0), `server.max_queue_depth` (int, 0).

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli_impl/test_queueing.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/cli_impl/test_queueing.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'muse.cli_impl.queueing'`

- [ ] **Step 3: Implement `queueing.py` + config rows**

```python
# src/muse/cli_impl/queueing.py
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
```

Config rows (insert in `SETTINGS` right after the `server.shutdown_grace_seconds` row in `src/muse/core/config.py`):

```python
    Setting("server.default_max_concurrency", "MUSE_DEFAULT_MAX_CONCURRENCY",
            "int", 0, "server",
            "Default per-model concurrent-request cap for models without "
            "capabilities.max_concurrency; 0 = unlimited."),
    Setting("server.queue_timeout_seconds", "MUSE_QUEUE_TIMEOUT_SECONDS",
            "float", 300.0, "server",
            "Max seconds a request is held waiting for a concurrency slot "
            "and/or capacity before a 503 queue_timeout; 0 disables waiting."),
    Setting("server.max_queue_depth", "MUSE_MAX_QUEUE_DEPTH",
            "int", 0, "server",
            "Per-model bound on parked waiters; exceeded requests fail fast "
            "503 queue_full; 0 = unbounded."),
```

Add the same three to the server table in `docs/CONFIG.md` (mirror the wording above).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/cli_impl/test_queueing.py tests/cli_impl/test_config_cli.py -q`
Expected: PASS (queueing tests green; config CLI still green since rows are additive)

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/queueing.py tests/cli_impl/test_queueing.py src/muse/core/config.py docs/CONFIG.md
git commit -m "feat(queueing): ConcurrencyGate + CapacityNotifier primitives + config rows"
```

---

### Task 2: director capacity signal (`retryable` flag + notifier hooks)

**Files:**
- Modify: `src/muse/admin/operations.py:108` (OperationError.__init__)
- Modify: `src/muse/cli_impl/load_director.py` (`__init__`, `release`, `_evict_lru_until_fits`)
- Test: `tests/cli_impl/test_load_director.py` (append a new class)

**Interfaces:**
- Consumes: nothing new.
- Produces: `OperationError(code, message, status=400, retryable=False)` with `.retryable: bool`; `LoadDirector.capacity_listener: Callable[[], None] | None` attribute (default None), fired (a) in `release()` when a refcount drops to 0 and (b) at the successful end of `_evict_lru_until_fits`; the `model_too_large_for_device` raise in `_evict_lru_until_fits` carries `retryable=True` iff any loaded entry in the SAME pool has refcount > 0 (in-use models will free capacity later), else `retryable=False`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli_impl/test_load_director.py` (reuse that file's existing fixtures/fakes for probes; the class below builds its own director with the same fake-probe pattern used by `TestRecentDecisions` in `tests/admin/routes/test_memory_routes.py`):

```python
class TestCapacitySignal:
    """Spec 2026-07-08: retryable flag + capacity_listener hooks."""

    def _director(self, *, gpu_free=1.0):
        from muse.cli_impl.load_director import LoadDirector
        return LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: gpu_free),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )

    def _entry(self, model_id, refcount):
        import time
        from muse.cli_impl.load_director import LoadEntry
        now = time.monotonic()
        return LoadEntry(model_id=model_id, worker_port=9001, memory_gb=4.0,
                         refcount=refcount, last_touched_at=now, loaded_at=now)

    def test_release_to_zero_fires_listener(self):
        d = self._director()
        fired = []
        d.capacity_listener = lambda: fired.append(1)
        d.loaded["m"] = self._entry("m", refcount=1)
        d.release("m")
        assert fired == [1]

    def test_release_above_zero_does_not_fire(self):
        d = self._director()
        fired = []
        d.capacity_listener = lambda: fired.append(1)
        d.loaded["m"] = self._entry("m", refcount=2)
        d.release("m")
        assert fired == []

    def test_listener_exception_swallowed(self):
        d = self._director()
        def boom():
            raise RuntimeError("listener broke")
        d.capacity_listener = boom
        d.loaded["m"] = self._entry("m", refcount=1)
        d.release("m")  # must not raise
        assert d.loaded["m"].refcount == 0

    def test_capacity_503_retryable_when_inuse_models_block(self):
        from muse.admin.operations import OperationError
        d = self._director(gpu_free=1.0)  # 1 GB free, need 8
        d.loaded["busy"] = self._entry("busy", refcount=1)  # in-use: not evictable
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda", required_gb=8.0,
            )
        assert exc.value.retryable is True

    def test_capacity_503_not_retryable_when_nothing_will_free(self):
        from muse.admin.operations import OperationError
        d = self._director(gpu_free=1.0)
        # No loaded models at all: nothing will ever release capacity.
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda", required_gb=8.0,
            )
        assert exc.value.retryable is False

    def test_operation_error_default_not_retryable(self):
        from muse.admin.operations import OperationError
        assert OperationError("x", "y").retryable is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/cli_impl/test_load_director.py::TestCapacitySignal -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'retryable'` / `AttributeError: capacity_listener` / assertion errors)

- [ ] **Step 3: Implement**

`src/muse/admin/operations.py` -- extend `OperationError.__init__`:

```python
    def __init__(self, code: str, message: str, status: int = 400,
                 retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        # Spec 2026-07-08: True marks a TRANSIENT capacity failure (in-use
        # models will release memory); the gateway parks and retries these
        # instead of surfacing the 503. False (default) = surface as today.
        self.retryable = retryable
```

`src/muse/cli_impl/load_director.py`:

1. In `__init__`, after `self._inflight_epoch = 0` add:

```python
        # Fired (fire-and-forget) whenever capacity MAY have freed: a
        # release() that drops a refcount to 0, or a completed eviction.
        # The supervisor wires this to CapacityNotifier.notify so gateway
        # capacity-waiters wake and re-decide. Never allowed to raise into
        # the caller (wrapped at each call site).
        self.capacity_listener: Callable[[], None] | None = None
```

2. Add a private helper next to `release`:

```python
    def _fire_capacity_listener(self) -> None:
        listener = self.capacity_listener
        if listener is None:
            return
        try:
            listener()
        except Exception:  # noqa: BLE001
            logger.warning("capacity_listener raised; ignoring", exc_info=True)
```

3. In `release()`, capture whether the refcount dropped to zero and fire AFTER the lock is released (never call foreign code under `self.lock`):

```python
    def release(self, model_id: str) -> None:
        dropped_to_zero = False
        with self.lock:
            entry = self.loaded.get(model_id)
            if entry is None:
                logger.debug("release(%r): model not in loaded set; ignoring", model_id)
                return
            before = entry.refcount
            entry.refcount = max(0, entry.refcount - 1)
            entry.last_touched_at = time.monotonic()
            dropped_to_zero = before > 0 and entry.refcount == 0
        if dropped_to_zero:
            self._fire_capacity_listener()
```

(keep the existing docstring; add one line noting the listener fire.)

4. In `_evict_lru_until_fits`, at the `return` points that mean "eviction freed enough / model now fits" add `self._fire_capacity_listener()` immediately before returning (there are two: the re-check `return` at the no-candidates branch and the normal loop-exit when the shortfall is covered; grep for `return` inside that method and add the fire before each). At the `raise OperationError("model_too_large_for_device", ...)` site, compute retryable under the lock snapshot already taken there:

```python
                    pool = self._resolve_pool_device(device)
                    any_inuse = any(
                        e.refcount > 0
                        and self._resolve_pool_device(
                            # LoadEntry has no device field; in-use entries in
                            # the LOADED set share the pool being contended
                            # here by construction of the candidates filter.
                            device,
                        ) == pool
                        for e in self.loaded.values()
                    )
                    raise OperationError(
                        "model_too_large_for_device",
                        ( ...existing message unchanged... ),
                        status=503,
                        retryable=any_inuse,
                    )
```

NOTE to implementer: the existing loaded-set is not pool-partitioned on the entry, so the practical rule is `any(e.refcount > 0 for e in self.loaded.values())` -- if ANY in-use model exists, capacity may free. Use that simpler form (drop the pool comparison shown above); single-pool boxes (both deployment targets) make it exact, and on a mixed box the cost of a spurious retry is one cheap `_decide` pass.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/cli_impl/test_load_director.py tests/cli_impl/test_load_director_concurrency.py tests/cli_impl/test_concurrency_fixes.py -q`
Expected: PASS (new class green; existing director suites unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/muse/admin/operations.py src/muse/cli_impl/load_director.py tests/cli_impl/test_load_director.py
git commit -m "feat(director): retryable capacity errors + capacity_listener hooks"
```

---

### Task 3: gateway integration (deadline + gate + capacity-wait + telemetry)

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py` (SupervisorState fields + wiring in `run_supervisor`)
- Modify: `src/muse/cli_impl/gateway.py` (`_route_via_director`, `_forward_with_release` signature)
- Modify: `src/muse/observability/events.py` (+ `queued_ms` column) and `src/muse/observability/store.py` (schema + migration)
- Create: `tests/cli_impl/test_queueing_gateway.py`
- Test (extend): `tests/observability/test_store.py`

**Interfaces:**
- Consumes: `ConcurrencyGate`, `CapacityNotifier`, `QueueFull`, `QueueTimeout` from Task 1; `OperationError.retryable` + `capacity_listener` from Task 2.
- Produces: `SupervisorState.concurrency_gate: ConcurrencyGate` and `SupervisorState.capacity_notifier: CapacityNotifier` (both `field(default_factory=...)`); `_forward_with_release(..., extra_release: Callable[[], None] | None = None)`; telemetry `request` events carry `queued_ms: float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli_impl/test_queueing_gateway.py
"""Gateway queueing integration (spec 2026-07-08).

Uses the same fake-director + TestClient style as test_gateway_lazy.py:
a real FastAPI gateway app over a FakeDirector and a stub worker, no
subprocesses. Focus: cap resolution from manifest/config, queue_timeout
and queue_full envelopes, capacity-wait retry on retryable 503s, and
release pairing on success/failure.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from muse.admin.operations import OperationError
from muse.cli_impl.queueing import ConcurrencyGate, CapacityNotifier


def _effective_cap_for(manifest: dict) -> int | None:
    from muse.cli_impl.gateway import _effective_max_concurrency
    return _effective_max_concurrency(manifest)


class TestEffectiveCap:
    def test_manifest_cap_wins(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEFAULT_MAX_CONCURRENCY", "4")
        assert _effective_cap_for(
            {"capabilities": {"max_concurrency": 1}}) == 1

    def test_config_default_when_undeclared(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEFAULT_MAX_CONCURRENCY", "4")
        assert _effective_cap_for({"capabilities": {}}) == 4

    def test_unlimited_when_neither(self, monkeypatch):
        monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)
        assert _effective_cap_for({"capabilities": {}}) is None

    def test_bad_manifest_value_falls_back(self, monkeypatch):
        monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)
        assert _effective_cap_for(
            {"capabilities": {"max_concurrency": "not-an-int"}}) is None


class TestCapacityWaitRetry:
    """_acquire_with_capacity_wait: park on retryable 503, retry after
    notify, propagate non-retryable immediately, 503 on deadline."""

    async def test_retryable_then_success_after_notify(self):
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()
        calls = []

        async def fake_acquire():
            calls.append(1)
            if len(calls) == 1:
                raise OperationError("model_too_large_for_device",
                                     "full", status=503, retryable=True)
            return 9001

        async def free_capacity_soon():
            await asyncio.sleep(0.05)
            notifier.notify()

        asyncio.create_task(free_capacity_soon())
        port = await _acquire_with_capacity_wait(
            fake_acquire, notifier, deadline=time.monotonic() + 5,
            model_id="m",
        )
        assert port == 9001 and len(calls) == 2

    async def test_non_retryable_propagates_immediately(self):
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()

        async def fake_acquire():
            raise OperationError("model_too_large_for_device",
                                 "impossible", status=503, retryable=False)

        with pytest.raises(OperationError) as exc:
            await _acquire_with_capacity_wait(
                fake_acquire, notifier, deadline=time.monotonic() + 5,
                model_id="m",
            )
        assert exc.value.retryable is False

    async def test_deadline_exhaustion_raises_queue_timeout(self):
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        from muse.cli_impl.queueing import QueueTimeout
        notifier = CapacityNotifier()

        async def fake_acquire():
            raise OperationError("model_too_large_for_device",
                                 "full", status=503, retryable=True)

        with pytest.raises(QueueTimeout):
            await _acquire_with_capacity_wait(
                fake_acquire, notifier, deadline=time.monotonic() + 0.1,
                model_id="m",
            )

    async def test_zero_timeout_degrades_to_immediate_503(self):
        """queue_timeout_seconds=0 -> deadline already passed -> the
        retryable 503 surfaces as-is (today's behavior)."""
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()

        async def fake_acquire():
            raise OperationError("model_too_large_for_device",
                                 "full", status=503, retryable=True)

        with pytest.raises(OperationError):
            await _acquire_with_capacity_wait(
                fake_acquire, notifier, deadline=time.monotonic(),  # now
                model_id="m",
            )


class TestQueuedMsColumn:
    def test_event_to_row_accepts_queued_ms(self):
        from muse.observability.events import event_to_row
        row = event_to_row("request", 1.0, queued_ms=42.0)
        assert row["queued_ms"] == 42.0

    def test_store_migrates_missing_column(self, tmp_path):
        """A pre-v0.55 telemetry.db (no queued_ms column) must be migrated
        in place by TelemetryStore so insert_many with the new column works."""
        import sqlite3
        from muse.observability.store import TelemetryStore
        from muse.observability.events import event_to_row
        db = tmp_path / "telemetry.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE events (ts REAL NOT NULL, type TEXT NOT NULL, "
            "model_id TEXT)")  # ancient schema: most columns missing
        conn.commit(); conn.close()
        store = TelemetryStore(db)
        store.insert_many([event_to_row("request", 1.0, queued_ms=5.0)])
        assert store.count() == 1
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/cli_impl/test_queueing_gateway.py -q`
Expected: FAIL (`ImportError: cannot import name '_effective_max_concurrency'` etc.)

- [ ] **Step 3: Implement**

3a. `src/muse/observability/events.py`: add `"queued_ms"` to `EVENT_COLUMNS` (after `"latency_ms"`).

3b. `src/muse/observability/store.py`: add `queued_ms REAL,` to the `CREATE TABLE` DDL (same position), and in `__init__` after the executescript add a generic migration so pre-existing DBs gain any missing columns:

```python
        # Migrate older DBs in place: add any EVENT_COLUMNS the existing
        # table lacks (new columns are always nullable in the sparse model,
        # so ALTER TABLE ADD COLUMN is safe and idempotent).
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        for col in EVENT_COLUMNS:
            if col not in have:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {col}")
        self._conn.commit()
```

(import `EVENT_COLUMNS` from `.events`; it is already used by `insert_many`'s row shape.)

3c. `src/muse/cli_impl/supervisor.py` -- `SupervisorState` gains two fields next to `cold_load_gates` (line ~150), with deferred import inside `field(default_factory=...)` lambdas NOT needed since `queueing.py` imports only stdlib + config (safe at module top):

```python
    concurrency_gate: "ConcurrencyGate" = field(default_factory=_new_gate)
    capacity_notifier: "CapacityNotifier" = field(default_factory=_new_notifier)
```

with module-level factories (keeps dataclass defaults picklable/clean):

```python
def _new_gate():
    from muse.cli_impl.queueing import ConcurrencyGate
    return ConcurrencyGate()


def _new_notifier():
    from muse.cli_impl.queueing import CapacityNotifier
    return CapacityNotifier()
```

In `run_supervisor`, right after the director is constructed and attached to state, wire the listener:

```python
    state.director.capacity_listener = state.capacity_notifier.notify
```

3d. `src/muse/cli_impl/gateway.py`:

Add module-level helpers near `_acquire_off_loop`:

```python
def _effective_max_concurrency(manifest: dict) -> int | None:
    """Per-model cap: capabilities.max_concurrency, else the config
    default, else None (unlimited). Lenient: junk values -> next tier."""
    caps = (manifest or {}).get("capabilities") or {}
    declared = caps.get("max_concurrency")
    if declared is not None:
        try:
            n = int(declared)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    default = config.get("server.default_max_concurrency") or 0
    try:
        return int(default) if int(default) > 0 else None
    except (TypeError, ValueError):
        return None


async def _acquire_with_capacity_wait(acquire_once, notifier, *, deadline,
                                      model_id: str) -> int:
    """Bounded retry around one acquire attempt.

    `acquire_once` is a zero-arg async callable (the coalesced acquire).
    On a retryable capacity OperationError: park on the notifier's
    generation event (armed BEFORE the attempt, so a release that lands
    mid-attempt still wakes us), bounded by the shared deadline, then
    retry. Non-retryable errors and every other exception propagate
    unchanged. Deadline exhaustion raises QueueTimeout.
    """
    from muse.cli_impl.queueing import QueueTimeout
    while True:
        event = notifier.snapshot()  # arm first: no missed wakeup
        try:
            return await acquire_once()
        except OperationError as exc:
            if not getattr(exc, "retryable", False):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise  # zero-budget: surface today's immediate 503
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise QueueTimeout(model_id) from None
```

Rework the middle of `_route_via_director` (currently step 3, the `worker_port = await _acquire_coalesced(...)` block at ~line 663). Replace with:

```python
    # 3. Queueing (spec 2026-07-08). One deadline covers gate-wait +
    # capacity-wait + retries. queue_timeout_seconds == 0 degrades to
    # today's no-wait behavior: an occupied slot / capacity 503 surfaces
    # immediately.
    queue_budget = config.get("server.queue_timeout_seconds") or 0.0
    deadline = time.monotonic() + max(0.0, float(queue_budget))
    gate = state.concurrency_gate
    cap = _effective_max_concurrency(manifest)
    queued_t0 = time.monotonic()

    slot_cm = gate.slot(model_id, cap, deadline=deadline)
    try:
        await slot_cm.__aenter__()
    except QueueTimeout:
        return _openai_error(
            503, "queue_timeout",
            f"waited {queue_budget:.0f}s for model {model_id!r} "
            f"(queue depth {gate.depth(model_id)})",
            error_type="server_error",
        )
    except QueueFull as exc:
        return _openai_error(
            503, "queue_full",
            f"queue for model {model_id!r} is full (depth {exc.depth})",
            error_type="server_error",
        )

    slot_released = False

    def _release_slot() -> None:
        nonlocal slot_released
        if slot_released:
            return
        slot_released = True
        # __aexit__ of the slot context releases the semaphore; it never
        # raises for a held slot. Schedule on the loop if called from a
        # non-loop thread (release paths run in the relay/finally).
        coro = slot_cm.__aexit__(None, None, None)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run_coroutine_threadsafe(coro, state.gateway_loop)

    try:
        worker_port = await _acquire_with_capacity_wait(
            lambda: _acquire_coalesced(state, model_id, manifest),
            state.capacity_notifier, deadline=deadline, model_id=model_id,
        )
    except QueueTimeout:
        _release_slot()
        return _openai_error(
            503, "queue_timeout",
            f"waited {queue_budget:.0f}s for capacity for model {model_id!r}",
            error_type="server_error",
        )
    except OperationError as exc:
        _release_slot()
        return _openai_error(
            exc.status, exc.code, exc.message,
            error_type=error_type_for_status(exc.status),
        )
    except Exception as exc:  # noqa: BLE001
        _release_slot()
        logger.error(
            "director.acquire(%r) raised an unexpected error",
            model_id, exc_info=True,
        )
        return _openai_error(
            503, "model_load_failed",
            f"load of {model_id!r} failed",
            error_type="server_error",
        )
    queued_ms = (time.monotonic() - queued_t0) * 1000.0
```

IMPLEMENTER NOTES for this block:
- `state.gateway_loop`: capture it once at app startup. In `build_gateway`, add a startup hook: `@app.on_event("startup")` (or the lifespan if one exists) doing `state.gateway_loop = asyncio.get_running_loop()`. If neither exists in the file, add `app.add_event_handler("startup", _capture_loop)` with `def _capture_loop(): state.gateway_loop = asyncio.get_event_loop()` -- follow whichever startup pattern `build_gateway` already uses; TestClient triggers startup handlers when used as a context manager, and plain calls still work because `_release_slot` first tries `get_running_loop()` (the common path: release fires inside the loop).
- Simplification permitted: since the gate's semaphore is a plain asyncio object, `slot_cm.__aexit__` only calls `sem.release()`; an equivalent and simpler `_release_slot` may call a new synchronous `gate.release_slot(model_id, cap)` method added to ConcurrencyGate that does `self._sems[model_id].release()` guarded by `cap and cap > 0`, with `loop.call_soon_threadsafe` when off-loop. If you take this route, add the method + a unit test in test_queueing.py and skip the __aenter__/__aexit__ juggling: acquire via `await gate.acquire_slot(model_id, cap, deadline=...)` and release via `gate.release_slot_threadsafe(...)`. EITHER shape is acceptable; the invariants that matter are: exactly-once release on all paths, off-loop-callable, FIFO, FakeException-wide.

Then thread the release into the forward leg. `_forward_with_release` gains a parameter:

```python
async def _forward_with_release(request, target_url, timeout, *, director,
                                model_id, extra_release=None):
```

and every place it currently calls `director.release(model_id)` (buffered finally, stream-relay finally, body-read except, stream-open except -- there are four; grep `director.release` within the function) additionally calls, immediately after and guarded:

```python
            if extra_release is not None:
                try:
                    extra_release()
                except Exception:  # noqa: BLE001
                    logger.warning("gate release failed", exc_info=True)
```

The call site in `_route_via_director` passes `extra_release=_release_slot`.

Finally telemetry: the existing `record("request", ...)` call at ~line 708 gains `queued_ms=queued_ms` (define `queued_ms = 0.0` on any early path that skips the queue block, or compute before the try as shown).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/cli_impl/test_queueing_gateway.py tests/cli_impl/test_gateway.py tests/cli_impl/test_gateway_lazy.py tests/cli_impl/test_concurrency_fixes.py tests/observability/ -q`
Expected: PASS (new integration tests green, all existing gateway/observability suites unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/gateway.py src/muse/cli_impl/supervisor.py src/muse/observability/events.py src/muse/observability/store.py tests/cli_impl/test_queueing_gateway.py
git commit -m "feat(gateway): per-model concurrency gate + bounded capacity-wait + queued_ms telemetry"
```

---

### Task 4: queue_depth observability (`/v1/admin/memory` + `/v1/telemetry/summary`)

**Files:**
- Modify: `src/muse/admin/routes/memory.py` (`_per_model_breakdown`)
- Modify: `src/muse/observability/dashboard.py` (`summary`)
- Test: `tests/admin/routes/test_memory_routes.py`, `tests/observability/test_dashboard_router.py` (or the existing dashboard test file; locate with `grep -rln "telemetry/summary" tests/`)

**Interfaces:**
- Consumes: `SupervisorState.concurrency_gate.depths()` / `.depth(model_id)` from Tasks 1+3.
- Produces: per-model breakdown records gain optional `queue_depth: int` (omitted when 0 AND no gate bound, present when a gate exists -- mirror the refcount omit-vs-0 discipline: present with value 0 when the gate exists, since 0 is meaningful there); summary `loaded[]` entries gain `queue_depth`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/admin/routes/test_memory_routes.py` inside `TestMemoryRoute` (reuses that file's `_seed_catalog`, `WorkerSpec`, `SupervisorState` imports):

```python
    def test_per_model_breakdown_includes_queue_depth(self, tmp_catalog):
        from muse.cli_impl.queueing import ConcurrencyGate
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(models=["kokoro-82m"],
                          python_path="/v/bin/python", port=9001)
        state = SupervisorState(workers=[spec], device="cpu")
        state.concurrency_gate = ConcurrencyGate()
        state.concurrency_gate._waiting["kokoro-82m"] = 3  # simulate parked waiters
        set_supervisor_state(state)
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert out[0]["queue_depth"] == 3
```

And a summary test in the dashboard test file (same style as its existing summary tests; find the fixture that builds the app):

```python
    def test_summary_includes_queue_depth(self, ...existing fixtures...):
        # state.concurrency_gate._waiting["m"] = 2 before the request, then:
        # r = client.get("/v1/telemetry/summary", headers=auth_headers)
        # entry = [e for e in r.json()["loaded"] if e["model_id"] == "m"][0]
        # assert entry["queue_depth"] == 2
```

(Implementer: adapt to the file's real fixtures; the assertion shape above is the contract.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/admin/routes/test_memory_routes.py -q -k queue_depth`
Expected: FAIL with `KeyError: 'queue_depth'`

- [ ] **Step 3: Implement**

`src/muse/admin/routes/memory.py` -- next to `_loaded_refcounts()` add:

```python
def _queue_depths() -> dict[str, int] | None:
    """Live per-model parked-waiter counts from the gateway's gate.

    None when no gate is bound (pre-boot / isolation) -> key omitted,
    mirroring the refcount discipline: unknown is never rendered as 0.
    """
    state = get_supervisor_state()
    gate = getattr(state, "concurrency_gate", None)
    if gate is None:
        return None
    return gate.depths()
```

In `_per_model_breakdown`, alongside the refcounts read add `depths = _queue_depths()`, and in the record loop after the refcount line:

```python
        if depths is not None:
            record["queue_depth"] = depths.get(model_id, 0)
```

`src/muse/observability/dashboard.py` -- `summary()` passes depths into `_loaded_entry_dict`: change the helper signature to `_loaded_entry_dict(model_id, entry, queue_depth=None)` adding `"queue_depth": queue_depth` to the dict, and in `summary()`:

```python
        gate = getattr(state, "concurrency_gate", None)
        depths = gate.depths() if gate is not None else {}
        loaded = [
            _loaded_entry_dict(model_id, entry,
                               queue_depth=depths.get(model_id, 0))
            for model_id, entry in director.loaded.items()
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/admin/ tests/observability/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/admin/routes/memory.py src/muse/observability/dashboard.py tests/admin/routes/test_memory_routes.py tests/observability/
git commit -m "feat(observability): per-model queue_depth in admin memory + telemetry summary"
```

---

### Task 5: slow e2e, docs, release prep

**Files:**
- Test: `tests/cli_impl/test_e2e_queueing.py` (new, `@pytest.mark.slow`)
- Modify: `CLAUDE.md` (new "Request queueing" subsection under the Lazy load section), `docs/CONFIG.md` (verify Task 1 rows landed), `pyproject.toml` (version 0.55.0)

**Interfaces:** consumes everything above; produces the release candidate.

- [ ] **Step 1: Write the slow e2e test**

```python
# tests/cli_impl/test_e2e_queueing.py
"""Slow e2e: a max_concurrency=1 model serializes concurrent requests.

In-process style (mirrors tests/cli_impl/test_e2e_supervisor.py /
test_gateway_lazy.py fakes): a real gateway app + real ConcurrencyGate +
FakeDirector whose worker is a stub HTTP server that sleeps 0.3s per
request and records (start, end) timestamps. Four concurrent requests to
the capped model must all return 200 with non-overlapping service
windows; total wall time ~1.2s (serialized), not ~0.3s (parallel).
"""
from __future__ import annotations

import threading
import time

import pytest

pytestmark = pytest.mark.slow


def test_cap_one_serializes_four_concurrent_requests(...):
    # Implementer: build on the fixture set in test_gateway_lazy.py
    # (fake director + stub worker via httpx MockTransport or a local
    # http.server). Manifest: {"capabilities": {"max_concurrency": 1,
    # "memory_gb": 0.1}}. Fire 4 requests from 4 threads via
    # TestClient; collect worker-side (start, end) pairs; assert:
    #   all(r.status_code == 200 for r in responses)
    #   windows sorted by start are pairwise non-overlapping
    #   len(windows) == 4
    ...
```

(Implementer: this is the one test in the plan whose scaffolding depends
on the existing fixture shapes in `test_gateway_lazy.py` -- reuse them
rather than inventing new ones; the assertions above are the contract.)

- [ ] **Step 2: Run it RED, implement fixture glue, run GREEN**

Run: `python -m pytest tests/cli_impl/test_e2e_queueing.py -q` (no `-m` filter; slow runs explicitly)
Expected: PASS after glue

- [ ] **Step 3: Docs**

CLAUDE.md, after the "Idle eviction" subsection, add:

```markdown
### Request queueing (v0.55.0+)

Two gateway-side mechanisms (spec docs/superpowers/specs/2026-07-08-request-queueing-design.md), both waiting ON the event loop (never in pool threads):

- **Per-model concurrency cap:** `capabilities.max_concurrency` (else `server.default_max_concurrency`, default 0 = unlimited = no gating). Excess requests park FIFO on a per-model asyncio.Semaphore in `muse.cli_impl.queueing.ConcurrencyGate`. The CPU box's 32B runs with `max_concurrency: 1`.
- **Capacity-wait:** the LoadDirector tags its capacity 503 `retryable=True` when in-use models block eviction; the gateway parks on `CapacityNotifier` (generation asyncio.Event, fired threadsafe by the director on release-to-zero and post-eviction) and retries under the shared deadline.

One budget covers both: `server.queue_timeout_seconds` (default 300; 0 = today's immediate-503). `server.max_queue_depth` (default 0 = unbounded) fast-fails 503 `queue_full`. Timeout -> 503 `queue_timeout`. Telemetry `request` events carry `queued_ms`; `/v1/admin/memory` and `/v1/telemetry/summary` expose per-model `queue_depth` (the seam a future queue-aware federation router consumes).
```

- [ ] **Step 4: Full verification**

Run: `MUSE_CONFIG=$(mktemp) MUSE_CATALOG_DIR=$(mktemp -d) python -m pytest tests/ -q -m "not slow"` then `python -m pytest tests/cli_impl/test_e2e_queueing.py -q`
Expected: fast lane >= baseline count, zero new failures (ignore the 4 config-cli env-override failures); e2e green.

- [ ] **Step 5: Version bump + final commit (NO tag/push -- release is user-gated)**

```bash
sed -i 's/^version = "0.54.5"/version = "0.55.0"/' pyproject.toml
git add -A
git commit -m "feat(queueing): request queueing v0.55.0 (docs + e2e + version bump)"
```

Post-merge deploy notes (for the session driver, not the implementer):
- Release ritual on user "go": FF to main, tag v0.55.0, gh release, build + twine (museq), deploy frodo (pull + `pip install -e . --no-deps` + supervisor restart).
- On the 64GB box (192.168.0.102): set the 32B cap in its catalog manifest:
  `python -c "import json,pathlib; p=pathlib.Path.home()/'.muse/catalog.json'; d=json.loads(p.read_text()); d['qwen2.5-32b-instruct-gguf-q4-k-m']['manifest']['capabilities']['max_concurrency']=1; p.write_text(json.dumps(d, indent=2))"` then restart that supervisor.
- Live validation: 3 concurrent 32B chats -> serialized responses, `queue_depth` visible in `/v1/admin/memory` mid-burst, `queued_ms` in telemetry.
```
