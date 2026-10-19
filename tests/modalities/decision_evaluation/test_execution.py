"""Cancellation-safe decision execution inside a worker."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from muse.cli_impl.queueing import QueueFull, QueueTimeout
from muse.modalities.decision_evaluation.execution import DecisionExecutor
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionInputError,
    PreparedBatch,
    PreparedQuestion,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.validation import parse_request


REQUEST = parse_request(
    b'{"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}'
)


class BlockingBackend:
    """evaluate() blocks until `gate` is set; records native concurrency."""

    model_id = "m"

    def __init__(self):
        self.gate = threading.Event()
        self.started = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.usable = True
        self.prepare_error = None

    def prepare(self, request):
        if self.prepare_error is not None:
            raise self.prepare_error
        return PreparedBatch("m", (PreparedQuestion("q", ("true", "false"), None, 3),), 3)

    def evaluate(self, batch, *, should_stop=None):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            assert self.gate.wait(10), "test never released the native call"
            return DecisionBatch((QuestionDistribution("q", {"true": 1.0, "false": 0.0}),), 3)
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        pass


def _deadline(s=10.0):
    return time.monotonic() + s


async def _until(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "condition never became true"
        await asyncio.sleep(0.005)


@pytest.fixture
async def executor():
    ex = DecisionExecutor("m", max_concurrency=1, max_queue_depth=32)
    yield ex
    await ex.close(timeout=1)


async def test_runs_off_loop_and_returns_the_batch(executor):
    backend = BlockingBackend()
    task = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    # The loop stays responsive while native work blocks.
    t0 = time.monotonic()
    await asyncio.sleep(0.01)
    assert time.monotonic() - t0 < 0.5
    backend.gate.set()
    batch = await asyncio.wait_for(task, 5)
    assert batch.answers[0].probabilities == {"true": 1.0, "false": 0.0}


async def test_one_runs_32_wait_and_the_next_is_refused(executor, monkeypatch):
    monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "0")  # global unbounded
    backend = BlockingBackend()
    threads_before = threading.active_count()
    first = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    waiters = [
        asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
        for _ in range(32)
    ]
    await _until(lambda: executor.queue_depth == 32)
    # Waiting costs no threads: only the executor's single worker thread exists.
    assert threading.active_count() <= threads_before + 1
    with pytest.raises(QueueFull):
        await executor.execute(backend, REQUEST, deadline=_deadline())
    backend.gate.set()
    await asyncio.wait_for(asyncio.gather(first, *waiters), 10)
    assert backend.calls == 33 and backend.max_active == 1
    assert executor._gate._entered.get("m", 0) == 0


async def test_positive_global_depth_lowers_the_ceiling(executor, monkeypatch):
    monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "2")
    backend = BlockingBackend()
    first = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    waiters = [asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
               for _ in range(2)]
    await _until(lambda: executor.queue_depth == 2)
    with pytest.raises(QueueFull):
        await executor.execute(backend, REQUEST, deadline=_deadline())
    backend.gate.set()
    await asyncio.gather(first, *waiters)


async def test_wait_deadline(executor):
    backend = BlockingBackend()
    first = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    with pytest.raises(QueueTimeout):
        await executor.execute(backend, REQUEST, deadline=_deadline(0.05))
    backend.gate.set()
    await first
    assert executor._gate._entered.get("m", 0) == 0


async def test_cancelled_waiter_leaves_the_others_intact(executor):
    backend = BlockingBackend()
    first = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    doomed = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    survivor = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(lambda: executor.queue_depth == 2)
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed
    assert executor.queue_depth == 1
    backend.gate.set()
    await asyncio.wait_for(asyncio.gather(first, survivor), 5)
    assert backend.calls == 2  # the cancelled waiter never reached native code
    assert executor._gate._entered.get("m", 0) == 0


async def test_cancelled_running_request_keeps_its_slot_until_native_completes(executor):
    backend = BlockingBackend()
    running = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert executor.running == 1  # native work continues without its caller

    nxt = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await asyncio.sleep(0.05)
    assert not nxt.done() and backend.calls == 1  # not admitted early

    backend.gate.set()
    await asyncio.wait_for(nxt, 5)
    assert backend.calls == 2 and backend.max_active == 1
    assert executor.running == 0


async def test_input_errors_propagate_and_release_the_slot(executor):
    backend = BlockingBackend()
    backend.prepare_error = DecisionInputError("input_too_long", "too long")
    with pytest.raises(DecisionInputError):
        await executor.execute(backend, REQUEST, deadline=_deadline())
    await _until(lambda: executor.running == 0)
    backend.prepare_error = None
    backend.gate.set()
    await executor.execute(backend, REQUEST, deadline=_deadline(0))
    assert backend.calls == 1


async def test_unusable_backend_is_refused_without_native_calls(executor):
    backend = BlockingBackend()
    backend.usable = False
    with pytest.raises(DecisionBackendError) as exc:
        await executor.execute(backend, REQUEST, deadline=_deadline())
    assert exc.value.code == "backend_unusable"
    assert backend.calls == 0


async def test_close_drains_running_work_then_refuses():
    ex = DecisionExecutor("m")
    backend = BlockingBackend()
    task = asyncio.ensure_future(ex.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    closer = asyncio.ensure_future(ex.close(timeout=5))
    await asyncio.sleep(0.02)
    assert not closer.done()
    backend.gate.set()
    assert await closer is True
    await task
    with pytest.raises(DecisionBackendError) as exc:
        await ex.execute(backend, REQUEST, deadline=_deadline())
    assert exc.value.code == "shutting_down"


async def test_close_reports_undrained_work():
    ex = DecisionExecutor("m")
    backend = BlockingBackend()
    task = asyncio.ensure_future(ex.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    assert await ex.close(timeout=0.05) is False
    backend.gate.set()
    await task


async def test_cancelling_a_running_request_asks_the_backend_to_stop(executor):
    backend = BlockingBackend()
    seen = []
    original = backend.evaluate

    def evaluate(batch, *, should_stop=None):
        seen.append(should_stop)
        return original(batch, should_stop=should_stop)

    backend.evaluate = evaluate
    running = asyncio.ensure_future(executor.execute(backend, REQUEST, deadline=_deadline()))
    await _until(backend.started.is_set)
    assert seen[0]() is False
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert seen[0]() is True
    backend.gate.set()
    await _until(lambda: executor.running == 0)
