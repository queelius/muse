"""Cancellation-safe execution of decision batches inside a worker.

The gateway already admits at most one request per decision model, but a
direct worker call (or a race across gateways) must not reach native
state concurrently either. Each backend gets one DecisionExecutor:

  - waiters park on the event loop in a ConcurrencyGate (cap 1, bounded
    queue), never in a thread each;
  - an admitted batch (prepare, then evaluate) runs on the executor's own
    single thread;
  - the slot is released by the native call's completion callback, NOT by
    the awaiting request. Cancelling a request whose batch is already
    running stops the wait and asks the backend to stop before its next
    question; the question in progress always finishes, and the next
    request is admitted only after the native work has returned.

A backend that reports `usable = False` is refused; the worker's /health
then fails so the supervisor replaces the worker.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading

from collections.abc import Callable
from typing import TypeVar

from muse.cli_impl.queueing import ConcurrencyGate
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionBatchRequest,
    DecisionCancelled,
    DecisionInputError,
    DecisionModel,
    DecisionRequest,
    PreparedBatch,
)


logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _run(
    backend: DecisionModel, request: DecisionRequest, stop: threading.Event,
) -> DecisionBatch:
    # Every question is prepared (tokenized, length-checked) before any is
    # evaluated, so an invalid question means no inference at all.
    return backend.evaluate(backend.prepare(request), should_stop=stop.is_set)


def _run_batch(
    backend: DecisionModel, request: DecisionBatchRequest, stop: threading.Event,
) -> list[DecisionBatch]:
    # Prepare every item before evaluating any, so an over-limit item fails
    # cleanly without running partial inference. Check cancellation between
    # every item (prepare and evaluate) as well as between every question.
    prepared: list[PreparedBatch] = []
    for i, item in enumerate(request.items):
        if stop.is_set():
            raise DecisionCancelled()
        item_req = DecisionRequest(
            model=request.model, state=item.state, questions=item.questions,
        )
        try:
            prepared.append(backend.prepare(item_req))
        except DecisionInputError as exc:
            raise DecisionInputError(
                exc.code, f"items[{i}]: {exc.message}", status=exc.status,
            ) from None

    results: list[DecisionBatch] = []
    for batch in prepared:
        if stop.is_set():
            raise DecisionCancelled()
        results.append(backend.evaluate(batch, should_stop=stop.is_set))
    return results


def _consume(future: asyncio.Future) -> None:
    # The awaiting request may be gone; retrieve the outcome so asyncio
    # does not log "exception was never retrieved".
    if not future.cancelled():
        future.exception()


class DecisionExecutor:
    def __init__(
        self, model_id: str, *, max_concurrency: int = 1, max_queue_depth: int = 32,
    ) -> None:
        self.model_id = model_id
        self._cap = max_concurrency
        self._depth = max_queue_depth
        self._gate = ConcurrencyGate()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix=f"decision-{model_id}",
        )
        # Mutated from pool threads (_finished) and read on the loop (close).
        self._running_lock = threading.Lock()
        self._running: set[concurrent.futures.Future] = set()
        self._closed = False

    @property
    def queue_depth(self) -> int:
        return self._gate.depth(self.model_id)

    @property
    def running(self) -> int:
        with self._running_lock:
            return len(self._running)

    async def execute(
        self, backend: DecisionModel, request: DecisionRequest, *, deadline: float,
    ) -> DecisionBatch:
        """Run one single-item request. Raises QueueFull / QueueTimeout while
        waiting, DecisionBackendError when the backend is unusable or closed,
        and whatever prepare/evaluate raise."""
        return await self._submit(backend, _run, request, deadline=deadline)

    async def execute_batch(
        self, backend: DecisionModel, request: DecisionBatchRequest, *, deadline: float,
    ) -> list[DecisionBatch]:
        """Run a multi-item batch under a single executor slot and shared
        queue deadline, checking cancellation between items and questions."""
        return await self._submit(backend, _run_batch, request, deadline=deadline)

    async def _submit(
        self,
        backend: DecisionModel,
        fn: Callable[..., _T],
        request: object,
        *,
        deadline: float,
    ) -> _T:
        self._check_open(backend)
        # Release only a slot actually taken (the gateway's rule): False
        # means "uncapped, nothing reserved".
        took_slot = await self._gate.acquire_slot(
            self.model_id, self._cap, deadline=deadline, max_queue_depth=self._depth,
        )
        stop = threading.Event()
        try:
            self._check_open(backend)
            future = self._pool.submit(fn, backend, request, stop)
        except BaseException:
            if took_slot:
                self._gate.release_slot(self.model_id)
            raise
        with self._running_lock:
            self._running.add(future)
        future.add_done_callback(
            lambda done: self._finished(done, release=took_slot),
        )
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(_consume)
        # shield: a cancelled request must not cancel the native work's
        # tracking future; the slot follows the native work, not the caller.
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _finished(
        self, future: concurrent.futures.Future, *, release: bool,
    ) -> None:
        with self._running_lock:
            self._running.discard(future)
        if release:
            self._gate.release_slot(self.model_id)  # threadsafe

    def _check_open(self, backend: DecisionModel) -> None:
        if self._closed:
            raise DecisionBackendError("shutting_down", "the worker is shutting down")
        if getattr(backend, "usable", True) is False:
            raise DecisionBackendError(
                "backend_unusable", "the model is unusable until its worker restarts",
            )

    async def close(self, *, timeout: float = 30.0) -> bool:
        """Refuse new work, wait up to `timeout` for running native work,
        then stop the thread. Returns whether everything drained."""
        self._closed = True
        with self._running_lock:
            running = list(self._running)
        pending = [asyncio.wrap_future(f) for f in running]
        drained = True
        if pending:
            _, still = await asyncio.wait(pending, timeout=timeout)
            drained = not still
            if still:
                logger.warning(
                    "decision model %s: native work still running at shutdown", self.model_id,
                )
        self._pool.shutdown(wait=False, cancel_futures=True)
        return drained
