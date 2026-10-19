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
    running stops the wait, not the native work, and the next request is
    admitted only after that work has finished.

A backend that reports `usable = False` is refused; the worker's /health
then fails so the supervisor replaces the worker.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from muse.cli_impl.queueing import ConcurrencyGate
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionModel,
    DecisionRequest,
)


logger = logging.getLogger(__name__)


def _run(backend: DecisionModel, request: DecisionRequest) -> DecisionBatch:
    # Every question is prepared (tokenized, length-checked) before any is
    # evaluated, so an invalid question means no inference at all.
    return backend.evaluate(backend.prepare(request))


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
        self._running: set[concurrent.futures.Future] = set()
        self._closed = False

    @property
    def queue_depth(self) -> int:
        return self._gate.depth(self.model_id)

    @property
    def running(self) -> int:
        return len(self._running)

    async def execute(
        self, backend: DecisionModel, request: DecisionRequest, *, deadline: float,
    ) -> DecisionBatch:
        """Run one batch. Raises QueueFull / QueueTimeout while waiting,
        DecisionBackendError when the backend is unusable or closed, and
        whatever prepare/evaluate raise."""
        self._check_open(backend)
        await self._gate.acquire_slot(
            self.model_id, self._cap, deadline=deadline, max_queue_depth=self._depth,
        )
        try:
            self._check_open(backend)
            future = self._pool.submit(_run, backend, request)
        except BaseException:
            self._gate.release_slot(self.model_id)
            raise
        self._running.add(future)
        future.add_done_callback(self._finished)
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(_consume)
        # shield: a cancelled request must not cancel the native work's
        # tracking future; the slot follows the native work, not the caller.
        return await asyncio.shield(wrapped)

    def _finished(self, future: concurrent.futures.Future) -> None:
        self._running.discard(future)
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
        pending = [asyncio.wrap_future(f) for f in list(self._running)]
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
