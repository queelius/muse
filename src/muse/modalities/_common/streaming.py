"""Producer-thread SSE streaming shared by speech and chat.

A backend's blocking iterator runs on a dedicated thread that holds the
model's inference lock for the whole stream. Items cross to the event loop
through a bounded queue, and the async generator returned here turns each
item into one SSE event.

The producer thread is started INSIDE the async generator, on its first
iteration. sse_starlette can cancel a response before its body iterator has
ever run (a client that disconnects while the response start is still being
sent), and an async generator that never started does not run its `finally`
on close. Starting the producer earlier would leave it holding the lock with
nobody left to tell it to stop.

The stream is stopped from the response's `__call__`, not only from the
generator's own `finally`: when sse_starlette's task group is cancelled
while a `send` is in progress, the generator stays paused at a `yield`
and sse_starlette 3.0.4 never closes it, so that `finally` would wait for
the async-generator finalizer (cyclic GC). `_ThreadedSSEResponse.__call__`
sets the cancel flag and closes the generator deterministically, and runs
`on_finish` itself when the body never started.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, AsyncIterator, Callable, Iterable, Sequence

from sse_starlette.sse import EventSourceResponse


logger = logging.getLogger(__name__)


class _ThreadedSSEResponse(EventSourceResponse):
    """EventSourceResponse that always stops its producer on the way out."""

    def __init__(self, events: AsyncIterator[dict], *, stop: Callable[[], Any],
                 **kwargs: Any) -> None:
        super().__init__(events, **kwargs)
        self._events = events
        self._stop = stop

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._stop()


def threaded_sse_response(
    *,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> EventSourceResponse:
    """`threaded_sse_events(**kwargs)` wrapped in a response that stops the
    producer, closes the generator and runs `on_finish` (if the producer
    never ran) when the response ends for any reason."""
    events, stop = _threaded_sse_events(**kwargs)
    return _ThreadedSSEResponse(events, stop=stop, headers=headers)


def _threaded_sse_events(
    *,
    lock: Any,
    open_stream: Callable[[], Iterable[Any]],
    encode: Callable[[Any], dict],
    done_events: Sequence[dict],
    error_event: Callable[[], dict],
    label: str,
    queue_depth: int,
    poll_seconds: float,
    cleanup_grace_seconds: float = 0.5,
    on_finish: Callable[[], None] | None = None,
) -> AsyncIterator[dict]:
    """Return an async generator of SSE event dicts for one stream.

    `open_stream` is called on the producer thread once `lock` is held and
    returns the backend iterator; the iterator is closed explicitly when the
    stream ends or the client leaves. `encode` turns one item into an SSE
    event on the event loop. A backend or encoding failure is logged with
    `label` and becomes `error_event()` followed by `done_events`; so the
    client never sees backend text. `on_finish` runs on the producer thread
    after the lock is released, or when the producer never ran.

    Returns the generator and an async `stop()` that sets the cancel flag,
    closes the generator (running its bounded cleanup wait) and runs
    `on_finish` if the producer never started. `stop()` is idempotent.
    """
    cancelled = threading.Event()
    producer_done = threading.Event()
    finish_lock = threading.Lock()
    state = {"producer_started": False, "finished": False}

    def _finish() -> None:
        with finish_lock:
            if state["finished"]:
                return
            state["finished"] = True
        if on_finish is not None:
            try:
                on_finish()
            except Exception:
                logger.exception("%s stream cleanup failed", label)

    async def _events() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_depth)
        queue_slots = threading.Semaphore(queue_depth)
        done_sentinel = object()
        error_sentinel = object()

        def _enqueue(item: object) -> bool:
            # The producer cannot await an asyncio.Queue; a matching
            # semaphore bounds scheduled + queued items and the consumer
            # releases one slot per get().
            while not cancelled.is_set():
                if not queue_slots.acquire(timeout=poll_seconds):
                    continue
                if cancelled.is_set():
                    queue_slots.release()
                    return False
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:
                    # The event loop can close during process shutdown.
                    queue_slots.release()
                    return False
                return True
            return False

        def _acquire_lock() -> bool:
            # Poll so a disconnected request waiting for the model gives up.
            while not cancelled.is_set():
                if lock.acquire(timeout=poll_seconds):
                    return True
            return False

        def _produce() -> None:
            stream = None
            lock_acquired = False
            failure_reported = False
            try:
                lock_acquired = _acquire_lock()
                if not lock_acquired or cancelled.is_set():
                    return
                stream = iter(open_stream())
                while not cancelled.is_set():
                    try:
                        item = next(stream)
                    except StopIteration:
                        break
                    if not _enqueue(item):
                        break
            except Exception:
                logger.exception("%s stream backend failed", label)
                failure_reported = _enqueue(error_sentinel)
            finally:
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.exception("%s stream close failed", label)
                            if not failure_reported:
                                failure_reported = _enqueue(error_sentinel)
                if lock_acquired:
                    lock.release()
                _finish()
                if not cancelled.is_set():
                    _enqueue(done_sentinel)
                producer_done.set()

        # A dedicated thread (not the default executor) so a long stream
        # never occupies one of the loop's bounded pool threads.
        producer = threading.Thread(
            target=_produce, name=f"muse-{label}-stream", daemon=True,
        )
        try:
            producer.start()
            state["producer_started"] = True
        except Exception:
            logger.exception("%s stream producer failed to start", label)
            _finish()
            yield error_event()
            for event in done_events:
                yield event
            return

        try:
            while True:
                item = await queue.get()
                queue_slots.release()
                if item is done_sentinel:
                    for event in done_events:
                        yield event
                    return
                if item is error_sentinel:
                    yield error_event()
                    for event in done_events:
                        yield event
                    return
                try:
                    event = encode(item)
                except Exception:
                    logger.exception("%s stream encoding failed", label)
                    yield error_event()
                    for done in done_events:
                        yield done
                    return
                yield event
        finally:
            # EventSourceResponse closes this generator on disconnect. Wake
            # a producer waiting for queue capacity or the model, then give
            # it a bounded chance to close the backend iterator and release
            # the lock. A backend blocked inside next() cannot be killed
            # safely, so the wait stays bounded and is reported.
            cancelled.set()
            deadline = loop.time() + cleanup_grace_seconds
            while not producer_done.is_set() and loop.time() < deadline:
                await asyncio.sleep(poll_seconds)
            if not producer_done.is_set():
                logger.warning(
                    "%s stream producer did not stop within %.2fs",
                    label, cleanup_grace_seconds,
                )

    events = _events()

    async def stop() -> None:
        cancelled.set()
        try:
            await events.aclose()
        except RuntimeError:
            # Still being iterated elsewhere; its own finally will clean up.
            logger.debug("%s stream generator busy at close", label)
        if not state["producer_started"]:
            _finish()

    return events, stop


def threaded_sse_events(**kwargs: Any) -> AsyncIterator[dict]:
    """The bare event generator (for callers that drive it directly).

    Prefer `threaded_sse_response`, which also stops the producer when the
    response is cancelled mid-send.
    """
    return _threaded_sse_events(**kwargs)[0]
