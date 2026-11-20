"""Lifecycle tests for the producer-thread SSE helper."""
import asyncio
import gc
import threading
import time

import pytest

from muse.modalities._common.streaming import threaded_sse_response


class _Endless:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.producing = threading.Event()
        self.finished = threading.Event()
        self.closed = []

    def stream(self):
        try:
            while not self.stop.is_set():
                self.producing.set()
                time.sleep(0.001)
                yield "x"
        finally:
            self.finished.set()


def _response(src, **kw):
    return threaded_sse_response(
        lock=src.lock,
        open_stream=src.stream,
        encode=lambda item: {"data": item},
        done_events=({"data": "[DONE]"},),
        error_event=lambda: {"event": "error", "data": "{}"},
        label="test",
        queue_depth=4,
        poll_seconds=0.02,
        on_finish=lambda: src.closed.append(True),
        **kw,
    )


async def _drive(response, *, disconnect_after: int):
    """Run the response as ASGI; disconnect once `disconnect_after` body
    chunks went out, while a yielding `send` is still in progress."""
    gone = asyncio.Event()
    sent = 0

    async def receive():
        if disconnect_after:
            await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal sent
        if message["type"] == "http.response.body" and message.get("body"):
            sent += 1
            if sent >= disconnect_after:
                gone.set()
        await asyncio.sleep(0.05)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "GET",
             "path": "/", "headers": []}
    await asyncio.wait_for(response(scope, receive, send), timeout=5)


async def _released(src, timeout=1.0):
    deadline = time.monotonic() + timeout
    while (src.lock.locked() or not src.finished.is_set()) and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return not src.lock.locked() and src.finished.is_set()


@pytest.mark.parametrize("disconnect_after", [1, 2])
async def test_disconnect_after_first_events_stops_producer_without_gc(disconnect_after):
    """sse_starlette leaves the generator paused at `yield` when its task
    group is cancelled mid-send; the response itself must stop the producer
    and release the lock, with no help from the GC."""
    src = _Endless()
    gc.disable()
    try:
        await _drive(_response(src), disconnect_after=disconnect_after)
        assert await _released(src), "producer still holds the lock"
    finally:
        gc.enable()
        src.stop.set()
    assert src.closed == [True]


async def test_never_started_body_runs_on_finish_and_never_locks():
    src = _Endless()
    gc.disable()
    try:
        await _drive(_response(src), disconnect_after=0)
        await asyncio.sleep(0.1)
        assert not src.producing.is_set()
        assert not src.lock.locked()
        assert src.closed == [True]
    finally:
        gc.enable()
        src.stop.set()
