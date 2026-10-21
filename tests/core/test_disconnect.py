"""unless_disconnected: abandon work whose HTTP client has gone away."""
from __future__ import annotations

import asyncio

import pytest

from muse.core.disconnect import ClientDisconnected, unless_disconnected


class FakeRequest:
    """receive() blocks until `leave()`, then reports http.disconnect."""

    def __init__(self):
        self.gone = asyncio.Event()

    def leave(self):
        self.gone.set()

    async def receive(self):
        await self.gone.wait()
        return {"type": "http.disconnect"}


async def test_returns_the_result_when_the_client_stays():
    async def work():
        return 7

    assert await unless_disconnected(FakeRequest(), work()) == 7


async def test_propagates_the_work_error():
    async def work():
        raise KeyError("x")

    with pytest.raises(KeyError):
        await unless_disconnected(FakeRequest(), work())


async def test_disconnect_cancels_the_work():
    request = FakeRequest()
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.ensure_future(unless_disconnected(request, work()))
    await asyncio.sleep(0.01)
    request.leave()
    with pytest.raises(ClientDisconnected):
        await asyncio.wait_for(task, 5)
    assert cancelled.is_set()


async def test_other_messages_are_not_a_disconnect():
    messages = [{"type": "http.request", "body": b"", "more_body": False}]
    gone = asyncio.Event()

    class Request:
        async def receive(self):
            if messages:
                return messages.pop()
            await gone.wait()
            return {"type": "http.disconnect"}

    async def work():
        await asyncio.sleep(0.02)
        return "done"

    assert await unless_disconnected(Request(), work()) == "done"


async def test_cancelling_the_caller_cancels_the_work():
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.ensure_future(unless_disconnected(FakeRequest(), work()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_a_failing_receive_channel_is_not_a_disconnect():
    class Broken:
        async def receive(self):
            raise RuntimeError("Unexpected message received")

    async def work():
        await asyncio.sleep(0.01)
        return "done"

    assert await unless_disconnected(Broken(), work()) == "done"
