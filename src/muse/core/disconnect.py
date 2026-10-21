"""Stop waiting on work whose HTTP client has gone away.

Neither Starlette nor uvicorn cancels a non-streaming handler when its
client disconnects; the handler simply finishes and its response is
dropped. For work worth abandoning, `unless_disconnected` races the work
against the request's `http.disconnect` message and cancels the work if
the client leaves first.

Call it only after the request body has been read in full: it consumes the
ASGI receive channel, whose next message is then `http.disconnect`.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ClientDisconnected(Exception):
    """The client disconnected before the work finished; the work was cancelled."""


async def _disconnected(request: Any) -> None:
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


async def _settle(task: asyncio.Future) -> None:
    task.cancel()
    await asyncio.wait({task})
    if not task.cancelled() and task.exception() is not None:
        logger.debug("abandoned work raised %s", type(task.exception()).__name__)


async def unless_disconnected(request: Any, work: Awaitable[T]) -> T:
    """Await `work`; raise ClientDisconnected (after cancelling it) if the
    client disconnects first. Cancelling the caller cancels `work` too. If
    the receive channel fails instead, the work simply runs to completion."""
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_disconnected(request))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if not task.done() and watcher.exception() is not None:
            logger.debug(
                "cannot watch for disconnect: %s", type(watcher.exception()).__name__,
            )
            await asyncio.wait({task})
    except BaseException:
        await _settle(task)
        raise
    finally:
        watcher.cancel()
        if watcher.done() and not watcher.cancelled():
            watcher.exception()  # retrieved, so asyncio does not warn
    if task.done():
        return task.result()
    await _settle(task)
    raise ClientDisconnected()
