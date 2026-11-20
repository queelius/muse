"""Run a route's synchronous model work off the event loop.

``offload_call`` is the one shape every image/video/3D route uses: the
backend call AND its output encoding run in a worker thread (through
``run_native_offload``), optionally under the backend's inference lock,
and the request-owned input images are closed exactly once, after the
native work has settled, even when the client disconnects mid-call.

``produce_each`` is the n-results loop that runs inside such a call:
each result is produced under the lock, encoded outside it, and its
images are closed as soon as they are encoded, so a failure part-way
through never strands earlier results.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from muse.modalities._common.images import close_decoded_images
from muse.modalities._native_offload import run_native_offload


T = TypeVar("T")
E = TypeVar("E")


async def offload_call(
    call: Callable[[], T],
    *,
    lock: Any = None,
    inputs: Iterable[Any] = (),
    stop: threading.Event | None = None,
) -> T:
    """Return ``call()`` run in a worker thread; owns and closes ``inputs``.

    ``lock`` (a backend's ``_inference_lock``) is held for the whole call;
    pass None when ``call`` takes it per item (see ``produce_each``).
    ``stop`` is set when the request is cancelled, so a multi-result call
    can skip the results nobody will read.
    """
    owned = list(inputs)

    def _run() -> T:
        if lock is None:
            return call()
        with lock:
            return call()

    abandoned = False
    try:
        return await run_native_offload(
            _run, cleanup_abandoned=lambda _result: close_decoded_images(owned),
        )
    except asyncio.CancelledError:
        abandoned = True
        if stop is not None:
            stop.set()
        raise
    finally:
        if not abandoned:
            close_decoded_images(owned)


def result_images(result: Any) -> list[Any]:
    """The closable images a generation result carries (`image`, `frames`)."""
    images = [getattr(result, "image", None)]
    images.extend(getattr(result, "frames", None) or ())
    return images


def produce_each(
    n: int,
    produce: Callable[[int], Any],
    encode: Callable[[Any], E],
    *,
    lock: Any,
    stop: threading.Event | None = None,
) -> list[E]:
    """``[encode(produce(i)) for i in range(n)]``, closing each result.

    Runs synchronously (inside ``offload_call``). ``produce`` runs under
    ``lock`` per item, so other requests can interleave between items;
    ``encode`` runs outside it. Stops early once ``stop`` is set.
    """
    out: list[E] = []
    for i in range(n):
        if stop is not None and stop.is_set():
            break
        with lock:
            result = produce(i)
        try:
            out.append(encode(result))
        finally:
            close_decoded_images(result_images(result))
    return out
