"""Route helpers that run synchronous model work off the event loop.

- `run_locked`: run a synchronous backend call off the event loop under the
  backend's `_inference_lock`; a failure is logged and becomes the opaque
  500 envelope (backend text can carry paths or driver detail).
- `check_text_batch` / `cap`: the shared batch-size and per-item length
  caps, read live with `config.get_positive` (0 or a bad value falls back
  to the setting's default instead of rejecting every request).
- `offload_call`: the one shape every image/video/3D route uses: the
  backend call AND its output encoding run in a worker thread (through
  `run_native_offload`), optionally under the backend's inference lock,
  and the request-owned input images are closed exactly once, after the
  native work has settled, even when the client disconnects mid-call.
- `produce_each`: the n-results loop that runs inside such a call: each
  result is produced under the lock, encoded outside it, and its images
  are closed as soon as they are encoded, so a failure part-way through
  never strands earlier results.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from fastapi.responses import JSONResponse

from muse.core import config
from muse.core.errors import error_response
from muse.modalities._common.images import close_decoded_images
from muse.modalities._native_offload import run_native_offload


logger = logging.getLogger(__name__)

T = TypeVar("T")
E = TypeVar("E")


async def run_locked(
    backend: Any,
    call: Callable[[], Any],
    *,
    label: str,
    message: str,
    passthrough: tuple[type[BaseException], ...] = (),
) -> Any:
    """Return `call()` run off-loop under `backend._inference_lock`.

    On failure, logs `label` with the traceback and returns a 500
    `internal_error` response carrying `message`; callers return any
    `JSONResponse` result as-is. Exceptions in `passthrough` (typed errors
    the route maps itself) are re-raised.
    """
    def _locked() -> Any:
        with backend._inference_lock:
            return call()

    try:
        return await asyncio.to_thread(_locked)
    except passthrough:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("%s failed", label)
        return error_response(500, "internal_error", message)


def cap(key: str) -> Any:
    """A positive cap from the config registry, read per request."""
    return config.get_positive(key)


def _env(key: str) -> str:
    return config.SETTINGS_BY_KEY[key].env


def check_text_batch(
    items: Sequence[str],
    *,
    max_batch_key: str,
    max_chars_key: str,
    field: str = "input",
) -> JSONResponse | None:
    """Return a 400 when `items` exceeds the batch or per-item caps."""
    max_batch = cap(max_batch_key)
    if len(items) > max_batch:
        return error_response(
            400, "invalid_parameter",
            f"{field} batch size {len(items)} exceeds "
            f"{_env(max_batch_key)}={max_batch}",
        )
    max_chars = cap(max_chars_key)
    too_long = next(
        (i for i, s in enumerate(items) if len(s) > max_chars), None,
    )
    if too_long is not None:
        return error_response(
            400, "invalid_parameter",
            f"{field}[{too_long}] exceeds {_env(max_chars_key)}={max_chars}",
        )
    return None



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
