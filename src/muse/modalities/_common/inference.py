"""Route helpers for locked backend calls and text-batch caps.

- `run_locked`: run a synchronous backend call off the event loop under the
  backend's `_inference_lock`; a failure is logged and becomes the opaque
  500 envelope (backend text can carry paths or driver detail).
- `check_text_batch`: the shared batch-size and per-item length caps,
  read live with `config.get_positive` (0 or a bad value falls back to the
  setting's default instead of rejecting every request).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Sequence

from fastapi.responses import JSONResponse

from muse.core import config
from muse.core.errors import error_response


logger = logging.getLogger(__name__)


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
