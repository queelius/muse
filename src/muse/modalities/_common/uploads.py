"""Multipart upload handling shared by the audio-input routes.

- `read_upload`: bounded read + close, mapping empty -> 400 and
  oversize -> 413 (`payload_too_large`).
- `run_on_spooled_upload`: write the bytes to a temp file and run the
  synchronous backend call on it off the event loop. The temp file is
  removed by the worker thread once the call finishes, so a cancelled
  request (client disconnect) cannot delete the file while the backend
  is still reading it; work cancelled before it started is cleaned up
  through `run_native_offload`'s abandoned-ownership hook.
- `looks_like_audio_decode_error`: message heuristic for decoder
  failures (librosa / soundfile / audioread / PyAV / ffmpeg /
  torchcodec) that routes map to 415.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from muse.core.errors import error_response
from muse.modalities._native_offload import run_native_offload


logger = logging.getLogger(__name__)

T = TypeVar("T")

_DECODE_ERROR_TOKENS = (
    "audioread",
    "decode",            # also matches "decoder"
    "ffmpeg",
    "format not recognised",
    "format not recognized",
    "invalid data",
    "invaliddata",       # PyAV's InvalidDataError class name, lowercased
    "no backend available",
    "soundfile",
    "torchcodec",
)


async def read_upload(
    upload: UploadFile,
    *,
    max_bytes: int,
    limit: str,
    label: str = "audio file",
) -> bytes | JSONResponse:
    """Read at most `max_bytes` from `upload`, then close it.

    Returns the bytes, or the error response to return as-is: 400 when
    empty, 413 when over the cap. `limit` names the knob in the 413
    message (e.g. ``"MUSE_AUDIO_QUALITY_MAX_BYTES=52428800"``).
    """
    try:
        data = await upload.read(max_bytes + 1)
    finally:
        try:
            await upload.close()
        except Exception:  # noqa: BLE001
            logger.debug("failed to close upload", exc_info=True)
    if not data:
        return error_response(400, "invalid_parameter", f"{label} is empty")
    if len(data) > max_bytes:
        return error_response(
            413, "payload_too_large", f"{label} exceeds {limit}",
        )
    return data


def upload_suffix(filename: str | None, default: str = ".wav") -> str:
    """The upload's extension (so decoders can sniff the format), else `default`."""
    return Path(filename or "").suffix or default


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def run_on_spooled_upload(
    data: bytes,
    call: Callable[[str], T],
    *,
    suffix: str,
) -> T:
    """Write `data` to a temp file and return `call(path)` run off-loop.

    The file is always removed, and never before `call` has returned.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    started = False

    def _work() -> T:
        nonlocal started
        started = True
        try:
            return call(path)
        finally:
            _unlink_quietly(path)

    def _cleanup_abandoned(_result: T | None) -> None:
        # Only work cancelled while still queued reaches here without
        # `_work` having run (and removed the file) already.
        if not started:
            _unlink_quietly(path)

    # Not unlinked on this coroutine's own unwind: a cancelled await
    # leaves the queued/running work alive, and it still needs the file.
    return await run_native_offload(_work, cleanup_abandoned=_cleanup_abandoned)


def looks_like_audio_decode_error(exc: BaseException) -> bool:
    """True when `exc`'s message reads like an audio decoder failure."""
    message = str(exc).lower()
    return any(token in message for token in _DECODE_ERROR_TOKENS)
