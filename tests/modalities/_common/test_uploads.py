"""Tests for the shared multipart upload helpers."""
from __future__ import annotations

import asyncio
import json
import os
import threading

import pytest

from muse.modalities._common.uploads import (
    looks_like_audio_decode_error,
    read_upload,
    run_on_spooled_upload,
    upload_suffix,
)


class _Upload:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.data[:size]

    async def close(self) -> None:
        self.closed = True


def _error(resp):
    return resp.status_code, json.loads(resp.body)["error"]


async def test_read_upload_returns_bytes_bounded_and_closes():
    up = _Upload(b"abc")
    assert await read_upload(up, max_bytes=10, limit="X=10") == b"abc"
    assert up.read_sizes == [11]
    assert up.closed


async def test_read_upload_empty_is_400():
    up = _Upload(b"")
    status, err = _error(await read_upload(up, max_bytes=10, limit="X=10"))
    assert status == 400
    assert err["code"] == "invalid_parameter"
    assert err["message"] == "audio file is empty"
    assert up.closed


async def test_read_upload_oversize_is_413():
    up = _Upload(b"x" * 20)
    status, err = _error(await read_upload(
        up, max_bytes=10, limit="MUSE_CAP=10", label="file[0]",
    ))
    assert status == 413
    assert err["code"] == "payload_too_large"
    assert err["message"] == "file[0] exceeds MUSE_CAP=10"


def test_upload_suffix():
    assert upload_suffix("clip.mp3") == ".mp3"
    assert upload_suffix(None) == ".wav"
    assert upload_suffix("noext", default="") == ""


def test_decode_error_heuristic():
    assert looks_like_audio_decode_error(RuntimeError("PyAV InvalidDataError"))
    assert looks_like_audio_decode_error(RuntimeError("audioread.NoBackendError"))
    assert not looks_like_audio_decode_error(RuntimeError("GPU out of memory"))


class InvalidDataError(Exception):
    pass


@pytest.mark.parametrize("exc", [
    InvalidDataError("[Errno 1094995529] whatever"),
    RuntimeError("Error opening 'x.wav': Format not recognised."),
    RuntimeError("Invalid data found when processing input"),
])
def test_decode_error_heuristic_accepts_decoder_rejections(exc):
    assert looks_like_audio_decode_error(exc)


def test_decode_error_heuristic_matches_subclasses():
    class EOFDecodeError(InvalidDataError):
        pass

    assert looks_like_audio_decode_error(EOFDecodeError("x"))


def test_decode_error_heuristic_follows_the_cause_chain():
    try:
        try:
            raise InvalidDataError("bad")
        except InvalidDataError as inner:
            raise RuntimeError("transcription failed") from inner
    except RuntimeError as outer:
        assert looks_like_audio_decode_error(outer)


@pytest.mark.parametrize("exc", [
    RuntimeError(
        "TorchCodec audio decoder is unavailable; run `muse pull` and "
        "ensure FFmpeg shared libraries are installed"
    ),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    RuntimeError("Error in decoder layer 3"),
    ModuleNotFoundError("No module named 'soundfile'"),
])
def test_decode_error_heuristic_rejects_server_faults(exc):
    """I2: a missing dependency or a model error is a 500, not a 415."""
    assert not looks_like_audio_decode_error(exc)


async def test_spooled_upload_passes_path_and_removes_file():
    seen = {}

    def _call(path):
        seen["path"] = path
        seen["data"] = open(path, "rb").read()
        return "ok"

    assert await run_on_spooled_upload(b"payload", _call, suffix=".flac") == "ok"
    assert seen["data"] == b"payload"
    assert seen["path"].endswith(".flac")
    assert not os.path.exists(seen["path"])


async def test_spooled_upload_removes_file_on_backend_error():
    seen = {}

    def _call(path):
        seen["path"] = path
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_on_spooled_upload(b"x", _call, suffix=".wav")
    assert not os.path.exists(seen["path"])


async def test_cancelled_request_keeps_file_until_backend_finishes():
    """A client disconnect must not delete the file under a running backend."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    seen = {}

    def _call(path):
        seen["path"] = path
        started.set()
        release.wait(5)
        seen["readable"] = open(path, "rb").read()
        finished.set()

    task = asyncio.create_task(run_on_spooled_upload(b"audio", _call, suffix=".wav"))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert os.path.exists(seen["path"])

    release.set()
    assert await asyncio.to_thread(finished.wait, 5)
    for _ in range(100):
        if not os.path.exists(seen["path"]):
            break
        await asyncio.sleep(0.01)
    assert seen["readable"] == b"audio"
    assert not os.path.exists(seen["path"])


async def test_spooled_upload_writes_the_body_off_the_event_loop(monkeypatch):
    import asyncio
    import builtins
    from muse.modalities._common import uploads

    on_loop = []
    real_open = builtins.open

    def _open(path, mode="r", *args, **kwargs):
        if "w" in mode:
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)
    got = await uploads.run_on_spooled_upload(
        b"abc", lambda p: real_open(p, "rb").read(), suffix=".wav",
    )
    assert got == b"abc"
    assert on_loop == [False]
