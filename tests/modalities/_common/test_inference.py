"""Tests for muse.modalities._common.inference (offload_call, produce_each)."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from muse.modalities._common.inference import offload_call, produce_each


def _result():
    return SimpleNamespace(image=MagicMock(), frames=[MagicMock(), MagicMock()])


def test_produce_each_encodes_then_closes_every_result():
    results = []

    def produce(i):
        results.append(_result())
        return results[-1]

    out = produce_each(3, produce, lambda r: id(r), lock=threading.Lock())
    assert out == [id(r) for r in results]
    for r in results:
        r.image.close.assert_called_once_with()
        for frame in r.frames:
            frame.close.assert_called_once_with()


def test_produce_each_closes_the_result_whose_encode_failed():
    r = _result()

    def encode(_r):
        raise RuntimeError("encode failed")

    with pytest.raises(RuntimeError):
        produce_each(2, lambda i: r, encode, lock=threading.Lock())
    r.image.close.assert_called_once_with()


def test_produce_each_stops_once_stop_is_set():
    stop = threading.Event()
    calls = []

    def produce(i):
        calls.append(i)
        stop.set()
        return _result()

    out = produce_each(4, produce, lambda r: r, lock=threading.Lock(), stop=stop)
    assert calls == [0]
    assert len(out) == 1


async def test_offload_call_runs_off_loop_holds_lock_and_closes_inputs():
    lock = threading.Lock()
    image = MagicMock()

    def call():
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        assert lock.locked()
        image.close.assert_not_called()
        return "ok"

    assert await offload_call(call, lock=lock, inputs=[image]) == "ok"
    image.close.assert_called_once_with()


async def test_offload_call_cancel_sets_stop_and_defers_input_cleanup():
    started = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    image = MagicMock()

    def call():
        started.set()
        assert release.wait(timeout=5)
        image.close.assert_not_called()

    task = asyncio.create_task(offload_call(call, inputs=[image], stop=stop))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stop.is_set()
    image.close.assert_not_called()
    release.set()
    for _ in range(100):
        if image.close.called:
            break
        await asyncio.sleep(0.01)
    image.close.assert_called_once_with()
