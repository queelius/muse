"""Fire-and-forget telemetry recorder.

`record(type, **fields)` is meant to be called from hot request-handling
paths, so it must never block and must never raise. Events are enqueued
onto a bounded queue and drained by a background daemon thread that
batches writes into the TelemetryStore. When the queue is full the event
is silently dropped and `dropped` is incremented, rather than blocking
the caller or losing more than the one event.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from muse.observability.events import event_to_row
from muse.observability.store import TelemetryStore

logger = logging.getLogger(__name__)


class TelemetryRecorder:
    """Background-flushing telemetry recorder backed by a TelemetryStore."""

    def __init__(
        self,
        store: TelemetryStore,
        *,
        max_queue: int = 10000,
        flush_interval: float = 0.5,
    ) -> None:
        self._store = store
        self._flush_interval = flush_interval
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        # Guards `dropped` increments: record() is called from arbitrary
        # request-handling threads across the process, so a bare `+= 1`
        # is a cross-thread read-modify-write that can under-report the
        # true drop count under concurrent callers.
        self._dropped_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _mark_dropped(self) -> None:
        with self._dropped_lock:
            self.dropped += 1

    def record(self, type: str, **fields: Any) -> None:
        # event_to_row() raises ValueError on an unknown field name (a
        # typo'd kwarg). record() must never raise -- treat a bad field
        # the same as a dropped event rather than letting it escape into
        # a hot request-handling path.
        try:
            row = event_to_row(type, time.time(), **fields)
        except ValueError:
            logger.warning(
                "telemetry recorder: dropping event with unknown field(s) "
                "type=%r fields=%r", type, sorted(fields), exc_info=True,
            )
            self._mark_dropped()
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._mark_dropped()

    def flush(self) -> None:
        rows: list[dict[str, Any]] = []
        while True:
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if rows:
            self._store.insert_many(rows)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="telemetry-recorder-flush", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._flush_interval * 10 + 1)
            self._thread = None
        # Final drain so nothing queued is silently lost on shutdown.
        try:
            self.flush()
        except Exception:
            logger.warning("telemetry recorder: final flush failed", exc_info=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._flush_interval)
            try:
                self.flush()
            except Exception:
                logger.warning("telemetry recorder: flush failed", exc_info=True)


class _NoopRecorder:
    """Silent stand-in used when telemetry is disabled or uninitialized."""

    dropped = 0

    def record(self, *a: Any, **k: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


_NOOP = _NoopRecorder()
_RECORDER: TelemetryRecorder | _NoopRecorder | None = None


def init_recorder(store: TelemetryStore, *, enabled: bool = True) -> None:
    global _RECORDER
    if enabled:
        _RECORDER = TelemetryRecorder(store)
    else:
        _RECORDER = _NoopRecorder()
    _RECORDER.start()


def get_recorder() -> TelemetryRecorder | _NoopRecorder:
    if _RECORDER is None:
        return _NOOP
    return _RECORDER


def record(type: str, **fields: Any) -> None:
    get_recorder().record(type, **fields)


def reset_recorder() -> None:
    global _RECORDER
    if _RECORDER is not None:
        _RECORDER.stop()
    _RECORDER = None
