"""Periodic sampler: free VRAM/RAM + loaded/in-flight model counts.

`Sampler` runs a daemon thread that periodically records a `sample`
telemetry event via the shared `record` function (Task 4's recorder).
Each sample captures a point-in-time view of resource pressure:

- `free_vram_gb`: live free VRAM via `gpu_free_gb()` (None on a CPU-only
  host or when pynvml is unavailable; passed through as-is, never
  coerced to 0.0, since that would fabricate data the nullable store
  column is designed to represent honestly).
- `free_ram_gb`: live free host RAM via `cpu_free_gb()`.
- `loaded_count`: number of currently loaded models (`len(loaded_fn())`).
- `in_flight_count`: number of in-flight requests (`inflight_fn()`).

`gpu_free_gb` and `cpu_free_gb` are imported at module top (not
called through an indirection layer) so tests can monkeypatch this
module's globals directly; `sample_once` must reference the bare
names so patched globals are actually observed at call time.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from muse.core.memory_probe import gpu_free_gb, cpu_free_gb
from muse.observability.recorder import record

logger = logging.getLogger(__name__)


class Sampler:
    """Background daemon that periodically records a `sample` event.

    `stop_event` is an optional external `threading.Event`. Pass the
    supervisor-wide `state.stop_event` so a single Ctrl+C / SIGTERM
    unblocks this sampler's loop along with every other supervisor-owned
    daemon thread (mirrors `IdleSweeper`'s `stop_event` parameter). If
    omitted, the sampler creates its own private Event (unchanged
    behavior for existing callers/tests).
    """

    def __init__(
        self,
        *,
        interval: float,
        loaded_fn: Callable[[], dict[str, Any]],
        inflight_fn: Callable[[], int],
        record_fn: Callable[..., None] = record,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.interval = interval
        self.loaded_fn = loaded_fn
        self.inflight_fn = inflight_fn
        self.record_fn = record_fn
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        free_vram_gb = gpu_free_gb()
        free_ram_gb = cpu_free_gb()
        loaded_count = len(self.loaded_fn())
        in_flight_count = self.inflight_fn()
        self.record_fn(
            "sample",
            free_vram_gb=free_vram_gb,
            free_ram_gb=free_ram_gb,
            loaded_count=loaded_count,
            in_flight_count=in_flight_count,
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="telemetry-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 10 + 1)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample_once()
            except Exception:
                logger.warning("sampler: sample_once failed", exc_info=True)
