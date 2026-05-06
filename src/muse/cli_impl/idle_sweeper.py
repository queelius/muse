"""IdleSweeper: per-model idle-timeout eviction (v0.40.1).

Background sweeper that unloads loaded models which have been idle (no
requests) for longer than the model's declared `idle_timeout_seconds`,
freeing memory without waiting for traffic-driven LRU eviction.

The sweeper is a CLIENT of LoadDirector's public surface
(`director.lock`, `director.loaded`, `director.disable_fn`,
`director.memory_probe`, `director.recent_decisions`); no source-side
changes to LoadDirector itself.

Lock discipline mirrors the on-demand LRU eviction path:

  1. Snapshot under `director.lock`. Drop lock.
  2. For each entry, evaluate skip conditions WITHOUT lock:
     - manifest's `capabilities.idle_timeout_seconds` is None or <= 0.
     - `entry.refcount > 0` (in-flight).
     - `time.monotonic() - entry.last_touched_at < idle_timeout`.
     - catalog_lookup raises (KeyError = model removed mid-tick): skip.
  3. For each candidate that passed all three checks: re-check under
     `director.lock` (refcount may have changed); pop from
     `director.loaded`; capture `free_before`; drop lock.
  4. Run `director.disable_fn(candidate.model_id)` OUTSIDE the lock.
     - On exception: re-insert the LoadEntry preserving its original
       `last_touched_at` + `refcount`, log warning, continue.
  5. Capture `free_after`, append a DecisionLogEntry with
     `reason=f"idle_timeout:{int(idle_timeout)}s"` under the lock.

Thread loop: `while not stop_event.is_set(): self.tick();
stop_event.wait(self.interval_seconds)`. The wait()-based sleep allows
fast shutdown when stop_event is set; tick() exceptions are logged with
exc_info and the loop continues so a bug in tick doesn't take down the
sweeper.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

from muse.cli_impl.load_director import DecisionLogEntry

if TYPE_CHECKING:
    from muse.cli_impl.load_director import LoadDirector, LoadEntry

logger = logging.getLogger(__name__)


class IdleSweeper:
    """Background sweeper that idle-evicts loaded models.

    Constructor injection:
      - director: LoadDirector. The sweeper reads `director.loaded`
        under `director.lock`, calls `director.disable_fn`, probes
        `director.memory_probe`, and appends to
        `director.recent_decisions`. Never mutates LoadDirector
        internals beyond what the public surface allows.
      - catalog_lookup: Callable[[str], dict]. model_id -> manifest
        dict. Per-tick lookup so a `muse models remove` mid-tick
        cleanly skips that candidate. Implementations should raise
        KeyError when a model is no longer in the catalog; the sweeper
        catches and silently continues.
      - interval_seconds: float. How often the thread loop ticks.
        Default 30.0 (matches `MUSE_IDLE_SWEEP_INTERVAL_SECONDS=30`).
      - stop_event: threading.Event | None. Optional external event so
        the supervisor can share `state.stop_event`. If None, the
        sweeper creates its own and `stop()` flips it.
    """

    def __init__(
        self,
        *,
        director: "LoadDirector",
        catalog_lookup: Callable[[str], dict],
        interval_seconds: float = 30.0,
        stop_event: threading.Event | None = None,
    ):
        self.director = director
        self.catalog_lookup = catalog_lookup
        self.interval_seconds = interval_seconds
        self._stop_event = stop_event if stop_event is not None else threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> list[str]:
        """Per-iteration body. Returns list of model_ids evicted on this
        tick.

        Exposed so unit tests can drive the sweeper without spinning up
        the real thread loop. The thread loop in `_run` calls this
        repeatedly between waits.
        """
        # 1. Snapshot under lock; drop lock for filtering.
        with self.director.lock:
            loaded_snapshot = list(self.director.loaded.values())

        # 2. Filter without lock.
        candidates: list[tuple["LoadEntry", float]] = []
        now = time.monotonic()
        for entry in loaded_snapshot:
            try:
                manifest = self.catalog_lookup(entry.model_id)
            except KeyError:
                # Model removed from catalog mid-tick. Silently skip;
                # LoadDirector cleanup handles it on the next acquire.
                continue
            except Exception:  # noqa: BLE001
                # Any other lookup error: log + skip this entry but
                # keep ticking. Do not let an upstream catalog bug
                # stop idle eviction for OTHER models.
                logger.warning(
                    "idle sweeper: catalog_lookup(%r) raised; skipping",
                    entry.model_id,
                    exc_info=True,
                )
                continue

            idle_timeout = (
                manifest.get("capabilities", {}).get("idle_timeout_seconds")
            )
            if idle_timeout is None:
                continue
            try:
                idle_timeout_f = float(idle_timeout)
            except (TypeError, ValueError):
                # Operator put a non-numeric in the manifest. Skip.
                logger.warning(
                    "idle sweeper: %r has non-numeric idle_timeout_seconds=%r; skipping",
                    entry.model_id, idle_timeout,
                )
                continue
            if idle_timeout_f <= 0:
                # Operator typo (0 or negative). Defensive skip.
                continue

            if entry.refcount > 0:
                continue

            if now - entry.last_touched_at < idle_timeout_f:
                continue

            candidates.append((entry, idle_timeout_f))

        # 3 + 4 + 5: per candidate, re-check + pop + disable_fn + log.
        evicted_ids: list[str] = []
        for candidate, idle_timeout_f in candidates:
            evicted_id = self._evict_candidate(candidate, idle_timeout_f)
            if evicted_id is not None:
                evicted_ids.append(evicted_id)

        return evicted_ids

    def start(self) -> threading.Thread:
        """Start the sweeper thread (daemon=True). Returns the Thread."""
        thread = threading.Thread(
            target=self._run,
            name="muse-idle-sweeper",
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self) -> None:
        """Set the stop_event so the thread exits on its next iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_candidate(
        self, candidate: "LoadEntry", idle_timeout_f: float,
    ) -> str | None:
        """Re-check, pop, disable_fn, log. Returns model_id if evicted."""
        device = "cpu"
        try:
            manifest = self.catalog_lookup(candidate.model_id)
            device = str(
                manifest.get("capabilities", {}).get("device", "cpu")
            ).lower()
        except KeyError:
            # Removed between snapshot and now: skip silently.
            return None
        except Exception:  # noqa: BLE001
            logger.warning(
                "idle sweeper: catalog_lookup(%r) raised during candidate handling; skipping",
                candidate.model_id, exc_info=True,
            )
            return None

        # Re-check under lock; the refcount may have changed since the
        # snapshot. If it did, the model has live traffic and must not
        # be evicted on this tick.
        with self.director.lock:
            current = self.director.loaded.get(candidate.model_id)
            if current is None:
                # Already evicted by a concurrent path (e.g. on-demand
                # LRU eviction from a cold acquire). Nothing to do.
                return None
            if current.refcount > 0:
                # Mid-flight request grabbed the model since snapshot;
                # try again on the next tick.
                return None

            # Capture free_before under the lock to keep the measurement
            # tightly bound to the eviction commitment.
            free_before_gb = self._free_for_device(device)

            # Pop now so other threads see the eviction commitment
            # immediately. If disable_fn raises we'll re-insert.
            self.director.loaded.pop(candidate.model_id, None)

        # ---- lock released: slow disable_fn step ----
        try:
            self.director.disable_fn(candidate.model_id)
        except Exception as exc:  # noqa: BLE001
            # Same remediation as on-demand LRU eviction's failed-victim
            # path: re-insert the popped LoadEntry so accounting matches
            # the still-alive worker. Preserve last_touched_at +
            # refcount verbatim so LRU ordering and idle-timeout
            # arithmetic remain coherent next tick.
            logger.warning(
                "idle sweeper: disable_fn(%r) raised: %s; re-inserted into loaded set",
                candidate.model_id, exc,
                exc_info=True,
            )
            with self.director.lock:
                self.director.loaded[candidate.model_id] = candidate
            return None

        # 5. Append DecisionLogEntry under lock.
        free_after_gb = self._free_for_device(device)
        with self.director.lock:
            self.director.recent_decisions.append(DecisionLogEntry(
                timestamp=time.time(),
                model_id=candidate.model_id,
                action="evict",
                memory_gb=candidate.memory_gb,
                free_before_gb=free_before_gb,
                free_after_gb=free_after_gb,
                reason=f"idle_timeout:{int(idle_timeout_f)}s",
                evicted=[candidate.model_id],
            ))

        return candidate.model_id

    def _free_for_device(self, device: str) -> float:
        """Live free memory in GB for the relevant device.

        Mirrors LoadDirector._free_for_device but called from the
        sweeper. Replicated locally rather than reaching into the
        director's private API so the sweeper remains a clean client
        of the documented public surface.
        """
        probe = self.director.memory_probe
        if device in ("cuda", "gpu"):
            free = probe.gpu_free_gb()
            return float(free) if free is not None else 0.0
        return float(probe.cpu_free_gb())

    def _run(self) -> None:
        """Thread loop body. Runs until `_stop_event` is set."""
        logger.info(
            "idle sweeper: starting (interval=%.1fs)",
            self.interval_seconds,
        )
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                # Catch-all so a tick bug doesn't kill the sweeper.
                # KeyboardInterrupt / SystemExit pass through.
                logger.exception("IdleSweeper.tick crashed; continuing")
            # wait()-based sleep so stop() unblocks immediately.
            self._stop_event.wait(self.interval_seconds)
        logger.info("idle sweeper: stopped")
