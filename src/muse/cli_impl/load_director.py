"""LoadDirector: lazy-load coordinator for v0.40.0.

Holds the in-memory map of currently-loaded models, gates concurrent loads
of the same model behind a per-model threading.Event (singleton-load
collapse), and routes decisions to an injected memory probe + enable/
disable callable seam so unit tests don't need a real supervisor.

Three-phase acquire (decide / load / commit):

  Decision (under self.lock)
    - Hot path: increment refcount, update last_touched_at, return port.
    - Cold path: claim ownership of the load by stashing a fresh
      threading.Event in `in_flight_loads`. Other threads racing on the
      same model_id see the Event, drop the lock, and wait. When woken,
      they re-enter the decision phase to read the freshly populated
      LoadEntry.
    - Eviction (Task C): if the requested memory does not fit live free
      (minus headroom), the decision phase returns ("evict_and_retry",
      shortfall_gb, device) WITHOUT claiming an in-flight slot. The
      acquire outer loop runs eviction outside the lock and re-calls
      _decide, which may classify the entry as hot (another thread
      loaded it during our eviction window), or cold-fits, or raise 503
      if eviction can't free enough memory. Re-decide on every retry
      protects against TOCTOU windows during the unlocked eviction.

  Load (lock NOT held)
    - Capture free_before via memory_probe.
    - Call enable_fn(model_id), which is the long-running worker spawn.
    - Capture free_after.
    - Failures here are caught in the commit phase: the in-flight Event
      is popped + set so waiters wake up, no LoadEntry is recorded, and
      the original exception propagates to the caller.

  Commit (under self.lock)
    - Insert LoadEntry, pop the in_flight_loads Event, append a
      DecisionLogEntry, set() the Event so any concurrent waiters wake.
    - Schedule the observed-peak writeback (Task D): a fire-and-forget
      daemon thread compares observed delta vs the recorded
      `measurements.<device>.peak_bytes` and raises it if observation
      exceeds the seed. Errors are logged + swallowed so the request
      hot path is unaffected.

This module does NOT import enable_model / disable_model from
muse.admin.operations at import time (it does lazy-import OperationError
inside the eviction failure path). The callable injection seam
(enable_fn, disable_fn, memory_probe) is what tests use; production
wiring (Task E) will inject the real supervisor's spawn / shutdown
callables.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

# Task D writeback uses the existing catalog read / atomic-write helpers.
# These are imported at module scope so tests can monkeypatch them on the
# load_director module's namespace (see tests/cli_impl/test_load_director.py
# `test_swallows_ioerror_during_write` etc.).
from muse.core.catalog import _read_catalog, _write_catalog

logger = logging.getLogger(__name__)


# Task D: serialize observed-peak writebacks across models so concurrent
# cold loads of DIFFERENT models cannot lose updates via interleaved
# read-modify-write. The atomic write-then-rename in `_write_catalog`
# ensures one writer's tmp -> final rename does not corrupt another's
# pending tmp file, but it does NOT prevent thread A from reading the
# catalog, thread B reading + rewriting it, and thread A then writing
# its (now stale) version back -- losing B's update. A single global
# Lock on the writeback path eliminates the read-modify-write race
# without coupling to LoadDirector.lock (which is reserved for in-memory
# state mutations on the request hot path).
_WRITEBACK_LOCK = threading.Lock()


# Default headroom margins in GB. Subtracted from live free before the
# fit check. Spec defaults: 1 GB GPU, 2 GB CPU. Constructor accepts
# overrides so tests can pin any value they like.
_DEFAULT_GPU_HEADROOM_GB = 1.0
_DEFAULT_CPU_HEADROOM_GB = 2.0


@dataclass
class LoadEntry:
    """One currently-loaded model.

    `memory_gb` is what we accounted for at load time (from the
    manifest's capabilities.memory_gb, falling back to 0.0). Task D
    will add observed-peak writeback so this drifts toward measured
    reality as cold loads happen.

    `last_touched_at` is monotonic seconds, updated on each acquire and
    release. Eviction sorts evictable candidates (refcount == 0) by this
    field ascending (LRU first).
    """
    model_id: str
    worker_port: int
    memory_gb: float
    refcount: int
    last_touched_at: float
    loaded_at: float


@dataclass
class DecisionLogEntry:
    """One load or evict decision, surfaced in /v1/admin/memory.

    `timestamp` is wall-clock seconds (time.time(), not monotonic) so
    the admin endpoint can render an ISO string for humans. The deque
    holding these is capped at maxlen=20 by LoadDirector.

    `evicted` is always a list (possibly empty) so the wire shape is
    uniform across load and evict actions.
    """
    timestamp: float
    model_id: str
    action: Literal["load", "evict"]
    memory_gb: float
    free_before_gb: float
    free_after_gb: float | None
    reason: str
    evicted: list[str] = field(default_factory=list)


class LoadDirector:
    """Coordinates lazy loads + LRU eviction for the supervisor.

    Constructor injection:
      - enable_fn(model_id) -> worker_port: the long-running call to
        spawn (or wake) the per-model worker. In production this is a
        thin wrapper around `muse.admin.operations.enable_model`; in
        tests it's a MagicMock returning a fixed port.
      - disable_fn(model_id) -> None: complement to enable_fn. Called
        by the eviction loop on each LRU victim. The OS reclaims the
        worker's memory after SIGTERM; the director polls free memory
        with a 2s budget per victim before retrying the fit check.
      - memory_probe: any object with .gpu_free_gb() and .cpu_free_gb()
        methods. In production this is a `muse.core.memory_probe.MemoryProbe`;
        in tests it's a MagicMock with .return_value set on each method.

    Concurrency:
      - `lock` is an RLock so the same thread can re-enter (e.g. status
        snapshot from inside an admin route that already holds it).
      - `in_flight_loads` maps model_id to threading.Event. The Event is
        created under the lock at decision time; waiters drop the lock,
        await it, then re-enter the decision phase. The winner sets the
        Event in the commit phase (or on exception in the cleanup path).
      - `recent_decisions` is a deque(maxlen=20) appended to under the
        lock; admin reads it via list() while holding the lock for a
        consistent snapshot.
    """

    def __init__(
        self,
        *,
        enable_fn: Callable[[str], int],
        disable_fn: Callable[[str], None],
        memory_probe: Any,
        gpu_budget_gb: float | None = None,
        cpu_budget_gb: float | None = None,
        gpu_headroom_gb: float = _DEFAULT_GPU_HEADROOM_GB,
        cpu_headroom_gb: float = _DEFAULT_CPU_HEADROOM_GB,
    ):
        self.enable_fn = enable_fn
        self.disable_fn = disable_fn
        self.memory_probe = memory_probe

        self.gpu_budget_gb = gpu_budget_gb
        self.cpu_budget_gb = cpu_budget_gb
        self.gpu_headroom_gb = gpu_headroom_gb
        self.cpu_headroom_gb = cpu_headroom_gb

        self.loaded: dict[str, LoadEntry] = {}
        self.in_flight_loads: dict[str, threading.Event] = {}
        self.recent_decisions: collections.deque = collections.deque(maxlen=20)
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, model_id: str, *, manifest: dict) -> int:
        """Increment refcount on a loaded model, or load it cold.

        The three phases run as documented in the module docstring.
        Returns the worker_port hosting `model_id`.

        Raises:
          OperationError("model_too_large_for_device", status=503): when
            memory doesn't fit and on-demand LRU eviction can't free
            enough (no evictable candidates, or sum of evictable memory
            < shortfall).
          Exception: re-raises any exception from enable_fn after
            cleaning up the in-flight Event so concurrent waiters wake
            and re-enter the decision phase.
        """
        return self._acquire_or_warmup(
            model_id, manifest=manifest, bump_refcount=True,
        )

    def warmup(self, model_id: str, *, manifest: dict) -> int:
        """Pre-load a model without serving a request.

        Like `acquire` but does NOT increment refcount on either the hot
        or cold path. The semantic is "load this model now so subsequent
        requests are hot." The loaded entry's initial refcount is 0,
        making it immediately eligible for LRU eviction if pressure
        arises before any request lands.

        Hot path (model already loaded): returns the existing port
        without touching refcount or last_touched_at. The intent is
        idempotency: repeated warmup calls must not skew LRU ordering
        or pin the model artificially.

        Cold path: runs the full decide / load / commit cycle including
        on-demand LRU eviction. The committed LoadEntry has refcount=0,
        which is the only behavioral difference from acquire.

        Returns the worker_port hosting `model_id` after the warmup.

        Raises:
          OperationError("model_too_large_for_device", status=503): when
            memory doesn't fit and on-demand LRU eviction can't free
            enough (same as acquire).
          Exception: re-raises any exception from enable_fn after
            cleaning up the in-flight Event so concurrent waiters wake
            and re-enter the decision phase.
        """
        return self._acquire_or_warmup(
            model_id, manifest=manifest, bump_refcount=False,
        )

    def _acquire_or_warmup(
        self, model_id: str, *, manifest: dict, bump_refcount: bool,
    ) -> int:
        """Shared body for acquire (bump_refcount=True) and warmup (False).

        The two operations differ only in:
          - Hot path: acquire bumps refcount + last_touched; warmup
            doesn't.
          - Cold commit: acquire seeds the LoadEntry refcount at 1;
            warmup seeds at 0.
        """
        # Loop because a thread that lost the singleton race and waited
        # on an in_flight Event may wake to find that the winner failed
        # (no LoadEntry was inserted). On wake, we re-enter the decision
        # phase rather than blindly trusting the winner.
        #
        # The same loop also handles the "evict_and_retry" cycle: when
        # _decide reports the model doesn't fit, we run eviction OUTSIDE
        # the lock, then re-decide. The state may have changed during
        # the unlocked eviction window: another thread may have loaded
        # the model already, or stolen our victim, or pushed yet more
        # memory pressure. Re-decide is the correct recovery.
        while True:
            decision = self._decide(
                model_id, manifest=manifest, bump_refcount=bump_refcount,
            )
            phase = decision[0]

            if phase == "hot":
                # _decide returns the port read under the same lock that
                # classified the entry as loaded. No TOCTOU window: an
                # eviction cannot race between classification and read
                # because both happen inside the lock.
                return decision[1]

            if phase == "wait":
                # We did not own the load; another thread did. Wait for
                # its Event to fire and then re-decide.
                event = self._get_in_flight_event(model_id)
                if event is not None:
                    event.wait()
                # Loop and re-decide. Either the entry is now hot (winner
                # succeeded) or it's still cold (winner raised) and this
                # thread will become the new winner.
                continue

            if phase == "evict_and_retry":
                # Run eviction outside the lock so concurrent acquires
                # for already-loaded models can hot-acquire during the
                # disable_fn + memory-release-poll window. Eviction
                # raises OperationError(503) if it can't free enough.
                _, shortfall_gb, device = decision
                self._evict_lru_until_fits(
                    model_id=model_id,
                    shortfall_gb=shortfall_gb,
                    device=device,
                )
                # Re-enter the decision phase. State may have changed:
                # another thread may have loaded our model, or evicted
                # additional models, or grown the memory pressure.
                continue

            # phase == "load": this thread won the singleton race.
            # The Event is already in self.in_flight_loads under our name.
            # Run the (long) load phase outside the lock, then commit.
            initial_refcount = 1 if bump_refcount else 0
            return self._load_and_commit(
                model_id, manifest=manifest, initial_refcount=initial_refcount,
            )

    def release(self, model_id: str) -> None:
        """Decrement refcount + bump last_touched_at.

        Release does NOT auto-evict: eviction is on-demand, triggered
        only from a cold acquire that needs room. The release call
        merely flips the refcount toward 0 (making the entry eligible
        for eviction on the next pressure point) and updates the LRU
        timestamp so traffic recency drives the eviction order.

        Releasing an unknown model_id is a no-op (defensive: an evicted
        model whose final request just finished should not crash the
        gateway's `finally:` clause).
        """
        with self.lock:
            entry = self.loaded.get(model_id)
            if entry is None:
                logger.debug("release(%r): model not in loaded set; ignoring", model_id)
                return
            entry.refcount = max(0, entry.refcount - 1)
            entry.last_touched_at = time.monotonic()

    def status(self) -> dict[str, dict[str, Any]]:
        """Snapshot of currently-loaded models for /v1/models lookup.

        Each value is `{"loaded": True, "worker_port", "last_touched_at",
        "refcount"}`. Models not currently loaded are absent.
        """
        with self.lock:
            return {
                mid: {
                    "loaded": True,
                    "worker_port": e.worker_port,
                    "last_touched_at": e.last_touched_at,
                    "refcount": e.refcount,
                }
                for mid, e in self.loaded.items()
            }

    def observed_peak(
        self,
        model_id: str,
        *,
        observed_peak_bytes: int,
        device: str,
    ) -> threading.Thread | None:
        """Schedule a passive writeback of the observed peak.

        Spec section "Lazy-load passive observation": every cold load
        compares the observed `free_before - free_after` delta against
        the recorded `catalog[model_id].measurements.<device>.peak_bytes`.
        If observed > recorded (or recorded is missing), the catalog gets
        the new larger value via the existing atomic write-then-rename
        in `muse.core.catalog._write_catalog`.

        This is fire-and-forget: a daemon thread does the
        read-modify-write so the request hot path is unaffected.
        Failures are logged at WARNING level and swallowed because a
        corrupted catalog file or transient filesystem error must NOT
        surface as a 500 to the user. Returns the Thread reference (so
        tests can deterministically join it via .join()); production
        callers can ignore the return value.

        `observed_peak_bytes <= 0` is treated as a no-op (negative deltas
        mean another process freed memory during our load window, or the
        measurement was meaningless; we do not corrupt a recorded peak
        with a zero or negative observation).

        `device` is the catalog measurements bucket key. Existing probe
        records use "cuda" / "cpu" / "mps"; the alias "gpu" normalizes
        to "cuda", and the manifest convention "auto" / "" is resolved
        to the device the load actually consumed (cuda if a GPU is
        available, else cpu) so the bucket key matches the probe's.
        """
        if observed_peak_bytes <= 0:
            return None

        # Normalize the bucket key. Three cases:
        #   "gpu" -> "cuda": legacy alias.
        #   "auto" / "": resolved to "cuda" if memory_probe reports a
        #     GPU is available, else "cpu". Without this, manifests
        #     declaring `device: "auto"` would split-brain against
        #     probe records that always write the resolved name.
        #   anything else: passed through verbatim ("cpu", "cuda",
        #     "mps", etc.).
        if device == "gpu":
            device_key = "cuda"
        elif device in ("auto", ""):
            gpu_free = self.memory_probe.gpu_free_gb()
            device_key = "cuda" if gpu_free is not None else "cpu"
        else:
            device_key = device

        thread = threading.Thread(
            target=self._observed_peak_writeback,
            args=(model_id, observed_peak_bytes, device_key),
            daemon=True,
            name=f"observed-peak-writeback-{model_id}",
        )
        thread.start()
        return thread

    @staticmethod
    def _observed_peak_writeback(
        model_id: str,
        observed_peak_bytes: int,
        device_key: str,
    ) -> None:
        """Body of the writeback thread; runs OFF the request hot path.

        Held under the module-level _WRITEBACK_LOCK so concurrent cold
        loads of different models do not lose updates via interleaved
        read-modify-write.

        Errors are logged at WARNING and swallowed: the catalog write
        is best-effort, not an invariant.
        """
        try:
            with _WRITEBACK_LOCK:
                catalog = _read_catalog()
                entry = catalog.get(model_id)
                if entry is None:
                    # Model removed (e.g. `muse models remove`) since the
                    # load completed. Nothing to write back; not an error.
                    logger.debug(
                        "observed_peak: %r not in catalog; skipping writeback",
                        model_id,
                    )
                    return

                measurements = entry.setdefault("measurements", {})
                bucket = measurements.get(device_key) or {}
                recorded = int(bucket.get("peak_bytes") or 0)
                if observed_peak_bytes <= recorded:
                    # Estimate is monotonically upward only.
                    return

                bucket["peak_bytes"] = int(observed_peak_bytes)
                bucket["device"] = device_key
                bucket["source"] = "lazy_load_observation"
                bucket["observed_at"] = datetime.now(timezone.utc).isoformat()
                # If `weights_bytes` is absent (no probe ever ran), seed
                # it with the observed peak as a conservative lower bound.
                # This keeps existing /v1/models renderers working even
                # when only lazy-load measurements exist.
                bucket.setdefault("weights_bytes", int(observed_peak_bytes))
                measurements[device_key] = bucket

                _write_catalog(catalog)

                logger.info(
                    "observed_peak writeback: %s (%s): %d bytes (was %d)",
                    model_id,
                    device_key,
                    observed_peak_bytes,
                    recorded,
                )
        except Exception as exc:  # noqa: BLE001
            # Any IO / OS / JSON failure is logged + swallowed. The
            # writeback is best-effort; the next cold load will retry.
            logger.warning(
                "observed_peak writeback failed for %s (%s): %s",
                model_id,
                device_key,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internals: decision / load / commit
    # ------------------------------------------------------------------

    def _decide(
        self, model_id: str, *, manifest: dict, bump_refcount: bool = True,
    ) -> tuple:
        """First phase, under the lock.

        Returns one of these tuple shapes:
          ("hot", port): entry already loaded; refcount + last_touched
            bumped under the lock (when bump_refcount=True); port is the
            worker_port read under the same lock (no TOCTOU window for
            an evictor to race in).
          ("wait",): another thread owns the load; caller awaits the Event.
          ("load",): we just claimed ownership; run load phase + commit.
          ("evict_and_retry", shortfall_gb, device): the model does not
            fit live free minus headroom. Caller runs eviction OUTSIDE
            the lock and re-calls _decide. No in-flight slot is claimed
            on this path so the eviction loop's lock-release-and-reacquire
            pattern is safe across the boundary.

        `bump_refcount` is True for acquire (refcount drives in-flight
        request accounting) and False for warmup (load idempotently
        without skewing LRU ordering or pinning the model).

        On the ("load",) return path, an Event has been stashed in
        self.in_flight_loads[model_id]. The caller is responsible for
        either committing (success) or popping + setting the Event
        (failure). _commit and _abort cover both cases.
        """
        with self.lock:
            entry = self.loaded.get(model_id)
            if entry is not None:
                if bump_refcount:
                    entry.refcount += 1
                    entry.last_touched_at = time.monotonic()
                # Read port under the same lock that classified the
                # entry; an evictor cannot race between the classification
                # and this read.
                return ("hot", entry.worker_port)

            if model_id in self.in_flight_loads:
                return ("wait",)

            # We're going to do this load. Decide whether it fits before
            # we claim the in-flight slot, so that if the answer is "no"
            # we don't strand an Event.
            memory_gb = float(manifest.get("capabilities", {}).get("memory_gb", 0.0) or 0.0)
            device = str(manifest.get("capabilities", {}).get("device", "cpu")).lower()

            _, available_gb = self._available_for_device(device)
            if memory_gb > available_gb:
                shortfall_gb = memory_gb - available_gb
                # Defer eviction to acquire(), which runs it outside the
                # lock so concurrent hot-acquires keep working during
                # the (potentially multi-second) disable_fn + poll
                # window. We do NOT claim an in_flight_loads slot here:
                # if eviction fails (503), there's nothing to clean up;
                # if eviction succeeds, the retry will re-decide and
                # claim the slot then.
                return ("evict_and_retry", shortfall_gb, device)

            # Claim ownership of the load.
            event = threading.Event()
            self.in_flight_loads[model_id] = event
            return ("load",)

    def _get_in_flight_event(self, model_id: str) -> threading.Event | None:
        """Read the current in-flight Event under the lock.

        The Event may have been removed between _decide returning "wait"
        and this call (the winner committed and pop'd it). In that case
        we return None and the caller falls through to re-decide.
        """
        with self.lock:
            return self.in_flight_loads.get(model_id)

    def _load_and_commit(
        self, model_id: str, *, manifest: dict, initial_refcount: int = 1,
    ) -> int:
        """Run the load phase outside the lock, then commit.

        On exception, performs cleanup so concurrent waiters wake up,
        no stale LoadEntry exists, and the exception propagates.

        `initial_refcount` is 1 for acquire (a request is being served)
        and 0 for warmup (no request, just pre-load).

        After a successful commit, schedules the observed-peak writeback
        (Task D): a daemon thread compares the observed `free_before -
        free_after` delta against the catalog's recorded peak and raises
        the recorded value if observation exceeds it. The thread starts
        OFF the request hot path so the caller is unaffected by catalog
        IO latency or filesystem failures.
        """
        memory_gb = float(manifest.get("capabilities", {}).get("memory_gb", 0.0) or 0.0)
        device = str(manifest.get("capabilities", {}).get("device", "cpu")).lower()

        free_before_gb = self._free_for_device(device)
        try:
            worker_port = self.enable_fn(model_id)
        except BaseException:
            # Cleanup: pop the Event, set it so waiters wake (they will
            # find no LoadEntry and become the new winner), and re-raise.
            self._abort(model_id)
            raise

        free_after_gb = self._free_for_device(device)

        with self.lock:
            now = time.monotonic()
            entry = LoadEntry(
                model_id=model_id,
                worker_port=worker_port,
                memory_gb=memory_gb,
                refcount=initial_refcount,
                last_touched_at=now,
                loaded_at=now,
            )
            self.loaded[model_id] = entry

            self.recent_decisions.append(DecisionLogEntry(
                timestamp=time.time(),
                model_id=model_id,
                action="load",
                memory_gb=memory_gb,
                free_before_gb=free_before_gb,
                free_after_gb=free_after_gb,
                reason="fit",
                evicted=[],
            ))

            event = self.in_flight_loads.pop(model_id, None)

        # event.set() outside the lock is fine; threading.Event is its
        # own synchronization. Doing it inside the lock would block any
        # awakened waiter on its first re-entry into _decide.
        if event is not None:
            event.set()

        # Task D: schedule the observed-peak writeback. Compute the delta
        # in bytes; observed_peak swallows non-positive values internally.
        # Spawn happens AFTER waiters wake up so the writeback IO can't
        # delay them.
        observed_bytes = int((free_before_gb - free_after_gb) * 1024**3)
        self.observed_peak(
            model_id,
            observed_peak_bytes=observed_bytes,
            device=device,
        )

        return worker_port

    def _abort(self, model_id: str) -> None:
        """Clean up after a failed load: drop + signal the Event.

        Called from the exception handler in _load_and_commit. After
        this, the model_id is not in self.in_flight_loads and any
        waiting threads have been woken up to re-enter the decision
        phase. No LoadEntry exists, so on re-decide they will become
        the new singleton winner.
        """
        with self.lock:
            event = self.in_flight_loads.pop(model_id, None)
        if event is not None:
            event.set()

    # ------------------------------------------------------------------
    # Eviction (Task C)
    # ------------------------------------------------------------------

    def _evict_lru_until_fits(
        self,
        *,
        model_id: str,
        shortfall_gb: float,
        device: str,
    ) -> None:
        """Evict refcount==0 models in LRU order until shortfall is met.

        Lock discipline (the critical detail of Task C):
          1. Acquire the lock briefly to snapshot evictable candidates
             and pop the LRU victim from self.loaded. Append a partial
             DecisionLogEntry under the lock so the admin endpoint sees
             the in-progress eviction as soon as it commits to one.
          2. RELEASE the lock for the slow steps (disable_fn invocation
             + memory release polling). Other threads can hot-acquire
             during this window: their _decide call takes the lock,
             observes the (still-loaded) other entries, and proceeds.
             Cold acquires that race here will see a smaller free pool
             until our poll completes, but that's accurate.
          3. REACQUIRE the lock to update the DecisionLogEntry's
             free_after_gb once the poll returns.
          4. If the cumulative freed memory still falls short of the
             original shortfall, loop and pick the next victim. If no
             candidates remain: lazy-import OperationError and raise 503.

        The reacquire step is intentionally minimal (just the entry
        mutation), so other admin endpoints don't stall behind the
        post-poll commit.
        """
        cumulative_freed_gb = 0.0
        # Track victims whose disable_fn raised so we don't re-pick them
        # on the next iteration. A re-inserted victim retains its
        # last_touched_at and would otherwise sort back to LRU position
        # and loop forever. The set is local to this call: a future
        # acquire is welcome to retry the failed victim, since the
        # worker may have died on its own by then.
        failed_victims: set[str] = set()

        while cumulative_freed_gb < shortfall_gb:
            # ---- under-lock: pick + pop a victim ----
            with self.lock:
                # Re-snapshot every iteration: another thread may have
                # released a model since the last pass, making it newly
                # evictable.
                candidates = [
                    e for e in self.loaded.values()
                    if e.refcount == 0 and e.model_id not in failed_victims
                ]
                if not candidates:
                    # Pop the in-flight Event for the requested model in
                    # case anything stranded one (defensive: the current
                    # _decide path doesn't, but a future path might).
                    self.in_flight_loads.pop(model_id, None)
                    # Lazy import to avoid a cycle: load_director ->
                    # admin.operations -> supervisor -> load_director
                    # (Task E will wire the supervisor to LoadDirector).
                    from muse.admin.operations import OperationError
                    raise OperationError(
                        "model_too_large_for_device",
                        (
                            f"cannot fit {model_id!r} on {device}: "
                            f"shortfall {shortfall_gb:.2f} GB; "
                            f"no evictable candidates remain "
                            f"(all loaded models have refcount > 0)"
                        ),
                        status=503,
                    )

                # LRU first.
                candidates.sort(key=lambda e: e.last_touched_at)
                victim = candidates[0]
                victim_id = victim.model_id
                victim_memory_gb = victim.memory_gb

                # Pop from loaded so other threads see the eviction
                # commitment immediately. If disable_fn raises, the
                # state is already consistent (no zombie LoadEntry
                # claiming a worker port that isn't running).
                self.loaded.pop(victim_id, None)

                free_before_gb = self._free_for_device(device)

                decision = DecisionLogEntry(
                    timestamp=time.time(),
                    model_id=victim_id,
                    action="evict",
                    memory_gb=victim_memory_gb,
                    free_before_gb=free_before_gb,
                    free_after_gb=None,
                    reason=f"evicted_for_{model_id}",
                    evicted=[victim_id],
                )
                self.recent_decisions.append(decision)

            # ---- lock released: slow steps ----
            try:
                self.disable_fn(victim_id)
            except Exception as exc:  # noqa: BLE001
                # disable_fn raised. The worker is still alive holding
                # memory: the OS has NOT reclaimed its slot. Polling for
                # release would observe ~0 freed and the loop would
                # iterate to the next victim, but the orphan worker
                # would persist indefinitely with no remediation path.
                #
                # Remediation: re-insert the victim's LoadEntry into
                # self.loaded so accounting matches reality (worker
                # alive, slot occupied). Append a structured "evict"
                # decision log entry recording the failure for the
                # admin endpoint. Skip cumulative_freed_gb (the slot
                # did not free). Continue to the next candidate.
                #
                # KeyboardInterrupt + SystemExit pass through (we catch
                # Exception, not BaseException) so the user can still
                # Ctrl-C out of a stuck eviction.
                logger.error(
                    "disable_fn failed for evicted victim %r; re-inserted into loaded set",
                    victim_id,
                    exc_info=True,
                )
                with self.lock:
                    # Re-insert the popped LoadEntry. refcount was 0 at
                    # eviction time (only refcount==0 entries are
                    # candidates) and we preserve that.
                    self.loaded[victim_id] = victim
                    # Update the existing decision entry's reason
                    # in-place. The deque already holds it; mutating
                    # the dataclass is sufficient. evicted=[] because
                    # nothing was actually evicted.
                    decision.reason = f"disable_fn_raised: {exc}"
                    decision.evicted = []
                    # free_after_gb stays None: no poll, no measurement.
                # Mark this victim as off-limits for the rest of this
                # eviction call. Re-picking would loop forever (the
                # re-inserted victim still has the oldest
                # last_touched_at and would sort to LRU position).
                failed_victims.add(victim_id)
                continue

            freed_this_round = self._wait_for_memory_release(
                min_freed_gb=victim_memory_gb,
                free_at_eviction_start_gb=free_before_gb,
                device=device,
            )

            # ---- reacquire the lock briefly: writeback free_after ----
            with self.lock:
                free_after_gb = self._free_for_device(device)
                decision.free_after_gb = free_after_gb

            cumulative_freed_gb += freed_this_round

    def _wait_for_memory_release(
        self,
        *,
        min_freed_gb: float,
        free_at_eviction_start_gb: float,
        device: str,
        timeout: float = 2.0,
        poll: float = 0.05,
    ) -> float:
        """Poll memory_probe until enough memory has freed (or timeout).

        Returns the actual freed_gb observed at exit. The caller compares
        this against the expected per-victim freeing to decide whether
        to evict another victim.

        The director lock is NOT held while this runs. Other threads can
        progress through their own decision phases during the poll.

        Negative `freed_gb` is clamped to 0.0: if memory pressure grew
        during the poll (another process allocated VRAM, or our own
        eviction observation lagged), we don't double-count it as a
        freeing.
        """
        deadline = time.monotonic() + timeout
        while True:
            current_free_gb = self._free_for_device(device)
            freed = current_free_gb - free_at_eviction_start_gb
            if freed < 0.0:
                freed = 0.0

            if freed >= min_freed_gb:
                return freed
            if time.monotonic() >= deadline:
                return freed

            time.sleep(poll)

    # ------------------------------------------------------------------
    # Memory accounting helpers
    # ------------------------------------------------------------------

    def _free_for_device(self, device: str) -> float:
        """Live free memory in GB for the relevant device.

        For CPU: cpu_free_gb. For CUDA: gpu_free_gb (which may return
        None if pynvml is unavailable; caller treats None as 0.0 so we
        don't accidentally classify GPU loads as fitting under unknown
        conditions).
        """
        if device in ("cuda", "gpu"):
            free = self.memory_probe.gpu_free_gb()
            return float(free) if free is not None else 0.0
        # cpu, mps, auto, anything else: treat as host-RAM accounting
        return float(self.memory_probe.cpu_free_gb())

    def _available_for_device(self, device: str) -> tuple[float, float]:
        """Return (free_before_gb, available_gb).

        available_gb = min(free, declared_budget) - headroom. Negative
        values are clamped to 0 so a request for a 0-byte model on a
        completely-loaded host still gets a sane fit-check.
        """
        free = self._free_for_device(device)
        if device in ("cuda", "gpu"):
            budget = self.gpu_budget_gb
            headroom = self.gpu_headroom_gb
        else:
            budget = self.cpu_budget_gb
            headroom = self.cpu_headroom_gb

        usable = free if budget is None else min(free, budget)
        available = usable - headroom
        if available < 0.0:
            available = 0.0
        return free, available
