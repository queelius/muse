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
    - Eviction: deferred to Task C. If the requested memory does not fit
      live free (minus headroom), this raises NotImplementedError with a
      "Task C" marker. Cleanup of the in-flight Event happens before the
      raise so a later acquire under different memory conditions is not
      blocked.

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

This module does NOT import enable_model / disable_model from
muse.admin.operations. The callable injection seam (enable_fn,
disable_fn, memory_probe) is what tests use; production wiring (Task E)
will inject the real supervisor's spawn / shutdown callables.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


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
    release. Task C's eviction sorts evictable candidates by this field
    ascending (LRU first).
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
      - disable_fn(model_id) -> None: complement to enable_fn. Used by
        Task C's eviction loop. Task B never calls this, but stores it
        for Task C.
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
          NotImplementedError: when memory does not fit live free minus
            headroom. Task C will replace this with an LRU-eviction
            attempt and (if that fails) an OperationError(503).
          Exception: re-raises any exception from enable_fn after
            cleaning up the in-flight Event so concurrent waiters wake
            and re-enter the decision phase.
        """
        # Loop because a thread that lost the singleton race and waited
        # on an in_flight Event may wake to find that the winner failed
        # (no LoadEntry was inserted). On wake, we re-enter the decision
        # phase rather than blindly trusting the winner.
        while True:
            phase_decision, port = self._decide(model_id, manifest=manifest)

            if phase_decision == "hot":
                # _decide returns the port read under the same lock that
                # classified the entry as loaded. No TOCTOU window: an
                # eviction (Task C) cannot race between classification
                # and read because both happen inside the lock.
                return port

            if phase_decision == "wait":
                # We did not own the load; another thread did. Wait for
                # its Event to fire and then re-decide.
                event = self._get_in_flight_event(model_id)
                if event is not None:
                    event.wait()
                # Loop and re-decide. Either the entry is now hot (winner
                # succeeded) or it's still cold (winner raised) and this
                # thread will become the new winner.
                continue

            # phase_decision == "load": this thread won the singleton race.
            # The Event is already in self.in_flight_loads under our name.
            # Run the (long) load phase outside the lock, then commit.
            return self._load_and_commit(model_id, manifest=manifest)

    def release(self, model_id: str) -> None:
        """Decrement refcount + bump last_touched_at.

        Task B does NOT auto-evict on release: eviction is on-demand,
        triggered only from a cold acquire that needs room (Task C).

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

    # ------------------------------------------------------------------
    # Internals: decision / load / commit
    # ------------------------------------------------------------------

    def _decide(
        self, model_id: str, *, manifest: dict,
    ) -> tuple[str, int | None]:
        """First phase, under the lock.

        Returns a (phase, port_or_none) tuple:
          ("hot", port): entry already loaded; refcount + last_touched
            bumped under the lock; port is the worker_port read under
            the same lock (no TOCTOU window for an evictor to race in).
          ("wait", None): another thread owns the load; caller awaits the Event.
          ("load", None): we just claimed ownership; run load phase + commit.

        On the ("load", None) return path, an Event has been stashed in
        self.in_flight_loads[model_id]. The caller is responsible for
        either committing (success) or popping + setting the Event
        (failure). _commit and _abort cover both cases.
        """
        with self.lock:
            entry = self.loaded.get(model_id)
            if entry is not None:
                entry.refcount += 1
                entry.last_touched_at = time.monotonic()
                # Read port under the same lock that classified the
                # entry; an evictor (Task C) cannot race between the
                # classification and this read.
                return ("hot", entry.worker_port)

            if model_id in self.in_flight_loads:
                return ("wait", None)

            # We're going to do this load. Decide whether it fits before
            # we claim the in-flight slot, so that if the answer is "no"
            # we don't strand an Event.
            memory_gb = float(manifest.get("capabilities", {}).get("memory_gb", 0.0) or 0.0)
            device = str(manifest.get("capabilities", {}).get("device", "cpu")).lower()

            free_before_gb, available_gb = self._available_for_device(device)
            if memory_gb > available_gb:
                # TODO(Task C): replace this raise with an LRU-eviction
                # loop. The eviction loop should:
                #   - identify candidates: loaded entries with refcount == 0
                #   - sort by last_touched_at ascending (LRU first)
                #   - call disable_fn(victim) for each, polling the
                #     memory_probe until freed_gb >= shortfall (or 2s
                #     timeout per eviction)
                #   - log a DecisionLogEntry per eviction with action="evict"
                #     and reason=f"evicted_for_{model_id}"
                #   - if we run out of candidates and still don't fit:
                #     raise OperationError("model_too_large_for_device", 503)
                # The decision above (memory_gb, available_gb, free_before_gb)
                # is what the eviction loop needs as input; preserve those
                # locals when implementing.
                raise NotImplementedError(
                    "model %r needs %.2f GB but only %.2f GB available on %s; "
                    "eviction handled in Task C of v0.40.0" % (
                        model_id, memory_gb, available_gb, device,
                    )
                )

            # Claim ownership of the load.
            event = threading.Event()
            self.in_flight_loads[model_id] = event
            return ("load", None)

    def _get_in_flight_event(self, model_id: str) -> threading.Event | None:
        """Read the current in-flight Event under the lock.

        The Event may have been removed between _decide returning "wait"
        and this call (the winner committed and pop'd it). In that case
        we return None and the caller falls through to re-decide.
        """
        with self.lock:
            return self.in_flight_loads.get(model_id)

    def _load_and_commit(self, model_id: str, *, manifest: dict) -> int:
        """Run the load phase outside the lock, then commit.

        On exception, performs cleanup so concurrent waiters wake up,
        no stale LoadEntry exists, and the exception propagates.
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
                refcount=1,
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
