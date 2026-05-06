"""End-to-end slow test for v0.40.1 idle-timeout eviction.

Task D of the v0.40.1 plan. Unit tests in `test_idle_sweeper.py` cover
each tick branch in isolation with mocked LoadDirector internals. This
file verifies the cross-cutting integration: a real LoadDirector + a
real IdleSweeper, with the model pre-loaded into `director.loaded` and
a real wall-clock wait crossing the model's declared
`idle_timeout_seconds=1` boundary.

Why slow lane:
  - The pass test sleeps ~1.2 seconds (just over the 1-second timeout)
    so `time.monotonic() - last_touched_at` arithmetic exercises the
    real clock. A mocked clock would short-circuit this; a real wait is
    the contract that "an idle model gets evicted after its declared
    timeout" actually holds end-to-end.
  - The counter-test (no idle_timeout in the manifest) also sleeps
    ~1.2s to prove that the absence of the field is honored even when
    the model has been idle long enough to trip a timeout.

Approach: in-process. We instantiate a real LoadDirector with mocked
`enable_fn` / `disable_fn` / `memory_probe` (no subprocess spawn), then
inject a LoadEntry directly into `director.loaded` via the lock so we
don't depend on `acquire` running cold. We build a real IdleSweeper
pointed at that director and call `sweeper.tick()` synchronously after
the wait. Driving `tick()` directly (not the background thread loop)
eliminates threading-cadence flakiness while still exercising the full
sweeper integration.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from muse.cli_impl.idle_sweeper import IdleSweeper
from muse.cli_impl.load_director import LoadDirector, LoadEntry


pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_probe(gpu_free: float | None = None, cpu_free: float = 64.0) -> MagicMock:
    """Default probe: deterministic free values."""
    probe = MagicMock()
    probe.gpu_free_gb.return_value = gpu_free
    probe.cpu_free_gb.return_value = cpu_free
    return probe


def _build_director(
    *, enable_fn=None, disable_fn=None, probe=None,
) -> tuple[LoadDirector, MagicMock, MagicMock]:
    """Build a LoadDirector with mocked seams.

    Returns the director plus the enable_fn + disable_fn mocks so the
    test can assert on call_count + args without re-deriving them.
    """
    enable_mock = enable_fn if enable_fn is not None else MagicMock(return_value=9001)
    disable_mock = disable_fn if disable_fn is not None else MagicMock()
    director = LoadDirector(
        enable_fn=enable_mock,
        disable_fn=disable_mock,
        memory_probe=probe if probe is not None else _make_probe(),
    )
    return director, enable_mock, disable_mock


def _preload_entry(
    director: LoadDirector,
    model_id: str,
    *,
    memory_gb: float = 0.5,
    refcount: int = 0,
    worker_port: int = 9001,
) -> LoadEntry:
    """Inject a fresh LoadEntry into director.loaded.

    `last_touched_at` is set to the current monotonic time so the entry
    is "fresh as of now"; the test then sleeps to cross the timeout.
    """
    now = time.monotonic()
    entry = LoadEntry(
        model_id=model_id,
        worker_port=worker_port,
        memory_gb=memory_gb,
        refcount=refcount,
        last_touched_at=now,
        loaded_at=now,
    )
    with director.lock:
        director.loaded[model_id] = entry
    return entry


# ----------------------------------------------------------------------
# Pass: idle_timeout_seconds=1 with 1.2s wait -> eviction fires
# ----------------------------------------------------------------------


class TestIdleEvictionAfterTimeoutBoundary:
    """A model with `idle_timeout_seconds=1` that has been idle for 1.2s
    must be evicted on the next tick. Verifies:
      - director.loaded no longer contains the model_id.
      - disable_fn was called with the model_id.
      - recent_decisions has an entry with action="evict",
        reason="idle_timeout:1s", and evicted=[model_id].
    """

    def test_evicts_idle_model_and_records_decision(self):
        director, _enable_mock, disable_mock = _build_director()

        model_id = "idle-evict-target"
        _preload_entry(director, model_id, memory_gb=0.5)

        # Manifest declares idle_timeout_seconds=1.
        manifest = {
            "model_id": model_id,
            "modality": "audio/speech",
            "capabilities": {
                "memory_gb": 0.5,
                "device": "cpu",
                "idle_timeout_seconds": 1,
            },
        }
        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lambda mid: manifest,
            interval_seconds=0.5,
        )

        # Sleep just past the timeout boundary so monotonic arithmetic
        # in tick() classifies the entry as expired.
        time.sleep(1.2)

        evicted = sweeper.tick()

        assert evicted == [model_id], (
            f"sweeper.tick returned {evicted!r}; expected [{model_id!r}]"
        )

        # Director state: model unloaded.
        with director.lock:
            assert model_id not in director.loaded, (
                "model_id remained in director.loaded after idle eviction tick"
            )

        # disable_fn was called exactly once with the right id.
        disable_mock.assert_called_once_with(model_id)

        # Decision log entry recorded with the v0.40.1 reason format.
        with director.lock:
            decisions = list(director.recent_decisions)
        assert len(decisions) == 1, (
            f"expected exactly one decision log entry; got {decisions!r}"
        )
        decision = decisions[0]
        assert decision.action == "evict"
        assert decision.model_id == model_id
        assert decision.reason == "idle_timeout:1s"
        assert decision.evicted == [model_id]


# ----------------------------------------------------------------------
# Counter: no idle_timeout_seconds -> no eviction even after long idle
# ----------------------------------------------------------------------


class TestNoIdleTimeoutMeansNoEviction:
    """A model with NO `idle_timeout_seconds` declared in its manifest
    must NOT be evicted no matter how long it sits idle. This is the
    v0.40.0-preserving default: opt-in only.
    """

    def test_absent_idle_timeout_skips_eviction(self):
        director, _enable_mock, disable_mock = _build_director()

        model_id = "no-idle-timeout-model"
        _preload_entry(director, model_id, memory_gb=0.5)

        # Manifest has NO idle_timeout_seconds key.
        manifest = {
            "model_id": model_id,
            "modality": "audio/speech",
            "capabilities": {
                "memory_gb": 0.5,
                "device": "cpu",
            },
        }
        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lambda mid: manifest,
            interval_seconds=0.5,
        )

        # Wait the same 1.2s as the pass test; the only difference is
        # the absent capability key.
        time.sleep(1.2)

        evicted = sweeper.tick()

        assert evicted == [], (
            f"sweeper.tick evicted {evicted!r}; expected [] when "
            "idle_timeout_seconds is absent from the manifest"
        )

        # Director state: model still loaded.
        with director.lock:
            assert model_id in director.loaded, (
                "model_id was unexpectedly removed from director.loaded "
                "even though no idle_timeout_seconds was declared"
            )

        # disable_fn was NOT called.
        disable_mock.assert_not_called()

        # No decision entries recorded for this no-op tick.
        with director.lock:
            decisions = list(director.recent_decisions)
        assert decisions == [], (
            f"expected no decision log entries; got {decisions!r}"
        )

    def test_explicit_null_idle_timeout_skips_eviction(self):
        """Manifests that explicitly set `idle_timeout_seconds: null`
        (rather than omitting the key) must behave identically to the
        absent case. The sweeper treats `None` as "no idle eviction"
        end-to-end.
        """
        director, _enable_mock, disable_mock = _build_director()

        model_id = "null-idle-timeout-model"
        _preload_entry(director, model_id, memory_gb=0.5)

        manifest = {
            "model_id": model_id,
            "modality": "audio/speech",
            "capabilities": {
                "memory_gb": 0.5,
                "device": "cpu",
                "idle_timeout_seconds": None,
            },
        }
        sweeper = IdleSweeper(
            director=director,
            catalog_lookup=lambda mid: manifest,
            interval_seconds=0.5,
        )

        time.sleep(1.2)

        evicted = sweeper.tick()

        assert evicted == []
        with director.lock:
            assert model_id in director.loaded
        disable_mock.assert_not_called()
