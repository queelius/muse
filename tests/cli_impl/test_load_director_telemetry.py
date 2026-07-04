"""Tests for LoadDirector telemetry instrumentation (Task 9).

Verifies that a cold load emits a "model_load" event and an eviction
emits a "model_evict" event via the fire-and-forget
`muse.observability.recorder.record` function. `record` is imported by
NAME into `muse.cli_impl.load_director` (not the module), so tests
monkeypatch that module-level name directly to capture calls without
touching the real TelemetryRecorder machinery.

Reuses the fake enable_fn / disable_fn / memory_probe harness from
tests/cli_impl/test_load_director.py (same shapes: `_make_probe` and
`_manifest` helpers, same eviction scenario as
`TestColdAcquireEvictsLRU.test_evicts_one_lru_candidate_to_make_room`).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from muse.cli_impl.load_director import LoadDirector


def _make_probe(gpu_free: float = 32.0, cpu_free: float = 64.0) -> MagicMock:
    """Default probe: lots of room. The 'fits' branch is exercised."""
    probe = MagicMock()
    probe.gpu_free_gb.return_value = gpu_free
    probe.cpu_free_gb.return_value = cpu_free
    return probe


def _manifest(memory_gb: float = 0.5, device: str = "cpu") -> dict:
    return {
        "model_id": "fake-model",
        "modality": "audio/speech",
        "capabilities": {
            "memory_gb": memory_gb,
            "device": device,
        },
    }


@pytest.fixture
def capture_records(monkeypatch):
    """Monkeypatch the `record` name on the load_director module.

    Calling the bare name `record(...)` inside LoadDirector's methods
    (rather than `observability.record(...)`) is what makes this
    monkeypatch effective: it replaces the module-global that the
    director's code actually calls.
    """
    calls: list[tuple[str, dict]] = []

    def fake_record(event_type, **kwargs):
        calls.append((event_type, kwargs))

    monkeypatch.setattr("muse.cli_impl.load_director.record", fake_record)
    return calls


class TestColdLoadEmitsTelemetry:
    def test_cold_load_emits_model_load_event(self, capture_records):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5, device="cpu"))

        load_events = [c for c in capture_records if c[0] == "model_load"]
        assert len(load_events) == 1

        _, kwargs = load_events[0]
        assert kwargs["model_id"] == "fake-model"
        assert kwargs["pool"] in ("cuda", "cpu")
        assert kwargs["gb"] == 0.5
        assert kwargs["cold_load_seconds"] >= 0.0

    def test_hot_acquire_does_not_emit_another_model_load_event(self, capture_records):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        capture_records.clear()

        # Hot path: no new cold load, so no new model_load event.
        director.acquire("fake-model", manifest=_manifest())
        assert capture_records == []


class TestEvictionEmitsTelemetry:
    def test_eviction_emits_model_evict_event(self, capture_records):
        # Mirrors TestColdAcquireEvictsLRU.test_evicts_one_lru_candidate_to_make_room
        # in tests/cli_impl/test_load_director.py.
        cpu_free_state = {"value": 10.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        def enable_side_effect(model_id: str) -> int:
            if model_id == "victim":
                cpu_free_state["value"] = 4.0
                return 9001
            if model_id == "newcomer":
                return 9002
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        def disable_side_effect(model_id: str) -> None:
            cpu_free_state["value"] = 9.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        # Pre-load "victim", release so it's evictable (refcount 0).
        director.acquire("victim", manifest=_manifest(memory_gb=5.0, device="cpu"))
        director.release("victim")
        capture_records.clear()

        # Cold-acquire "newcomer": shortfall forces eviction of "victim".
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=6.0, device="cpu"))
        assert port == 9002

        evict_events = [c for c in capture_records if c[0] == "model_evict"]
        assert len(evict_events) == 1

        _, kwargs = evict_events[0]
        assert kwargs["model_id"] == "victim"
        assert kwargs["pool"] in ("cuda", "cpu")
        assert kwargs["reason"] == "evicted_for_newcomer"

        # A model_load event for "newcomer" is also emitted.
        load_events = [c for c in capture_records if c[0] == "model_load"]
        assert any(kw["model_id"] == "newcomer" for _, kw in load_events)
