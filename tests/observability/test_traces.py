from unittest.mock import MagicMock

from muse.cli_impl.load_director import LoadDirector
from muse.observability.traces import begin_request_trace, reset_request_trace


def _manifest(memory_gb=1.0):
    return {
        "model_id": "new",
        "capabilities": {"memory_gb": memory_gb, "device": "cpu"},
    }


def test_director_annotates_exact_request_cold_load_and_eviction(monkeypatch):
    events = []
    monkeypatch.setattr(
        "muse.cli_impl.load_director.record",
        lambda event, **fields: events.append((event, fields)),
    )
    free = {"value": 10.0}
    probe = MagicMock()
    probe.gpu_free_gb.return_value = 32.0
    probe.cpu_free_gb.side_effect = lambda: free["value"]

    def enable(model_id):
        if model_id == "victim":
            free["value"] = 4.0
            return 9001
        return 9002

    def disable(_model_id):
        free["value"] = 9.0

    director = LoadDirector(
        enable_fn=enable, disable_fn=disable, memory_probe=probe,
        cpu_headroom_gb=2.0,
    )
    director.acquire("victim", manifest=_manifest(5.0))
    director.release("victim")
    events.clear()

    trace, token = begin_request_trace("new", "audio/speech")
    try:
        director.acquire("new", manifest=_manifest(6.0))
        snapshot = trace.snapshot()
    finally:
        reset_request_trace(token)

    assert snapshot["cold"] is True
    assert snapshot["load_ms"] >= 0
    assert snapshot["evicted_models"] == ["victim"]
    assert {
        fields["request_id"] for _, fields in events
        if fields.get("request_id") is not None
    } == {trace.request_id}


def test_director_does_not_record_eviction_when_disable_fails(monkeypatch):
    import pytest
    from muse.admin.operations import OperationError

    events = []
    monkeypatch.setattr(
        "muse.cli_impl.load_director.record",
        lambda event, **fields: events.append((event, fields)),
    )
    free = {"value": 10.0}
    probe = MagicMock()
    probe.gpu_free_gb.return_value = 32.0
    probe.cpu_free_gb.side_effect = lambda: free["value"]

    def enable(model_id):
        if model_id == "victim":
            free["value"] = 4.0
            return 9001
        return 9002

    def disable(_model_id):
        raise RuntimeError("teardown exploded")

    director = LoadDirector(
        enable_fn=enable, disable_fn=disable, memory_probe=probe,
        cpu_headroom_gb=2.0,
    )
    director.acquire("victim", manifest=_manifest(5.0))
    director.release("victim")
    events.clear()

    trace, token = begin_request_trace("new", "audio/speech")
    try:
        with pytest.raises(OperationError):
            director.acquire("new", manifest=_manifest(6.0))
        snapshot = trace.snapshot()
    finally:
        reset_request_trace(token)

    assert snapshot["evicted_models"] == []
    assert [e for e in events if e[0] == "model_evict"] == []


def test_request_trace_timing_helpers_and_decode_evicted_models():
    from muse.observability.events import decode_evicted_models

    trace, token = begin_request_trace("tts", "audio/speech")
    try:
        trace.add_queued_ms(12.5)
        trace.add_queued_ms(7.5)
        trace.mark_forward_start(100.0)
        trace.finish_forward(100.25)
        # Second non-forced call must not overwrite already-finished forward_ms.
        trace.finish_forward(105.0)
        fields = trace.record_fields(
            status=200, stream=False, peak_vram_gb=3.25, ended_at=trace.started_at + 0.5,
        )
    finally:
        reset_request_trace(token)

    assert fields["queued_ms"] == 20.0
    assert fields["forward_ms"] == 250.0
    assert fields["latency_ms"] == 500.0
    assert fields["peak_vram_gb"] == 3.25
    assert fields["evicted_models"] == "[]"

    assert decode_evicted_models(None) == []
    assert decode_evicted_models("") == []
    assert decode_evicted_models("[]") == []
    assert decode_evicted_models('["a", "b"]') == ["a", "b"]
    assert decode_evicted_models('"single"') == ["single"]
    assert decode_evicted_models("123") == ["123"]
    assert decode_evicted_models("not-json") == ["not-json"]
    assert decode_evicted_models(["x", 1]) == ["x", "1"]
