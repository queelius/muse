import time, pytest
from muse.observability.store import TelemetryStore
from muse.observability import recorder as rec


@pytest.fixture(autouse=True)
def _reset():
    rec.reset_recorder(); yield; rec.reset_recorder()


def test_record_enqueues_and_flush_writes(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, flush_interval=0.05)
    r.record("request", model_id="m", latency_ms=5.0, status=200)
    r.flush()
    assert store.summary_counts()["total"] == 1
    r.stop(); store.close()


def test_overflow_drops_not_raises(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, max_queue=2)
    for _ in range(10):
        r.record("sample", free_vram_gb=1.0)   # must never raise
    assert r.dropped >= 1
    r.stop(); store.close()


def test_module_record_is_noop_until_init(tmp_path):
    rec.record("request", model_id="m")   # no recorder yet -> silent no-op, no raise
    assert rec.get_recorder().dropped == 0


def test_disabled_init_is_noop(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    rec.init_recorder(store, enabled=False)
    rec.record("request", model_id="m")
    assert store.summary_counts()["total"] == 0


def test_record_with_unknown_field_does_not_raise_and_counts_as_dropped(tmp_path):
    """Regression: event_to_row() ran BEFORE the try/except in record(),
    so an unknown kwarg (e.g. a typo'd field name) raised ValueError
    straight out of record(), violating its never-raises contract.
    """
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store)
    r.record("model_load", bogus_field=1)  # must not raise
    assert r.dropped == 1
    r.stop(); store.close()


def test_dropped_counter_increments_are_lock_protected(tmp_path):
    """Sanity check that the dropped counter still increments correctly
    under normal (single-threaded) use across both drop sites: queue
    overflow and a bad field name.
    """
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, max_queue=1)
    assert hasattr(r, "_dropped_lock")
    r.record("sample", free_vram_gb=1.0)
    r.record("sample", free_vram_gb=1.0)  # queue full -> dropped
    r.record("model_load", bogus_field=1)  # bad field -> dropped
    assert r.dropped == 2
    r.stop(); store.close()
