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
