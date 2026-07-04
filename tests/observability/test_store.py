import pytest

from muse.observability.store import TelemetryStore
from muse.observability.events import event_to_row


@pytest.fixture
def store(tmp_path):
    s = TelemetryStore(tmp_path / "t.db")
    yield s
    s.close()


def test_insert_and_request_rate_bucketing(store):
    rows = [event_to_row("request", ts, model_id="m", latency_ms=10.0, status=200)
            for ts in (100.0, 101.0, 102.0, 160.0)]
    store.insert_many(rows)
    out = store.series("request_rate", since_ts=0.0, bucket_seconds=60.0)
    assert out["metric"] == "request_rate"
    # bucket [60,120) has 3 requests, [120,180) has 1
    counts = {p["t"]: p["count"] for p in out["points"]}
    assert counts[120.0] == 3 and counts[180.0] == 1


def test_latency_series(store):
    store.insert_many([event_to_row("request", 61.0, model_id="m", latency_ms=x, status=200)
                       for x in (10.0, 20.0, 30.0)])
    out = store.series("latency", since_ts=0.0, bucket_seconds=60.0)
    p = out["points"][0]
    assert p["avg"] == 20.0 and p["max"] == 30.0


def test_prune(store):
    store.insert_many([event_to_row("sample", ts, free_vram_gb=1.0) for ts in (10.0, 5000.0)])
    removed = store.prune(older_than_ts=100.0)
    assert removed == 1
    assert store.summary_counts()["total"] == 1
