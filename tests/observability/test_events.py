from muse.observability.events import EVENT_COLUMNS, event_to_row


def test_event_to_row_fills_all_columns_with_none():
    row = event_to_row("request", 1.0, model_id="m", latency_ms=12.5, status=200)
    assert set(row) == set(EVENT_COLUMNS)
    assert row["type"] == "request" and row["ts"] == 1.0
    assert row["model_id"] == "m" and row["latency_ms"] == 12.5 and row["status"] == 200
    assert row["pool"] is None and row["free_vram_gb"] is None  # unset -> None


def test_event_to_row_rejects_unknown_field():
    import pytest
    with pytest.raises(ValueError):
        event_to_row("request", 1.0, bogus=1)
