"""Unit tests for muse.cli_impl.telemetry CLI helpers and commands."""
from __future__ import annotations

import csv
import json
import time

import pytest

from muse.cli_impl import telemetry
from muse.core import config
from muse.observability.events import EVENT_COLUMNS, event_to_row
from muse.observability.store import TelemetryStore


@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    config.reset_config()
    yield tmp_path / "catalog"
    config.reset_config()


def _seed_store(catalog, rows):
    catalog.mkdir(parents=True, exist_ok=True)
    with TelemetryStore(catalog / "telemetry.db") as store:
        store.insert_many(rows)


@pytest.mark.parametrize("bad", ["", "   ", "0s", "-5m", "nan", "inf", "abc", "1x"])
def test_parse_duration_rejects_invalid(bad):
    with pytest.raises(ValueError):
        telemetry.parse_duration(bad)


def test_fmt_helpers_reject_bool_and_non_finite():
    assert telemetry._fmt_ms(True) == "--"
    assert telemetry._fmt_ms(float("nan")) == "--"
    assert telemetry._fmt_ms(float("inf")) == "--"
    assert telemetry._fmt_ms(42.5) == "42.5"
    assert telemetry._fmt_ms(125.4) == "125"
    assert telemetry._fmt_gb(False) == "--"
    assert telemetry._fmt_gb(float("nan")) == "--"
    assert telemetry._fmt_gb(3.5) == "3.50"


def test_run_status_missing_and_populated(capsys, _isolated_catalog):
    assert telemetry.run_status(as_json=True) == 0
    missing = json.loads(capsys.readouterr().out)
    assert missing["exists"] is False
    assert missing["first_ts"] is None
    assert missing["last_ts"] is None

    now = time.time()
    _seed_store(_isolated_catalog, [
        event_to_row("request", now - 60.0, model_id="tts", status=200),
        event_to_row("sample", now - 10.0, free_ram_gb=8.0),
    ])
    assert telemetry.run_status(as_json=False) == 0
    out = capsys.readouterr().out
    assert "Range (UTC):" in out
    assert "request: 1" in out
    assert "sample: 1" in out


def test_run_traces_includes_queue_ms_and_filters(capsys, _isolated_catalog):
    now = time.time()
    _seed_store(_isolated_catalog, [
        event_to_row(
            "request", now - 20.0, request_id="r1", model_id="tts",
            modality="audio/speech", cold=True, latency_ms=500.0,
            load_ms=300.0, forward_ms=150.0, queued_ms=50.0,
            peak_vram_gb=4.0, evicted_models='["old"]', status=200,
        ),
        event_to_row(
            "request", now - 10.0, request_id="r2", model_id="llm",
            modality="chat/completions", cold=None, latency_ms=80.0,
            queued_ms=12.5, status=200,
        ),
    ])

    assert telemetry.run_traces(since="1h", model_id="tts") == 0
    out = capsys.readouterr().out
    assert "Queue ms" in out
    assert "50.0" in out
    assert "tts" in out
    assert "llm" not in out


def test_run_vram_all_zero_does_not_render_wall_of_hashes(capsys, _isolated_catalog):
    now = time.time()
    _seed_store(_isolated_catalog, [
        event_to_row("sample", now - 20.0, gpu_used_gb=0.0, free_vram_gb=12.0),
        event_to_row("sample", now - 10.0, gpu_used_gb=0.0, free_vram_gb=12.0),
    ])

    assert telemetry.run_vram(since="1h", width=20) == 0
    out = capsys.readouterr().out
    assert "peak 0.00 GB" in out
    assert "#" not in out


def test_run_export_validates_upfront_and_writes_csv_header_even_when_empty(
    tmp_path, _isolated_catalog,
):
    nested = tmp_path / "never-created" / "out.csv"
    with pytest.raises(ValueError, match="format"):
        telemetry.run_export(since="1h", output=nested, format="yaml")
    assert not nested.parent.exists()

    target_dir = tmp_path / "existing-dir"
    target_dir.mkdir()
    with pytest.raises(ValueError, match="directory"):
        telemetry.run_export(since="1h", output=target_dir, format="csv")

    empty_csv = tmp_path / "empty.csv"
    assert telemetry.run_export(since="1h", output=empty_csv, format="csv") == 0
    with empty_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [list(EVENT_COLUMNS)]

    now = time.time()
    _seed_store(_isolated_catalog, [
        event_to_row("request", now - 5.0, model_id="tts", status=200),
    ])
    populated_csv = tmp_path / "populated.csv"
    assert telemetry.run_export(since="1h", output=populated_csv, format="csv") == 0
    with populated_csv.open("r", encoding="utf-8", newline="") as handle:
        dict_rows = list(csv.DictReader(handle))
    assert len(dict_rows) == 1
    assert dict_rows[0]["model_id"] == "tts"


def test_run_prune_dry_run_and_delete(capsys, _isolated_catalog):
    now = time.time()
    _seed_store(_isolated_catalog, [
        event_to_row("sample", now - 7200.0, free_ram_gb=8.0),
        event_to_row("sample", now - 10.0, free_ram_gb=8.0),
    ])

    assert telemetry.run_prune(older_than="1h", dry_run=True) == 0
    assert "Would prune 1 events" in capsys.readouterr().out

    assert telemetry.run_prune(older_than="1h", dry_run=False) == 0
    assert "Pruned 1 events" in capsys.readouterr().out

    with TelemetryStore(_isolated_catalog / "telemetry.db") as store:
        assert store.summary_counts()["total"] == 1
