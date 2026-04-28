"""Tests for `muse models probe` via run_probe().

run_probe() is the parent-side dispatcher: it validates the model, finds
its venv, spawns the in-venv probe worker via subprocess.run, parses the
JSON record on the last line of stdout, and persists it under
catalog['<id>']['measurements']['<device>'].

These tests stub subprocess.run with crafted stdout so we don't actually
spawn anything. The probe-worker's own behavior is covered separately
in test_probe_worker.py.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from muse.cli_impl.probe import run_probe
from muse.core.catalog import (
    _read_catalog,
    _reset_known_models_cache,
    _write_catalog,
)


@pytest.fixture(autouse=True)
def _isolate_catalog_cache():
    _reset_known_models_cache()
    yield
    _reset_known_models_cache()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    yield tmp_path


def _seed_pulled_entry(tmp_catalog, model_id="kokoro-82m"):
    """Write a fake pulled-entry catalog record so is_pulled returns True."""
    catalog_state = {
        model_id: {
            "pulled_at": "2026-04-28T00:00:00+00:00",
            "hf_repo": "hexgrad/Kokoro-82M",
            "local_dir": str(tmp_catalog / "weights" / model_id),
            "venv_path": str(tmp_catalog / "venvs" / model_id),
            "python_path": str(tmp_catalog / "venvs" / model_id / "bin" / "python"),
            "enabled": True,
        },
    }
    _write_catalog(catalog_state)


def test_run_probe_unknown_model_returns_error(tmp_catalog, capsys):
    rc = run_probe(
        model_id="does-not-exist",
        no_inference=False,
        device=None,
        as_json=False,
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown model" in err
    assert "does-not-exist" in err


def test_run_probe_not_pulled_returns_error(tmp_catalog, capsys):
    """Existing bundled id but not pulled: tell the user to pull."""
    rc = run_probe(
        model_id="kokoro-82m",
        no_inference=False,
        device=None,
        as_json=False,
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not pulled" in err
    assert "muse pull kokoro-82m" in err


def test_run_probe_subprocess_failure_returns_error(tmp_catalog, capsys):
    _seed_pulled_entry(tmp_catalog)
    fake_completed = MagicMock()
    fake_completed.returncode = 7
    fake_completed.stdout = ""
    fake_completed.stderr = "boom"
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed):
        rc = run_probe(
            model_id="kokoro-82m",
            no_inference=False,
            device=None,
            as_json=False,
        )
    assert rc == 7
    err = capsys.readouterr().err
    assert "exited 7" in err
    assert "boom" in err


def test_run_probe_persists_measurement_to_catalog(tmp_catalog):
    _seed_pulled_entry(tmp_catalog)
    record = {
        "model_id": "kokoro-82m",
        "modality": "audio/speech",
        "device": "cpu",
        "weights_bytes": 500_000_000,
        "peak_bytes": 600_000_000,
        "load_seconds": 1.5,
        "ran_inference": True,
        "shape": "5s synthesis",
        "probed_at": "2026-04-28T00:00:00+00:00",
    }
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = "baseline log\n" + json.dumps(record) + "\n"
    fake_completed.stderr = ""
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed):
        rc = run_probe(
            model_id="kokoro-82m",
            no_inference=False,
            device=None,
            as_json=False,
        )
    assert rc == 0
    catalog = _read_catalog()
    assert "measurements" in catalog["kokoro-82m"]
    assert "cpu" in catalog["kokoro-82m"]["measurements"]
    persisted = catalog["kokoro-82m"]["measurements"]["cpu"]
    assert persisted["peak_bytes"] == 600_000_000
    assert persisted["shape"] == "5s synthesis"


def test_run_probe_json_output_emits_record(tmp_catalog, capsys):
    _seed_pulled_entry(tmp_catalog)
    record = {
        "model_id": "kokoro-82m",
        "device": "cpu",
        "weights_bytes": 1024,
        "peak_bytes": 2048,
        "ran_inference": False,
        "load_seconds": 0.5,
        "probed_at": "2026-04-28T00:00:00+00:00",
    }
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = json.dumps(record)
    fake_completed.stderr = ""
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed):
        rc = run_probe(
            model_id="kokoro-82m",
            no_inference=True,
            device=None,
            as_json=True,
        )
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["peak_bytes"] == 2048


def test_run_probe_no_output_returns_error(tmp_catalog, capsys):
    _seed_pulled_entry(tmp_catalog)
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = ""
    fake_completed.stderr = ""
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed):
        rc = run_probe(
            model_id="kokoro-82m",
            no_inference=False,
            device=None,
            as_json=False,
        )
    assert rc == 4
    err = capsys.readouterr().err
    assert "no output" in err


def test_run_probe_non_json_output_returns_error(tmp_catalog, capsys):
    _seed_pulled_entry(tmp_catalog)
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = "not json at all"
    fake_completed.stderr = ""
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed):
        rc = run_probe(
            model_id="kokoro-82m",
            no_inference=False,
            device=None,
            as_json=False,
        )
    assert rc == 4
    err = capsys.readouterr().err
    assert "not JSON" in err


def test_run_probe_passes_no_inference_flag_to_subprocess(tmp_catalog):
    _seed_pulled_entry(tmp_catalog)
    record = {
        "model_id": "kokoro-82m",
        "device": "cpu",
        "weights_bytes": 1,
        "peak_bytes": 1,
        "ran_inference": False,
        "load_seconds": 0.0,
        "probed_at": "2026-04-28T00:00:00+00:00",
    }
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = json.dumps(record)
    fake_completed.stderr = ""
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed) as mock_run:
        run_probe(
            model_id="kokoro-82m",
            no_inference=True,
            device=None,
            as_json=False,
        )
    cmd = mock_run.call_args.args[0]
    assert "--no-inference" in cmd


def test_run_probe_uses_per_model_venv_python(tmp_catalog):
    _seed_pulled_entry(tmp_catalog)
    record = {
        "model_id": "kokoro-82m", "device": "cpu",
        "weights_bytes": 1, "peak_bytes": 1, "ran_inference": False,
        "load_seconds": 0.0, "probed_at": "2026-04-28T00:00:00+00:00",
    }
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = json.dumps(record)
    fake_completed.stderr = ""
    with patch("muse.cli_impl.probe.subprocess.run", return_value=fake_completed) as mock_run:
        run_probe(
            model_id="kokoro-82m", no_inference=False, device=None, as_json=False,
        )
    cmd = mock_run.call_args.args[0]
    expected_py = str(tmp_catalog / "venvs" / "kokoro-82m" / "bin" / "python")
    assert cmd[0] == expected_py


def test_run_probe_timeout_returns_error(tmp_catalog, capsys):
    _seed_pulled_entry(tmp_catalog)
    import subprocess as sp
    with patch(
        "muse.cli_impl.probe.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="x", timeout=600),
    ):
        rc = run_probe(
            model_id="kokoro-82m", no_inference=False, device=None, as_json=False,
        )
    assert rc == 3
    err = capsys.readouterr().err
    assert "timed out" in err
