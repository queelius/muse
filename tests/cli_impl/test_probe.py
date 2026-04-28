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


def test_run_probe_all_with_no_enabled_models_returns_1(tmp_path, monkeypatch, capsys):
    """No enabled+pulled models -> error to stderr, return 1."""
    from muse.cli_impl.probe import run_probe_all

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    with patch("muse.cli_impl.probe.known_models", return_value={}):
        rc = run_probe_all(no_inference=False, device=None, as_json=False)
    assert rc == 1
    captured = capsys.readouterr()
    assert "no enabled" in captured.err.lower()


def test_run_probe_all_iterates_enabled_models(tmp_path, monkeypatch):
    """run_probe_all should call run_probe for each enabled+pulled model."""
    from muse.core.catalog import CatalogEntry
    from muse.cli_impl.probe import run_probe_all

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    # Build fake catalog state with 3 models, 2 enabled
    fake_known = {
        "alpha": CatalogEntry("alpha", "audio/speech", "x:Y", "h/r", "", (), (), {}),
        "beta": CatalogEntry("beta", "audio/speech", "x:Y", "h/r", "", (), (), {}),
        "gamma": CatalogEntry("gamma", "audio/speech", "x:Y", "h/r", "", (), (), {}),
    }

    def fake_is_pulled(mid):
        return mid in {"alpha", "beta"}  # gamma not pulled

    fake_catalog_data = {
        "alpha": {"enabled": True},
        "beta": {"enabled": False},  # disabled
    }

    call_log = []
    def fake_run_probe(*, model_id, **kwargs):
        call_log.append(model_id)
        return 0

    with patch("muse.cli_impl.probe.known_models", return_value=fake_known), \
         patch("muse.cli_impl.probe.is_pulled", side_effect=fake_is_pulled), \
         patch("muse.cli_impl.probe._read_catalog", return_value=fake_catalog_data), \
         patch("muse.cli_impl.probe.run_probe", side_effect=fake_run_probe):
        rc = run_probe_all(no_inference=False, device=None, as_json=False)

    # Only "alpha" qualifies: pulled AND enabled. beta is pulled but disabled. gamma is not pulled.
    assert call_log == ["alpha"]
    assert rc == 0


def test_run_probe_all_continues_past_individual_failures(tmp_path, monkeypatch, capsys):
    """One probe failing must not abort the loop; final return is 1."""
    from muse.core.catalog import CatalogEntry
    from muse.cli_impl.probe import run_probe_all

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_known = {
        "alpha": CatalogEntry("alpha", "audio/speech", "x:Y", "h/r", "", (), (), {}),
        "beta": CatalogEntry("beta", "audio/speech", "x:Y", "h/r", "", (), (), {}),
        "gamma": CatalogEntry("gamma", "audio/speech", "x:Y", "h/r", "", (), (), {}),
    }

    fake_catalog_data = {mid: {"enabled": True} for mid in fake_known}

    call_log = []
    def fake_run_probe(*, model_id, **kwargs):
        call_log.append(model_id)
        return 0 if model_id != "beta" else 2  # beta fails

    with patch("muse.cli_impl.probe.known_models", return_value=fake_known), \
         patch("muse.cli_impl.probe.is_pulled", return_value=True), \
         patch("muse.cli_impl.probe._read_catalog", return_value=fake_catalog_data), \
         patch("muse.cli_impl.probe.run_probe", side_effect=fake_run_probe):
        rc = run_probe_all(no_inference=False, device=None, as_json=False)

    # All three got tried; only beta failed
    assert call_log == ["alpha", "beta", "gamma"]
    assert rc == 1
    captured = capsys.readouterr()
    assert "1 failed" in captured.out.lower() or "failed: beta" in captured.out.lower()
