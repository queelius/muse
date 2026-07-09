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


# v0.40.0 probe-on-pull --------------------------------------------------


def test_run_probe_for_pull_dispatches_for_curated_alias(tmp_path, monkeypatch):
    """run_probe_for_pull resolves a curated alias to its model_id and
    dispatches run_probe with that id."""
    from muse.cli_impl.probe import run_probe_for_pull

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    # Stub the catalog so the resolved id ends up in known_models +
    # catalog reads. The curated alias "qwen3-8b-q4" must resolve to
    # itself when present in the catalog.
    fake_catalog = {"qwen3-8b-q4": {"enabled": True}}
    monkeypatch.setattr(
        "muse.cli_impl.probe._read_catalog", lambda: fake_catalog,
    )

    call_log: list[str] = []
    def fake_run_probe(*, model_id, **kwargs):
        call_log.append(model_id)
        return 0

    with patch("muse.cli_impl.probe.run_probe", side_effect=fake_run_probe):
        rc = run_probe_for_pull("qwen3-8b-q4", before_keys=set())

    assert rc == 0
    assert call_log == ["qwen3-8b-q4"]


def test_run_probe_for_pull_resolves_uri_via_diff(tmp_path, monkeypatch):
    """When the identifier is a URI, run_probe_for_pull diffs the catalog
    against `before_keys` to find the new entry's model_id."""
    from muse.cli_impl.probe import run_probe_for_pull

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_catalog = {
        "existing-pre-pull": {"enabled": True},
        "freshly-pulled-id": {"enabled": True},
    }
    monkeypatch.setattr(
        "muse.cli_impl.probe._read_catalog", lambda: fake_catalog,
    )

    call_log: list[str] = []
    def fake_run_probe(*, model_id, **kwargs):
        call_log.append(model_id)
        return 0

    before = {"existing-pre-pull"}
    with patch("muse.cli_impl.probe.run_probe", side_effect=fake_run_probe):
        rc = run_probe_for_pull(
            "hf://some/repo@v",
            before_keys=before,
        )

    assert rc == 0
    assert call_log == ["freshly-pulled-id"]


def test_run_probe_for_pull_swallows_subprocess_failure(tmp_path, monkeypatch, capsys):
    """A probe that fails post-pull must NOT propagate. The pull already
    succeeded; probe is a best-effort polish step. Caller should warn
    but exit zero."""
    from muse.cli_impl.probe import run_probe_for_pull

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    fake_catalog = {"x": {"enabled": True}}
    monkeypatch.setattr(
        "muse.cli_impl.probe._read_catalog", lambda: fake_catalog,
    )

    def fake_run_probe(*, model_id, **kwargs):
        raise RuntimeError("simulated probe failure")

    with patch("muse.cli_impl.probe.run_probe", side_effect=fake_run_probe):
        # No exception escapes.
        rc = run_probe_for_pull("x", before_keys=set())

    # Non-zero is fine -- the caller propagates with a warning -- but the
    # function MUST NOT raise.
    assert isinstance(rc, int)
    err = capsys.readouterr().err
    assert "warning" in err.lower() or "probe" in err.lower()


def test_run_probe_for_pull_returns_nonzero_on_unidentified_pull(tmp_path, monkeypatch, capsys):
    """When the diff finds zero new entries (URI pull added nothing
    visible to us) we cannot identify the model. Return non-zero +
    warning so the caller knows."""
    from muse.cli_impl.probe import run_probe_for_pull

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))

    # Catalog hasn't grown.
    fake_catalog = {"existing": {"enabled": True}}
    monkeypatch.setattr(
        "muse.cli_impl.probe._read_catalog", lambda: fake_catalog,
    )

    with patch("muse.cli_impl.probe.run_probe") as mock_run_probe:
        rc = run_probe_for_pull(
            "hf://opaque/repo", before_keys={"existing"},
        )

    assert rc != 0
    mock_run_probe.assert_not_called()
    err = capsys.readouterr().err
    assert "could not identify" in err.lower() or "skipping probe" in err.lower()


class TestPynvmlFallback:
    """#330 (v0.57.1): when the venv lacks torch but the resolved device is
    cuda (GGUF split, or any torch-free runtime), the probe must measure
    VRAM via pynvml free-deltas and record device=cuda -- NOT fall back to
    RSS-as-cpu, which mis-pools the model for admission."""

    def test_pynvml_meter_free_delta(self, monkeypatch):
        from muse.cli_impl import probe_worker
        readings = iter([10.0, 7.5, 6.0])  # free GB: baseline, post-load, post-inference
        monkeypatch.setattr(
            probe_worker, "_gpu_free_gb", lambda: next(readings))
        meter = probe_worker.PynvmlVramMeter()
        assert meter.start() is True          # baseline 10.0
        assert meter.delta_bytes() == int(2.5 * 1024**3)   # 10.0 - 7.5
        assert meter.delta_bytes() == int(4.0 * 1024**3)   # 10.0 - 6.0 (monotonic max ok)

    def test_pynvml_meter_unavailable(self, monkeypatch):
        from muse.cli_impl import probe_worker
        monkeypatch.setattr(probe_worker, "_gpu_free_gb", lambda: None)
        meter = probe_worker.PynvmlVramMeter()
        assert meter.start() is False
        assert meter.delta_bytes() == 0
