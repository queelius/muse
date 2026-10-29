"""A failed post-pull probe fails `muse pull` and leaves the model disabled.

Regression for the 2026-10-18 install audit: the CLI ignored the probe's
return code, so a model that could not load or run inference still pulled
with exit 0 and `enabled: true`.
"""
import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import muse.cli_impl.probe as probe_mod
import muse.core.catalog as catalog
from muse.cli import app
from muse.core.catalog import (
    _read_catalog,
    _reset_known_models_cache,
    _reset_read_catalog_cache,
)
from tests import darwin_process_semantics


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _reset_read_catalog_cache()
    _reset_known_models_cache()
    yield tmp_path
    _reset_read_catalog_cache()
    _reset_known_models_cache()


def _fake_pull(tmp_path, model_id="m1"):
    def _pull(identifier, **kw):
        data = {model_id: {"local_dir": str(tmp_path / "w"), "enabled": True}}
        (tmp_path / "catalog.json").write_text(json.dumps(data))
        _reset_read_catalog_cache()
        return model_id
    return _pull


def _run(monkeypatch, tmp_path, probe_rc=None, probe_exc=None, args=("pull", "hf://o/m1")):
    monkeypatch.setattr(catalog, "pull", _fake_pull(tmp_path))
    calls = []

    def _probe(*, model_id, **kw):
        calls.append(model_id)
        if probe_exc is not None:
            raise probe_exc
        return probe_rc

    monkeypatch.setattr(probe_mod, "run_probe", _probe)
    result = CliRunner().invoke(app, list(args))
    _reset_read_catalog_cache()
    return result, calls


def test_failed_probe_exits_nonzero_and_disables(monkeypatch, tmp_catalog):
    result, calls = _run(monkeypatch, tmp_catalog, probe_rc=2)

    assert result.exit_code == 1
    assert calls == ["m1"]
    entry = _read_catalog()["m1"]
    assert entry["enabled"] is False
    assert entry["local_dir"]  # weights kept for a retry without re-download
    assert "muse models probe m1" in result.output
    assert "muse models enable m1" in result.output
    assert "--no-probe" in result.output


def test_probe_internal_error_also_fails_the_pull(monkeypatch, tmp_catalog):
    result, _ = _run(monkeypatch, tmp_catalog, probe_exc=RuntimeError("boom"))

    assert result.exit_code == 1
    assert _read_catalog()["m1"]["enabled"] is False
    assert "boom" in result.output


def test_successful_probe_keeps_model_enabled(monkeypatch, tmp_catalog):
    result, calls = _run(monkeypatch, tmp_catalog, probe_rc=0)

    assert result.exit_code == 0, result.output
    assert calls == ["m1"]
    assert _read_catalog()["m1"]["enabled"] is True


def test_no_probe_skips_and_keeps_model_enabled(monkeypatch, tmp_catalog):
    result, calls = _run(
        monkeypatch, tmp_catalog, probe_rc=2, args=("pull", "hf://o/m1", "--no-probe"),
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert _read_catalog()["m1"]["enabled"] is True


def test_pull_returns_the_installed_id_for_a_bundled_model(tmp_catalog):
    with patch("muse.core.catalog_pull.create_venv"), \
         patch("muse.core.catalog_pull.install_into_venv"), \
         patch("muse.core.catalog_pull.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog_pull.check_system_packages", return_value=[]):
        assert catalog.pull("kokoro-82m") == "kokoro-82m"


@pytest.mark.parametrize("platform", darwin_process_semantics.PLATFORMS)
def test_admin_pull_job_fails_when_the_cli_exits_nonzero(platform, monkeypatch):
    """The admin pull job runs `muse pull` as a subprocess, so the new
    nonzero exit marks the job failed with the CLI's stderr.

    ``no-waitid`` is the macOS path (CPython before 3.13 does not build
    ``os.waitid`` there), which used to fail every admin job as "subprocess
    identity is ambiguous".
    """
    import sys

    darwin_process_semantics.apply_platform(platform, monkeypatch)

    from muse.admin import operations
    from muse.admin.jobs import JobStore

    store = JobStore()
    job = store.create("pull", "m1")
    cmd = [
        sys.executable, "-c",
        "import sys; sys.stderr.write('error: m1 was installed but failed "
        "its post-pull probe'); sys.exit(1)",
    ]
    operations._run_subprocess_into_job(cmd, store=store, job=job, success_op="pull")

    final = store.get(job.job_id)
    assert final.state == "failed"
    assert "failed its post-pull probe" in final.error
