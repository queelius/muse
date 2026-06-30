"""Tests for the `muse models set-device` CLI verb.

The verb writes a per-model `device_override` into the catalog (operator
knob), overriding the manifest device pin + --device flag at load time
(see catalog.load_backend precedence). Catalog-only: takes effect on the
model's next cold load.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import typer


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    yield tmp_path


def _pull_soprano():
    from muse.core.catalog import pull
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")


def test_set_device_writes_override(tmp_catalog):
    from muse.cli import models_set_device, Device
    from muse.core.catalog import _read_catalog
    _pull_soprano()

    models_set_device("soprano-80m", Device.cuda, clear=False)

    assert _read_catalog()["soprano-80m"]["device_override"] == "cuda"


def test_set_device_clear_removes_override(tmp_catalog):
    from muse.cli import models_set_device, Device
    from muse.core.catalog import _read_catalog
    _pull_soprano()

    models_set_device("soprano-80m", Device.cuda, clear=False)
    models_set_device("soprano-80m", None, clear=True)

    assert "device_override" not in _read_catalog()["soprano-80m"]


def test_set_device_unknown_model_exits_2(tmp_catalog):
    from muse.cli import models_set_device, Device
    with pytest.raises(typer.Exit) as exc:
        models_set_device("never-pulled-xyz", Device.cuda, clear=False)
    assert exc.value.exit_code == 2


def test_set_device_no_device_and_no_clear_exits_2(tmp_catalog):
    """Calling with neither a device nor --clear is a usage error."""
    from muse.cli import models_set_device
    _pull_soprano()
    with pytest.raises(typer.Exit) as exc:
        models_set_device("soprano-80m", None, clear=False)
    assert exc.value.exit_code == 2


def test_main_propagates_nonzero_exit_for_unknown_model(tmp_catalog):
    """The shipped `muse` binary (muse.cli:main) must return the nonzero
    exit code, not swallow it to 0.

    Regression: with standalone_mode=False, click handles typer.Exit
    internally and *returns* the exit code as app()'s return value; main
    discarded that and hardcoded `return 0`, so every error path exited 0
    on the real binary (the subprocess tests use `python -m muse.cli` ->
    app() directly, so they never exercised main()).
    """
    from muse.cli import main
    rc = main(["models", "set-device", "ghost-model-xyz", "cuda"])
    assert rc == 2


def test_main_returns_zero_on_success(tmp_catalog):
    """A successful command still returns 0 through main()."""
    from muse.cli import main
    _pull_soprano()
    assert main(["models", "set-device", "soprano-80m", "cuda"]) == 0
