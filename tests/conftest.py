"""Suite-wide guards against host-state mutation from editable installs."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_host_configuration(tmp_path, monkeypatch):
    """Keep editable-install tests from reading or mutating operator state."""
    from muse.core import config
    from muse.core.catalog import (
        _reset_known_models_cache,
        _reset_read_catalog_cache,
    )

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("MUSE_STORAGE_AUTO_PRUNE_BEFORE_PULL", "0")
    config.reset_config()
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    config.reset_config()
    _reset_known_models_cache()
    _reset_read_catalog_cache()


@pytest.fixture(autouse=True)
def _darwin_process_semantics_everywhere(monkeypatch):
    """``MUSE_TEST_DARWIN_PROCESS_SEMANTICS=1``: run the suite with macOS's
    zombie rules for ``killpg`` / ``getpgid`` (see
    ``tests/darwin_process_semantics.py``)."""
    from tests import darwin_process_semantics

    if darwin_process_semantics.globally_enabled():
        if not darwin_process_semantics.supported():
            pytest.skip("darwin process-semantics emulation needs Linux /proc")
        darwin_process_semantics.install(monkeypatch)
    yield
