"""pixel-tools: the weightless, CPU-only bundled model for image/pixelart."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from muse.models.pixel_tools import MANIFEST, Model
from muse.modalities.image_pixelart.runtime import PixelTools


def test_pixel_tools_manifest():
    assert MANIFEST["model_id"] == "pixel-tools"
    assert MANIFEST["modality"] == "image/pixelart"
    assert MANIFEST["weights"] == "none"
    for key in ("hf_repo", "revision", "allow_patterns", "hf_artifacts"):
        assert key not in MANIFEST, key
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "cpu"
    # Ruling F: measured peak ~1.3 GB at the 4096 px input cap.
    assert caps["memory_gb"] == 1.5
    assert {"Pillow", "numpy"} <= {e.split(">")[0].split("=")[0] for e in MANIFEST["pip_extras"]}
    for key in ("summary", "strengths", "limits", "quality_tier", "license", "description"):
        assert MANIFEST.get(key), key


def test_model_is_the_runtime():
    assert Model is PixelTools


def test_constructor_accepts_loader_kwargs():
    m = Model(hf_repo="", local_dir="/nowhere", device="cpu", model_id="pixel-tools",
              revision=None, memory_gb=1.5, future_kwarg=1)
    assert m.model_id == "pixel-tools"


def _fake_create_venv(target: Path) -> None:
    python = Path(target) / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)


def _forbidden(*_a, **_k):
    raise AssertionError("pixel-tools is weightless: nothing may be downloaded")


@pytest.fixture
def fresh_catalog():
    from muse.core.catalog_models import _reset_known_models_cache
    from muse.core.catalog_store import _reset_read_catalog_cache

    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


def test_pull_then_load_backend_end_to_end(fresh_catalog):
    """The Task 1 weightless seam works for the real bundled script."""
    from muse.core.catalog import _read_catalog, load_backend, model_load_blocker, pull
    from muse.core.catalog_store import _catalog_dir, _reset_read_catalog_cache

    with patch("muse.core.catalog_pull.create_venv", side_effect=_fake_create_venv), \
         patch("muse.core.catalog_pull.install_into_venv") as install, \
         patch("muse.core.catalog_pull.install_python_sources"), \
         patch("muse.core.catalog_pull.check_system_packages", return_value=[]), \
         patch("muse.core.catalog_pull.dedupe_after_install"), \
         patch("muse.core.catalog_pull.snapshot_download", side_effect=_forbidden), \
         patch("muse.core.catalog_pull.seed_repo", side_effect=_forbidden), \
         patch("muse.core.artifacts.download_hf_artifact_bundle", side_effect=_forbidden):
        assert pull("pixel-tools") == "pixel-tools"
    assert install.called
    _reset_read_catalog_cache()
    row = _read_catalog()["pixel-tools"]
    assert str(_catalog_dir()) in row["local_dir"]
    assert row["weights"] == "none" and row["enabled"] is True
    assert model_load_blocker("pixel-tools", row) is None

    backend = load_backend("pixel-tools")
    assert isinstance(backend, PixelTools)
    assert backend.model_id == "pixel-tools"

    from muse.modalities.image_pixelart import PROBE_DEFAULTS

    PROBE_DEFAULTS["call"](backend)
