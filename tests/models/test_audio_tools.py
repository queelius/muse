"""audio-tools: the weightless, CPU-only bundled model for ``audio/process``."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from muse.modalities.audio_process.runtime import AudioTools
from muse.models.audio_tools import MANIFEST, Model


def test_audio_tools_manifest():
    assert MANIFEST["model_id"] == "audio-tools"
    assert MANIFEST["modality"] == "audio/process"
    assert MANIFEST["weights"] == "none"
    for key in ("hf_repo", "revision", "allow_patterns", "hf_artifacts"):
        assert key not in MANIFEST, key
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "cpu"
    assert caps["memory_gb"] == 0.5
    assert {"numpy", "scipy"} <= {
        e.split(">")[0].split("=")[0] for e in MANIFEST["pip_extras"]
    }
    for key in ("summary", "strengths", "limits", "quality_tier", "license", "description"):
        assert MANIFEST.get(key), key


_BLOCK_SCIPY = """
import importlib.abc, sys

class _NoScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked for this test")
        return None

sys.meta_path.insert(0, _NoScipy())
"""


def test_discovered_and_importable_without_scipy(tmp_path):
    """scipy is not a base muse dependency: script discovery and modality import
    must succeed without importing scipy."""
    code = _BLOCK_SCIPY + (
        "import muse.models.audio_tools as m\n"
        "assert m.MANIFEST['model_id'] == 'audio-tools'\n"
        "from muse.core.catalog import known_models\n"
        "assert 'audio-tools' in known_models(), sorted(known_models())\n"
        "import muse.modalities.audio_process\n"
        "assert 'scipy' not in sys.modules, 'scipy was imported'\n"
        "print('ok')\n"
    )
    env = {**os.environ, "MUSE_CATALOG_DIR": str(tmp_path / "catalog")}
    env.pop("MUSE_MODELS_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("ok")
    assert "audio_tools.py: import failed" not in proc.stderr


def test_model_is_the_runtime():
    assert Model is AudioTools


def test_constructor_accepts_loader_kwargs():
    m = Model(
        hf_repo="",
        local_dir="/nowhere",
        device="cpu",
        model_id="audio-tools",
        revision=None,
        memory_gb=0.5,
        future_kwarg=1,
    )
    assert m.model_id == "audio-tools"
    assert m.device == "cpu"


def _fake_create_venv(target: Path) -> None:
    python = Path(target) / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)


def _forbidden(*_a, **_k):
    raise AssertionError("audio-tools is weightless: nothing may be downloaded")


@pytest.fixture
def fresh_catalog(tmp_path, monkeypatch):
    from muse.core.catalog_models import _reset_known_models_cache
    from muse.core.catalog_store import _reset_read_catalog_cache

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


def test_pull_then_load_backend_end_to_end(fresh_catalog):
    """Weightless pull + load_backend + PROBE_DEFAULTS works end-to-end."""
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
        assert pull("audio-tools") == "audio-tools"
    assert install.called
    _reset_read_catalog_cache()
    row = _read_catalog()["audio-tools"]
    assert str(_catalog_dir()) in row["local_dir"]
    assert row["weights"] == "none" and row["enabled"] is True
    assert model_load_blocker("audio-tools", row) is None

    backend = load_backend("audio-tools")
    assert isinstance(backend, AudioTools)
    assert backend.model_id == "audio-tools"

    from muse.modalities.audio_process import PROBE_DEFAULTS

    metrics = PROBE_DEFAULTS["call"](backend)
    assert metrics["peak"] <= 0.999
    assert metrics["frames"] > 0
