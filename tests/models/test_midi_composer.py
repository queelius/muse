"""Tests for the bundled ``midi-composer`` weightless CPU model script."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from muse.modalities.music_midi import PROBE_DEFAULTS
from muse.modalities.music_midi.runtimes.procedural import ProceduralMidiComposer
from muse.models.midi_composer import MANIFEST, Model


def test_midi_composer_manifest():
    assert MANIFEST["model_id"] == "midi-composer"
    assert MANIFEST["modality"] == "music/midi"
    assert MANIFEST["weights"] == "none"
    for forbidden in ("hf_repo", "revision", "allow_patterns", "hf_artifacts"):
        assert forbidden not in MANIFEST
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "cpu"
    assert caps["memory_gb"] == 0.25
    assert caps["supports_layers"] is True
    assert MANIFEST["pip_extras"] == ()
    assert MANIFEST["system_packages"] == ()
    for key in ("summary", "strengths", "limits", "quality_tier", "license", "description"):
        assert MANIFEST.get(key), key


def test_model_aliases_procedural_runtime_and_accepts_loader_kwargs():
    assert Model is ProceduralMidiComposer
    m = Model(
        hf_repo="",
        local_dir="/nowhere",
        device="cpu",
        model_id="midi-composer",
        revision=None,
        memory_gb=0.25,
        future_kwarg=123,
    )
    assert m.model_id == "midi-composer"


def _fake_create_venv(target: Path) -> None:
    python = Path(target) / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)


def _forbidden(*_a, **_k):
    raise AssertionError("midi-composer is weightless: nothing may be downloaded")


@pytest.fixture
def fresh_catalog():
    from muse.core.catalog_models import _reset_known_models_cache
    from muse.core.catalog_store import _reset_read_catalog_cache

    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


def test_pull_then_load_backend_and_probe(fresh_catalog, tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    from muse.core.catalog import _read_catalog, load_backend, model_load_blocker, pull
    from muse.core.catalog_store import _reset_read_catalog_cache

    with patch("muse.core.catalog_pull.create_venv", side_effect=_fake_create_venv), \
         patch("muse.core.catalog_pull.install_into_venv") as install, \
         patch("muse.core.catalog_pull.install_python_sources"), \
         patch("muse.core.catalog_pull.check_system_packages", return_value=[]), \
         patch("muse.core.catalog_pull.dedupe_after_install"), \
         patch("muse.core.catalog_pull.snapshot_download", side_effect=_forbidden), \
         patch("muse.core.catalog_pull.seed_repo", side_effect=_forbidden), \
         patch("muse.core.artifacts.download_hf_artifact_bundle", side_effect=_forbidden):
        assert pull("midi-composer") == "midi-composer"
    assert install.called
    _reset_read_catalog_cache()
    row = _read_catalog()["midi-composer"]
    assert row["weights"] == "none"
    assert row["enabled"] is True
    assert model_load_blocker("midi-composer", row) is None

    backend = load_backend("midi-composer")
    assert isinstance(backend, ProceduralMidiComposer)
    assert backend.model_id == "midi-composer"

    probe_out = PROBE_DEFAULTS["call"](backend)
    assert probe_out["bars"] == 4
    assert probe_out["tracks"] >= 1
