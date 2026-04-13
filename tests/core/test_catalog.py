"""Tests for the KNOWN_MODELS catalog and pull()."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.core.catalog import (
    CatalogEntry,
    KNOWN_MODELS,
    pull,
    is_pulled,
    list_known,
    load_backend,
    remove,
)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    """Point catalog state at a temp file."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    yield tmp_path


def test_known_models_entries_have_valid_modality():
    valid = {"audio.speech", "images.generations"}
    for model_id, entry in KNOWN_MODELS.items():
        assert entry.modality in valid, \
            f"model {model_id} has invalid modality {entry.modality!r}"


def test_known_models_seeded_with_required_entries():
    assert "soprano-80m" in KNOWN_MODELS
    assert "kokoro-82m" in KNOWN_MODELS
    assert "bark-small" in KNOWN_MODELS
    assert "sd-turbo" in KNOWN_MODELS


def test_list_known_filters_by_modality():
    audio = list_known("audio.speech")
    assert all(e.modality == "audio.speech" for e in audio)
    assert len(audio) >= 1
    images = list_known("images.generations")
    assert all(e.modality == "images.generations" for e in images)
    assert len(images) >= 1


def test_list_known_all():
    all_entries = list_known()
    modalities = {e.modality for e in all_entries}
    assert "audio.speech" in modalities
    assert "images.generations" in modalities


def test_is_pulled_false_when_not_in_catalog(tmp_catalog):
    assert not is_pulled("soprano-80m")


def test_pull_installs_pip_downloads_and_writes_catalog(tmp_catalog):
    with patch("muse.core.catalog.install_pip_extras") as mock_pip, \
         patch("muse.core.catalog.snapshot_download") as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        mock_download.return_value = "/fake/cache/soprano"
        pull("soprano-80m")
        mock_pip.assert_called_once()
        mock_download.assert_called_once()
        assert is_pulled("soprano-80m")


def test_pull_unknown_raises():
    with pytest.raises(KeyError, match="unknown model"):
        pull("does-not-exist-xyz")


def test_pull_warns_on_missing_system_packages(tmp_catalog, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    with patch("muse.core.catalog.install_pip_extras"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=["espeak-ng"]):
        pull("kokoro-82m")
        assert "espeak-ng" in caplog.text


def test_remove_clears_from_catalog(tmp_catalog):
    with patch("muse.core.catalog.install_pip_extras"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
        assert is_pulled("soprano-80m")
        remove("soprano-80m")
        assert not is_pulled("soprano-80m")


def test_load_backend_raises_when_not_pulled(tmp_catalog):
    with pytest.raises(RuntimeError, match="not pulled"):
        load_backend("soprano-80m")


def test_load_backend_imports_and_constructs(tmp_catalog):
    with patch("muse.core.catalog.install_pip_extras"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.SopranoModel = fake_class
    with patch("muse.core.catalog.importlib.import_module", return_value=fake_module):
        load_backend("soprano-80m", device="cpu")
    fake_class.assert_called_once()
    # Verify the constructor got hf_repo, local_dir, and device kwargs
    kwargs = fake_class.call_args.kwargs
    assert "hf_repo" in kwargs
    assert "local_dir" in kwargs
    assert kwargs["device"] == "cpu"
