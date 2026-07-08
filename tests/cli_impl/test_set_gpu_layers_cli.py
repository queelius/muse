"""Tests for the `muse models set-gpu-layers` CLI verb (spec 2026-07-08).

Mirrors test_set_device_cli.py: the verb writes a per-model
`gpu_layers_override` into the catalog; load_backend honors it on the
next cold load. GGUF-only: models without capabilities.gguf_file are
refused (honest error beats a silently ignored pin).
"""
from __future__ import annotations

import json

import pytest
import typer


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    from muse.core.catalog import _reset_known_models_cache
    _reset_known_models_cache()
    yield tmp_path
    _reset_known_models_cache()


def _seed_gguf(tmp_path, model_id="test-gguf", capabilities=None):
    from muse.core.catalog import _reset_known_models_cache
    entry = {
        "pulled_at": "...", "hf_repo": "org/repo", "local_dir": "/w",
        "venv_path": "/v", "python_path": "/v/bin/python",
        "enabled": True, "source": "hf://org/repo",
        "manifest": {
            "model_id": model_id, "modality": "chat/completion",
            "hf_repo": "org/repo",
            "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
            "capabilities": capabilities if capabilities is not None
                            else {"gguf_file": "m.gguf"},
        },
    }
    (tmp_path / "catalog.json").write_text(json.dumps({model_id: entry}))
    _reset_known_models_cache()


def test_set_writes_override(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 30, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 30


def test_clear_removes_override(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 30, clear=False)
    models_set_gpu_layers("test-gguf", None, clear=True)
    assert "gpu_layers_override" not in _read_catalog()["test-gguf"]


def test_minus_one_and_zero_accepted(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", -1, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1
    models_set_gpu_layers("test-gguf", 0, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 0


def test_below_minus_one_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", -2, clear=False)
    assert exc.value.exit_code == 2


def test_no_n_and_no_clear_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", None, clear=False)
    assert exc.value.exit_code == 2


def test_unknown_model_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("never-pulled", 10, clear=False)
    assert exc.value.exit_code == 2


def test_non_gguf_model_refused(tmp_catalog):
    """A model without capabilities.gguf_file is refused: the pin would be
    silently ignored by non-llama.cpp runtimes."""
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog, capabilities={})  # no gguf_file
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", 30, clear=False)
    assert exc.value.exit_code == 2
    assert "gpu_layers_override" not in _read_catalog()["test-gguf"]


def test_clear_on_non_gguf_still_allowed(tmp_catalog):
    """--clear must work even on a non-GGUF entry (e.g. an operator
    removing a stale pin after a manifest change)."""
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog, capabilities={})
    models_set_gpu_layers("test-gguf", None, clear=True)  # must not raise


def test_info_renders_pin(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 30, clear=False)
    from muse.cli_impl.models_info_display import format_info
    from muse.core.catalog import _read_catalog, known_models
    text = format_info(
        "test-gguf",
        catalog_known=known_models(),
        catalog_data=_read_catalog().get("test-gguf", {}),
        online_status=None,
    )
    assert "gpu layers" in text and "30" in text
