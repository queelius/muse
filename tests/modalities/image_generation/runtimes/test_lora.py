"""Tests for the shared LoRA base resolution helper."""
from __future__ import annotations

import sys
import subprocess

import pytest

from muse.modalities.image_generation.runtimes._lora import resolve_lora_base


def test_missing_base_raises_actionable():
    with pytest.raises(RuntimeError) as e:
        resolve_lora_base("my-lora", None, resolve=lambda ref: ref)
    msg = str(e.value)
    assert msg.startswith("my-lora: lora_adapter is set but base_model is missing")
    assert "muse pull <id> --base <muse-id-or-hf-repo>" in msg
    with pytest.raises(RuntimeError, match="base_model is missing"):
        resolve_lora_base("my-lora", "", resolve=lambda ref: ref)


def test_unpulled_muse_id_raises_actionable():
    # resolve_model_source echoes an unresolved muse id back verbatim.
    with pytest.raises(RuntimeError) as e:
        resolve_lora_base("my-lora", "flux2-klein-4b", resolve=lambda ref: ref)
    assert str(e.value) == (
        "my-lora: LoRA base 'flux2-klein-4b' is not pulled (it may have been "
        "removed); run `muse pull flux2-klein-4b` and retry"
    )


def test_hf_repo_id_passes_through():
    assert resolve_lora_base(
        "my-lora", "black-forest-labs/FLUX.2-klein-4B", resolve=lambda ref: ref,
    ) == "black-forest-labs/FLUX.2-klein-4B"


def test_pulled_muse_id_maps_to_local_dir():
    assert resolve_lora_base(
        "my-lora", "sdxl-turbo", resolve={"sdxl-turbo": "/w/sdxl-turbo"}.get,
    ) == "/w/sdxl-turbo"


def test_default_resolver_reads_catalog(tmp_path, monkeypatch):
    from muse.core.catalog import _reset_read_catalog_cache, _write_catalog

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _write_catalog({"sdxl-turbo": {"local_dir": "/w/sdxl-turbo", "enabled": True}})
    _reset_read_catalog_cache()
    assert resolve_lora_base("my-lora", "sdxl-turbo") == "/w/sdxl-turbo"
    with pytest.raises(RuntimeError, match="is not pulled"):
        resolve_lora_base("my-lora", "flux2-klein-4b")


def test_module_is_import_light():
    code = (
        "import sys; import muse.modalities.image_generation.runtimes._lora; "
        "bad = [m for m in ('torch', 'diffusers', 'muse.core.catalog') "
        "if m in sys.modules]; print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "[]"
