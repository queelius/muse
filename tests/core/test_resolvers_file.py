"""Tests for muse.core.resolvers_file (local file:// GGUF drop-in resolver)."""
from __future__ import annotations

import errno
import hashlib
import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from muse.core import config
from muse.core.curated import find_curated
from muse.core.resolvers import (
    ResolverError,
    _reset_registry_for_tests,
    get_resolver,
    register_resolver,
    resolve,
    search,
)
from muse.core.resolvers_file import FileResolver
from muse.core.runtime_helpers import GGUF_DEFAULT_CONTEXT_LENGTH


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    register_resolver(FileResolver())
    yield
    _reset_registry_for_tests()


def test_import_registers_file_resolver():
    _reset_registry_for_tests()
    from muse.core import resolvers_file

    importlib.reload(resolvers_file)
    assert isinstance(
        get_resolver("file:///tmp/model.gguf"), resolvers_file.FileResolver
    )


def test_resolve_single_gguf_file_and_stage(tmp_path):
    src = tmp_path / "My_Model.Q4_K_M.gguf"
    payload = b"GGUF" + b"\x00" * 2048
    src.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()

    rm = resolve(f"file://{src}")
    assert rm.backend_path == (
        "muse.modalities.chat_completion.runtimes.llama_server:LlamaServerModel"
    )
    m = rm.manifest
    assert m["model_id"] == "my-model.q4-k-m"
    assert m["modality"] == "chat/completion"
    assert m["hf_repo"] == ""
    assert m["pip_extras"] == []
    assert m["system_packages"] == []
    caps = m["capabilities"]
    assert caps["gguf_file"] == src.name
    assert caps["device"] == "auto"
    assert caps["parallel"] == 4
    assert caps["cont_batching"] is True
    assert caps["cache_prompt"] is True
    assert caps["context_length"] == GGUF_DEFAULT_CONTEXT_LENGTH

    assert rm.artifact_provenance == (
        {
            "kind": "file",
            "path": str(src.resolve()),
            "filename": src.name,
            "sha256": expected_sha,
            "size_bytes": len(payload),
        },
    )

    weights_cache = tmp_path / "weights"
    staged_dir = rm.download(weights_cache)
    assert staged_dir == weights_cache / "local-gguf" / f"my-model.q4-k-m-{expected_sha[:12]}"
    staged_file = staged_dir / src.name
    assert staged_file.exists()
    assert staged_file.read_bytes() == payload

    # Deleting the staged copy (as on `muse models remove --purge`) leaves source intact.
    staged_file.unlink()
    assert src.exists()
    assert src.read_bytes() == payload


def test_resolve_tilde_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    gguf = model_dir / "tiny-npc.gguf"
    gguf.write_bytes(b"GGUF-tilde")

    rm = resolve("file://~/models/tiny-npc.gguf")
    assert rm.manifest["model_id"] == "tiny-npc"
    assert rm.manifest["capabilities"]["gguf_file"] == "tiny-npc.gguf"
    assert rm.artifact_provenance[0]["path"] == str(gguf.resolve())


def test_resolve_directory_with_single_gguf(tmp_path):
    model_dir = tmp_path / "dropin"
    model_dir.mkdir()
    (model_dir / "README.md").write_text("notes")
    gguf = model_dir / "single-model.gguf"
    gguf.write_bytes(b"GGUF-dir")

    rm = resolve(f"file://{model_dir}")
    assert rm.manifest["model_id"] == "single-model"
    assert rm.manifest["capabilities"]["gguf_file"] == "single-model.gguf"


def test_resolve_split_shards_from_shard_or_directory(tmp_path):
    model_dir = tmp_path / "sharded"
    model_dir.mkdir()
    s1 = model_dir / "big-q4-00001-of-00002.gguf"
    s2 = model_dir / "big-q4-00002-of-00002.gguf"
    s1.write_bytes(b"GGUF-shard-1")
    s2.write_bytes(b"GGUF-shard-2")

    # Resolving either shard 2 or the directory yields shard 1 as primary.
    for uri in (f"file://{s2}", f"file://{model_dir}"):
        rm = resolve(uri)
        assert rm.manifest["model_id"] == "big-q4"
        assert rm.manifest["capabilities"]["gguf_file"] == s1.name
        assert rm.artifact_provenance[0]["filename"] == s1.name

        staged = rm.download(tmp_path / "cache")
        assert (staged / s1.name).read_bytes() == b"GGUF-shard-1"
        assert (staged / s2.name).read_bytes() == b"GGUF-shard-2"


def test_resolve_missing_split_shard_raises(tmp_path):
    s1 = tmp_path / "broken-00001-of-00002.gguf"
    s1.write_bytes(b"GGUF-only-1")

    with pytest.raises(ResolverError, match="missing GGUF shard"):
        resolve(f"file://{s1}")


def test_resolve_directory_multiple_ggufs_raises(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"a")
    (tmp_path / "b.gguf").write_bytes(b"b")

    with pytest.raises(ResolverError, match="multiple GGUF models"):
        resolve(f"file://{tmp_path}")


def test_resolve_directory_without_gguf_raises(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")

    with pytest.raises(ResolverError, match="no .gguf files"):
        resolve(f"file://{tmp_path}")


def test_resolve_rejects_invalid_paths_and_symlinks(tmp_path):
    # Missing file
    with pytest.raises(ResolverError, match="does not exist"):
        resolve(f"file://{tmp_path / 'nope.gguf'}")

    # Non-GGUF file
    other = tmp_path / "weights.safetensors"
    other.write_bytes(b"nope")
    with pytest.raises(ResolverError, match="not a .gguf file"):
        resolve(f"file://{other}")

    # Relative path
    with pytest.raises(ResolverError, match="absolute path"):
        resolve("file://relative/model.gguf")

    # Traversal segment
    with pytest.raises(ResolverError, match="traversal"):
        resolve(f"file://{tmp_path}/../model.gguf")

    # Symlink file
    real = tmp_path / "real.gguf"
    real.write_bytes(b"GGUF")
    link = tmp_path / "link.gguf"
    link.symlink_to(real)
    with pytest.raises(ResolverError, match="symlink"):
        resolve(f"file://{link}")

    # Symlink directory
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "m.gguf").write_bytes(b"GGUF")
    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ResolverError, match="symlink"):
        resolve(f"file://{link_dir}")


def test_resolve_applies_chat_format_hints(tmp_path):
    gguf = tmp_path / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF-llama")

    rm = resolve(f"file://{gguf}")
    caps = rm.manifest["capabilities"]
    assert "chat_format" in caps or "supports_tools" in caps


def test_download_falls_back_to_copy_on_exdev(tmp_path):
    src = tmp_path / "crossfs.gguf"
    src.write_bytes(b"GGUF-cross-device")
    rm = resolve(f"file://{src}")

    with patch(
        "muse.core.resolvers_file.os.link",
        side_effect=OSError(errno.EXDEV, "Invalid cross-device link"),
    ):
        staged = rm.download(tmp_path / "weights")

    assert (staged / src.name).read_bytes() == b"GGUF-cross-device"


def test_search_scans_configured_gguf_dir(tmp_path, monkeypatch):
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "alpha-q4.gguf").write_bytes(b"a" * 1000)
    sub = gguf_dir / "nested"
    sub.mkdir()
    (sub / "beta-q8.gguf").write_bytes(b"b" * 2000)
    (sub / "gamma-00001-of-00002.gguf").write_bytes(b"c" * 1500)
    (sub / "gamma-00002-of-00002.gguf").write_bytes(b"d" * 1500)
    (gguf_dir / "ignore.txt").write_text("not gguf")
    (gguf_dir / "sym.gguf").symlink_to(gguf_dir / "alpha-q4.gguf")

    monkeypatch.setenv("MUSE_GGUF_DIR", str(gguf_dir))
    config.reset_config()

    all_rows = list(search("", backend="file"))
    assert [r.model_id for r in all_rows] == ["alpha-q4", "beta-q8", "gamma"]
    assert all(r.modality == "chat/completion" for r in all_rows)
    assert all(r.uri.startswith("file://") for r in all_rows)

    filtered = list(search("BETA", backend="file"))
    assert [r.model_id for r in filtered] == ["beta-q8"]

    limited = list(search("", backend="file", limit=2))
    assert len(limited) == 2

    wrong_mod = list(search("", backend="file", modality="audio/speech"))
    assert wrong_mod == []

    # Missing directory returns empty rather than raising.
    monkeypatch.setenv("MUSE_GGUF_DIR", str(tmp_path / "does-not-exist"))
    assert list(search("", backend="file")) == []


def test_pull_uri_as_error_mentions_file_scheme():
    from muse.core.catalog_pull import pull

    with pytest.raises(ValueError, match=r"file://"):
        pull("bare-id", model_id="custom-id")


def test_file_pull_persists_catalog_without_hf_seed(tmp_path, monkeypatch):
    """End-to-end pull of a file:// URI writes a valid catalog.json without calling seed_repo."""
    from muse.core import catalog_pull
    from muse.core.catalog_models import known_models
    from muse.core.catalog_store import _read_catalog, _reset_read_catalog_cache

    catalog_dir = tmp_path / "catalog"
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(catalog_dir))
    _reset_read_catalog_cache()

    src = tmp_path / "local-npc.gguf"
    src.write_bytes(b"GGUF-local-model-bytes")

    def _fake_ensure_venv(venv_path, creator=None):
        (venv_path / "bin").mkdir(parents=True, exist_ok=True)
        (venv_path / "bin" / "python").write_text("#!/bin/sh\n")

    with (
        patch.object(catalog_pull, "ensure_venv", side_effect=_fake_ensure_venv),
        patch.object(catalog_pull, "install_into_venv"),
        patch.object(catalog_pull, "install_python_sources"),
        patch.object(catalog_pull, "dedupe_after_install"),
        patch.object(catalog_pull, "_run_automatic_storage_maintenance"),
        patch.object(
            catalog_pull, "seed_repo",
            side_effect=AssertionError("seed_repo must not be called for file:// pulls"),
        ),
    ):
        pulled_id = catalog_pull.pull(f"file://{src}")

    assert pulled_id == "local-npc"
    catalog = _read_catalog()
    entry = catalog["local-npc"]
    assert entry["source"] == f"file://{src}"
    assert "hf_repo" not in entry
    assert entry["artifact_provenance"][0]["kind"] == "file"
    assert entry["artifact_provenance"][0]["size_bytes"] == len(b"GGUF-local-model-bytes")
    assert Path(entry["local_dir"], "local-npc.gguf").exists()
    assert "local-npc" in known_models()


def test_catalog_store_rejects_malformed_file_receipt(tmp_path, monkeypatch):
    from muse.core.catalog_store import (
        CatalogError,
        _reset_read_catalog_cache,
        _write_catalog,
    )

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    _reset_read_catalog_cache()

    bad = {
        "bad-file": {
            "pulled_at": "2026-11-25T00:00:00+00:00",
            "local_dir": str(tmp_path),
            "venv_path": str(tmp_path),
            "python_path": str(tmp_path / "python"),
            "enabled": True,
            "source": "file:///tmp/bad.gguf",
            "artifact_provenance": [
                {
                    "kind": "file",
                    "path": "/tmp/bad.gguf",
                    "filename": "bad.gguf",
                    "sha256": "not-a-sha",
                    "size_bytes": 10,
                }
            ],
        }
    }
    with pytest.raises(CatalogError, match="sha256"):
        _write_catalog(bad)

    bad["bad-file"]["artifact_provenance"][0]["sha256"] = "a" * 64
    bad["bad-file"]["artifact_provenance"][0]["size_bytes"] = 0
    with pytest.raises(CatalogError, match="size_bytes"):
        _write_catalog(bad)


def test_curated_borrowed_stars_v21_entries():
    expected = {
        "borrowed-stars-npc-qwen3.5-0.8b": (
            "hf://queelius/borrowed-stars-npc-qwen3.5-0.8b-32k-GGUF@q4_k_m",
            0.33,
            {
                "device": "auto",
                "memory_gb": 0.6,
                "parallel": 4,
                "cont_batching": True,
                "cache_prompt": True,
                "no_jinja": True,
                "context_length": 8192,
            },
        ),
        "borrowed-stars-npc-lfm2.5-350m": (
            "hf://queelius/borrowed-stars-npc-lfm2.5-350m-GGUF@q4_k_m",
            0.22,
            {
                "device": "auto",
                "memory_gb": 0.45,
                "parallel": 4,
                "cont_batching": True,
                "cache_prompt": True,
                "context_length": 8192,
            },
        ),
        "borrowed-stars-npc-gemma3-1b": (
            "hf://queelius/borrowed-stars-npc-gemma3-1b-32k-GGUF@q4_k_m",
            0.50,
            {
                "device": "auto",
                "memory_gb": 0.85,
                "parallel": 4,
                "cont_batching": True,
                "cache_prompt": True,
                "context_length": 8192,
            },
        ),
    }
    for cid, (uri, size_gb, caps) in expected.items():
        entry = find_curated(cid)
        assert entry is not None, f"missing curated entry {cid}"
        assert entry.uri == uri
        assert entry.modality == "chat/completion"
        assert entry.size_gb == size_gb
        assert entry.quality_tier == "good"
        assert entry.capabilities == caps
