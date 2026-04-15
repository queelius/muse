"""Tests for muse.core.discovery.

Discovery scans directories of .py files (models) or subpackages
(modalities) and extracts MANIFEST + Model class (models) or
MODALITY tag + build_router (modalities). Errors during discovery
are logged and skipped; discovery never raises.
"""
import textwrap
from pathlib import Path

import pytest

from muse.core.discovery import (
    DiscoveredModel,
    discover_models,
    discover_modalities,
)


def _write_model_script(tmp_path: Path, filename: str, content: str) -> Path:
    """Helper: write a .py file with given content to tmp_path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def _write_modality_package(tmp_path: Path, name: str, content: str) -> Path:
    """Helper: write a subpackage (__init__.py only) under tmp_path/name/."""
    pkg = tmp_path / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(textwrap.dedent(content).lstrip())
    return pkg


# ---------- Model discovery ----------

class TestDiscoverModels:
    def test_empty_directory_yields_no_models(self, tmp_path):
        result = discover_models([tmp_path])
        assert result == {}

    def test_script_with_manifest_and_model_class_is_discovered(self, tmp_path):
        _write_model_script(tmp_path, "fake_model.py", """
            MANIFEST = {
                "model_id": "fake-model",
                "modality": "audio/speech",
                "hf_repo": "fake/repo",
            }
            class Model:
                model_id = "fake-model"
        """)
        result = discover_models([tmp_path])
        assert "fake-model" in result
        entry = result["fake-model"]
        assert isinstance(entry, DiscoveredModel)
        assert entry.manifest["model_id"] == "fake-model"
        assert entry.manifest["modality"] == "audio/speech"
        assert entry.model_class.__name__ == "Model"
        assert entry.source_path == tmp_path / "fake_model.py"

    def test_script_without_manifest_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "noisy.py", """
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "MANIFEST" in caplog.text or "noisy" in caplog.text

    def test_script_without_model_class_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "manifest_only.py", """
            MANIFEST = {
                "model_id": "half-model",
                "modality": "audio/speech",
                "hf_repo": "x/y",
            }
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "Model" in caplog.text or "half-model" in caplog.text or "manifest_only" in caplog.text

    def test_script_with_import_error_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "broken.py", """
            import definitely_not_a_real_module_xyz
            MANIFEST = {"model_id": "x", "modality": "y", "hf_repo": "z"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        # Discovery must not raise; just log
        assert "broken" in caplog.text or "ImportError" in caplog.text or "definitely_not" in caplog.text

    def test_files_starting_with_underscore_are_ignored(self, tmp_path):
        _write_model_script(tmp_path, "_private.py", """
            MANIFEST = {"model_id": "p", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        _write_model_script(tmp_path, "__init__.py", "")
        result = discover_models([tmp_path])
        assert result == {}

    def test_manifest_missing_required_fields_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "bad_manifest.py", """
            MANIFEST = {"model_id": "x", "modality": "audio/speech"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "hf_repo" in caplog.text or "required" in caplog.text.lower()

    def test_multiple_directories_scanned_in_order(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_model_script(d1, "model_a.py", """
            MANIFEST = {"model_id": "a", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        _write_model_script(d2, "model_b.py", """
            MANIFEST = {"model_id": "b", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        result = discover_models([d1, d2])
        assert {"a", "b"} == set(result.keys())

    def test_first_found_wins_on_model_id_collision(self, tmp_path, caplog):
        d1 = tmp_path / "bundled"
        d2 = tmp_path / "user"
        d1.mkdir()
        d2.mkdir()
        _write_model_script(d1, "m.py", """
            MANIFEST = {"model_id": "collide", "modality": "m", "hf_repo": "bundled-repo"}
            class Model: ...
        """)
        _write_model_script(d2, "m.py", """
            MANIFEST = {"model_id": "collide", "modality": "m", "hf_repo": "user-repo"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([d1, d2])
        assert len(result) == 1
        assert result["collide"].manifest["hf_repo"] == "bundled-repo"
        assert "collide" in caplog.text

    def test_nonexistent_directory_is_silently_skipped(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        result = discover_models([missing])
        assert result == {}


# ---------- Modality discovery ----------

class TestDiscoverModalities:
    def test_empty_directory_yields_no_modalities(self, tmp_path):
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_subpackage_with_MODALITY_and_build_router_is_discovered(self, tmp_path):
        _write_modality_package(tmp_path, "fake_modality", """
            MODALITY = "fake/type"
            def build_router(registry):
                from fastapi import APIRouter
                return APIRouter()
        """)
        result = discover_modalities([tmp_path])
        assert "fake/type" in result
        build_fn = result["fake/type"]
        assert callable(build_fn)

    def test_subpackage_without_MODALITY_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "no_tag", """
            def build_router(registry):
                return None
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "MODALITY" in caplog.text or "no_tag" in caplog.text

    def test_subpackage_without_build_router_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "no_router", """
            MODALITY = "x/y"
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "build_router" in caplog.text or "no_router" in caplog.text

    def test_plain_py_files_are_not_treated_as_modalities(self, tmp_path):
        (tmp_path / "not_a_package.py").write_text(
            'MODALITY = "wrong/form"\ndef build_router(r): pass\n'
        )
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_first_found_wins_on_modality_tag_collision(self, tmp_path, caplog):
        d1 = tmp_path / "bundled"
        d2 = tmp_path / "escape"
        d1.mkdir()
        d2.mkdir()
        _write_modality_package(d1, "my_mod", """
            MODALITY = "collide/tag"
            def build_router(r): return ("bundled",)
        """)
        _write_modality_package(d2, "my_mod", """
            MODALITY = "collide/tag"
            def build_router(r): return ("escape",)
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([d1, d2])
        assert len(result) == 1
        assert result["collide/tag"](None) == ("bundled",)
        assert "collide/tag" in caplog.text
