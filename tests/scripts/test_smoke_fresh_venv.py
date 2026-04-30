"""Unit tests for scripts/smoke_fresh_venv.py.

Heavy operations (venv creation, subprocess runs, HF download) are
mocked so the unit tests do not actually spawn pip or download weights.
The CI workflow exercises the full flow end-to-end.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_smoke_module():
    """Import scripts/smoke_fresh_venv.py as a module by file path.

    The script lives outside any importable package; tests load it
    directly via importlib.util.spec_from_file_location so the test
    file does not need a sys.path hack. Module is registered in
    sys.modules before exec_module so dataclasses.dataclass can resolve
    the module's __module__ attribute back to its globals dict.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke_fresh_venv.py"
    spec = importlib.util.spec_from_file_location(
        "muse_smoke_fresh_venv", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load_smoke_module()


@pytest.fixture
def fake_known_models():
    """Provide a known_models()-shaped dict with one fake bundled entry."""
    entry = MagicMock()
    entry.model_id = "fake-model"
    entry.modality = "embedding/text"
    entry.pip_extras = ("torch>=2.1.0", "transformers>=4.36.0")
    return {"fake-model": entry}


def test_smoke_one_success_path(tmp_path, fake_known_models):
    """All four shell-outs succeed; SmokeResult.ok is True; label is OK."""
    with patch.object(smoke, "_create_venv") as mock_create, \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(smoke, "_run_load_only", return_value=(0, '{"ok": 1}')), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("fake-model", tmp_path)

    assert result.ok is True
    assert result.error is None
    assert result.model_id == "fake-model"
    assert "OK" in result.label
    mock_create.assert_called_once()


def test_smoke_one_unknown_model(tmp_path, fake_known_models):
    """Model not in known_models() returns ok=False with 'unknown model'."""
    with patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("missing-model", tmp_path)

    assert result.ok is False
    assert "unknown" in result.error
    assert "unknown model" in result.label


def test_smoke_one_pip_install_muse_fails(tmp_path, fake_known_models):
    """muse[server] install fails; result is FAIL with pip mention."""
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse",
                      return_value=(1, "ERROR: no matching distribution\n")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(smoke, "_run_load_only", return_value=(0, "")), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("fake-model", tmp_path)

    assert result.ok is False
    assert "muse[server]" in result.error
    assert "FAIL" in result.label


def test_smoke_one_pip_extras_fails(tmp_path, fake_known_models):
    """pip_extras install fails; result is FAIL with pip extras mention."""
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras",
                      return_value=(1, "ERROR: no matching distribution\n")), \
         patch.object(smoke, "_run_load_only", return_value=(0, "")), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("fake-model", tmp_path)

    assert result.ok is False
    assert "pip_extras" in result.error
    assert "pip extras" in result.label


def test_smoke_one_load_fails_with_missing_dep(tmp_path, fake_known_models):
    """Worker stderr has ModuleNotFoundError; label mentions the missing dep."""
    captured = (
        "Traceback (most recent call last):\n"
        "  File \".../catalog.py\", line 565, in load_backend\n"
        "    return cls(hf_repo=...)\n"
        "ModuleNotFoundError: No module named 'librosa'\n"
    )
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(smoke, "_run_load_only", return_value=(1, captured)), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("fake-model", tmp_path)

    assert result.ok is False
    assert "missing dep: librosa" in result.label
    assert result.error is not None


def test_smoke_one_load_fails_with_generic_error(tmp_path, fake_known_models):
    """Worker fails without a recognized signature; label uses fallback."""
    captured = "load failed: model file not found\n"
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(smoke, "_run_load_only", return_value=(1, captured)), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("fake-model", tmp_path)

    assert result.ok is False
    assert "load failed" in result.label


def test_extract_failure_reason_modulenotfound():
    """_extract_failure_reason picks the missing module name."""
    captured = "ModuleNotFoundError: No module named 'safetensors'\n"
    assert smoke._extract_failure_reason(captured) == "missing dep: safetensors"


def test_extract_failure_reason_importerror():
    """_extract_failure_reason picks ImportError lines."""
    captured = (
        "Traceback ...\n"
        "ImportError: cannot import name 'foo' from 'bar'\n"
    )
    out = smoke._extract_failure_reason(captured)
    assert "ImportError" in out


def test_extract_failure_reason_load_failed():
    """_extract_failure_reason picks 'load failed:' lines from the probe worker."""
    captured = "baseline RAM=1.2 GB\nload failed: model file not found\n"
    out = smoke._extract_failure_reason(captured)
    assert "load failed" in out


def test_extract_failure_reason_empty():
    """_extract_failure_reason handles empty input gracefully."""
    assert smoke._extract_failure_reason("") == "no output"


def test_main_human_output(tmp_path, fake_known_models, capsys):
    """main() without --json prints the SmokeResult.label to stdout."""
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(smoke, "_run_load_only", return_value=(0, "")), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        rc = smoke.main([
            "--model_id", "fake-model",
            "--venv_root", str(tmp_path),
        ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "fake-model" in out
    assert "OK" in out


def test_main_json_output(tmp_path, fake_known_models, capsys):
    """main() with --json prints a parseable JSON record."""
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(smoke, "_run_load_only", return_value=(0, "")), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        rc = smoke.main([
            "--model_id", "fake-model",
            "--venv_root", str(tmp_path),
            "--json",
        ])

    assert rc == 0
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["model_id"] == "fake-model"
    assert record["ok"] is True
    assert record["error"] is None
    assert "duration_s" in record
    assert "label" in record


def test_main_failure_returns_non_zero(tmp_path, fake_known_models, capsys):
    """main() returns 1 when the smoke test fails."""
    with patch.object(smoke, "_create_venv"), \
         patch.object(smoke, "_install_muse", return_value=(0, "")), \
         patch.object(smoke, "_install_pip_extras", return_value=(0, "")), \
         patch.object(
             smoke, "_run_load_only",
             return_value=(1, "ModuleNotFoundError: No module named 'librosa'\n"),
         ), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        rc = smoke.main([
            "--model_id", "fake-model",
            "--venv_root", str(tmp_path),
            "--json",
        ])

    assert rc == 1
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["ok"] is False
    assert "librosa" in record["label"]


def test_repo_root_finds_pyproject():
    """_repo_root() locates the muse repo via pyproject.toml ancestor walk."""
    root = smoke._repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "muse").exists()


def test_venv_python_path_layout():
    """_venv_python returns the POSIX layout path."""
    p = Path("/tmp/some-venv")
    assert smoke._venv_python(p) == p / "bin" / "python"
