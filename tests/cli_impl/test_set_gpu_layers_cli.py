"""Tests for the `muse models set-gpu-layers` CLI verb (spec 2026-07-08).

Mirrors test_set_device_cli.py: the verb writes a per-model
`gpu_layers_override` into the catalog; load_backend honors it on the
next cold load. GGUF-only: models without capabilities.gguf_file are
refused (honest error beats a silently ignored pin).

click has no negative-number heuristic for positional arguments, so a
bare `-1` (all layers on GPU -- the verb's primary documented value) dies
at the shell with "Error: No such option: -1" before any Python code
here runs. That failure mode can only be caught by an actual subprocess
invocation, not a direct function call -- hence the `_run_cli` subprocess
tests below alongside the direct-call unit tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

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


def test_zero_positional_accepted(tmp_catalog):
    """0 (pure CPU) is a normal positional value -- reachable from a real
    shell, unlike -1."""
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", 0, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 0


def test_all_flag_writes_minus_one(tmp_catalog):
    """--all is the shell-safe way to request -1 (all layers on GPU)."""
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", None, all_layers=True, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1


def test_direct_call_with_negative_one_still_accepted(tmp_catalog):
    """A DIRECT function call with n=-1 (bypassing click's positional
    parsing) is still accepted and writes -1: the setter allows it, and
    only the shell path is unreachable for -1, not the Python API."""
    from muse.cli import models_set_gpu_layers
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    models_set_gpu_layers("test-gguf", -1, clear=False)
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1


def test_below_minus_one_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", -2, clear=False)
    assert exc.value.exit_code == 2


def test_no_n_and_no_clear_exits_2(tmp_catalog):
    """Neither a layer count, --all, nor --clear given: usage error."""
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", None, clear=False)
    assert exc.value.exit_code == 2


def test_n_and_all_together_exits_2(tmp_catalog):
    """Both a layer count and --all given: ambiguous, usage error."""
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", 30, all_layers=True, clear=False)
    assert exc.value.exit_code == 2


def test_n_and_clear_together_exits_2(tmp_catalog):
    """Both a layer count and --clear given: ambiguous, usage error."""
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", 30, clear=True)
    assert exc.value.exit_code == 2


def test_all_and_clear_together_exits_2(tmp_catalog):
    """Both --all and --clear given: ambiguous, usage error."""
    from muse.cli import models_set_gpu_layers
    _seed_gguf(tmp_catalog)
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("test-gguf", None, all_layers=True, clear=True)
    assert exc.value.exit_code == 2


def test_unknown_model_exits_2(tmp_catalog):
    from muse.cli import models_set_gpu_layers
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("never-pulled", 10, clear=False)
    assert exc.value.exit_code == 2


def test_unknown_model_error_names_not_pulled_not_gguf(tmp_catalog, capsys):
    """FINDING 2: an unknown/never-pulled model given N must get the
    uniform 'not pulled' error, not the misleading 'is not a GGUF model'
    refusal (which comes from treating a missing catalog entry as an
    entry with empty capabilities)."""
    from muse.cli import models_set_gpu_layers
    with pytest.raises(typer.Exit) as exc:
        models_set_gpu_layers("never-pulled", 10, clear=False)
    assert exc.value.exit_code == 2
    err = capsys.readouterr().err
    assert "never-pulled" in err
    assert "not pulled" in err
    assert "GGUF" not in err


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


# ---------------------------------------------------------------------------
# Subprocess tests: exercise the REAL click/typer argument parser.
#
# FINDING 1 was invisible to every test above because they call
# models_set_gpu_layers() directly in-process, bypassing click's option
# parser entirely. Only a real `python -m muse.cli ...` invocation can
# catch "bare -1 dies at the shell" (or confirm --all fixes it).
# ---------------------------------------------------------------------------

def _run_cli(tmp_path, *args, timeout=30):
    env = dict(os.environ)
    env["MUSE_CATALOG_DIR"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, "-m", "muse.cli", "models", "set-gpu-layers", *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def test_subprocess_positional_n_writes_override(tmp_catalog):
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    r = _run_cli(tmp_catalog, "test-gguf", "30")
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 30


def test_subprocess_all_flag_writes_minus_one(tmp_catalog):
    """FINDING 1 regression test: `muse models set-gpu-layers <id> --all`
    must succeed at the real shell. A bare `-1` positional cannot (click
    has no negative-number heuristic and dies with "No such option: -1"
    before any muse code runs) -- --all is the shell-safe replacement."""
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    r = _run_cli(tmp_catalog, "test-gguf", "--all")
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1


def test_subprocess_clear_removes_override(tmp_catalog):
    from muse.core.catalog import _read_catalog
    _seed_gguf(tmp_catalog)
    r1 = _run_cli(tmp_catalog, "test-gguf", "30")
    assert r1.returncode == 0, f"stdout={r1.stdout!r} stderr={r1.stderr!r}"
    r2 = _run_cli(tmp_catalog, "test-gguf", "--clear")
    assert r2.returncode == 0, f"stdout={r2.stdout!r} stderr={r2.stderr!r}"
    assert "gpu_layers_override" not in _read_catalog()["test-gguf"]


def test_subprocess_n_and_all_together_exits_2(tmp_catalog):
    _seed_gguf(tmp_catalog)
    r = _run_cli(tmp_catalog, "test-gguf", "30", "--all")
    assert r.returncode == 2, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_subprocess_neither_n_nor_flag_exits_2(tmp_catalog):
    _seed_gguf(tmp_catalog)
    r = _run_cli(tmp_catalog, "test-gguf")
    assert r.returncode == 2, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_subprocess_bare_negative_one_dies_at_shell(tmp_catalog):
    """Documents FINDING 1's root cause for posterity: click has no
    negative-number heuristic for positional arguments, so a bare -1 is
    parsed as an unrecognized option and the process exits nonzero before
    any muse code runs. This is why --all exists; it is not something the
    muse-layer code can intercept or fix."""
    _seed_gguf(tmp_catalog)
    r = _run_cli(tmp_catalog, "test-gguf", "-1")
    assert r.returncode != 0
    assert "no such option" in (r.stdout + r.stderr).lower()
