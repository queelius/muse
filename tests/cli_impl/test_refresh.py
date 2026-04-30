"""Tests for `muse models refresh` (#140)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.refresh import (
    MODALITY_EXTRAS,
    RefreshResult,
    _infer_extras,
    _muse_repo_root,
    _pip_target,
    _select_targets,
    refresh_one,
    run_refresh,
)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data):
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _make_python_path(tmp_path: Path, name: str) -> str:
    """Create a fake python_path file so Path(p).exists() is True."""
    venv_dir = tmp_path / "venvs" / name / "bin"
    venv_dir.mkdir(parents=True, exist_ok=True)
    p = venv_dir / "python"
    p.write_text("#!/bin/sh\necho fake\n")
    p.chmod(0o755)
    return str(p)


class TestInferExtras:
    def test_known_modality_returns_mapped_extras(self):
        assert _infer_extras("audio/speech") == ["audio"]
        assert _infer_extras("image/generation") == ["images"]
        assert _infer_extras("embedding/text") == ["embeddings"]

    def test_unknown_modality_returns_empty(self):
        assert _infer_extras("totally/unknown") == []

    def test_modality_with_no_extras_returns_empty(self):
        assert _infer_extras("text/rerank") == []


class TestPipTarget:
    def test_includes_server_unconditionally(self):
        target = _pip_target([])
        assert "[server]" in target

    def test_appends_modality_extras(self):
        target = _pip_target(["audio"])
        assert "[server,audio]" in target

    def test_multiple_extras_comma_separated(self):
        target = _pip_target(["images", "embeddings"])
        assert "[server,images,embeddings]" in target

    def test_path_points_at_repo_root(self):
        target = _pip_target([])
        # The path part precedes the bracket
        path = target.split("[", 1)[0]
        assert (Path(path) / "pyproject.toml").exists()


class TestRefreshOne:
    def test_invokes_pip_install_with_muse_extras(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "kokoro-82m")
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(tmp_path / "venvs" / "kokoro-82m"),
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "model_id": "kokoro-82m",
            "modality": "audio/speech",
            "pip_extras": ("kokoro", "soundfile"),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("kokoro-82m")
        assert result.state == "ok"
        # Two calls: muse[server,audio] then pip_extras
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0].args[0]
        assert first_cmd[0] == py
        assert first_cmd[1:5] == ["-m", "pip", "install", "--upgrade"]
        assert first_cmd[5] == "-e"
        target = first_cmd[6]
        assert "[server,audio]" in target

    def test_appends_pip_extras_in_second_call(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(tmp_path / "venvs" / "x"),
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "model_id": "x",
            "modality": "audio/speech",
            "pip_extras": ("kokoro", "misaki[en]"),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            refresh_one("x")
        second_cmd = mock_run.call_args_list[1].args[0]
        assert second_cmd[0] == py
        assert second_cmd[1:5] == ["-m", "pip", "install", "--upgrade"]
        assert "kokoro" in second_cmd
        assert "misaki[en]" in second_cmd

    def test_no_extras_flag_skips_extras_step(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(tmp_path / "venvs" / "x"),
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "modality": "audio/speech",
            "pip_extras": ("kokoro",),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("x", no_extras=True)
        assert result.state == "ok"
        # Only ONE call (the muse[server,audio] one); extras call skipped
        assert mock_run.call_count == 1

    def test_skips_missing_catalog_entry(self, tmp_catalog):
        _seed_catalog({})
        result = refresh_one("does-not-exist")
        assert result.state == "skipped"
        assert "not in catalog" in result.message

    def test_skips_missing_python_path(self, tmp_catalog):
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": "/nonexistent/python",
                "enabled": True,
            },
        })
        result = refresh_one("x")
        assert result.state == "skipped"
        assert "python_path" in result.message

    def test_failed_pip_returns_failed_with_output(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        with patch("muse.cli_impl.refresh.get_manifest", return_value={"modality": ""}), \
             patch("muse.cli_impl.refresh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="ERROR: Could not find muse",
            )
            result = refresh_one("x")
        assert result.state == "failed"
        assert "muse[server] install failed" in result.message
        assert "Could not find muse" in result.pip_output

    def test_failed_extras_install_returns_failed(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "modality": "audio/speech",
            "pip_extras": ("kokoro",),
        }
        # First call ok, second fails
        results = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="kokoro install failed"),
        ]
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.subprocess.run", side_effect=results):
            result = refresh_one("x")
        assert result.state == "failed"
        assert "pip_extras install failed" in result.message
        assert "kokoro install failed" in result.pip_output

    def test_no_pip_extras_in_manifest_skips_second_step(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {"modality": "audio/speech", "pip_extras": ()}
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            refresh_one("x")
        # Only the muse[server,audio] call; no extras pass.
        assert mock_run.call_count == 1


class TestSelectTargets:
    def test_all_returns_alphabetical(self, tmp_catalog):
        _seed_catalog({
            "zebra": {"python_path": "/x", "enabled": True},
            "alpha": {"python_path": "/y", "enabled": True},
            "mango": {"python_path": "/z", "enabled": False},
        })
        targets = _select_targets(model_id=None, all_=True, enabled_only=False)
        assert targets == ["alpha", "mango", "zebra"]

    def test_enabled_only_filters_disabled(self, tmp_catalog):
        _seed_catalog({
            "yes1": {"python_path": "/x", "enabled": True},
            "yes2": {"python_path": "/y", "enabled": True},
            "no1": {"python_path": "/z", "enabled": False},
        })
        targets = _select_targets(model_id=None, all_=False, enabled_only=True)
        assert targets == ["yes1", "yes2"]

    def test_single_id_returns_singleton(self, tmp_catalog):
        _seed_catalog({"x": {}})
        targets = _select_targets(model_id="x", all_=False, enabled_only=False)
        assert targets == ["x"]

    def test_no_flags_returns_none(self, tmp_catalog):
        _seed_catalog({})
        targets = _select_targets(model_id=None, all_=False, enabled_only=False)
        assert targets is None


class TestRunRefresh:
    def test_no_targets_prints_usage_returns_2(self, tmp_catalog, capsys):
        _seed_catalog({})
        rc = run_refresh()
        assert rc == 2
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_empty_catalog_with_all_returns_0(self, tmp_catalog, capsys):
        _seed_catalog({})
        rc = run_refresh(all_=True)
        assert rc == 0
        captured = capsys.readouterr()
        assert "no targets" in captured.out.lower()

    def test_all_iterates_alphabetically(self, tmp_catalog, tmp_path):
        py_a = _make_python_path(tmp_path, "alpha")
        py_z = _make_python_path(tmp_path, "zebra")
        _seed_catalog({
            "zebra": {"python_path": py_z, "enabled": True},
            "alpha": {"python_path": py_a, "enabled": True},
        })
        manifest = {"modality": "", "pip_extras": ()}
        called: list[str] = []

        def fake_refresh_one(mid, **kw):
            called.append(mid)
            return RefreshResult(mid, "ok")

        with patch("muse.cli_impl.refresh.refresh_one", side_effect=fake_refresh_one):
            rc = run_refresh(all_=True)
        assert rc == 0
        assert called == ["alpha", "zebra"]

    def test_enabled_only_filters(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "yes")
        py2 = _make_python_path(tmp_path, "no")
        _seed_catalog({
            "yes-id": {"python_path": py, "enabled": True},
            "no-id": {"python_path": py2, "enabled": False},
        })
        called: list[str] = []
        with patch(
            "muse.cli_impl.refresh.refresh_one",
            side_effect=lambda mid, **kw: (called.append(mid), RefreshResult(mid, "ok"))[1],
        ):
            rc = run_refresh(enabled_only=True)
        assert rc == 0
        assert called == ["yes-id"]

    def test_continues_past_failures(self, tmp_catalog, tmp_path):
        py_a = _make_python_path(tmp_path, "a")
        py_b = _make_python_path(tmp_path, "b")
        py_c = _make_python_path(tmp_path, "c")
        _seed_catalog({
            "a": {"python_path": py_a, "enabled": True},
            "b": {"python_path": py_b, "enabled": True},
            "c": {"python_path": py_c, "enabled": True},
        })

        def fake(mid, **kw):
            if mid == "b":
                return RefreshResult(mid, "failed", "boom", "boom output")
            return RefreshResult(mid, "ok")

        called: list[str] = []
        with patch("muse.cli_impl.refresh.refresh_one",
                   side_effect=lambda mid, **kw: (called.append(mid), fake(mid))[1]):
            rc = run_refresh(all_=True)
        # All three were attempted
        assert called == ["a", "b", "c"]
        # Exit code is 1 because one failed
        assert rc == 1

    def test_json_output_is_parseable(self, tmp_catalog, tmp_path, capsys):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({"x": {"python_path": py, "enabled": True}})
        with patch(
            "muse.cli_impl.refresh.refresh_one",
            return_value=RefreshResult("x", "ok", extras=["audio"]),
        ):
            run_refresh(model_id="x", as_json=True)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["model_id"] == "x"
        assert parsed[0]["state"] == "ok"
        assert parsed[0]["extras"] == ["audio"]

    def test_human_output_includes_per_target_and_summary(self, tmp_catalog, tmp_path, capsys):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({"x": {"python_path": py, "enabled": True}})
        with patch(
            "muse.cli_impl.refresh.refresh_one",
            return_value=RefreshResult("x", "ok"),
        ):
            run_refresh(model_id="x")
        captured = capsys.readouterr()
        assert "x" in captured.out
        assert "OK" in captured.out
        assert "1 ok" in captured.out

    def test_failed_result_includes_pip_tail_in_human_output(self, tmp_catalog, tmp_path, capsys):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({"x": {"python_path": py, "enabled": True}})
        bad = RefreshResult("x", "failed", "boom", "line1\nline2\nline3\nline4\nline5\nline6")
        with patch("muse.cli_impl.refresh.refresh_one", return_value=bad):
            run_refresh(model_id="x")
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "line6" in captured.out
        # First line was clipped (only last 5 shown)
        assert "line1" not in captured.out


class TestModalityExtrasMap:
    def test_all_modality_keys_have_list_values(self):
        for k, v in MODALITY_EXTRAS.items():
            assert isinstance(v, list), f"{k} maps to non-list {type(v)}"

    def test_known_muse_modalities_are_mapped(self):
        """Sanity: every modality the muse server can serve has a row."""
        # Spot-check the public set; not a hard guard against new modalities.
        for mod in (
            "audio/speech",
            "audio/transcription",
            "image/generation",
            "embedding/text",
            "chat/completion",
            "video/generation",
        ):
            assert mod in MODALITY_EXTRAS


def test_muse_repo_root_finds_pyproject():
    root = _muse_repo_root()
    assert (root / "pyproject.toml").exists()
