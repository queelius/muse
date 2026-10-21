"""Hardlinking identical package files across model venvs."""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from muse.core import venv_dedupe
from muse.core.venv_dedupe import dedupe_after_install, dedupe_venvs

OLD = time.time() - 3600


@pytest.fixture(autouse=True)
def small_files(monkeypatch):
    monkeypatch.setattr(venv_dedupe, "MIN_BYTES", 4)


def _venv(root: Path, name: str) -> Path:
    v = root / name
    (v / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (v / "bin").mkdir()
    (v / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return v


def _put(venv: Path, rel: str, data: bytes, *, mode=0o644, mtime=OLD) -> Path:
    p = venv / "lib" / "python3.12" / "site-packages" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    os.chmod(p, mode)
    os.utime(p, (mtime, mtime))
    return p


def _same(a: Path, b: Path) -> bool:
    return os.stat(a).st_ino == os.stat(b).st_ino


def test_identical_files_become_one_inode(tmp_path):
    a, b, c = (_venv(tmp_path, n) for n in "abc")
    files = [_put(v, "nvidia/lib/libcublas.so.13", b"cublas-bytes") for v in (a, b, c)]
    result = dedupe_venvs(tmp_path)
    assert _same(files[0], files[1]) and _same(files[1], files[2])
    assert result.linked_files == 2
    assert result.reclaimed_bytes == 2 * len(b"cublas-bytes")
    assert all(f.read_bytes() == b"cublas-bytes" for f in files)
    # A second pass has nothing left to do.
    assert dedupe_venvs(tmp_path).linked_files == 0


@pytest.mark.parametrize("second, mode", [(b"other-bytes!", 0o644), (b"cublas-bytes", 0o755)])
def test_different_content_or_mode_is_never_linked(tmp_path, second, mode):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    x = _put(a, "lib.so", b"cublas-bytes")
    y = _put(b, "lib.so", second, mode=mode)
    assert dedupe_venvs(tmp_path).linked_files == 0
    assert not _same(x, y)


def test_dry_run_changes_nothing(tmp_path):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    result = dedupe_venvs(tmp_path, dry_run=True)
    assert result.linked_files == 1 and result.reclaimed_bytes == len(b"payload")
    assert not _same(x, y)


def test_recently_changed_files_are_left_alone(tmp_path):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    _put(a, "lib.so", b"payload")
    _put(b, "lib.so", b"payload", mtime=time.time())
    result = dedupe_venvs(tmp_path)
    assert result.linked_files == 0 and result.skipped_files == 1


def test_only_replaces_files_in_the_named_venv(tmp_path):
    a, b, fresh = (_venv(tmp_path, n) for n in ("a", "b", "fresh"))
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    z = _put(fresh, "lib.so", b"payload", mtime=time.time())  # just installed
    result = dedupe_venvs(tmp_path, only=fresh)
    assert result.linked_files == 1
    assert not _same(x, y)            # other venvs untouched
    assert _same(z, x) or _same(z, y)


def test_staging_dirs_bin_scripts_and_small_files_are_ignored(tmp_path, monkeypatch):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    staging = _venv(tmp_path, ".gone.transaction-x")
    for v in (a, b, staging):
        (v / "bin" / "tool").write_bytes(b"#!/v/bin/python\n")
        os.utime(v / "bin" / "tool", (OLD, OLD))
        _put(v, "pkg/__pycache__/m.cpython-312.pyc", b"bytecode")
    monkeypatch.setattr(venv_dedupe, "MIN_BYTES", 1 << 20)
    _put(a, "tiny.so", b"tiny"), _put(b, "tiny.so", b"tiny")
    monkeypatch.setattr(venv_dedupe, "MIN_BYTES", 4)
    result = dedupe_venvs(tmp_path)
    assert result.scanned_files == 2  # only a/ and b/ tiny.so
    assert not _same(a / "bin" / "tool", b / "bin" / "tool")


def test_a_file_changed_after_hashing_is_not_replaced(tmp_path):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    real = venv_dedupe._digest

    def digest_then_edit(path):
        out = real(path)
        if path == y:
            y.write_bytes(b"PAYLOAD")  # same size, new content, new mtime
        return out

    with patch.object(venv_dedupe, "_digest", side_effect=digest_then_edit):
        result = dedupe_venvs(tmp_path)
    assert not _same(x, y) and y.read_bytes() == b"PAYLOAD"
    assert result.linked_files == 0


def test_a_pip_style_reinstall_breaks_only_its_own_link(tmp_path):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    dedupe_venvs(tmp_path)
    assert _same(x, y)
    # pip's wheel installer unlinks before writing the new contents.
    os.unlink(y)
    y.write_bytes(b"upgrade")
    assert x.read_bytes() == b"payload"


def test_after_install_hook_respects_the_setting_and_never_raises(tmp_path, monkeypatch):
    a = _venv(tmp_path, "a")
    monkeypatch.setenv("MUSE_STORAGE_DEDUPE_VENVS", "false")
    with patch.object(venv_dedupe, "dedupe_venvs") as run:
        dedupe_after_install(a)
    run.assert_not_called()
    monkeypatch.setenv("MUSE_STORAGE_DEDUPE_VENVS", "true")
    with patch.object(venv_dedupe, "dedupe_venvs", side_effect=OSError("boom")) as run:
        dedupe_after_install(a)
    run.assert_called_once_with(tmp_path, only=a, cache=tmp_path.parent / "venv-digests.json")


def test_cli_reports_and_links(tmp_path, monkeypatch, capsys):
    import json

    from muse.cli_impl.storage import run_storage_dedupe

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    a, b = _venv(tmp_path / "venvs", "a"), _venv(tmp_path / "venvs", "b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    assert run_storage_dedupe(dry_run=True, json_output=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] and report["linked_files"] == 1 and not _same(x, y)
    assert run_storage_dedupe() == 0
    assert "linked 1 of 2" in capsys.readouterr().out and _same(x, y)


def test_digest_cache_skips_rehashing_unchanged_inodes(tmp_path):
    root = tmp_path / "venvs"
    a, b = _venv(root, "a"), _venv(root, "b")
    _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    cache = tmp_path / "venv-digests.json"
    assert dedupe_venvs(root, cache=cache).linked_files == 1
    c = _venv(root, "c")
    _put(c, "lib.so", b"payload")
    real = venv_dedupe._digest
    with patch.object(venv_dedupe, "_digest", side_effect=real) as hashed:
        assert dedupe_venvs(root, cache=cache).linked_files == 1
    assert [call.args[0].parent.parent.parent.parent.name for call in hashed.call_args_list] == ["c"]


def test_a_corrupt_digest_cache_only_costs_rehashing(tmp_path):
    root = tmp_path / "venvs"
    a, b = _venv(root, "a"), _venv(root, "b")
    _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    cache = tmp_path / "venv-digests.json"
    cache.write_text("{not json")
    assert dedupe_venvs(root, cache=cache).linked_files == 1


def test_a_venv_in_a_transaction_is_left_alone(tmp_path):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "qwen3.5-b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    (tmp_path / ".qwen3.5-b.transaction-abc123").mkdir()
    assert dedupe_venvs(tmp_path).linked_files == 0
    assert not _same(x, y)


def test_a_stale_cached_digest_never_links_different_bytes(tmp_path):
    root = tmp_path / "venvs"
    a, b = _venv(root, "a"), _venv(root, "b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"PAYLOAD")
    with patch.object(venv_dedupe, "_digest", return_value="same"):
        result = dedupe_venvs(root)
    assert result.linked_files == 0 and not _same(x, y)
    assert y.read_bytes() == b"PAYLOAD"


def test_a_keeper_replaced_while_linking_is_not_linked(tmp_path):
    a, b = _venv(tmp_path, "a"), _venv(tmp_path, "b")
    x, y = _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    real_link = os.link

    def swap_keeper_first(src, dst):
        # pip replaces the keeper (new inode, same bytes) just before the link.
        Path(src).unlink()
        Path(src).write_bytes(b"PAYLOAD")
        real_link(src, dst)

    with patch.object(venv_dedupe.os, "link", side_effect=swap_keeper_first):
        result = dedupe_venvs(tmp_path)
    assert result.linked_files == 0
    assert (a / "lib/python3.12/site-packages/lib.so").read_bytes() != (
        b / "lib/python3.12/site-packages/lib.so").read_bytes()
    assert not list(tmp_path.rglob("*.muse-dedupe-*"))


def test_the_digest_cache_drops_inodes_that_no_longer_exist(tmp_path):
    import json

    root = tmp_path / "venvs"
    a, b = _venv(root, "a"), _venv(root, "b")
    _put(a, "lib.so", b"payload"), _put(b, "lib.so", b"payload")
    cache = tmp_path / "venv-digests.json"
    cache.write_text(json.dumps({"1:999999": [7, 0, "gone"]}))
    dedupe_venvs(root, cache=cache)
    kept = json.loads(cache.read_text())
    assert "1:999999" not in kept and kept
