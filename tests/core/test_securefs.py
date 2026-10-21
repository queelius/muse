"""Unit tests for the shared secure-file primitives (muse.core.securefs)."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from muse.core import securefs

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX ownership/no-follow semantics",
)


class Boom(RuntimeError):
    pass


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _other_uid(monkeypatch) -> None:
    real = os.geteuid()
    monkeypatch.setattr(securefs.os, "geteuid", lambda: real + 1)


# --- directories -----------------------------------------------------------


def test_ensure_directory_creates_private_despite_umask(tmp_path):
    target = tmp_path / "a" / "b"
    previous = os.umask(0)
    try:
        securefs.ensure_directory(target, error=Boom, label="dir", private=True)
    finally:
        os.umask(previous)
    assert _mode(target) == 0o700


def test_ensure_directory_private_repairs_existing_mode(tmp_path):
    target = tmp_path / "d"
    target.mkdir()
    target.chmod(0o777)
    securefs.ensure_directory(target, error=Boom, label="dir", private=True)
    assert _mode(target) == 0o700


def test_ensure_directory_refuses_group_writable_when_not_private(tmp_path):
    target = tmp_path / "d"
    target.mkdir()
    target.chmod(0o770)
    with pytest.raises(Boom, match="dir is group/other writable"):
        securefs.ensure_directory(target, error=Boom, label="dir")


def test_ensure_directory_refuses_symlink_leaf_without_touching_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o755)
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Boom, match="not a safe directory"):
        securefs.ensure_directory(link, error=Boom, label="dir", private=True)
    assert _mode(outside) == 0o755


def test_ensure_directory_wraps_dangling_symlink_creation(tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(Boom, match="cannot create dir"):
        securefs.ensure_directory(link, error=Boom, label="dir")


def test_validate_directory_refuses_foreign_owner(tmp_path, monkeypatch):
    _other_uid(monkeypatch)
    with pytest.raises(Boom, match="dir is not owned by this user"):
        securefs.validate_directory(tmp_path, error=Boom, label="dir")


def test_validate_directory_trusted_owners(tmp_path, monkeypatch):
    real = os.geteuid()
    _other_uid(monkeypatch)
    assert securefs.validate_directory(
        tmp_path, error=Boom, label="dir", trusted_owners=(real,),
    )
    with pytest.raises(Boom, match="dir has an untrusted owner"):
        securefs.validate_directory(
            tmp_path, error=Boom, label="dir", trusted_owners=(real + 2,),
        )


def test_validate_directory_missing(tmp_path):
    missing = tmp_path / "missing"
    assert securefs.validate_directory(
        missing, error=Boom, label="dir", missing_ok=True,
    ) is False
    with pytest.raises(FileNotFoundError):
        securefs.validate_directory(missing, error=Boom, label="dir")


def test_validate_directory_refuses_regular_file(tmp_path):
    path = tmp_path / "file"
    path.write_text("x")
    with pytest.raises(Boom, match="not a safe directory|not a directory"):
        securefs.validate_directory(path, error=Boom, label="dir")


# --- regular files ---------------------------------------------------------


def test_open_regular_file_refuses_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("secret")
    link = tmp_path / "link"
    link.symlink_to(outside)
    with pytest.raises(Boom, match="cannot safely open thing"):
        securefs.open_regular_file(link, os.O_RDONLY, error=Boom, label="thing")


def test_open_regular_file_action_in_message(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "anything")
    with pytest.raises(Boom, match="cannot safely read thing"):
        securefs.open_regular_file(
            link, os.O_RDONLY, error=Boom, label="thing", action="read",
        )


def test_open_regular_file_missing_is_raw(tmp_path):
    with pytest.raises(FileNotFoundError):
        securefs.open_regular_file(
            tmp_path / "missing", os.O_RDONLY, error=Boom, label="thing",
        )


def test_open_regular_file_refuses_directory(tmp_path):
    with pytest.raises(Boom):
        securefs.open_regular_file(tmp_path, os.O_RDONLY, error=Boom, label="thing")


def test_open_regular_file_private_and_links(tmp_path):
    path = tmp_path / "f"
    path.write_text("x")
    path.chmod(0o660)
    os.close(securefs.open_regular_file(path, os.O_RDONLY, error=Boom, label="f"))
    with pytest.raises(Boom, match="f is group/other writable"):
        securefs.open_regular_file(
            path, os.O_RDONLY, error=Boom, label="f", require_private=True,
        )
    path.chmod(0o600)
    os.link(path, tmp_path / "g")
    with pytest.raises(Boom, match="f has multiple links"):
        securefs.open_regular_file(
            path, os.O_RDONLY, error=Boom, label="f", require_single_link=True,
        )


def test_open_regular_file_refuses_foreign_owner(tmp_path, monkeypatch):
    path = tmp_path / "f"
    path.write_text("x")
    _other_uid(monkeypatch)
    with pytest.raises(Boom, match="f is not owned by this user"):
        securefs.open_regular_file(path, os.O_RDONLY, error=Boom, label="f")


# --- lock files ------------------------------------------------------------


def test_open_lock_file_creates_private_single_byte(tmp_path):
    lock = tmp_path / ".lock"
    previous = os.umask(0)
    try:
        handle = securefs.open_lock_file(lock, error=Boom, label="lock")
    finally:
        os.umask(previous)
    handle.close()
    assert _mode(lock) == 0o600
    assert lock.read_bytes() == b"\0"


def test_open_lock_file_refuses_group_writable_and_symlink(tmp_path):
    lock = tmp_path / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o660)
    with pytest.raises(Boom, match="group/other writable"):
        securefs.open_lock_file(lock, error=Boom, label="lock")

    outside = tmp_path / "outside"
    outside.write_text("keep")
    link = tmp_path / ".link-lock"
    link.symlink_to(outside)
    with pytest.raises(Boom, match="cannot safely open lock"):
        securefs.open_lock_file(link, error=Boom, label="lock")
    assert outside.read_text() == "keep"


def test_open_lock_file_without_create(tmp_path):
    lock = tmp_path / ".lock"
    with pytest.raises(FileNotFoundError):
        securefs.open_lock_file(lock, error=Boom, label="lock", create=False)
    assert not lock.exists()


def test_lock_file_nonblocking_conflict(tmp_path):
    lock = tmp_path / ".lock"
    first = securefs.open_lock_file(lock, error=Boom, label="lock")
    second = securefs.open_lock_file(lock, error=Boom, label="lock")
    try:
        assert securefs.lock_file(first) is True
        assert securefs.lock_file(second, blocking=False) is False
        securefs.unlock_file(first)
        assert securefs.lock_file(second, blocking=False) is True
        securefs.unlock_file(second)
    finally:
        first.close()
        second.close()


def test_shared_locks_coexist(tmp_path):
    lock = tmp_path / ".lock"
    securefs.open_lock_file(lock, error=Boom, label="lock").close()
    first = securefs.open_lock_file(lock, error=Boom, label="lock", create=False)
    second = securefs.open_lock_file(lock, error=Boom, label="lock", create=False)
    try:
        assert securefs.lock_file(first, exclusive=False)
        assert securefs.lock_file(second, exclusive=False, blocking=False)
    finally:
        first.close()
        second.close()


# --- durable publication ---------------------------------------------------


def test_atomic_write_text_private_despite_umask(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old")
    target.chmod(0o644)
    previous = os.umask(0)
    try:
        securefs.atomic_write_text(target, "new\n")
    finally:
        os.umask(previous)
    assert target.read_bytes() == b"new\n"
    assert _mode(target) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_atomic_write_text_replaces_symlink_not_its_target(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("keep")
    target = tmp_path / "out"
    target.symlink_to(outside)
    securefs.atomic_write_text(target, "new")
    assert not target.is_symlink()
    assert outside.read_text() == "keep"


def test_atomic_write_text_failure_keeps_old_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / "out"
    target.write_text("old")

    def fail(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(securefs.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        securefs.atomic_write_text(target, "new")
    with pytest.raises(Boom, match="cannot write .*replace failed"):
        securefs.atomic_write_text(target, "new", error=Boom)
    assert target.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_atomic_create_text(tmp_path):
    target = tmp_path / "new"
    assert securefs.atomic_create_text(target, "first") is True
    assert securefs.atomic_create_text(target, "second") is False
    assert target.read_text() == "first"
    assert target.stat().st_nlink == 1
    assert _mode(target) == 0o600

    outside = tmp_path / "outside"
    outside.write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(outside)
    assert securefs.atomic_create_text(link, "x") is False
    assert outside.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link", "new", "outside"]
