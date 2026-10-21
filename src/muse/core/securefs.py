"""Secure-file primitives shared by Muse's on-disk state owners.

The catalog (``catalog.json`` + its locks), the resource registry
(``runtime/resources.json``) and the config file (``config.yaml``) all live
under ``~/.muse`` and all defend the same trust boundary: a co-tenant or a
planted symlink must not be able to redirect a write, a chmod, or a lock to
a path Muse does not own. This module is the single implementation of the
four primitives they share:

* :func:`ensure_directory` / :func:`validate_directory` -- an owned
  directory, opened without following its final component, refused when it
  is a symlink/junction, not a directory, owned by another user, or
  group/other writable (``private=True`` instead forces mode 0700).
* :func:`open_regular_file` / :func:`open_lock_file` -- a no-follow open
  that accepts only an owned regular file, optionally private and with a
  single link (hard links would let a lock or config be shared with a path
  outside Muse's directory).
* :func:`lock_file` / :func:`unlock_file` -- advisory whole-file locks
  (``flock`` on POSIX, byte-0 ``msvcrt.locking`` on Windows).
* :func:`atomic_write_text` / :func:`atomic_create_text` -- durable
  publication through a mode-0600 same-directory temp file, fsyncing the
  file before publication and the directory after it.

Every validating primitive takes ``error=``, the exception class to raise,
and ``label=``, the noun used in messages, so each owner keeps its own error
type (``CatalogError``, ``ResourceRegistryError``, ``ConfigError``) and
readable messages. ``FileNotFoundError`` is deliberately re-raised raw from
the open/validate primitives: callers on read paths treat a missing file as
"empty", which is not a safety failure.

Only the final path component is checked. Parents may legitimately be
symlinked (an operator who points ``MUSE_CATALOG_DIR`` through a symlinked
home); the leaf Muse is about to populate, chmod or lock must not be.

Stdlib-only, so importing it from ``config`` (imported by everything) costs
nothing and introduces no cycle.
"""
from __future__ import annotations

import errno
import os
import stat
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import BinaryIO

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
_GROUP_OTHER_WRITE = 0o022

_IS_POSIX = os.name == "posix"


def _trusted(owners: Collection[int] | None) -> Collection[int]:
    return (os.geteuid(),) if owners is None else owners


def _owner_message(label: str, path: Path, owners: Collection[int] | None) -> str:
    if owners is None:
        return f"{label} is not owned by this user: {path}"
    return f"{label} has an untrusted owner: {path}"


# --- directories -----------------------------------------------------------


def validate_directory(
    path: Path,
    *,
    error: type[Exception],
    label: str,
    private: bool = False,
    missing_ok: bool = False,
    trusted_owners: Collection[int] | None = None,
) -> bool:
    """Validate ``path`` as a safe directory without following its leaf.

    POSIX: the directory is opened ``O_NOFOLLOW | O_DIRECTORY`` and checked
    through the descriptor, so a swap between check and use cannot redirect
    the check. It must be owned by the effective user (or one of
    ``trusted_owners``). With ``private`` it is then forced to mode 0700;
    otherwise a group/other-writable directory is refused.

    Windows: a symlink or junction leaf is refused; ``private`` best-effort
    chmods the directory.

    Returns False only when the directory is missing and ``missing_ok``;
    a missing directory otherwise raises ``FileNotFoundError``.
    """
    if not _IS_POSIX:  # pragma: no cover - Windows fallback
        try:
            info = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        except OSError as exc:
            raise error(f"cannot inspect {label} {path}: {exc}") from exc
        is_junction = getattr(path, "is_junction", lambda: False)
        if stat.S_ISLNK(info.st_mode) or is_junction() or not path.is_dir():
            raise error(f"{label} is not a safe directory: {path}")
        if private:
            path.chmod(PRIVATE_DIR_MODE)
        return True

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    except OSError as exc:
        raise error(f"{label} is not a safe directory: {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise error(f"{label} is not a directory: {path}")
        if info.st_uid not in _trusted(trusted_owners):
            raise error(_owner_message(label, path, trusted_owners))
        if private:
            os.fchmod(descriptor, PRIVATE_DIR_MODE)
        elif info.st_mode & _GROUP_OTHER_WRITE:
            raise error(f"{label} is group/other writable: {path}")
    finally:
        os.close(descriptor)
    return True


def ensure_directory(
    path: Path,
    *,
    error: type[Exception],
    label: str,
    private: bool = False,
) -> None:
    """Create ``path`` (mode 0700, with parents) and validate it.

    Creation failures and a directory vanishing between creation and
    validation both raise ``error``; see :func:`validate_directory` for the
    checks applied.
    """
    try:
        path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    except OSError as exc:
        raise error(f"cannot create {label} {path}: {exc}") from exc
    try:
        validate_directory(path, error=error, label=label, private=private)
    except FileNotFoundError as exc:
        raise error(f"{label} is not a safe directory: {path}: {exc}") from exc


# --- regular files ---------------------------------------------------------


def open_regular_file(
    path: Path,
    flags: int,
    *,
    error: type[Exception],
    label: str,
    mode: int | None = None,
    require_single_link: bool = False,
    require_private: bool = False,
    trusted_owners: Collection[int] | None = None,
    action: str = "open",
) -> int:
    """Open one regular file without following its final component.

    Returns the raw descriptor (``O_CLOEXEC``). The opened file must be a
    regular file; on POSIX it must be owned by the effective user (or one of
    ``trusted_owners``) and, with ``require_private``, not group/other
    writable. ``require_single_link`` refuses a hard-linked file.
    ``FileNotFoundError`` propagates raw; every other failure raises
    ``error``. The descriptor is closed on any validation failure.
    """
    open_flags = flags | getattr(os, "O_CLOEXEC", 0)
    if _IS_POSIX:
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
    else:  # pragma: no cover - Windows fallback
        try:
            before = path.lstat()
        except FileNotFoundError:
            before = None
        if before is not None and stat.S_ISLNK(before.st_mode):
            raise error(f"refusing symlink {label} {path}")
        open_flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = (
            os.open(path, open_flags)
            if mode is None
            else os.open(path, open_flags, mode)
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise error(f"cannot safely {action} {label} {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise error(f"{label} is not a regular file: {path}")
        if _IS_POSIX:
            if info.st_uid not in _trusted(trusted_owners):
                raise error(_owner_message(label, path, trusted_owners))
            if require_private and info.st_mode & _GROUP_OTHER_WRITE:
                raise error(f"{label} is group/other writable: {path}")
        if require_single_link and info.st_nlink != 1:
            raise error(f"{label} has multiple links: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_lock_file(
    path: Path,
    *,
    error: type[Exception],
    label: str,
    create: bool = True,
) -> BinaryIO:
    """Open an advisory lock file as an unbuffered binary handle.

    The lock must be a private, single-link regular file owned by this user.
    With ``create`` it is created mode 0600 if absent, re-chmodded 0600, and
    given one byte (byte-range locking on Windows needs a concrete byte; it
    is written on every platform so a catalog moves across them unchanged).
    Without ``create`` it is opened read-only and ``FileNotFoundError``
    propagates for a missing lock.
    """
    descriptor = open_regular_file(
        path,
        os.O_CREAT | os.O_RDWR if create else os.O_RDONLY,
        error=error,
        label=label,
        mode=PRIVATE_FILE_MODE if create else None,
        require_single_link=True,
        require_private=True,
    )
    try:
        handle = os.fdopen(descriptor, "r+b" if create else "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise
    if not create:
        return handle
    try:
        if _IS_POSIX:
            os.fchmod(handle.fileno(), PRIVATE_FILE_MODE)
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            os.fsync(handle.fileno())
    except BaseException as exc:
        handle.close()
        if isinstance(exc, OSError):
            raise error(f"cannot prepare {label} {path}: {exc}") from exc
        raise
    return handle


# --- advisory locks --------------------------------------------------------


def lock_file(handle: BinaryIO, *, exclusive: bool = True, blocking: bool = True) -> bool:
    """Take an advisory lock on an open lock-file handle.

    Returns False only when ``blocking`` is false and another owner holds
    the lock; other failures raise ``OSError`` (or ``ImportError`` on a
    platform without the locking module) for the caller to translate.
    """
    if os.name == "nt":  # pragma: no cover - platform-specific fallback
        import msvcrt

        # msvcrt.locking locks bytes from the current file position, so
        # every lock consistently covers byte 0 (written by open_lock_file).
        handle.seek(0)
        if blocking:
            operation = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
        else:
            operation = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
        try:
            msvcrt.locking(handle.fileno(), operation, 1)
        except OSError as exc:
            if not blocking and exc.errno in {
                errno.EACCES, errno.EAGAIN, errno.EDEADLK,
            }:
                return False
            raise
        return True

    import fcntl

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), operation)
    except BlockingIOError:
        if blocking:
            raise
        return False
    return True


def unlock_file(handle: BinaryIO) -> None:
    """Release a lock taken by :func:`lock_file`."""
    if os.name == "nt":  # pragma: no cover - platform-specific fallback
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# --- durable publication ---------------------------------------------------


def fsync_directory(path: Path) -> None:
    """Best-effort fsync of a directory after publishing an entry in it.

    Opened no-follow. Some filesystems/platforms cannot fsync a directory;
    the published file itself was already fsynced, so failures are ignored.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_temp(target: Path, text: str, mode: int) -> Path:
    """Write ``text`` to a fsynced, ``mode`` temp file beside ``target``.

    The temp file is created O_EXCL by ``mkstemp`` (so never through a
    planted name) and chmodded through its descriptor before any content is
    written, so secrets are never exposed through the umask.
    """
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    tmp = Path(raw_tmp)
    open_fd: int | None = descriptor
    try:
        if _IS_POSIX:
            os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            open_fd = None
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if open_fd is not None:
            os.close(open_fd)
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def atomic_write_text(
    target: Path,
    text: str,
    *,
    mode: int = PRIVATE_FILE_MODE,
    error: type[Exception] | None = None,
) -> None:
    """Durably replace ``target`` with ``text`` (UTF-8, LF newlines).

    Rename within one directory is atomic on POSIX and replaces atomically
    on Windows (``os.replace`` wraps ``MoveFileEx``). The caller owns the
    validation of ``target.parent`` and any transaction lock. On failure
    the temp file is removed and the old ``target`` is untouched. With
    ``error``, an ``OSError`` is raised as ``error("cannot write ...")``;
    otherwise it propagates raw.
    """
    try:
        tmp = _write_temp(target, text, mode)
        try:
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        fsync_directory(target.parent)
    except OSError as exc:
        if error is None:
            raise
        raise error(f"cannot write {target}: {exc}") from exc


def atomic_create_text(
    target: Path,
    text: str,
    *,
    mode: int = PRIVATE_FILE_MODE,
    error: type[Exception] | None = None,
) -> bool:
    """Durably create ``target``; return False if any entry already exists.

    On POSIX publication is a same-directory hard link, which has no
    replacement mode: an existing file, symlink, or directory makes it fail,
    so even a non-cooperating creator cannot be overwritten between a check
    and the write. The temp name is then unlinked so the published file has
    exactly one link (secure readers reject hard-linked files). Windows uses
    ``os.rename``, which also refuses an existing destination.
    """
    published = False
    tmp: Path | None = None
    try:
        tmp = _write_temp(target, text, mode)
        try:
            if _IS_POSIX:
                os.link(tmp, target)
                published = True
                try:
                    tmp.unlink()
                except BaseException:
                    target.unlink(missing_ok=True)
                    published = False
                    raise
            else:  # pragma: no cover - Windows no-replace rename semantics
                os.rename(tmp, target)
                published = True
        except FileExistsError:
            return False
        fsync_directory(target.parent)
        return True
    except BaseException as exc:
        if published:
            target.unlink(missing_ok=True)
        if isinstance(exc, OSError) and error is not None:
            raise error(f"cannot write {target}: {exc}") from exc
        raise
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
