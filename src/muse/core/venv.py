"""Venv management helpers.

Each pulled model gets its own venv under ~/.muse/venvs/<model-id>/.
This module handles creation and pip-install-into-venv; the catalog
records the resulting Python interpreter path per model.

Output discipline (v0.40.3+): `install_into_venv` defaults to quiet
mode (`pip install -q` + captured stdout/stderr, only emitted on
non-zero exit). The CLI's `muse pull -v` / `--verbose` flips into
pass-through mode via the `install_output_mode` context manager.
The verbose mode preserves the v0.40.2 firehose for dep-resolution
debugging.
"""
from __future__ import annotations

import collections.abc
import contextlib
import errno
import logging
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from muse.core.owned_command import _OS_NAME, run_owned_command

logger = logging.getLogger(__name__)


_VENV_CREATE_TIMEOUT = 120


# Thread-local verbose flag. The CLI flips this via
# `install_output_mode(verbose=True)` when `muse pull -v` is set;
# default behavior (no flag) is quiet. Thread-local so concurrent
# pulls in different threads don't stomp each other (not a current
# muse use-case but cheap correctness insurance).
_local = threading.local()


def _is_verbose() -> bool:
    return bool(getattr(_local, "verbose", False))


@contextlib.contextmanager
def install_output_mode(verbose: bool):
    """Toggle pip + HF download output for everything called inside.

    Used by the CLI's `pull` command:

        with install_output_mode(verbose=args.verbose):
            pull(identifier)

    Inside the block: `install_into_venv` runs pip without `-q` and
    streams pip's stdout/stderr to the user; outside (or with
    verbose=False), pip is run with `-q` and stdout/stderr is
    captured, only emitted on non-zero exit.

    Catalog.py wraps snapshot_download similarly using the
    `HF_HUB_DISABLE_PROGRESS_BARS` env var whose value is read from
    `_is_verbose()`.
    """
    prev = getattr(_local, "verbose", False)
    _local.verbose = bool(verbose)
    try:
        yield
    finally:
        _local.verbose = prev


def venv_python(venv_path: Path) -> Path:
    """Return the Python interpreter path inside a venv.

    POSIX layout only (bin/python). The Windows layout (Scripts/python.exe)
    is not supported because muse is Linux/macOS-focused.
    """
    return venv_path / "bin" / "python"


def _emit_failure_output(exc: subprocess.CalledProcessError | subprocess.TimeoutExpired) -> None:
    stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None)
    stderr = getattr(exc, "stderr", None)
    if stdout:
        sys.stderr.write(stdout)
    if stderr:
        sys.stderr.write(stderr)


def remove_owned_entry(path: Path, parent: Path) -> None:
    """Remove exactly one owned entry below ``parent`` without following links.

    A symlink or non-directory is unlinked as one entry, so a substituted
    link's target is never traversed. Directories are removed recursively
    only when ``shutil.rmtree`` advertises its fd-safe implementation. A
    missing path is not an error. Raises ValueError for a refused removal
    and OSError when the filesystem fails.
    """
    if path.parent != parent:
        raise ValueError(f"refusing cleanup outside its parent: {path}")
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            path.unlink()
            return
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise ValueError(f"fd-safe recursive deletion is unavailable: {path}")
        shutil.rmtree(path)
    except FileNotFoundError:
        # Another cleanup owner may already have removed the exact path.
        return


def _cleanup_venv_staging(staging: Path, parent: Path) -> None:
    """Remove exactly one private venv staging entry without following links."""
    try:
        remove_owned_entry(staging, parent)
    except ValueError as exc:
        logger.error("refusing venv staging cleanup: %s", exc)
    except OSError as exc:
        logger.warning("could not clean venv staging path %s: %s", staging, exc)


def _validate_created_venv(staging: Path) -> None:
    """Require the staged environment to contain a usable interpreter."""
    python = venv_python(staging)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f"venv creation completed without an executable interpreter: {python}"
        )


def create_venv(target: Path) -> None:
    """Create a fresh venv at `target`, using the same Python that muse runs on.

    Using `sys.executable` guarantees ABI compatibility: the venv's Python
    is the same version as the muse-supervisor's Python, so torch/CUDA
    wheels built for one will load in the other.

    Honors `install_output_mode(verbose=...)`: in quiet mode captures
    stdout/stderr and only emits on error.
    """
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"refusing to replace existing venv path: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    logger.info("creating venv at %s", target)
    cmd = [sys.executable, "-m", "venv", str(staging)]
    quiet = not _is_verbose()
    try:
        run_owned_command(
            cmd, timeout=_VENV_CREATE_TIMEOUT,
            capture_output=quiet, check=True,
        )
        _validate_created_venv(staging)
        staging.rename(target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if quiet:
            _emit_failure_output(exc)
        raise
    finally:
        _cleanup_venv_staging(staging, parent)


def ensure_venv(
    target: Path,
    *,
    creator: collections.abc.Callable[[Path], None] = create_venv,
) -> None:
    """Create a missing venv or reject an unsafe/incomplete existing one."""
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        creator(target)
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"existing venv path is not a regular directory: {target}")
    python = venv_python(target)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f"existing venv is incomplete (missing executable {python}); "
            "move it aside or purge it, then re-pull the model"
        )


def _clone_file(prior: Path, target: Path):
    """copytree copy_function for cloning `prior` back to `target`.

    Large package files are hardlinked rather than copied. The prior venv
    stays a valid rollback because pip unlinks an installed file before
    writing its replacement (the invariant venv_dedupe relies on), and the
    clone keeps the links dedupe made. Anything else is copied.
    """
    from muse.core.venv_dedupe import is_package_file

    def clone(src: str, dst: str) -> str:
        rel = Path(dst).relative_to(target)
        try:
            if is_package_file(rel, os.lstat(src).st_size):
                os.link(src, dst)
                return dst
        except OSError:
            pass  # e.g. EXDEV or a link-count limit: fall back to a copy
        return shutil.copy2(src, dst)

    return clone


def _venv_clone_bytes(source: Path) -> int:
    """Conservatively estimate bytes needed to clone one venv tree (files
    the clone hardlinks cost nothing; see `_clone_file`)."""
    from muse.core.venv_dedupe import is_package_file

    total = 0
    try:
        for root, directories, files in os.walk(source, followlinks=False):
            entries = [Path(root), *(Path(root) / name for name in directories)]
            entries.extend(Path(root) / name for name in files)
            for entry in entries:
                metadata = entry.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                if stat.S_ISREG(metadata.st_mode) and is_package_file(
                    entry.relative_to(source), metadata.st_size,
                ):
                    continue
                allocated = int(getattr(metadata, "st_blocks", 0)) * 512
                total += max(4096, int(metadata.st_size), allocated)
    except OSError as exc:
        raise RuntimeError(
            f"could not size existing venv for transactional update: {source}"
        ) from exc
    return total


def _insufficient_venv_space(
    target: Path, *, required: int, available: int | None = None,
) -> RuntimeError:
    detail = f"requires approximately {required} bytes"
    if available is not None:
        detail += f", but only {available} bytes are available"
    return RuntimeError(
        f"insufficient disk space to transactionally update venv {target}: "
        f"{detail}. Free space or purge unused models, then retry"
    )


def _is_venv_space_error(exc: BaseException) -> bool:
    """Recognize direct and copytree-aggregated disk exhaustion errors."""
    disk_errnos = {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}
    if isinstance(exc, OSError) and exc.errno in disk_errnos:
        return True
    if isinstance(exc, shutil.Error):
        details = exc.args[0] if exc.args else ()
        for detail in details if isinstance(details, list) else ():
            cause = detail[2] if isinstance(detail, tuple) and len(detail) >= 3 else detail
            if isinstance(cause, BaseException) and _is_venv_space_error(cause):
                return True
            text = str(cause)
            if any(f"[Errno {number}]" in text for number in disk_errnos):
                return True
    return False


@dataclass
class _VenvTransaction:
    """Rollback-safe mutation of one canonical model venv.

    An existing venv is renamed into a private sibling workspace and cloned
    back to its original path. Callers therefore mutate the same canonical
    path embedded in venv scripts and .pth files while the prior environment
    remains available for rollback. This temporarily needs roughly one extra
    venv's worth of disk space.

    Exceptions, cancellation, and ordinary exits without commit restore the
    prior venv. A SIGKILL or power loss between filesystem and catalog commits
    cannot be made unambiguous without a cross-resource transaction; private
    transaction directories are therefore never heuristically promoted or
    restored on a later run.
    """

    path: Path
    _workspace: Path | None = None
    _prior: Path | None = None
    _entered: bool = False
    _committed: bool = False

    def __enter__(self) -> "_VenvTransaction":
        if self._entered:
            raise RuntimeError("venv transaction cannot be entered more than once")
        self._entered = True
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)

        try:
            self.path.lstat()
        except FileNotFoundError:
            return self

        # Reject symlinks and incomplete environments before moving anything.
        ensure_venv(self.path)
        required = _venv_clone_bytes(self.path)
        try:
            available = int(shutil.disk_usage(parent).free)
        except OSError as exc:
            raise RuntimeError(
                f"could not determine free space for transactional venv update: "
                f"{parent}"
            ) from exc
        if available < required:
            raise _insufficient_venv_space(
                self.path, required=required, available=available,
            )

        workspace = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.transaction-", dir=parent)
        )
        prior = workspace / "prior"
        self._workspace = workspace
        self._prior = prior
        prior_exists = False
        try:
            self.path.rename(prior)
            prior_exists = True
            shutil.copytree(
                prior,
                self.path,
                symlinks=True,
                copy_function=_clone_file(prior, self.path),
            )
            ensure_venv(self.path)
        except BaseException as exc:
            try:
                if prior_exists:
                    self._restore()
                else:
                    self._prior = None
                    _cleanup_venv_staging(workspace, parent)
                    self._workspace = None
            except BaseException as rollback_exc:
                raise RuntimeError(
                    f"failed to clone venv {self.path} and could not restore "
                    "the prior environment"
                ) from rollback_exc
            if _is_venv_space_error(exc):
                raise _insufficient_venv_space(
                    self.path, required=required, available=available,
                ) from exc
            raise
        return self

    def commit(self) -> None:
        """Keep the mutated canonical venv when the context exits."""
        if not self._entered:
            raise RuntimeError("venv transaction has not been entered")
        self._committed = True

    def _move_current_aside(self) -> Path | None:
        workspace = self._workspace
        if workspace is None:
            # A first pull has no backup workspace. Reserve one only when
            # rollback actually has a newly-created target to remove.
            try:
                self.path.lstat()
            except FileNotFoundError:
                return None
            workspace = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.path.name}.transaction-", dir=self.path.parent,
                )
            )
            self._workspace = workspace
        discard = workspace / "discard"
        try:
            self.path.rename(discard)
        except FileNotFoundError:
            return None
        return discard

    def _restore(self) -> None:
        """Atomically restore the prior path, then remove failed state."""
        self._move_current_aside()
        prior = self._prior
        if prior is not None:
            prior.rename(self.path)
            self._prior = None
        workspace = self._workspace
        if workspace is not None:
            _cleanup_venv_staging(workspace, self.path.parent)
            self._workspace = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._committed:
            try:
                self._restore()
            except BaseException as rollback_exc:
                raise RuntimeError(
                    f"could not restore venv {self.path} after failed update"
                ) from rollback_exc
            return False

        workspace = self._workspace
        if workspace is not None:
            _cleanup_venv_staging(workspace, self.path.parent)
            self._workspace = None
            self._prior = None
        return False


def venv_transaction(target: Path) -> _VenvTransaction:
    """Return a rollback-safe transaction for one canonical model venv."""
    return _VenvTransaction(Path(target))


def _venv_site_packages(venv_path: Path) -> Path:
    if _OS_NAME == "nt":
        return venv_path / "Lib" / "site-packages"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv_path / "lib" / version / "site-packages"


def find_free_port(start: int = 9001, end: int = 9999) -> int:
    """Find an unbound local port in [start, end]. Raises RuntimeError if exhausted.

    The port is probed by briefly binding then releasing it; the returned
    number is a *hint*, not a reservation. A TOCTOU window exists between
    this function returning and the caller binding the port, so the caller
    MUST verify the worker actually bound it (e.g., via /health check)
    and retry with a different port on startup failure.
    """
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"no free port in range [{start}, {end}]")
