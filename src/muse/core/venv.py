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
import configparser
import errno
import logging
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from muse.core.owned_command import _OS_NAME, _run_owned, run_owned_command
from muse.core.resolvers import COMMIT_RE

logger = logging.getLogger(__name__)


_VENV_CREATE_TIMEOUT = 120
_SOURCE_INSTALL_TIMEOUT = 1800

_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GITHUB_GIT_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$"
)
_GIT_SAFE_CONFIG = (
    "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always",
    "-c", "credential.interactive=false",
    "-c", "credential.helper=",
    "-c", f"core.hooksPath={os.devnull}",
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
)


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


@dataclass(frozen=True)
class _GitSubmoduleSpec:
    path: str
    url: str
    revision: str


@dataclass(frozen=True)
class _GitPythonSourceSpec:
    name: str
    url: str
    revision: str
    sparse_paths: tuple[str, ...]
    required_paths: tuple[str, ...]
    pth_path: str
    submodules: tuple[_GitSubmoduleSpec, ...]


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


def _source_relative_path(
    value: object, *, field: str, allow_dot: bool = False,
) -> str:
    """Validate one repository-relative POSIX path from a manifest."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"python source {field} must be a non-empty string")
    if "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"python source {field} contains unsafe characters")
    if allow_dot and value == ".":
        return value
    path = PurePosixPath(value)
    if path.is_absolute() or value != str(path):
        raise ValueError(f"python source {field} must be a normalized relative path")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"python source {field} escapes its checkout")
    return value


def _source_string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"python source {field} must be a non-empty list")
    paths = tuple(
        _source_relative_path(item, field=f"{field}[]") for item in value
    )
    if len(paths) != len(set(paths)):
        raise ValueError(f"python source {field} contains duplicates")
    return paths


def _parse_git_python_source(raw: object) -> _GitPythonSourceSpec:
    """Parse and fail closed on one JSON-serializable source declaration."""
    if not isinstance(raw, Mapping):
        raise ValueError("python source declaration must be a mapping")
    allowed = {
        "type", "name", "url", "revision", "sparse_paths",
        "required_paths", "pth_path", "submodules",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown python source fields: {sorted(unknown)}")
    if raw.get("type") != "git":
        raise ValueError("python source type must be 'git'")

    name = raw.get("name")
    if not isinstance(name, str) or not _SOURCE_NAME_RE.fullmatch(name):
        raise ValueError("python source name must be a filesystem-safe identifier")
    url = raw.get("url")
    if not isinstance(url, str) or not _GITHUB_GIT_URL_RE.fullmatch(url):
        raise ValueError(
            "python source url must be an HTTPS github.com repository ending in .git"
        )
    revision = raw.get("revision")
    if not isinstance(revision, str) or not COMMIT_RE.fullmatch(revision):
        raise ValueError("python source revision must be a full lowercase commit SHA")

    sparse_paths = _source_string_sequence(
        raw.get("sparse_paths"), field="sparse_paths",
    )
    required_paths = _source_string_sequence(
        raw.get("required_paths"), field="required_paths",
    )
    pth_path = _source_relative_path(
        raw.get("pth_path", "."), field="pth_path", allow_dot=True,
    )

    raw_submodules = raw.get("submodules", ())
    if not isinstance(raw_submodules, (list, tuple)):
        raise ValueError("python source submodules must be a list")
    submodules: list[_GitSubmoduleSpec] = []
    seen_submodule_paths: set[str] = set()
    for item in raw_submodules:
        if not isinstance(item, Mapping) or set(item) != {"path", "url", "revision"}:
            raise ValueError(
                "each python source submodule requires path, url, and revision"
            )
        path = _source_relative_path(item.get("path"), field="submodules[].path")
        sub_url = item.get("url")
        if not isinstance(sub_url, str) or not _GITHUB_GIT_URL_RE.fullmatch(sub_url):
            raise ValueError("python source submodule url must be HTTPS GitHub .git")
        sub_revision = item.get("revision")
        if (
            not isinstance(sub_revision, str)
            or not COMMIT_RE.fullmatch(sub_revision)
        ):
            raise ValueError(
                "python source submodule revision must be a full lowercase commit SHA"
            )
        if path in seen_submodule_paths:
            raise ValueError("python source submodule paths must be unique")
        if not any(path == root or path.startswith(f"{root}/") for root in sparse_paths):
            raise ValueError("python source submodule must be inside a sparse path")
        seen_submodule_paths.add(path)
        submodules.append(_GitSubmoduleSpec(path, sub_url, sub_revision))

    return _GitPythonSourceSpec(
        name=name,
        url=url,
        revision=revision,
        sparse_paths=sparse_paths,
        required_paths=required_paths,
        pth_path=pth_path,
        submodules=tuple(submodules),
    )


def _source_timeout(deadline: float, command: Sequence[str]) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(
            cmd=list(command), timeout=_SOURCE_INSTALL_TIMEOUT,
        )
    return remaining


def _run_source_git(
    checkout: Path,
    args: Sequence[str],
    *,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *_GIT_SAFE_CONFIG, "-C", str(checkout), *args]
    # A reviewed URL is meaningless if inherited user configuration can
    # rewrite it, redirect the worktree/object database, or register checkout
    # filters. Preserve ordinary process/TLS/proxy environment while removing
    # every Git-specific override, then install only our non-interactive global
    # config boundary.
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    git_env.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "Never",
    })
    try:
        return _run_owned(
            command,
            timeout=_source_timeout(deadline, command),
            capture_output=True,
            env=git_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is required to install a model's reviewed Python source"
        ) from exc


def _source_git_output(
    checkout: Path,
    args: Sequence[str],
    *,
    deadline: float,
) -> str:
    result = _run_source_git(checkout, args, deadline=deadline)
    return (result.stdout or "").strip()


def _require_path_inside(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved_root = root.resolve()
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} escapes the model-owned venv") from exc
    return resolved


def _validate_submodule_metadata(
    checkout: Path, submodules: tuple[_GitSubmoduleSpec, ...],
) -> None:
    if not submodules:
        return
    metadata = checkout / ".gitmodules"
    if not metadata.is_file() or metadata.is_symlink():
        raise RuntimeError("reviewed Python source is missing a regular .gitmodules file")
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with metadata.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeError, configparser.Error) as exc:
        raise RuntimeError("could not parse reviewed Python source submodules") from exc
    for submodule in submodules:
        section = f'submodule "{submodule.path}"'
        if not parser.has_section(section):
            raise RuntimeError(f"reviewed submodule {submodule.path!r} is not declared")
        if parser.get(section, "path", fallback=None) != submodule.path:
            raise RuntimeError(f"reviewed submodule {submodule.path!r} path changed")
        if parser.get(section, "url", fallback=None) != submodule.url:
            raise RuntimeError(f"reviewed submodule {submodule.path!r} URL changed")
        if set(parser.options(section)) != {"path", "url"}:
            raise RuntimeError(
                f"reviewed submodule {submodule.path!r} options changed"
            )


def _validate_git_checkout(
    checkout: Path,
    spec: _GitPythonSourceSpec,
    *,
    deadline: float,
) -> None:
    actual_url = _source_git_output(
        checkout, ("remote", "get-url", "origin"), deadline=deadline,
    )
    if actual_url != spec.url:
        raise RuntimeError(
            f"Python source {spec.name!r} origin changed: {actual_url!r}"
        )
    actual_revision = _source_git_output(
        checkout, ("rev-parse", "HEAD"), deadline=deadline,
    )
    if actual_revision != spec.revision:
        raise RuntimeError(
            f"Python source {spec.name!r} is at unexpected commit "
            f"{actual_revision!r}"
        )
    dirty = _source_git_output(
        checkout,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        deadline=deadline,
    )
    if dirty:
        raise RuntimeError(f"Python source {spec.name!r} contains local changes")

    _validate_submodule_metadata(checkout, spec.submodules)
    for submodule in spec.submodules:
        submodule_root = checkout.joinpath(*PurePosixPath(submodule.path).parts)
        _require_path_inside(
            submodule_root, checkout, label=f"submodule {submodule.path!r}",
        )
        actual = _source_git_output(
            submodule_root, ("rev-parse", "HEAD"), deadline=deadline,
        )
        if actual != submodule.revision:
            raise RuntimeError(
                f"Python source submodule {submodule.path!r} is at "
                f"unexpected commit {actual!r}"
            )
        submodule_dirty = _source_git_output(
            submodule_root,
            ("status", "--porcelain=v1", "--untracked-files=all"),
            deadline=deadline,
        )
        if submodule_dirty:
            raise RuntimeError(
                f"Python source submodule {submodule.path!r} contains local changes"
            )

    for relative in spec.required_paths:
        required = checkout.joinpath(*PurePosixPath(relative).parts)
        _require_path_inside(required, checkout, label=f"required path {relative!r}")
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(
                f"Python source {spec.name!r} is missing regular file {relative!r}"
            )


def _materialize_git_source(
    source_root: Path,
    spec: _GitPythonSourceSpec,
    *,
    deadline: float,
) -> Path:
    name_root = source_root / spec.name
    name_root.mkdir(parents=True, exist_ok=True)
    _require_path_inside(name_root, source_root, label="Python source directory")
    if name_root.is_symlink():
        raise RuntimeError("Python source directory must not be a symlink")
    target = name_root / spec.revision

    if target.exists():
        if not target.is_dir() or target.is_symlink():
            raise RuntimeError("existing Python source target is not a regular directory")
        _validate_git_checkout(target, spec, deadline=deadline)
        return target

    temp = Path(tempfile.mkdtemp(prefix=f".{spec.revision}.", dir=name_root))
    try:
        _run_source_git(temp, ("init", "--quiet"), deadline=deadline)
        _run_source_git(
            temp, ("remote", "add", "origin", spec.url), deadline=deadline,
        )
        _run_source_git(
            temp, ("sparse-checkout", "init", "--cone"), deadline=deadline,
        )
        _run_source_git(
            temp, ("sparse-checkout", "set", *spec.sparse_paths), deadline=deadline,
        )
        _run_source_git(
            temp,
            (
                "fetch", "--depth", "1", "--filter=blob:none", "--no-tags",
                "origin", spec.revision,
            ),
            deadline=deadline,
        )
        _run_source_git(
            temp, ("checkout", "--quiet", "--detach", "FETCH_HEAD"),
            deadline=deadline,
        )
        _validate_submodule_metadata(temp, spec.submodules)
        for submodule in spec.submodules:
            _run_source_git(
                temp,
                (
                    "submodule", "update", "--init", "--depth", "1",
                    "--filter=blob:none", "--", submodule.path,
                ),
                deadline=deadline,
            )
        _validate_git_checkout(temp, spec, deadline=deadline)
        try:
            temp.rename(target)
        except OSError as exc:
            try:
                target_mode = target.lstat().st_mode
            except FileNotFoundError:
                target_is_regular_dir = False
            else:
                target_is_regular_dir = (
                    stat.S_ISDIR(target_mode) and not stat.S_ISLNK(target_mode)
                )
            if (
                exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}
                or not target_is_regular_dir
            ):
                raise
            # A concurrent refresh may have won promotion. Validate its exact
            # identity before discarding our private, still-owned temporary.
            _validate_git_checkout(target, spec, deadline=deadline)
        return target
    finally:
        _cleanup_source_staging(temp, name_root)


def _cleanup_source_staging(temp: Path, name_root: Path) -> None:
    """Remove exactly one installer-owned staging path without traversal risk."""
    try:
        remove_owned_entry(temp, name_root)
    except ValueError as exc:
        logger.error("refusing Python source staging cleanup: %s", exc)
    except OSError as exc:
        # Cleanup must not mask the source/Git failure that led here. Leaving
        # an inert staging directory is safer than widening deletion scope.
        logger.warning("could not remove Python source staging path %s: %s", temp, exc)


def _venv_site_packages(venv_path: Path) -> Path:
    if _OS_NAME == "nt":
        return venv_path / "Lib" / "site-packages"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv_path / "lib" / version / "site-packages"


def _source_pth_target(
    venv_path: Path,
    spec: _GitPythonSourceSpec,
) -> Path:
    site_packages = _venv_site_packages(venv_path)
    if not site_packages.is_dir() or site_packages.is_symlink():
        raise RuntimeError(
            f"venv site-packages is missing or unsafe: {site_packages}"
        )
    _require_path_inside(site_packages, venv_path, label="venv site-packages")
    return site_packages / f"muse-source-{spec.name}.pth"


def _disable_source_pth(venv_path: Path, spec: _GitPythonSourceSpec) -> None:
    """Fail closed before revalidating a previously exposed source tree."""
    target = _source_pth_target(venv_path, spec)
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        raise RuntimeError(f"Python source .pth target is an unsafe directory: {target}")
    try:
        target.unlink()
    except FileNotFoundError:
        pass


def _disable_all_source_pths(venv_path: Path) -> None:
    """Remove every hook in Muse's reserved reviewed-source namespace."""
    site_packages = _venv_site_packages(venv_path)
    try:
        site_mode = site_packages.lstat().st_mode
    except FileNotFoundError:
        # No site-packages means there is no executable .pth surface to revoke.
        return
    if not stat.S_ISDIR(site_mode) or stat.S_ISLNK(site_mode):
        raise RuntimeError(
            f"venv site-packages is missing or unsafe: {site_packages}"
        )
    _require_path_inside(site_packages, venv_path, label="venv site-packages")

    unsafe_directories: list[Path] = []
    for target in sorted(site_packages.glob("muse-source-*.pth")):
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            unsafe_directories.append(target)
            continue
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    if unsafe_directories:
        raise RuntimeError(
            "Python source .pth target is an unsafe directory: "
            f"{unsafe_directories[0]}"
        )


def _write_source_pth(
    venv_path: Path,
    checkout: Path,
    spec: _GitPythonSourceSpec,
) -> None:
    target = _source_pth_target(venv_path, spec)
    python_root = checkout.joinpath(*PurePosixPath(spec.pth_path).parts)
    python_root = _require_path_inside(
        python_root, checkout, label="Python source import path",
    )
    if not python_root.is_dir():
        raise RuntimeError("Python source import path is not a directory")

    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    temp = Path(raw_temp)
    open_fd: int | None = fd
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            open_fd = None
            handle.write(f"{python_root}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        if open_fd is not None:
            os.close(open_fd)
        temp.unlink(missing_ok=True)
        raise


def install_python_sources(
    venv_path: Path,
    sources: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    """Install immutable non-packaged Python sources into one model venv.

    Each declaration is validated before network work. Git operations share
    one wall-clock deadline and use the same exact process-group ownership as
    venv/pip commands. Only reviewed sparse paths and explicit submodules are
    materialized; a plain `.pth` then exposes the checkout to that venv.
    """
    if isinstance(sources, (str, bytes)):
        raise ValueError("python sources must be a sequence of mappings")

    venv_path = Path(venv_path)
    try:
        venv_mode = venv_path.lstat().st_mode
    except FileNotFoundError:
        if not sources:
            return ()
        raise RuntimeError(f"model venv is missing or unsafe: {venv_path}")
    if not stat.S_ISDIR(venv_mode) or stat.S_ISLNK(venv_mode):
        raise RuntimeError(f"model venv is missing or unsafe: {venv_path}")
    # Revoke every old hook before parsing/fetching the new declaration set.
    # This covers removals, renames, empty sets, and invalid replacements.
    _disable_all_source_pths(venv_path)

    specs = tuple(_parse_git_python_source(source) for source in sources)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("python source names must be unique per model")
    if not specs:
        return ()

    source_root = venv_path / "muse-sources"
    source_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_path_inside(source_root, venv_path, label="Python source root")
    if source_root.is_symlink():
        raise RuntimeError("Python source root must not be a symlink")

    deadline = time.monotonic() + _SOURCE_INSTALL_TIMEOUT
    installed: list[Path] = []
    activated = False
    try:
        try:
            for spec in specs:
                installed.append(_materialize_git_source(
                    source_root, spec, deadline=deadline,
                ))
            # Activate only after every checkout has passed validation.
            for spec, checkout in zip(specs, installed):
                _write_source_pth(venv_path, checkout, spec)
            activated = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if not _is_verbose():
                _emit_failure_output(exc)
            raise
    finally:
        if not activated:
            for spec in specs:
                try:
                    _disable_source_pth(venv_path, spec)
                except (OSError, RuntimeError) as exc:
                    logger.error(
                        "could not fail-closed Python source %r: %s",
                        spec.name, exc,
                    )
    return tuple(installed)


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
