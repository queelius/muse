"""Reviewed, non-packaged Python sources installed into a model's venv.

A model manifest may declare `python_sources`: GitHub repositories pinned
to full commit SHAs, checked out sparsely (plus explicit pinned submodules)
under `<venv>/muse-sources/` and exposed to that venv through a
`muse-source-<name>.pth` file. Declarations are validated before any network
work, checkouts are re-validated before activation, and every previous hook
is revoked first so a removed declaration cannot stay active.
"""
from __future__ import annotations

import configparser
import errno
import logging
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from muse.core.owned_command import _run_owned
from muse.core.resolvers import COMMIT_RE
from muse.core.venv import (
    _emit_failure_output,
    _is_verbose,
    _venv_site_packages,
    remove_owned_entry,
)

logger = logging.getLogger(__name__)


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
