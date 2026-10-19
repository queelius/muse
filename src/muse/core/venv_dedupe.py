"""Hardlink identical package files across model venvs.

Every model venv carries its own copy of the same large payloads: Torch,
Triton, CUDA runtime libraries, llama.cpp builds. Content-identical files
(SHA-256, size, mode and owner equal) become hardlinks to one inode, which
saves the disk space and lets workers that load the same shared library map
the same page-cache pages. VRAM is unaffected: CUDA loads kernels per
process.

Safe because nothing edits installed package files in place. pip unlinks a
file before writing its replacement (pip/_internal/operations/install/
wheel.py), and uninstall moves files aside, so an upgrade in one venv
breaks that venv's link and never changes the shared copy. Only regular
files of at least MIN_BYTES under `lib/python*/site-packages` are linked;
`bin/` scripts (whose shebangs pip rewrites in place) are never touched.
Transaction staging directories (dot-prefixed) are skipped.

Each replacement re-checks both files, links to a temporary name and
renames it over the target, so a reader sees either the old or the new
file, and a running worker keeps the inode it already has open.
"""
from __future__ import annotations

import hashlib
import logging
import os
import stat
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_BYTES = 1 << 20
# Files changed this recently may still be mid-install.
SETTLE_SECONDS = 60.0


@dataclass(frozen=True)
class DedupeResult:
    scanned_files: int
    linked_files: int
    reclaimed_bytes: int
    skipped_files: int
    dry_run: bool


@dataclass(frozen=True)
class _File:
    path: Path
    dev: int
    ino: int
    size: int
    mode: int
    uid: int
    mtime_ns: int
    nlink: int

    @property
    def inode(self) -> tuple[int, int]:
        return (self.dev, self.ino)


def _stat(path: Path) -> _File | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return _File(
        path, st.st_dev, st.st_ino, st.st_size, stat.S_IMODE(st.st_mode),
        st.st_uid, st.st_mtime_ns, st.st_nlink,
    )


def model_venvs(root: Path) -> list[Path]:
    """Canonical venvs under root: not staging (dot) dirs, not symlinks."""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    return [
        p for p in entries
        if not p.name.startswith(".") and not p.is_symlink() and (p / "pyvenv.cfg").is_file()
    ]


def _package_files(venv: Path) -> list[_File]:
    out = []
    for site in sorted(venv.glob("lib/python*/site-packages")):
        if site.is_symlink():
            continue
        for dirpath, dirnames, filenames in os.walk(site):  # never follows links
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                f = _stat(Path(dirpath) / name)
                if f is not None and f.size >= MIN_BYTES:
                    out.append(f)
    return out


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _unchanged(f: _File) -> bool:
    now = _stat(f.path)
    return (
        now is not None and now.inode == f.inode and now.size == f.size
        and now.mtime_ns == f.mtime_ns
    )


def _replace_with_link(keeper: _File, target: _File) -> bool:
    if not (_unchanged(keeper) and _unchanged(target)):
        return False
    tmp = target.path.with_name(f".{target.path.name}.muse-dedupe-{os.getpid()}")
    try:
        os.link(keeper.path, tmp)
    except OSError as exc:
        logger.debug("cannot link %s: %s", target.path, exc)
        return False
    try:
        os.replace(tmp, target.path)
    except OSError as exc:
        logger.debug("cannot replace %s: %s", target.path, exc)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    return True


def dedupe_venvs(
    root: Path, *, only: Path | None = None, dry_run: bool = False,
    now: float | None = None,
) -> DedupeResult:
    """Link identical package files across the venvs under root.

    With `only`, replace files in that venv alone (linking them to
    identical files in the others), which is cheap enough to run after
    every pull or refresh. The caller vouches that `only` is fully
    installed, so its files skip the settle check; other venvs' files
    changed in the last SETTLE_SECONDS are left alone.
    """
    now = time.time() if now is None else now
    settled_before = int((now - SETTLE_SECONDS) * 1e9)
    venvs = model_venvs(root)
    files = [f for v in venvs for f in _package_files(v)]
    if only is not None:
        only = Path(only)
        sizes = {f.size for f in files if f.path.is_relative_to(only)}
        files = [f for f in files if f.size in sizes]

    by_size: dict[int, list[_File]] = defaultdict(list)
    for f in files:
        by_size[f.size].append(f)

    linked = reclaimed = skipped = 0
    for size, group in by_size.items():
        inodes: dict[tuple[int, int], list[_File]] = defaultdict(list)
        for f in group:
            inodes[f.inode].append(f)
        if len(inodes) < 2:
            continue
        by_content: dict[tuple, list[list[_File]]] = defaultdict(list)
        for paths in inodes.values():
            rep = paths[0]
            fresh_ok = only is not None and rep.path.is_relative_to(only)
            if rep.mtime_ns > settled_before and not fresh_ok:
                skipped += len(paths)
                continue
            try:
                digest = _digest(rep.path)
            except OSError:
                skipped += len(paths)
                continue
            by_content[(digest, rep.dev, rep.mode, rep.uid)].append(paths)
        for same in by_content.values():
            if len(same) < 2:
                continue
            # Keep the most-shared inode; prefer one outside `only`.
            same.sort(key=lambda ps: (
                only is not None and all(p.path.is_relative_to(only) for p in ps),
                -ps[0].nlink,
            ))
            keeper = same[0][0]
            for paths in same[1:]:
                targets = [
                    p for p in paths if only is None or p.path.is_relative_to(only)
                ]
                done = 0
                for target in targets:
                    if dry_run or _replace_with_link(keeper, target):
                        done += 1
                    else:
                        skipped += 1
                linked += done
                # Space comes back only once no other name holds the inode.
                if done and done == paths[0].nlink:
                    reclaimed += size
    return DedupeResult(len(files), linked, reclaimed, skipped, dry_run)


def dedupe_after_install(venv: Path) -> None:
    """Best-effort `only=venv` pass after a pull or refresh; never raises."""
    from muse.core import config

    if not config.get("storage.dedupe_venvs"):
        return
    try:
        result = dedupe_venvs(Path(venv).parent, only=Path(venv))
    except Exception:  # noqa: BLE001 - an optimization must not fail the install
        logger.warning("venv dedupe after install of %s failed", venv, exc_info=True)
        return
    if result.linked_files:
        logger.info(
            "linked %d file(s) in %s to identical copies in other venvs (%.1f GiB reclaimed)",
            result.linked_files, venv, result.reclaimed_bytes / (1 << 30),
        )
