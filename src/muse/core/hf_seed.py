"""Seed Muse's weights cache from the shared Hugging Face cache.

Pulls download into Muse's own cache (`<catalog_dir>/weights`), so a repo
already in the shared HF cache (`~/.cache/huggingface/hub`) would be
downloaded again. Before a download, `seed_repo` hardlinks the shared
cache's blobs for that repo into Muse's cache. huggingface_hub then finds
each blob it needs under its etag name and only creates the snapshot
symlink; anything missing or different still downloads.

huggingface_hub trusts a blob by name, so every linked blob is verified:
an LFS blob's name is the SHA-256 of its content, a regular blob's name is
its git blob SHA-1. The link is made first and the linked inode verified,
so a swap of the shared path cannot slip in unverified content. Muse and
the shared cache then share the inode, so only blobs this user owns are
linked, and their group/world write bits are cleared first.
A blob that fails any check, or sits on another filesystem, is skipped
(and downloaded instead). Deleting the shared copy later leaves Muse's link
intact.

`drop_unused_seeds` removes seeded blobs the finished download did not
reference, so seeding never leaves extra weights behind.
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")


def shared_hub_cache() -> Path | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:  # noqa: BLE001 - seeding is an optimization
        return None
    return Path(HF_HUB_CACHE).expanduser()


def repo_folder(repo_id: str, repo_type: str = "model") -> str:
    return f"{repo_type}s--{repo_id.replace('/', '--')}"


def _make_private(blob: Path) -> bool:
    """Own the shared inode outright: it must belong to this user, and any
    group or world write bit is cleared before the content is verified."""
    if not hasattr(os, "getuid"):
        return False
    st = os.lstat(blob)
    if st.st_uid != os.getuid():
        return False
    if st.st_mode & 0o022:
        os.chmod(blob, stat.S_IMODE(st.st_mode) & ~0o022)
    return True


def _verified(blob: Path, name: str) -> bool:
    """`blob`'s content hashes to `name` (its blob name in the cache)."""
    if _SHA256.fullmatch(name):
        h = hashlib.sha256()
    elif _SHA1.fullmatch(name):
        h = hashlib.sha1(b"blob %d\0" % blob.stat().st_size)
    else:
        return False
    with open(blob, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == name


def _snapshots(repo: Path, revision: str | None) -> list[Path]:
    """The requested revision's snapshot if the shared cache has it, else all
    of them (unchanged files keep their blob across revisions)."""
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return []
    if revision:
        try:
            commit = (repo / "refs" / revision).read_text().strip()
        except OSError:
            commit = revision
        if commit and "/" not in commit and (snapshots / commit).is_dir():
            return [snapshots / commit]
    return list(snapshots.iterdir())


def _wanted_blobs(
    repo: Path, allow_patterns: Iterable[str] | None, revision: str | None = None,
) -> set[str]:
    """Blob names referenced by shared snapshot files matching the patterns."""
    # huggingface_hub reads a trailing "/" as everything under that folder.
    patterns = (
        [p + "*" if p.endswith("/") else p for p in allow_patterns]
        if allow_patterns else None
    )
    names: set[str] = set()
    for snap in _snapshots(repo, revision):
        for dirpath, _, filenames in os.walk(snap):
            for f in filenames:
                path = Path(dirpath) / f
                rel = path.relative_to(snap).as_posix()
                if patterns is not None and not any(fnmatch.fnmatch(rel, p) for p in patterns):
                    continue
                try:
                    target = path.resolve(strict=True)
                except OSError:
                    continue
                if target.parent == (repo / "blobs").resolve():
                    names.add(target.name)
    return names


def seed_repo(
    cache_dir: Path, repo_id: str, *, allow_patterns: Iterable[str] | None = None,
    revision: str | None = None, repo_type: str = "model", shared: Path | None = None,
) -> list[Path]:
    """Hardlink verified shared blobs for repo_id into cache_dir; return the
    blob paths created (pass them to `drop_unused_seeds`). Never raises."""
    try:
        shared = shared if shared is not None else shared_hub_cache()
        if shared is None or not shared.is_dir():
            return []
        cache_dir = Path(cache_dir)
        if shared.resolve() == cache_dir.resolve():
            return []
        src_repo = shared / repo_folder(repo_id, repo_type)
        if not (src_repo / "blobs").is_dir():
            return []
        dst_blobs = cache_dir / repo_folder(repo_id, repo_type) / "blobs"
        seeded: list[Path] = []
        for name in sorted(_wanted_blobs(src_repo, allow_patterns, revision)):
            src, dst = src_repo / "blobs" / name, dst_blobs / name
            if dst.exists() or not src.is_file() or src.is_symlink():
                continue
            tmp = dst_blobs / f".{name}.muse-seed-{os.getpid()}"
            try:
                dst_blobs.mkdir(parents=True, exist_ok=True)
                os.link(src, tmp)
                if not _make_private(tmp):
                    logger.debug("not seeding %s/%s: owned by another user", repo_id, name)
                elif not _verified(tmp, name):
                    logger.warning("not seeding %s/%s: content does not match its name", repo_id, name)
                else:
                    os.replace(tmp, dst)
                    seeded.append(dst)
                    continue
            except OSError as exc:
                logger.debug("not seeding %s/%s: %s", repo_id, name, exc)
            tmp.unlink(missing_ok=True)
        if seeded:
            logger.info("reusing %d file(s) of %s from the shared Hugging Face cache", len(seeded), repo_id)
        return seeded
    except Exception:  # noqa: BLE001 - an optimization must not fail a pull
        logger.warning("seeding %s from the shared Hugging Face cache failed", repo_id, exc_info=True)
        return []


def drop_unused_seeds(seeded: list[Path]) -> None:
    """Delete seeded blobs no snapshot in their repo now points to."""
    if not seeded:
        return
    try:
        repo = seeded[0].parent.parent
        used = set()
        snapshots = repo / "snapshots"
        if snapshots.is_dir():
            for dirpath, _, filenames in os.walk(snapshots):
                for f in filenames:
                    try:
                        used.add(Path(dirpath, f).resolve(strict=True))
                    except OSError:
                        continue
        for blob in seeded:
            if blob.resolve() not in used:
                blob.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        logger.warning("could not remove unused seeded blobs", exc_info=True)
