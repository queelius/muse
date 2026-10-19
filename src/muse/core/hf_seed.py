"""Seed Muse's weights cache from the shared Hugging Face cache.

Pulls download into Muse's own cache (`<catalog_dir>/weights`), so a repo
already in the shared HF cache (`~/.cache/huggingface/hub`) would be
downloaded again. Before a download, `seed_repo` hardlinks the shared
cache's blobs for that repo into Muse's cache. huggingface_hub then finds
each blob it needs under its etag name and only creates the snapshot
symlink; anything missing or different still downloads.

huggingface_hub trusts a blob by name, so every blob is verified first: an
LFS blob's name is the SHA-256 of its content, a regular blob's name is its
git blob SHA-1. A blob that fails verification, is incomplete, or sits on
another filesystem is skipped (and downloaded instead). Blobs are written
once and never edited in place, so sharing an inode with the shared cache
is safe; deleting the shared copy later leaves Muse's link intact.

`drop_unused_seeds` removes seeded blobs the finished download did not
reference, so seeding never leaves extra weights behind.
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
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


def _verified(blob: Path) -> bool:
    name = blob.name
    h = hashlib.sha256() if _SHA256.fullmatch(name) else hashlib.sha1()
    if not _SHA256.fullmatch(name):
        if not _SHA1.fullmatch(name):
            return False
        h.update(b"blob %d\0" % blob.stat().st_size)
    with open(blob, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == name


def _wanted_blobs(
    repo: Path, allow_patterns: Iterable[str] | None,
) -> set[str]:
    """Blob names referenced by any shared snapshot file matching the patterns."""
    patterns = list(allow_patterns) if allow_patterns else None
    names: set[str] = set()
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return names
    for snap in snapshots.iterdir():
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
    repo_type: str = "model", shared: Path | None = None,
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
        for name in sorted(_wanted_blobs(src_repo, allow_patterns)):
            src, dst = src_repo / "blobs" / name, dst_blobs / name
            if dst.exists() or not src.is_file() or src.is_symlink():
                continue
            try:
                if not _verified(src):
                    logger.warning("not seeding %s/%s: content does not match its name", repo_id, name)
                    continue
                dst_blobs.mkdir(parents=True, exist_ok=True)
                tmp = dst_blobs / f".{name}.muse-seed-{os.getpid()}"
                os.link(src, tmp)
                os.replace(tmp, dst)
            except OSError as exc:
                logger.debug("not seeding %s/%s: %s", repo_id, name, exc)
                continue
            seeded.append(dst)
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
