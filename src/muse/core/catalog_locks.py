"""Per-model and storage-wide advisory locks for catalog mutations.

Pulls, removals, storage pruning and live workers coordinate through these
thread + file locks under `<catalog_dir>/locks/`: one pull lock per model
identity, one lifetime resource lease per model, and one lock for the
Muse-owned weights cache. The catalog.json transaction lock itself lives
in `muse.core.catalog_store`.
"""
from __future__ import annotations

import contextlib
import hashlib
import threading
from pathlib import Path

from muse.core import securefs
from muse.core.catalog_store import (
    ModelInUseError,
    StorageBusyError,
    _catalog_dir,
    _ensure_owned_directory,
    _open_catalog_lock,
)


# The catalog transaction lock protects JSON consistency, but two processes
# installing into the same venv can corrupt site-packages before either one
# writes catalog.json. Pair a per-identity thread lock with an advisory file
# lock that spans the complete pull helper call. Resolver URIs are digested so
# untrusted identifier text never becomes part of a filesystem path.
_PULL_LOCKS_GUARD = threading.Lock()
_PULL_THREAD_LOCKS: dict[str, threading.RLock] = {}
_RESOURCE_LOCKS_GUARD = threading.Lock()
_RESOURCE_THREAD_LOCKS: dict[str, threading.Lock] = {}
_STORAGE_THREAD_LOCK = threading.RLock()


@contextlib.contextmanager
def _advisory_lock(path: Path, thread_lock, *, wait: bool, busy_exc: Exception):
    """Hold ``thread_lock`` then an advisory file lock at ``path``.

    Acquires the in-process lock first and the file lock second, releasing
    in reverse. With ``wait=False`` either being busy raises ``busy_exc``.
    """
    if not thread_lock.acquire(blocking=wait):
        raise busy_exc
    try:
        _ensure_owned_directory(path.parent, private=True)
        handle = _open_catalog_lock(path)
        try:
            if not securefs.lock_file(handle, blocking=wait):
                raise busy_exc
            try:
                yield
            finally:
                securefs.unlock_file(handle)
        finally:
            handle.close()
    finally:
        thread_lock.release()


def _model_pull_lock(identity: str, *, wait: bool = True):
    """Thread/process lock covering one model's complete pull transaction."""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    with _PULL_LOCKS_GUARD:
        thread_lock = _PULL_THREAD_LOCKS.setdefault(digest, threading.RLock())
    return _advisory_lock(
        _catalog_dir() / "locks" / f"pull-{digest}.lock",
        thread_lock,
        wait=wait,
        busy_exc=ModelInUseError(
            f"model {identity!r} has an active pull or update; retry later"
        ),
    )


def _storage_cache_lock(*, wait: bool = True):
    """Serialize mutation of the Muse-owned weights cache across processes."""
    return _advisory_lock(
        _catalog_dir() / "locks" / "storage-cache.lock",
        _STORAGE_THREAD_LOCK,
        wait=wait,
        busy_exc=StorageBusyError(
            "Muse's weights cache is busy with another pull or cleanup"
        ),
    )


def _model_resource_lease(model_id: str, *, wait: bool = False):
    """Exclusive lifetime lease for one model's venv and weight resources.

    Workers acquire a blocking lease for their whole lifetime. Commands that
    mutate or probe the environment use ``wait=False`` and fail clearly when
    a live worker still owns it. The advisory file lock makes this contract
    work across the supervisor, worker venv, and direct CLI processes.
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model resource lease requires a non-empty model id")
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
    with _RESOURCE_LOCKS_GUARD:
        thread_lock = _RESOURCE_THREAD_LOCKS.setdefault(digest, threading.Lock())
    return _advisory_lock(
        _catalog_dir() / "locks" / f"resource-{digest}.lock",
        thread_lock,
        wait=wait,
        busy_exc=ModelInUseError(
            f"model {model_id!r} is in use; stop or unload it and retry"
        ),
    )
