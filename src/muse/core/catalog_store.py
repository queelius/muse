"""catalog.json storage: location, locking, schema validation, read/write.

The on-disk catalog lives at `<catalog_dir>/catalog.json` (see
`muse.core.catalog` for the entry shape). This module owns the file's
secure I/O, the process-wide catalog transaction lock, the mtime-keyed read
cache, and the schema checks applied on every read and write. It sits at
the bottom of the catalog modules and imports none of them.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any

from muse.core import config, securefs
from muse.core.resolvers import COMMIT_RE

logger = logging.getLogger(__name__)

_MAX_CATALOG_BYTES = 16 * 1024 * 1024
_MAX_CATALOG_ENTRIES = 10_000
_MAX_CATALOG_DEPTH = 24
_MAX_CATALOG_CONTAINER_ITEMS = 10_000
_MAX_CATALOG_STRING_CHARS = 1_000_000


class CatalogError(RuntimeError):
    """Raised when catalog.json is corrupt and no cached good copy exists.

    Invalid JSON, invalid UTF-8, or a malformed catalog shape must never be
    treated as an empty-but-valid catalog: every consumer (get_manifest,
    known_models,
    /v1/models, is_pulled) would then silently behave as "no models
    installed" -- 404-ing requests and emptying listings -- with no
    signal that the catalog file itself needs attention. When a prior
    good read is cached, `_read_catalog` serves that instead of raising;
    this error is reserved for the case where there is nothing good to
    fall back to.
    """


class ModelInUseError(CatalogError):
    """Raised when a live worker/probe owns a model's mutable resources."""


class ModelIdConflictError(CatalogError):
    """Raised when a pull would replace a catalog entry owned by another source."""


class StorageBusyError(CatalogError):
    """Raised when another process is mutating Muse's weights cache."""


def _ensure_owned_directory(path: Path, *, private: bool = False) -> None:
    """Create and validate an owned directory without following its leaf.

    The configured catalog directory may itself intentionally live below a
    symlinked parent.  What must never be accepted is a symlink in place of
    the directory Muse is about to chmod or populate (notably ``locks/``).
    See `muse.core.securefs.ensure_directory` for the checks applied.
    """
    securefs.ensure_directory(
        path, error=CatalogError, label="catalog path", private=private,
    )


def _open_catalog_regular_file(
    path: Path,
    flags: int,
    *,
    mode: int | None = None,
    require_single_link: bool = False,
    require_private: bool = False,
) -> int:
    """Open one owned regular file without following its final component."""
    return securefs.open_regular_file(
        path,
        flags,
        error=CatalogError,
        label="catalog file",
        mode=mode,
        require_single_link=require_single_link,
        require_private=require_private,
    )


def _open_catalog_lock(path: Path):
    """Return a private binary handle suitable for the advisory lock."""
    return securefs.open_lock_file(path, error=CatalogError, label="catalog lock")


# _CATALOG_WRITE_LOCK (below) serializes every catalog read-modify-write.
# The atomic write-then-rename in _write_catalog prevents corruption but not
# lost updates between concurrent read -> mutate -> write sequences (probe
# writes, admin enable/disable, load_director's observed-peak writeback).
# Hold it for the whole sequence.
#
# Lock ordering (acquire in this order to prevent deadlocks):
#   1. _KNOWN_MODELS_LOCK  (only when building the known_models cache)
#   2. _CATALOG_WRITE_LOCK (when doing a catalog RMW)
# state.lock and _CATALOG_WRITE_LOCK are never held together: supervisor
# and admin operations plan under state.lock and write the catalog after
# releasing it.


class _CatalogWriteLock:
    """Re-entrant thread and process lock for catalog transactions.

    The old plain ``threading.Lock`` protected threads in one interpreter,
    but Muse also mutates the catalog from CLI/admin/probe subprocesses. An
    advisory lock file in the catalog directory serializes those processes.
    Re-entrancy lets `_write_catalog` defend standalone writes while callers
    continue to hold this same lock around the complete read-modify-write.
    """

    def __init__(self) -> None:
        self._thread_lock = threading.RLock()
        self._local = threading.local()

    def acquire(self) -> bool:
        self._thread_lock.acquire()
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            return True

        handle = None
        try:
            catalog_dir = _catalog_dir()
            _ensure_owned_directory(catalog_dir, private=True)
            lock_path = catalog_dir / ".catalog.lock"
            handle = _open_catalog_lock(lock_path)
            securefs.lock_file(handle)
            self._local.handle = handle
            self._local.depth = 1

            # Another process may have replaced catalog.json while this one
            # waited. Force the first read in the transaction to hit disk,
            # even on filesystems whose mtimes are too coarse to expose it.
            global _read_catalog_cache
            _read_catalog_cache = None
            return True
        except BaseException:
            if handle is not None:
                handle.close()
            self._thread_lock.release()
            raise

    def release(self) -> None:
        depth = getattr(self._local, "depth", 0)
        if depth <= 0:
            raise RuntimeError("cannot release an un-acquired catalog lock")
        try:
            if depth > 1:
                self._local.depth = depth - 1
                return

            handle = self._local.handle
            try:
                securefs.unlock_file(handle)
            finally:
                handle.close()
                del self._local.handle
                self._local.depth = 0
        finally:
            self._thread_lock.release()

    def __enter__(self) -> "_CatalogWriteLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


_CATALOG_WRITE_LOCK = _CatalogWriteLock()


def _validated_persisted_manifest(
    catalog_model_id: str, manifest: object,
) -> dict:
    """Validate one resolver manifest without poisoning unrelated entries."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    required = ("model_id", "modality", "backend_path", "hf_repo")
    for field_name in required:
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest {field_name} must be a non-empty string")
    if manifest["model_id"] != catalog_model_id:
        raise ValueError(
            f"manifest model_id {manifest['model_id']!r} does not match "
            f"catalog key {catalog_model_id!r}"
        )
    backend_module, separator, backend_name = manifest["backend_path"].partition(":")
    if not separator or not backend_module or not backend_name:
        raise ValueError("manifest backend_path must use 'module:attribute' form")
    description = manifest.get("description", "")
    if not isinstance(description, str):
        raise ValueError("manifest description must be a string")
    for field_name in ("pip_extras", "system_packages"):
        values = manifest.get(field_name, ())
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(item, str) and item for item in values
        ):
            raise ValueError(f"manifest {field_name} must be a list of strings")
    sources = manifest.get("python_sources", ())
    if not isinstance(sources, (list, tuple)) or not all(
        isinstance(item, dict) for item in sources
    ):
        raise ValueError("manifest python_sources must be a list of objects")
    capabilities = manifest.get("capabilities", {})
    if capabilities is None:
        capabilities = {}
    if not isinstance(capabilities, dict):
        raise ValueError("manifest capabilities must be a JSON object")
    validated = dict(manifest)
    validated["capabilities"] = dict(capabilities)
    return validated


_MODEL_ID_CHARSET_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_model_id_for_fs(model_id: str) -> None:
    """Reject a model_id that is unsafe as a filesystem path component.

    model_id becomes `<catalog_dir>/venvs/<model_id>` (and, for
    resolver-pulled models, feeds into the weights cache path too).
    Applied at every pull entry point -- the bare-id branch of pull(),
    the finalized model_id in `_pull_via_resolver`, and the top of
    `_pull_bundled` -- covering every identifier shape (bare id,
    curated alias resolution result, resolver-synthesized id) BEFORE
    any venv/weights path is constructed. A hostile or malformed
    identifier (path traversal via "..", a path separator via "/")
    can therefore never escape the muse-owned catalog directories.
    """
    if model_id in (".", ".."):
        raise ValueError(
            f"invalid model id {model_id!r}: must not be '.' or '..'"
        )
    if not _MODEL_ID_CHARSET_RE.match(model_id):
        raise ValueError(
            f"invalid model id {model_id!r}: must match "
            f"{_MODEL_ID_CHARSET_RE.pattern!r} "
            f"(letters, digits, '.', '_', '-' only)"
        )


_CatalogFileKey = tuple[str, int, int, int, int, int]


def _catalog_file_key(path: Path, info: os.stat_result) -> _CatalogFileKey:
    """Return a replacement-sensitive identity for one opened catalog."""
    return (
        str(path),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _catalog_dir() -> Path:
    """Resolve the catalog directory identically to `config.py`'s own
    bootstrap resolution (env+default only), so catalog.json always
    co-locates with config.yaml. Delegates to `config._catalog_dir()`
    rather than `config.get("paths.catalog_dir")`: catalog_dir is
    bootstrap state, and a config.yaml value must never be able to
    redirect catalog.json away from where config.yaml itself lives.
    """
    return config._catalog_dir()


def _catalog_path() -> Path:
    return _catalog_dir() / "catalog.json"


# mtime-based cache for _read_catalog. The catalog is consulted on the
# gateway hot path (every request: get_manifest -> _read_catalog) and by
# admin / CLI flows. Cache stores (path_str, mtime_ns, parsed_dict);
# invalidates whenever the file's mtime advances (writes go through
# `_write_catalog`'s atomic rename, which updates mtime). A path-keyed
# lookup means tests using `tmp_path` + the MUSE_CATALOG_DIR env var
# don't accidentally hit cache from a prior run; first read after path
# change is always a fresh disk read.
_read_catalog_cache: tuple[_CatalogFileKey, dict] | None = None


class _CatalogSchemaError(ValueError):
    """Internal marker for syntactically valid but unusable catalog JSON."""


def _validate_catalog_json_value(value: Any, *, path: str, depth: int = 0) -> None:
    """Bound arbitrary extension data before caching or copying it."""
    if depth > _MAX_CATALOG_DEPTH:
        raise _CatalogSchemaError(
            f"{path} exceeds maximum nesting depth {_MAX_CATALOG_DEPTH}"
        )
    if isinstance(value, dict):
        if len(value) > _MAX_CATALOG_CONTAINER_ITEMS:
            raise _CatalogSchemaError(f"{path} contains too many fields")
        for key, child in value.items():
            if not isinstance(key, str):
                raise _CatalogSchemaError(f"{path} contains a non-string key")
            _validate_catalog_json_value(
                child,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_CATALOG_CONTAINER_ITEMS:
            raise _CatalogSchemaError(f"{path} contains too many items")
        for index, child in enumerate(value):
            _validate_catalog_json_value(
                child,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
        return
    if isinstance(value, str):
        if len(value) > _MAX_CATALOG_STRING_CHARS:
            raise _CatalogSchemaError(f"{path} contains an oversized string")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _CatalogSchemaError(f"{path} must contain finite numbers")
        return
    raise _CatalogSchemaError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _validate_artifact_receipt(value: Any, *, path: str) -> None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or not value
    ):
        raise _CatalogSchemaError(f"{path} must be a non-empty list of objects")
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise _CatalogSchemaError(f"{item_path} must be an object")
        for field_name in ("repo_id", "revision", "subdir"):
            field = item.get(field_name)
            if not isinstance(field, str):
                raise _CatalogSchemaError(
                    f"{item_path}.{field_name} must be a string"
                )
        revision = item["revision"]
        if COMMIT_RE.fullmatch(revision) is None:
            raise _CatalogSchemaError(
                f"{item_path}.revision must be a 40-character commit"
            )
        for field_name in ("allow_patterns", "required_patterns"):
            patterns = item.get(field_name)
            if patterns is None:
                continue
            if (
                isinstance(patterns, (str, bytes))
                or not isinstance(patterns, (list, tuple))
                or not patterns
                or not all(isinstance(pattern, str) and pattern for pattern in patterns)
            ):
                raise _CatalogSchemaError(
                    f"{item_path}.{field_name} must be a non-empty list of strings"
                )


def _validate_measurements(value: Any, *, path: str) -> None:
    if not isinstance(value, dict):
        raise _CatalogSchemaError(f"{path} must be an object")
    for device, bucket in value.items():
        bucket_path = f"{path}.{device}"
        if not isinstance(device, str) or not device:
            raise _CatalogSchemaError(f"{path} device keys must be strings")
        if not isinstance(bucket, dict):
            raise _CatalogSchemaError(f"{bucket_path} must be an object")
        for field_name, field in bucket.items():
            if field_name.endswith("_bytes") and (
                type(field) is not int or field < 0
            ):
                raise _CatalogSchemaError(
                    f"{bucket_path}.{field_name} must be a non-negative integer"
                )
        for field_name in ("device", "source", "observed_at", "shape"):
            if field_name in bucket and not isinstance(bucket[field_name], str):
                raise _CatalogSchemaError(
                    f"{bucket_path}.{field_name} must be a string"
                )


def _validate_catalog_entry(
    model_id: str,
    entry: dict,
    *,
    strict_manifest: bool,
) -> None:
    path = f"entry {model_id!r}"
    if "enabled" in entry and type(entry["enabled"]) is not bool:
        raise _CatalogSchemaError(f"{path}.enabled must be a boolean")
    if "device_override" in entry and entry["device_override"] not in {
        "auto", "cpu", "cuda", "mps",
    }:
        raise _CatalogSchemaError(
            f"{path}.device_override must be auto, cpu, cuda, or mps"
        )
    if "gpu_layers_override" in entry and (
        type(entry["gpu_layers_override"]) is not int
        or entry["gpu_layers_override"] < -1
    ):
        raise _CatalogSchemaError(
            f"{path}.gpu_layers_override must be an integer >= -1"
        )
    for field_name in (
        "pulled_at",
        "hf_repo",
        "local_dir",
        "venv_path",
        "python_path",
        "source",
        "revision",
        "code_revision",
        "base_override",
    ):
        if field_name in entry and (
            not isinstance(entry[field_name], str) or not entry[field_name]
        ):
            raise _CatalogSchemaError(
                f"{path}.{field_name} must be a non-empty string"
            )
    if "measurements" in entry:
        _validate_measurements(entry["measurements"], path=f"{path}.measurements")
    if "artifact_provenance" in entry:
        _validate_artifact_receipt(
            entry["artifact_provenance"],
            path=f"{path}.artifact_provenance",
        )
    if strict_manifest and "manifest" in entry:
        try:
            manifest = _validated_persisted_manifest(model_id, entry["manifest"])
        except ValueError as exc:
            raise _CatalogSchemaError(f"{path}.manifest is invalid: {exc}") from exc
        if "artifact_provenance" in manifest:
            _validate_artifact_receipt(
                manifest["artifact_provenance"],
                path=f"{path}.manifest.artifact_provenance",
            )


def _validate_catalog_shape(
    data: Any,
    *,
    strict_manifests: bool = False,
) -> dict:
    """Validate stable semantics while retaining unknown extension fields.

    Internal writes validate persisted manifests eagerly. Reads leave those
    model-specific semantics to `_validated_persisted_manifest`, whose callers
    quarantine only the bad resolver entry instead of making every otherwise
    valid model unavailable because one legacy/external entry is malformed.
    """
    if not isinstance(data, dict):
        raise _CatalogSchemaError("top level must be a JSON object")
    if len(data) > _MAX_CATALOG_ENTRIES:
        raise _CatalogSchemaError(
            f"catalog cannot contain more than {_MAX_CATALOG_ENTRIES} entries"
        )
    for model_id, entry in data.items():
        if not isinstance(model_id, str) or not model_id:
            raise _CatalogSchemaError(
                "model identifiers must be non-empty strings"
            )
        try:
            _validate_model_id_for_fs(model_id)
        except ValueError as exc:
            raise _CatalogSchemaError(str(exc)) from exc
        if not isinstance(entry, dict):
            raise _CatalogSchemaError(
                f"entry for model {model_id!r} must be a JSON object"
            )
        _validate_catalog_json_value(entry, path=f"entry {model_id!r}")
        _validate_catalog_entry(
            model_id,
            entry,
            strict_manifest=strict_manifests,
        )
    return data


def _catalog_read_failure(
    path: Path,
    cached: tuple[_CatalogFileKey, dict] | None,
    reason: str,
) -> dict:
    """Serve last-known-good data or raise the public corruption error."""
    if cached is not None and cached[0][0] == str(path):
        logger.warning(
            "catalog at %s corrupt (%s); serving last-known-good cached parse",
            path,
            reason,
        )
        return _deep_copy_catalog(cached[1])
    logger.error(
        "catalog at %s corrupt (%s); no cached copy to fall back to",
        path,
        reason,
    )
    raise CatalogError(
        f"catalog at {path} is corrupt ({reason}) and no last-known-good "
        "cached copy is available; inspect or restore the file before continuing"
    )


def _read_catalog() -> dict:
    """Return the parsed catalog.json contents, with mtime-based caching.

    Hot path. Returns a fresh dict each call (deep copy of the cached
    parse) so callers can mutate without polluting the cache.

    A corrupt (invalid-JSON) catalog file does NOT silently degrade to
    an empty dict, which would look like a valid "no models" catalog to
    every caller. Instead: if a last-known-good parse is cached for this
    same path, serve that (stale-but-real data, logged as a warning);
    if there is no cache to fall back to, raise CatalogError so the
    caller surfaces the problem instead of masking it.
    """
    global _read_catalog_cache
    p = _catalog_path()
    try:
        path_info = p.lstat()
    except FileNotFoundError:
        # Drop any cached state from a prior pulled+removed cycle so a
        # subsequent write produces a cache-miss instead of returning a
        # stale dict.
        _read_catalog_cache = None
        return {}
    except OSError as exc:
        return _catalog_read_failure(p, _read_catalog_cache, f"cannot inspect file: {exc}")
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        return _catalog_read_failure(
            p, _read_catalog_cache, "path is not a regular non-symlink file",
        )
    if path_info.st_size > _MAX_CATALOG_BYTES:
        return _catalog_read_failure(
            p,
            _read_catalog_cache,
            f"file exceeds {_MAX_CATALOG_BYTES} bytes",
        )
    path_key = _catalog_file_key(p, path_info)

    cached = _read_catalog_cache
    if (
        cached is not None
        and cached[0] == path_key
    ):
        # Deep copy: callers (`pull`, `remove`, `set_enabled`) mutate the
        # returned dict in-place, then write back. Sharing the cached
        # reference would let those mutations bleed into later cache hits.
        return _deep_copy_catalog(cached[1])

    try:
        descriptor = _open_catalog_regular_file(p, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            opened_info = os.fstat(handle.fileno())
            if opened_info.st_size > _MAX_CATALOG_BYTES:
                return _catalog_read_failure(
                    p,
                    cached,
                    f"file exceeds {_MAX_CATALOG_BYTES} bytes",
                )
            text = handle.read(_MAX_CATALOG_BYTES + 1)
            if len(text.encode("utf-8")) > _MAX_CATALOG_BYTES:
                return _catalog_read_failure(
                    p,
                    cached,
                    f"file exceeds {_MAX_CATALOG_BYTES} bytes",
                )
            opened_key = _catalog_file_key(p, opened_info)
        data = _validate_catalog_shape(json.loads(text))
    except CatalogError as exc:
        return _catalog_read_failure(p, cached, str(exc))
    except json.JSONDecodeError:
        return _catalog_read_failure(p, cached, "invalid JSON")
    except RecursionError:
        return _catalog_read_failure(p, cached, "JSON nesting is too deep")
    except _CatalogSchemaError as exc:
        return _catalog_read_failure(p, cached, f"invalid schema: {exc}")
    except UnicodeError:
        return _catalog_read_failure(p, cached, "invalid UTF-8")
    except OSError as exc:
        return _catalog_read_failure(p, cached, f"cannot read file: {exc}")
    # Backfill enabled=True for pre-enable-flag entries (migration path).
    # Non-destructive: only affects the in-memory dict on read.
    for entry in data.values():
        entry.setdefault("enabled", True)
    _read_catalog_cache = (opened_key, _deep_copy_catalog(data))
    return data


def _deep_copy_catalog(data: dict) -> dict:
    """Shallow-then-shallow copy of the catalog dict-of-dicts.

    Catalog entries are plain JSON shapes (str/number/bool/None plus dicts
    and lists). The persisted `manifest` field is the deepest structure,
    and callers mutate it (e.g. `_pull_via_resolver` does
    `manifest = dict(resolved.manifest)` before storing). So a top-level
    deep copy via `json.loads(json.dumps(data))` is the most defensive
    cheap option; profile if this becomes a hot spot.
    """
    return json.loads(json.dumps(data))


def _reset_read_catalog_cache() -> None:
    """Test hook: clear the catalog read cache."""
    global _read_catalog_cache
    _read_catalog_cache = None


def _write_catalog(data: dict) -> None:
    """Durably write catalog JSON through a unique same-directory temp file.

    Rename within the same filesystem is atomic on POSIX and near-atomic
    on Windows (Python 3.3+ Path.replace wraps MoveFileEx with REPLACE_EXISTING).

    Invalidates the read cache so the next `_read_catalog()` sees this
    write. The mtime check would catch this on its own under normal
    filesystems, but explicit invalidation removes a class of race that
    surfaces on coarse-mtime filesystems (e.g. some FAT, network mounts):
    consecutive writes within the same mtime tick would otherwise serve
    stale data on the read between them.
    """
    global _read_catalog_cache
    try:
        _validate_catalog_shape(data, strict_manifests=True)
    except (_CatalogSchemaError, RecursionError) as exc:
        raise CatalogError(f"invalid catalog schema: {exc}") from exc
    try:
        serialized = json.dumps(data, indent=2, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CatalogError(f"catalog contains non-JSON data: {exc}") from exc
    if len(serialized.encode("utf-8")) > _MAX_CATALOG_BYTES:
        raise CatalogError(
            f"catalog cannot exceed {_MAX_CATALOG_BYTES} bytes"
        )
    with _CATALOG_WRITE_LOCK:
        p = _catalog_path()
        _ensure_owned_directory(p.parent)
        securefs.atomic_write_text(p, serialized)
        _read_catalog_cache = None


def is_pulled(model_id: str) -> bool:
    return model_id in _read_catalog()
