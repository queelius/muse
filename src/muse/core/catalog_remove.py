"""`muse models remove`: unregister a model and optionally purge its files.

Removal runs under the model's pull lock and resource lease. A purge
deletes only directories Muse owns (the model's venv and its weights
ownership unit) and preserves any path another catalog entry still
references.
"""
from __future__ import annotations

import logging
from pathlib import Path

from muse.core.catalog_locks import (
    _model_pull_lock,
    _model_resource_lease,
    _storage_cache_lock,
)
from muse.core.catalog_models import _reset_known_models_cache
from muse.core.catalog_store import (
    _CATALOG_WRITE_LOCK,
    CatalogError,
    _catalog_dir,
    _read_catalog,
    _write_catalog,
)
from muse.core.resolvers import COMMIT_RE
from muse.core.venv import remove_owned_entry

logger = logging.getLogger(__name__)


def _strict_owned_purge_path(
    raw_path: str,
    root: Path,
    *,
    field_name: str,
    model_id: str,
    root_label: str = "venv",
) -> Path:
    """Resolve a target and require this model's exact owned directory."""
    try:
        resolved_root = root.expanduser().resolve()
        resolved_path = Path(raw_path).expanduser().resolve()
        relative = resolved_path.relative_to(resolved_root)
    except (TypeError, ValueError, OSError) as exc:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog {field_name} "
            f"does not resolve inside the owned {root_label} root"
        ) from exc
    if relative.parts != (model_id,):
        expected_path = resolved_root / model_id
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog {field_name} "
            f"does not point at expected model directory {expected_path}"
        )
    return resolved_path


def _owned_weights_purge_path(
    raw_path: str,
    weights_root: Path,
    *,
    model_id: str,
) -> Path | None:
    """Return an owned weights target; preserve legitimate external caches."""
    try:
        resolved_root = weights_root.expanduser().resolve()
        resolved_path = Path(raw_path).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: invalid catalog local_dir"
        ) from exc
    if resolved_path == resolved_root:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog local_dir points "
            f"at owned root {resolved_root}, not a model directory"
        )
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        # Legacy bundled pulls may live in Hugging Face's shared cache. Muse
        # does not own an external location, so unregister while preserving it.
        return None

    # Hugging Face snapshots contain symlinks into a sibling ``blobs``
    # directory. Removing only ``snapshots/<commit>`` reclaims almost no
    # space and strands the actual weights. Treat one cache repository as
    # the deletion unit; reference checks below preserve it while any other
    # catalog entry points at any revision in that repository.
    if relative.parts and relative.parts[0].startswith("models--"):
        if (
            len(relative.parts) != 3
            or relative.parts[1] != "snapshots"
            or COMMIT_RE.fullmatch(relative.parts[2]) is None
        ):
            raise CatalogError(
                f"refusing to purge model {model_id!r}: catalog local_dir "
                "does not have a canonical Hugging Face snapshot layout"
            )
        return resolved_root / relative.parts[0]
    return resolved_path


def _other_catalog_path_reference(
    catalog: dict,
    *,
    model_id: str,
    field_name: str,
    target: Path,
) -> str | None:
    """Return another entry whose resource path overlaps ``target``.

    Recursive deletion of a parent damages a child reference, while deleting
    a child can damage a model that treats the parent as its snapshot root.
    Preserve either relationship, not only byte-for-byte path equality.
    """
    for other_model_id, other_entry in catalog.items():
        if other_model_id == model_id or not isinstance(other_entry, dict):
            continue
        raw_path = other_entry.get(field_name)
        if not raw_path:
            continue
        try:
            other_target = Path(raw_path).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            continue
        try:
            target.relative_to(other_target)
            overlaps = True
        except ValueError:
            try:
                other_target.relative_to(target)
                overlaps = True
            except ValueError:
                overlaps = False
        if overlaps:
            return other_model_id
    return None


def _purge_owned_directory(
    target: Path,
    *,
    model_id: str,
    label: str,
) -> None:
    """Remove one validated owned path without hiding cleanup failures."""
    try:
        remove_owned_entry(target, target.parent)
    except ValueError as exc:
        raise CatalogError(
            f"refusing to purge {label} for model {model_id!r}: {exc}"
        ) from exc
    except OSError as exc:
        raise CatalogError(
            f"could not purge {label} for model {model_id!r}: {target}: {exc}"
        ) from exc


def remove(model_id: str, *, purge: bool = False) -> None:
    """Serialize unregister/purge against every pull of the same model."""
    with _model_pull_lock(model_id):
        with _model_resource_lease(model_id):
            _remove_locked(model_id, purge=purge)


def _remove_locked(model_id: str, *, purge: bool = False) -> None:
    """Unregister `model_id` from the catalog.

    By default this only edits `catalog.json`; the per-model venv at
    `~/.muse/venvs/<model_id>/` stays on disk. Mirrors `apt remove`'s
    "metadata only" semantics.

    When `purge=True`:
      - rmtree the venv directory unless another catalog entry references it.
      - rmtree a Muse-owned artifact bundle, or the complete Hugging Face
        cache repository containing the entry's snapshot, IF no other
        catalog entry references that ownership unit. Legacy bundled pulls
        stored in the shared HF cache (`~/.cache/huggingface`) are left alone;
        Muse does not own that location.
      - rmtree a weightless model's empty owned `<catalog>/local/<id>` dir.

    Tolerates either path being already gone. Once catalog removal commits,
    every independent owned target is attempted; unsafe or failed filesystem
    cleanup raises ``CatalogError`` instead of reporting a false success.

    Holds _CATALOG_WRITE_LOCK for the full read->pop->write sequence. The
    rmtree steps run outside that transaction lock but remain under the
    per-model pull lock held by `remove`, so a re-pull cannot recreate a
    directory between unregister and deletion.
    """
    venv_target: Path | None = None
    weights_target: Path | None = None
    local_target: Path | None = None
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        entry = catalog.get(model_id, {}) or {}
        if purge:
            venv_path = entry.get("venv_path")
            local_dir = entry.get("local_dir")
            if venv_path:
                venv_target = _strict_owned_purge_path(
                    venv_path,
                    _catalog_dir() / "venvs",
                    field_name="venv_path",
                    model_id=model_id,
                )
            if local_dir and entry.get("weights") == "none":
                # A weightless model's empty owned directory; never a
                # shared weights cache, so no cross-reference check.
                local_target = _strict_owned_purge_path(
                    local_dir,
                    _catalog_dir() / "local",
                    field_name="local_dir",
                    model_id=model_id,
                    root_label="local",
                )
            elif local_dir:
                weights_target = _owned_weights_purge_path(
                    local_dir,
                    _catalog_dir() / "weights",
                    model_id=model_id,
                )
            if venv_target is not None:
                other_model_id = _other_catalog_path_reference(
                    catalog,
                    model_id=model_id,
                    field_name="venv_path",
                    target=venv_target,
                )
                if other_model_id is not None:
                    logger.info(
                        "preserving venv for model %s because model %s "
                        "also references %s",
                        model_id,
                        other_model_id,
                        venv_target,
                    )
                    venv_target = None
            if weights_target is not None:
                other_model_id = _other_catalog_path_reference(
                    catalog,
                    model_id=model_id,
                    field_name="local_dir",
                    target=weights_target,
                )
                if other_model_id is not None:
                    logger.info(
                        "preserving weights for model %s because model %s "
                        "also references %s",
                        model_id,
                        other_model_id,
                        weights_target,
                    )
                    weights_target = None
        catalog.pop(model_id, None)
        _write_catalog(catalog)
    # Resolver-pulled entries appear in known_models() via the persisted
    # manifest path; once removed, that cache must drop them too or
    # `muse models list` keeps reporting a model that no longer exists.
    _reset_known_models_cache()
    if not purge:
        return
    failures: list[str] = []
    if venv_target is not None:
        try:
            _purge_owned_directory(
                venv_target,
                model_id=model_id,
                label="venv",
            )
        except CatalogError as exc:
            failures.append(str(exc))
    if local_target is not None:
        try:
            _purge_owned_directory(
                local_target,
                model_id=model_id,
                label="local dir",
            )
        except CatalogError as exc:
            failures.append(str(exc))
    if weights_target is not None:
        try:
            # Pulls and storage pruning take this same process-wide/file lock.
            # The model lock remains outermost, preserving the established
            # model -> resource -> storage ordering used by pull/remove.
            with _storage_cache_lock():
                # A pull for a *different* model id may have downloaded and
                # cataloged another revision of this HF repository after the
                # unregister transaction above. Re-read while holding the
                # storage lock before deleting the repository ownership unit.
                # Pulls cannot add another weight reference until this lock is
                # released; removals can only make deletion more conservative.
                with _CATALOG_WRITE_LOCK:
                    current_catalog = _read_catalog()
                    other_model_id = _other_catalog_path_reference(
                        current_catalog,
                        model_id=model_id,
                        field_name="local_dir",
                        target=weights_target,
                    )
                if other_model_id is not None:
                    logger.info(
                        "preserving weights for model %s because model %s "
                        "acquired a reference to %s",
                        model_id,
                        other_model_id,
                        weights_target,
                    )
                else:
                    _purge_owned_directory(
                        weights_target,
                        model_id=model_id,
                        label="weights",
                    )
        except CatalogError as exc:
            # Attempt both independently-owned targets even when one fails.
            failures.append(str(exc))
    if failures:
        raise CatalogError("; ".join(failures))
