"""Known-models catalog: what can be pulled, what's been pulled.

The set of known models is not hardcoded. It is discovered at first
access by scanning `src/muse/models/*.py` for scripts that define a
top-level `MANIFEST` dict and a `Model` class (see `muse.core.discovery`).
Each MANIFEST's fields are projected onto the stable `CatalogEntry`
shape that the rest of muse (CLI, server, worker) consumes.

Structure:
    known_models() -> dict[model_id, CatalogEntry]  # cached, discovery-driven
    catalog.json (on disk): dict[model_id, {
        pulled_at,                     # ISO 8601 timestamp
        hf_repo,                       # original HF repo id
        revision,                      # immutable primary HF commit
        artifact_provenance,           # complete immutable download receipt
        local_dir,                     # HF snapshot_download cache path
        venv_path,                     # dedicated venv for this model
        python_path,                   # <venv_path>/bin/python for workers
    }]
"""
from __future__ import annotations

import copy

import importlib
from importlib import metadata as importlib_metadata
import contextlib
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from muse.core.catalog_locks import (
    _PULL_LOCKS_GUARD,
    _PULL_THREAD_LOCKS,
    _RESOURCE_LOCKS_GUARD,
    _RESOURCE_THREAD_LOCKS,
    _STORAGE_THREAD_LOCK,
    _advisory_lock,
    _model_pull_lock,
    _model_resource_lease,
    _storage_cache_lock,
)
from muse.core.catalog_models import (
    _KNOWN_MODELS_LOCK,
    _MISSING_CATALOG_KEY,
    CatalogEntry,
    _apply_manifest_overlays,
    _bundled_models_dir,
    _catalog_cache_key,
    _env_models_dir,
    _manifest_to_catalog_entry,
    _model_dirs,
    _persisted_manifest_to_catalog_entry,
    _reset_known_models_cache,
    _script_model_ids,
    _user_models_dir,
    get_manifest,
    known_models,
    list_known,
)
from muse.core.catalog_store import (
    _CATALOG_WRITE_LOCK,
    _MAX_CATALOG_BYTES,
    _MAX_CATALOG_CONTAINER_ITEMS,
    _MAX_CATALOG_DEPTH,
    _MAX_CATALOG_ENTRIES,
    _MAX_CATALOG_STRING_CHARS,
    _MODEL_ID_CHARSET_RE,
    CatalogError,
    ModelIdConflictError,
    ModelInUseError,
    StorageBusyError,
    _CatalogFileKey,
    _CatalogSchemaError,
    _CatalogWriteLock,
    _catalog_dir,
    _catalog_file_key,
    _catalog_path,
    _catalog_read_failure,
    _deep_copy_catalog,
    _ensure_owned_directory,
    _open_catalog_lock,
    _open_catalog_regular_file,
    _read_catalog,
    _reset_read_catalog_cache,
    _validate_artifact_receipt,
    _validate_catalog_entry,
    _validate_catalog_json_value,
    _validate_catalog_shape,
    _validate_measurements,
    _validate_model_id_for_fs,
    _validated_persisted_manifest,
    _write_catalog,
    is_pulled,
)
from muse.core.curated import find_curated, find_curated_by_uri
from muse.core.hf_seed import drop_unused_seeds, seed_repo
from muse.core.resolvers import COMMIT_RE
from muse.core.venv_dedupe import dedupe_after_install
from muse.core.venv import (
    _is_verbose,
    create_venv,
    ensure_venv,
    install_into_venv,
    install_python_sources,
    remove_owned_entry,
    venv_python,
    venv_transaction,
)


@contextlib.contextmanager
def _hf_quiet_if_needed():
    """Suppress huggingface_hub tqdm progress bars when in quiet mode.

    The bars are useful when a 1GB+ download is in progress and the
    user is staring at the terminal, but during `muse pull <id>` they
    interleave with subsequent stages and add noise. Honor the
    `install_output_mode(verbose=...)` flag set by the CLI: quiet =
    bars off; verbose = bars stay on.

    The `HF_HUB_DISABLE_PROGRESS_BARS` env var is read by
    huggingface_hub at module import time, NOT per call, so setting
    it inside this context manager has no effect after huggingface_hub
    is already loaded. Use the runtime API
    `huggingface_hub.utils.disable_progress_bars()` /
    `enable_progress_bars()` instead; both are idempotent and cheap.
    Fall back to the env var only if the runtime API isn't available
    (very old huggingface_hub versions, very rare in practice).
    """
    if _is_verbose():
        yield
        return
    try:
        from huggingface_hub.utils import (
            disable_progress_bars,
            enable_progress_bars,
        )
        disable_progress_bars()
        try:
            yield
        finally:
            # Restore default. Note: this re-enables globally; if a
            # caller above us had already disabled bars for their own
            # reasons, they get re-enabled. Acceptable since the only
            # in-tree disabler is this context manager itself.
            enable_progress_bars()
    except ImportError:
        # Old huggingface_hub: fall back to the env var. It only
        # works if the import happens after we set it; not great,
        # but better than nothing.
        prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev

logger = logging.getLogger(__name__)


def check_system_packages(packages: list[str]) -> list[str]:
    """Return the subset of system packages not found on PATH."""
    return [p for p in packages if shutil.which(p) is None]


# Published distribution name on PyPI. The importable package, CLI, and
# repo are all `muse`, but the wheel is `museq`. A wheel/PyPI install
# installs muse-into-venv from this dist, not from an editable source tree.
_PYPI_DIST = "museq"


def _is_muse_pyproject(pyproject: Path) -> bool:
    """True when pyproject.toml declares the museq project (name = "museq").

    A cheap sniff so `_muse_repo_root` only claims a directory that is
    actually the muse source tree, never an unrelated parent project.
    """
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return f'name = "{_PYPI_DIST}"' in text or f"name = '{_PYPI_DIST}'" in text


def _muse_repo_root() -> Path | None:
    """Resolve the muse source tree that contains this catalog.py, or None.

    Walks parents of this file for a pyproject.toml that actually declares
    the museq project. Returns None from a wheel/PyPI install (no such
    pyproject in any parent, e.g. under site-packages), so callers install
    the published `museq` distribution instead of editable-installing
    site-packages -- which `pip install -e` rejects ("neither setup.py nor
    pyproject.toml found"), previously making `muse pull` fail outright on a
    PyPI install.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and _is_muse_pyproject(pyproject):
            return parent
    return None


def _muse_server_install_args() -> list[str]:
    """pip args to install muse[server] itself into a per-model venv.

    From a source checkout: ``-e <root>[server]`` (editable, tracks the
    working tree). From a wheel/PyPI install, pin the exact invoking release
    so the worker cannot silently run newer protocol/catalog code than its
    host. Mirrors cli_impl.refresh._pip_target_args.
    """
    root = _muse_repo_root()
    if root is not None:
        return ["-e", f"{root}[server]"]
    return [_installed_muse_requirement(["server"])]


def _installed_muse_requirement(extras: list[str]) -> str:
    """Return an exact requirement for the invoking installed Muse release."""
    try:
        installed_version = importlib_metadata.version(_PYPI_DIST)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"cannot create a compatible worker environment: installed "
            f"distribution metadata for {_PYPI_DIST!r} is unavailable"
        ) from exc
    if not installed_version or any(char.isspace() for char in installed_version):
        raise RuntimeError(
            f"cannot create a compatible worker environment: invalid installed "
            f"{_PYPI_DIST} version {installed_version!r}"
        )
    extras_spec = f"[{','.join(extras)}]" if extras else ""
    return f"{_PYPI_DIST}{extras_spec}=={installed_version}"


def _validate_lora_capabilities(manifest: dict) -> None:
    """Reject unservable LoRA manifests at pull time, post-overlay.

    A lora_adapter entry without a base_model can never load; a
    muse-id base that is not pulled would fail at first request with a
    from_pretrained error. Both fail here, BEFORE the expensive venv
    creation and download, with the fix in the message. Runs after the
    curated/--base capabilities overlay merge so a --base override can
    satisfy a tagless adapter repo.
    """
    from muse.core.resolvers import ResolverError

    caps = manifest.get("capabilities") or {}
    if not caps.get("lora_adapter"):
        return
    model_id = manifest.get("model_id", "<unknown>")
    base = caps.get("base_model")
    if not base:
        raise ResolverError(
            f"LoRA adapter {model_id!r} declares no base model and none was "
            f"given; re-run with: muse pull <identifier> --base "
            f"<muse-id-or-hf-repo>"
        )
    if "/" not in base:
        entry = _read_catalog().get(base)
        if not entry or not entry.get("local_dir"):
            raise ResolverError(
                f"LoRA base {base!r} is not pulled; run `muse pull {base}` "
                f"first, then retry"
            )


def _run_automatic_storage_maintenance() -> None:
    """Run best-effort low-space cleanup before any pull locks are acquired."""
    # The deferred import avoids the catalog <-> storage module cycle. The
    # automatic policy never includes unreferenced resources or shared caches.
    try:
        from muse.core.storage import automatic_prune_before_pull

        maintenance = automatic_prune_before_pull()
    except Exception as exc:  # noqa: BLE001 - never hide the pull's own result
        logger.warning("automatic storage maintenance skipped: %s", exc)
        return
    if maintenance is None:
        return
    reclaimed = maintenance.result.reclaimed_bytes
    if reclaimed:
        logger.info(
            "automatic storage maintenance reclaimed %.1f GiB",
            reclaimed / 1024**3,
        )
    if maintenance.result.failures:
        logger.warning(
            "automatic storage maintenance could not clean %d item(s); "
            "run `muse doctor storage` for details",
            len(maintenance.result.failures),
        )
    if maintenance.free_bytes_after < maintenance.threshold_bytes:
        logger.warning(
            "disk space remains below the configured pre-pull reserve "
            "(%.1f GiB free); run `muse doctor storage`",
            maintenance.free_bytes_after / 1024**3,
        )


def pull(
    identifier: str,
    *,
    base_override: str | None = None,
    model_id: str | None = None,
) -> str:
    """Pull a model and return the catalog id it was installed under.

    Dispatch by identifier shape, with curated alias resolution.

    Resolution order:
      1. Curated alias (e.g. "qwen3-8b-q4" from src/muse/curated.yaml):
         expands to the underlying URI or bundled id. The curated id is
         preserved as the catalog key, so the user sees the friendly id
         in `muse models list` rather than a synthesized resolver one.
      2. Resolver URI (contains "://", e.g. "hf://Qwen/Qwen3-8B-GGUF@q4_k_m"):
         routed to the matching resolver, which synthesizes a manifest.
      3. Bare model_id (e.g. "kokoro-82m"): looked up in `known_models()`
         and pulled via the bundled-script path.

    All paths create a per-model venv at `<MUSE_CATALOG_DIR>/venvs/<id>/`,
    install museq[server] (editable) + pip_extras, fetch weights, and
    record the venv's Python path so `muse serve` can spawn workers
    with the right interpreter.

    `base_override` applies to LoRA adapter pulls: resolver-URI,
    curated-by-URI/curated-alias, AND resolver-sourced bare-id re-pulls.
    It is threaded to `_pull_via_resolver` as its own `base_override`
    kwarg (a top-level catalog field, mirroring `device_override`), NOT
    merged into the capabilities overlay: see `_pull_via_resolver` for
    why that durability matters. It is warned-and-ignored only for true
    bundled-script pulls, which have no LoRA base to set.

    `model_id` (`muse pull --id`) installs a resolver URI under an explicit
    catalog id, e.g. a second publisher's repo whose synthesized id is
    already taken. It applies only to resolver URIs; the id must not name
    a bundled script or a different curated entry.
    """
    if model_id is not None:
        return _pull_uri_as(identifier, model_id, base_override=base_override)

    curated = find_curated(identifier)
    if curated is not None:
        _run_automatic_storage_maintenance()
        if curated.uri:
            # Resolver-pulled curated entry, stored under the friendly
            # curated id (e.g. qwen3-8b-q4), not the synthesized one.
            with _model_pull_lock(curated.id):
                return _pull_via_resolver(
                    curated.uri,
                    model_id_override=curated.id,
                    base_override=base_override,
                    allow_source_change=True,
                    **_curated_resolver_kwargs(curated),
                )
        # Bundled curated entry: id equals an existing bundled script's
        # model_id. Fall through to the bundled path with that id.
        with _model_pull_lock(curated.id):
            return _pull_bundled(curated.id)

    if "://" in identifier:
        # Inherit curated capabilities for this URI even when the user
        # didn't go through the curated id. Without this, copying a URI
        # out of `muse search` output and pasting it into `muse pull`
        # silently strips overlay fields like `safe_labels` (KoalaAI)
        # or `trust_remote_code` (Nomic/Jina) that are required
        # for the model to behave correctly. The curated id, if any,
        # is also preserved so the catalog key stays friendly.
        uri_curated = find_curated_by_uri(identifier)
        existing_source_id = next(
            (
                model_id
                for model_id, entry in _read_catalog().items()
                if isinstance(entry, dict) and entry.get("source") == identifier
            ),
            None,
        )
        # Before resolution we do not yet know the synthesized model id.
        # Two spelling variants can resolve to the same id, so serialize all
        # first-time, non-curated resolver pulls under one identity. Re-pulls
        # use their stable catalog id and retain per-model concurrency.
        lock_identity = (
            uri_curated.id if uri_curated is not None
            else existing_source_id or "__unresolved_resolver_pull__"
        )
        _run_automatic_storage_maintenance()
        with _model_pull_lock(lock_identity):
            if uri_curated is not None:
                return _pull_via_resolver(
                    identifier,
                    model_id_override=uri_curated.id,
                    base_override=base_override,
                    allow_source_change=True,
                    **_curated_resolver_kwargs(uri_curated),
                )
            return _pull_via_resolver(identifier, base_override=base_override)

    # Bare id: a bundled script OR a resolver-pulled model (whose persisted
    # manifest also lists it in known_models()). A resolver model, detected
    # by its persisted `source` URI, re-pulls through the resolver under the
    # same id with any curated overlay: _pull_bundled would write an entry
    # without `manifest`/`source` that known_models() then drops.
    from muse.core.curated import load_curated

    _validate_model_id_for_fs(identifier)

    existing = _read_catalog().get(identifier, {}) or {}
    source_uri = existing.get("source")
    if source_uri:
        # Resolver-sourced: thread base_override through instead of
        # warn-and-ignore. The warning below is reserved for pulls with
        # no resolver source at all (true bundled scripts).
        uri_curated = find_curated_by_uri(source_uri)
        _run_automatic_storage_maintenance()
        with _model_pull_lock(identifier):
            if uri_curated is not None:
                # No allow_source_change: a bare-id re-pull keeps its source.
                return _pull_via_resolver(
                    source_uri,
                    model_id_override=identifier,
                    base_override=base_override,
                    **_curated_resolver_kwargs(uri_curated),
                )
            return _pull_via_resolver(
                source_uri,
                model_id_override=identifier,
                base_override=base_override,
                revision_override=existing.get("revision"),
                code_revision_override=existing.get("code_revision"),
            )

    if base_override:
        logger.warning(
            "--base only applies to resolver-pulled LoRA adapters; "
            "ignored for %s", identifier,
        )

    catalog_known = known_models()
    if identifier in catalog_known:
        _run_automatic_storage_maintenance()
        with _model_pull_lock(identifier):
            return _pull_bundled(identifier)

    curated_ids = [c.id for c in load_curated()]
    all_known = sorted(set(list(catalog_known) + curated_ids))
    from difflib import get_close_matches
    suggestions = get_close_matches(identifier, all_known, n=3, cutoff=0.5)
    msg = f"unknown model {identifier!r}"
    if suggestions:
        msg += f"; did you mean: {', '.join(suggestions)}?"
    else:
        msg += " (run `muse models list` to see all model ids)"
    raise KeyError(msg)


def _pull_uri_as(uri: str, model_id: str, *, base_override: str | None) -> str:
    """`pull(uri, model_id=...)`: install a resolver URI under an explicit id."""
    if "://" not in uri:
        raise ValueError(
            f"--id only applies to resolver URIs (hf://...), not {uri!r}"
        )
    _validate_model_id_for_fs(model_id)
    curated_owner = find_curated(model_id)
    if curated_owner is not None and not (
        curated_owner.uri and _same_source(curated_owner.uri, uri)
    ):
        raise ModelIdConflictError(
            f"model id {model_id!r} is a curated id for "
            f"{curated_owner.uri or 'a bundled model'}; choose another --id"
        )
    uri_curated = find_curated_by_uri(uri)
    _run_automatic_storage_maintenance()
    with _model_pull_lock(model_id):
        return _pull_via_resolver(
            uri,
            model_id_override=model_id,
            base_override=base_override,
            **_curated_resolver_kwargs(uri_curated),
        )


def _curated_resolver_kwargs(entry) -> dict[str, Any]:
    """Resolver-pull options a curated entry contributes (none for None).

    The capabilities overlay carries runtime settings such as
    trust_remote_code or chat_format into the persisted manifest.
    """
    if entry is None:
        return {}
    return {
        "capabilities_overlay": dict(entry.capabilities or {}) or None,
        "modality_override": entry.modality,
        "revision_override": entry.revision,
        "code_revision_override": entry.code_revision,
    }


def _source_key(uri: str) -> tuple[str, str]:
    """Case-insensitive (scheme, repo) of a resolver URI (Hub ids are).

    The variant is ignored: switching one repo's variant under the same id
    is an update, not another publisher taking the id.
    """
    from muse.core.resolvers import parse_uri

    scheme, ref, _variant = parse_uri(uri)
    return scheme.lower(), ref.lower()


def _same_source(a: str, b: str) -> bool:
    try:
        return _source_key(a) == _source_key(b)
    except Exception:  # noqa: BLE001 - unparseable legacy value
        return a == b


def _check_resolver_target(
    model_id: str, uri: str, *, allow_source_change: bool,
) -> None:
    """Refuse a resolver pull that would replace another source's entry.

    Resolver plugins derive ids from repo basenames, so two publishers'
    same-named repos collide. A script id is never a valid target: the
    script shadows the persisted manifest but would load the new weights.
    A curated pull owns its id and may follow a changed curated URI.
    """
    if model_id in _script_model_ids():
        raise ModelIdConflictError(
            f"{uri} resolves to model id {model_id!r}, which is a bundled "
            f"model script; pull it with `muse pull {uri} --id <new-id>`"
        )
    if allow_source_change:
        return
    existing = _read_catalog().get(model_id)
    if not isinstance(existing, dict):
        return
    existing_source = existing.get("source")
    if isinstance(existing_source, str) and existing_source:
        if _same_source(existing_source, uri):
            return
        owner = existing_source
    elif "manifest" in existing:
        # Legacy resolver entry without a recorded source URI.
        repo = str(existing.get("hf_repo") or "")
        try:
            new_repo = _source_key(uri)[1]
        except Exception:  # noqa: BLE001
            new_repo = uri
        if repo.lower() == new_repo:
            return
        owner = repo or "another source"
    else:
        # Bundled-shaped entry whose script is gone: nothing to protect.
        return
    raise ModelIdConflictError(
        f"model id {model_id!r} is already installed from {owner}; "
        f"refusing to replace it with {uri}. Install it under another id "
        f"with `muse pull {uri} --id <new-id>`, or `muse models remove "
        f"{model_id}` first"
    )


def _pull_bundled(model_id: str) -> str:
    """Pull one bundled model while excluding live resource users."""
    with _model_resource_lease(model_id):
        _pull_bundled_with_lease(model_id)
    return model_id


def _pull_bundled_with_lease(model_id: str) -> None:
    """Pull a bundled (script-discovered) model by bare id.

    Callers (only `pull()`) verify the id is in `known_models()` first
    and produce a user-friendly error for unknown ids; this defensive
    check guards against internal mistakes.
    """
    _validate_model_id_for_fs(model_id)
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(
            f"unknown model {model_id!r} (internal dispatch bug in "
            f"_pull_bundled; use pull() to get a better error)"
        )
    entry = catalog_known[model_id]

    venvs_root = _catalog_dir() / "venvs"
    _ensure_owned_directory(venvs_root, private=True)
    venv_path = venvs_root / model_id

    with venv_transaction(venv_path) as tx:
        _pull_bundled_transaction(model_id, entry, tx.path)
        tx.commit()
    dedupe_after_install(venv_path)


@dataclass(frozen=True)
class _BundledDownloadSpec:
    """Validated immutable inputs and receipt for one script-based model."""

    revision: str | None
    allow_patterns: tuple[str, ...] | None
    artifacts: tuple[Any, ...] | None
    artifact_provenance: tuple[dict[str, Any], ...] | None


def _artifact_provenance_item(
    *,
    repo_id: str,
    revision: str,
    subdir: str,
    allow_patterns: tuple[str, ...] | None,
    required_patterns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe canonical artifact receipt item."""
    item = {
        "repo_id": repo_id,
        "revision": revision,
        "subdir": subdir,
        "allow_patterns": (
            list(allow_patterns) if allow_patterns is not None else None
        ),
    }
    if required_patterns is not None:
        item["required_patterns"] = list(required_patterns)
    return item


def _bundled_download_spec(
    model_id: str,
    entry: CatalogEntry,
) -> _BundledDownloadSpec:
    """Validate script download metadata before any environment mutation.

    Packaged manifests must be immutable. User drop-in scripts retain the
    historical option of an unpinned snapshot, but receive the same receipt
    protection whenever they opt into a concrete revision.
    """
    raw_revision = entry.extra.get("revision")
    if raw_revision is None:
        if entry.bundled:
            raise CatalogError(
                f"bundled model {model_id!r} must declare an immutable "
                "40-character revision"
            )
        revision = None
    elif (
        not isinstance(raw_revision, str)
        or COMMIT_RE.fullmatch(raw_revision) is None
    ):
        raise CatalogError(
            f"bundled model {model_id!r} has an invalid immutable revision"
        )
    else:
        revision = raw_revision

    raw_allow_patterns = entry.extra.get("allow_patterns")
    if raw_allow_patterns is not None and (
        isinstance(raw_allow_patterns, (str, bytes))
        or not isinstance(raw_allow_patterns, (list, tuple))
        or not all(
            isinstance(pattern, str) and pattern
            for pattern in raw_allow_patterns
        )
    ):
        raise CatalogError(
            f"bundled model {model_id!r} has invalid allow_patterns"
        )
    allow_patterns = (
        tuple(raw_allow_patterns) if raw_allow_patterns is not None else None
    )

    if "hf_artifacts" not in entry.extra:
        provenance = None
        if revision is not None:
            provenance = (
                _artifact_provenance_item(
                    repo_id=entry.hf_repo,
                    revision=revision,
                    subdir="",
                    allow_patterns=allow_patterns,
                ),
            )
        return _BundledDownloadSpec(
            revision=revision,
            allow_patterns=allow_patterns,
            artifacts=None,
            artifact_provenance=provenance,
        )

    from muse.core.artifacts import ArtifactBundleError, normalize_hf_artifacts

    try:
        artifacts = normalize_hf_artifacts(entry.extra.get("hf_artifacts"))
    except ArtifactBundleError as exc:
        raise CatalogError(
            f"invalid hf_artifacts for bundled model {model_id!r}: {exc}"
        ) from exc
    primary = artifacts[0]
    if primary.repo_id != entry.hf_repo:
        raise CatalogError(
            f"bundled model {model_id!r} primary artifact repo "
            f"{primary.repo_id!r} does not match hf_repo {entry.hf_repo!r}"
        )
    if revision is not None and revision != primary.revision:
        raise CatalogError(
            f"bundled model {model_id!r} revision does not match its "
            "primary artifact revision"
        )
    revision = primary.revision
    provenance = tuple(
        _artifact_provenance_item(
            repo_id=artifact.repo_id,
            revision=artifact.revision,
            subdir=artifact.subdir,
            allow_patterns=artifact.allow_patterns,
            required_patterns=artifact.required_patterns,
        )
        for artifact in artifacts
    )
    return _BundledDownloadSpec(
        revision=revision,
        allow_patterns=allow_patterns,
        artifacts=artifacts,
        artifact_provenance=provenance,
    )


_OPERATOR_STATE_FIELDS = (
    "enabled",
    "device_override",
    "gpu_layers_override",
)


def _preserve_operator_state(
    replacement: dict[str, Any],
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge live operator controls into freshly pulled model metadata.

    Pull work deliberately runs outside the catalog lock.  Re-read and merge
    these fields only at the final locked commit so an admin edit made while a
    download is running cannot be replaced by the pull's stale defaults.
    """
    if current is None:
        return replacement
    for field in _OPERATOR_STATE_FIELDS:
        if field in current:
            replacement[field] = copy.deepcopy(current[field])
    return replacement


def _pull_bundled_transaction(
    model_id: str,
    entry: CatalogEntry,
    venv_path: Path,
) -> None:
    """Build and register one bundled model inside a venv transaction."""

    download_spec = _bundled_download_spec(model_id, entry)

    ensure_venv(venv_path, creator=create_venv)

    install_into_venv(venv_path, _muse_server_install_args())

    if entry.pip_extras:
        install_into_venv(venv_path, list(entry.pip_extras))

    if entry.system_packages:
        missing = check_system_packages(list(entry.system_packages))
        if missing:
            logger.warning(
                "model %s needs system packages not found on PATH: %s "
                "(install via apt/brew before running)",
                model_id, missing,
            )

    # Synchronize even when empty: removing a reviewed source declaration must
    # revoke its old .pth activation from a cloned/reused environment.
    install_python_sources(venv_path, entry.python_sources)

    # Bundled MANIFESTs may declare `capabilities.allow_patterns` to
    # restrict the snapshot_download manifest (mirrors what the resolver
    # plugins do for fp16-shaped or BIN-only repos). This avoids hauling
    # down fp32 siblings, .bin/.h5 dupes, and standalone single-file
    # checkpoints when the diffusers/transformers runtime only needs the
    # subfolder weights.
    allow_patterns = download_spec.allow_patterns
    revision = download_spec.revision
    weights_cache = _catalog_dir() / "weights"
    _ensure_owned_directory(weights_cache, private=True)
    with _storage_cache_lock():
        if download_spec.artifacts is not None:
            from muse.core.artifacts import download_hf_artifact_bundle

            with _hf_quiet_if_needed():
                local_dir = download_hf_artifact_bundle(
                    weights_cache,
                    bundle_name=model_id,
                    artifacts=download_spec.artifacts,
                    snapshot_download_fn=snapshot_download,
                )
        else:
            download_kwargs: dict[str, Any] = {
                "repo_id": entry.hf_repo,
                "cache_dir": str(weights_cache),
            }
            if allow_patterns:
                download_kwargs["allow_patterns"] = list(allow_patterns)
            if revision:
                download_kwargs["revision"] = revision
            seeded = seed_repo(
                weights_cache, entry.hf_repo,
                allow_patterns=allow_patterns, revision=revision,
            )
            try:
                with _hf_quiet_if_needed():
                    local_dir = snapshot_download(**download_kwargs)
            finally:
                drop_unused_seeds(seeded)
        # Keep the storage lock through the catalog commit. Otherwise a
        # concurrent cleaner can observe a fully-downloaded repository before
        # its catalog reference exists and misclassify it as unreferenced.
        with _CATALOG_WRITE_LOCK:
            catalog = _read_catalog()
            replacement = {
                "pulled_at": datetime.now(timezone.utc).isoformat(),
                "hf_repo": entry.hf_repo,
                "local_dir": str(local_dir),
                "venv_path": str(venv_path),
                "python_path": str(venv_python(venv_path)),
                "enabled": True,
            }
            if revision:
                replacement["revision"] = revision
            if download_spec.artifact_provenance is not None:
                replacement["artifact_provenance"] = copy.deepcopy(
                    list(download_spec.artifact_provenance)
                )
            catalog[model_id] = _preserve_operator_state(
                replacement,
                catalog.get(model_id),
            )
            _write_catalog(catalog)
    _reset_known_models_cache()


def _pull_via_resolver(
    uri: str,
    *,
    model_id_override: str | None = None,
    capabilities_overlay: dict | None = None,
    modality_override: str | None = None,
    base_override: str | None = None,
    revision_override: str | None = None,
    code_revision_override: str | None = None,
    allow_source_change: bool = False,
    _resolved: object | None = None,
) -> str:
    """Pull a model via a resolver URI (e.g. hf://Qwen/Qwen3-8B-GGUF@q4_k_m).

    Resolves the URI, builds the per-model venv, downloads the weights and
    persists the synthesized manifest plus `source: <uri>` in catalog.json.

    - `model_id_override`: catalog id to use instead of the synthesized one
      (a curated alias's friendly id, or `muse pull --id`).
    - `modality_override`: a curated `modality:` that bypasses sniff dispatch
      (rerankers ship as sentence-transformers repos, for example).
    - `capabilities_overlay`: curated capabilities, shallow-merged over the
      synthesized ones (overlay wins) and passed to the runtime.
    - `revision_override`: the curated, immutable Hugging Face snapshot.
      `code_revision_override` pins a repo referenced by Transformers
      `auto_map`, persisted as the `code_revision` capability.
    - `allow_source_change`: curated pulls own their id, so a changed
      curated URI replaces the old source; other pulls refuse that (see
      `_check_resolver_target`).
    - `base_override`: the operator's `--base` LoRA pin. Like
      `device_override` it is a top-level catalog field, applied after
      curated capabilities by `known_models()` and `get_manifest()`, so the
      operator beats both the tag-declared and curated base. It is also
      passed to the resolver (turbo bases re-derive steps/guidance) and
      written into `capabilities.base_model`. A re-pull without `--base`
      keeps the prior entry's pin.
    """
    from muse.core.resolvers import resolve

    # A set `modality_override` makes the resolver bypass sniff dispatch,
    # so curated yaml's modality declaration beats the priority pick.
    resolved = _resolved
    if resolved is None:
        resolved = resolve(
            uri,
            modality=modality_override,
            base_override=base_override,
            revision=revision_override,
        )
    manifest = dict(resolved.manifest)
    # Resolver may put backend_path in the manifest itself, or only on
    # the ResolvedModel. Persist it consistently so load_backend can
    # find it without consulting the resolver again.
    manifest.setdefault("backend_path", resolved.backend_path)

    if capabilities_overlay:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps.update(capabilities_overlay)
        manifest["capabilities"] = merged_caps

    if code_revision_override is not None:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps["code_revision"] = code_revision_override
        manifest["capabilities"] = merged_caps

    if model_id_override:
        manifest["model_id"] = model_id_override
    model_id = manifest["model_id"]
    _validate_model_id_for_fs(model_id)

    if _resolved is None:
        # Fail fast before taking the lease; re-checked under it below.
        _check_resolver_target(
            model_id, uri, allow_source_change=allow_source_change,
        )
        # Resolver URIs do not reveal their synthesized model id until after
        # resolution. Re-enter with that immutable result so the complete
        # venv/download/catalog mutation runs under the model's lifetime
        # resource lease without resolving or downloading twice.
        with _model_resource_lease(model_id):
            return _pull_via_resolver(
                uri,
                model_id_override=model_id_override,
                capabilities_overlay=capabilities_overlay,
                modality_override=modality_override,
                base_override=base_override,
                revision_override=revision_override,
                code_revision_override=code_revision_override,
                allow_source_change=allow_source_change,
                _resolved=resolved,
            )

    _check_resolver_target(model_id, uri, allow_source_change=allow_source_change)

    # Preserve a prior operator --base pin across a re-pull that omits
    # --base, so re-pulling never silently reverts operator intent.
    effective_base_override = base_override
    if not effective_base_override:
        prior_entry = _read_catalog().get(model_id, {}) or {}
        effective_base_override = prior_entry.get("base_override")

    if effective_base_override:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps["base_model"] = effective_base_override
        manifest["capabilities"] = merged_caps

    capabilities = manifest.get("capabilities") or {}
    if capabilities.get("trust_remote_code"):
        resolved_revision = manifest.get("revision")
        if not isinstance(resolved_revision, str) or not (
            COMMIT_RE.fullmatch(resolved_revision)
        ):
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"refusing to enable trust_remote_code for {model_id!r} "
                "without an immutable Hugging Face commit"
            )
        if revision_override is not None and resolved_revision != revision_override:
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"resolver did not honor reviewed revision {revision_override} "
                f"for {model_id!r}; got {resolved_revision!r}"
            )
        code_revision = capabilities.get("code_revision")
        if code_revision is not None and (
            not isinstance(code_revision, str)
            or not COMMIT_RE.fullmatch(code_revision)
        ):
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"refusing to enable external remote code for {model_id!r} "
                "without an immutable code commit"
            )
        if revision_override is None:
            logger.warning(
                "model %s enables trust_remote_code at discovered commit %s; "
                "this code was not selected by Muse's curated review pins",
                model_id,
                resolved_revision,
            )

    _validate_lora_capabilities(manifest)

    venvs_root = _catalog_dir() / "venvs"
    _ensure_owned_directory(venvs_root, private=True)
    venv_path = venvs_root / model_id

    with venv_transaction(venv_path) as tx:
        _pull_resolved_transaction(
            uri=uri,
            model_id=model_id,
            manifest=manifest,
            effective_base_override=effective_base_override,
            resolved=resolved,
            venv_path=tx.path,
        )
        tx.commit()
    dedupe_after_install(venv_path)
    return model_id


def _pull_resolved_transaction(
    *,
    uri: str,
    model_id: str,
    manifest: dict[str, Any],
    effective_base_override: str | None,
    resolved: Any,
    venv_path: Path,
) -> None:
    """Build and register one resolved model inside a venv transaction."""

    ensure_venv(venv_path, creator=create_venv)

    install_into_venv(venv_path, _muse_server_install_args())

    pip_extras = manifest.get("pip_extras") or ()
    if pip_extras:
        install_into_venv(venv_path, list(pip_extras))

    system_packages = manifest.get("system_packages") or ()
    if system_packages:
        missing = check_system_packages(list(system_packages))
        if missing:
            logger.warning(
                "model %s needs system packages not found on PATH: %s "
                "(install via apt/brew before running)",
                model_id, missing,
            )

    python_sources = manifest.get("python_sources") or ()
    # Synchronize even when empty so a re-pull revokes removed source hooks.
    install_python_sources(venv_path, python_sources)

    raw_receipt = getattr(resolved, "artifact_provenance", ())
    if isinstance(raw_receipt, (str, bytes)) or not isinstance(raw_receipt, Sequence):
        from muse.core.resolvers import ResolverError

        raise ResolverError("resolver artifact provenance must be a sequence")
    artifact_provenance: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_receipt):
        if not isinstance(raw_item, Mapping):
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"resolver artifact provenance item {index} is not an object"
            )
        artifact_provenance.append(copy.deepcopy(dict(raw_item)))
    if artifact_provenance:
        manifest["artifact_provenance"] = copy.deepcopy(artifact_provenance)

    weights_cache = _catalog_dir() / "weights"
    _ensure_owned_directory(weights_cache, private=True)
    # Seed only the files the download will fetch: a GGUF repo can hold
    # many quants, and hashing and linking all of them would be wasted work.
    seed_patterns = getattr(resolved, "allow_patterns", None)
    if seed_patterns is None:
        seed_patterns = next(
            (
                item.get("allow_patterns") for item in artifact_provenance
                if item.get("repo_id") == manifest["hf_repo"]
            ),
            None,
        )
    with _storage_cache_lock():
        seeded = seed_repo(
            weights_cache, manifest["hf_repo"],
            allow_patterns=seed_patterns, revision=manifest.get("revision"),
        )
        try:
            with _hf_quiet_if_needed():
                local_dir = resolved.download(weights_cache)
        finally:
            drop_unused_seeds(seeded)
        # Commit the reference before releasing the storage lock; see the
        # bundled path above for the cleaner race this prevents.
        with _CATALOG_WRITE_LOCK:
            catalog = _read_catalog()
            entry = {
                "pulled_at": datetime.now(timezone.utc).isoformat(),
                "hf_repo": manifest["hf_repo"],
                "local_dir": str(local_dir),
                "venv_path": str(venv_path),
                "python_path": str(venv_python(venv_path)),
                "enabled": True,
                "source": uri,
                "manifest": manifest,
            }
            resolved_revision = manifest.get("revision")
            if resolved_revision:
                entry["revision"] = resolved_revision
            if artifact_provenance:
                entry["artifact_provenance"] = copy.deepcopy(artifact_provenance)
            code_revision = (manifest.get("capabilities") or {}).get("code_revision")
            if code_revision:
                entry["code_revision"] = code_revision
            if effective_base_override:
                entry["base_override"] = effective_base_override
            catalog[model_id] = _preserve_operator_state(
                entry,
                catalog.get(model_id),
            )
            _write_catalog(catalog)
    _reset_known_models_cache()


def _strict_owned_purge_path(
    raw_path: str,
    root: Path,
    *,
    field_name: str,
    model_id: str,
) -> Path:
    """Resolve a venv target and require this model's exact owned directory."""
    try:
        resolved_root = root.expanduser().resolve()
        resolved_path = Path(raw_path).expanduser().resolve()
        relative = resolved_path.relative_to(resolved_root)
    except (TypeError, ValueError, OSError) as exc:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog {field_name} "
            "does not resolve inside the owned venv root"
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
            if local_dir:
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


def is_enabled(model_id: str) -> bool:
    """Return True if model is pulled AND enabled in the catalog."""
    catalog = _read_catalog()
    if model_id not in catalog:
        return False
    return catalog[model_id].get("enabled", True)


def set_enabled(model_id: str, enabled: bool) -> None:
    """Toggle the `enabled` flag for a pulled model.

    Raises KeyError if the model is not in the catalog (not pulled).
    Other catalog fields are preserved.

    Holds _CATALOG_WRITE_LOCK for the full read->mutate->write sequence so
    concurrent mutations on different keys do not lose each other's updates.
    """
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        catalog[model_id]["enabled"] = bool(enabled)
        _write_catalog(catalog)
    # `enabled` flows through known_models() into the CatalogEntry
    # consumers see; without the reset, `muse models list` would
    # display the stale state until the process restarts.
    _reset_known_models_cache()


VALID_DEVICE_OVERRIDES = ("auto", "cpu", "cuda", "mps")


def set_device_override(model_id: str, device: str | None) -> None:
    """Set or clear the per-model device override for a pulled model.

    `device` in {auto, cpu, cuda, mps} pins the model's load device,
    overriding both the manifest `capabilities.device` pin and the
    supervisor `--device` flag (see `load_backend`'s precedence). The
    special value "auto" un-pins a cpu-pinned model so the runtime's
    `select_device` picks cuda when a GPU is present. Passing ``None``
    removes the override entirely (revert to manifest pin / --device).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for an unrecognized device label and KeyError when
    the model is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write so concurrent mutations on different keys do not
    clobber each other (mirrors `set_enabled`).
    """
    if device is not None and device not in VALID_DEVICE_OVERRIDES:
        raise ValueError(
            f"invalid device {device!r}; expected one of "
            f"{', '.join(VALID_DEVICE_OVERRIDES)} or None to clear"
        )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if device is None:
            catalog[model_id].pop("device_override", None)
        else:
            catalog[model_id]["device_override"] = device
        _write_catalog(catalog)
    # device_override is read live from the catalog in load_backend, but
    # `muse models info` surfaces it via known_models()-adjacent reads;
    # reset for display consistency (mirrors set_enabled).
    _reset_known_models_cache()


def set_gpu_layers_override(model_id: str, n: int | None) -> None:
    """Set or clear the per-model llama.cpp GPU-layer pin for a pulled model.

    `n` is the llama.cpp `n_gpu_layers` value: -1 = offload every layer the
    GPU fits, 0 = pure CPU, N > 0 = first N layers on GPU (rest on CPU).
    Stored as the TOP-LEVEL catalog field `gpu_layers_override` (operator
    state, mirroring `device_override` -- NOT part of the manifest). Passing
    ``None`` removes the pin (revert to capabilities.n_gpu_layers / the
    runtime default).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for a non-int or n < -1 and KeyError when the model
    is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write (mirrors `set_device_override`).
    """
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < -1:
            raise ValueError(
                f"invalid gpu layers {n!r}; expected an int >= -1 "
                "(-1 = all layers on GPU, 0 = pure CPU) or None to clear"
            )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if n is None:
            catalog[model_id].pop("gpu_layers_override", None)
        else:
            catalog[model_id]["gpu_layers_override"] = n
        _write_catalog(catalog)
    _reset_known_models_cache()


def _import_backend_module(module_path: str):
    """Local indirection for `importlib.import_module`.

    Why this wrapper exists: tests need to stub out the catalog's backend
    import to install a fake module without fetching real ML deps. Patching
    `importlib.import_module` directly (or `muse.core.catalog.importlib.
    import_module`) mutates the shared global `importlib` module object, so
    the stub also intercepts calls from `muse.core.discovery` and anywhere
    else that imports at test time. Indirecting through a catalog-local
    function gives tests a patching target (`muse.core.catalog.
    _import_backend_module`) that only affects catalog's import path.
    """
    return importlib.import_module(module_path)


def _validate_remote_code_catalog_pins(
    model_id: str,
    manifest: dict,
    entry_data: dict,
) -> None:
    """Refuse a local remote-code snapshot that predates current review pins."""
    capabilities = manifest.get("capabilities") or {}
    if not capabilities.get("trust_remote_code"):
        return

    mismatches: list[str] = []
    expected_revision = manifest.get("revision")
    if expected_revision and entry_data.get("revision") != expected_revision:
        mismatches.append("model revision")
    expected_code_revision = capabilities.get("code_revision")
    if (
        expected_code_revision
        and entry_data.get("code_revision") != expected_code_revision
    ):
        mismatches.append("external code revision")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current reviewed "
            f"remote-code pin ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_bundled_catalog_provenance(
    model_id: str,
    entry: CatalogEntry,
    entry_data: dict,
) -> None:
    """Reject script snapshots that do not match their complete receipt."""
    # Resolver downloads persist their own manifest/source and use resolver
    # pin validation. A receipt is written only by the script-based pull path.
    if "manifest" in entry_data:
        return
    if not entry.bundled and "artifact_provenance" not in entry_data:
        return

    spec = _bundled_download_spec(model_id, entry)
    expected = (
        list(spec.artifact_provenance)
        if spec.artifact_provenance is not None
        else None
    )
    mismatches: list[str] = []
    if entry_data.get("revision") != spec.revision:
        mismatches.append("primary revision")
    if entry_data.get("artifact_provenance") != expected:
        mismatches.append("artifact receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current immutable "
            f"artifact provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_resolver_catalog_provenance(
    model_id: str,
    manifest: dict,
    entry_data: dict,
) -> None:
    """Reject resolver snapshots stale against current pins or their receipt."""
    if "manifest" not in entry_data or not entry_data.get("source"):
        return

    mismatches: list[str] = []
    expected_revision = manifest.get("revision")
    if expected_revision and entry_data.get("revision") != expected_revision:
        mismatches.append("primary revision")

    expected_receipt = manifest.get("artifact_provenance")
    if expected_receipt is not None:
        if entry_data.get("artifact_provenance") != expected_receipt:
            mismatches.append("artifact receipt")
        if expected_revision:
            primary_repo = manifest.get("hf_repo")
            primary_revisions = {
                item.get("revision")
                for item in expected_receipt
                if isinstance(item, dict) and item.get("repo_id") == primary_repo
            } if isinstance(expected_receipt, list) else set()
            if expected_revision not in primary_revisions:
                mismatches.append("manifest artifact receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current immutable resolver "
            f"provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_model_provenance(
    model_id: str, entry: CatalogEntry, entry_data: dict, manifest: dict,
) -> None:
    """One artifact policy for loader execution and serve preflight."""
    _validate_remote_code_catalog_pins(model_id, manifest, entry_data)
    _validate_resolver_catalog_provenance(model_id, manifest, entry_data)
    _validate_bundled_catalog_provenance(model_id, entry, entry_data)


def model_load_blocker(model_id: str, entry_data: dict) -> str | None:
    """Return a local cold-load prerequisite failure without loading a model.

    This checks reviewed provenance and the recorded worker environment,
    not dependency imports, inference, or memory capacity. It neither
    downloads nor rewrites artifacts. Legacy embedded ``load_backend``
    callers need not use a per-model interpreter; serving does.
    """
    try:
        manifest = get_manifest(model_id)
        entry = known_models()[model_id]
        _validate_model_provenance(model_id, entry, entry_data, manifest)
    except KeyError:
        return f"model {model_id!r} is not discoverable; re-run `muse pull {model_id}`"
    except (RuntimeError, ValueError, OSError) as exc:
        return str(exc)

    try:
        local_dir = entry_data.get("local_dir")
        if not local_dir or not Path(local_dir).is_dir():
            return f"model files are missing; re-run `muse pull {model_id}`"
        python_path = entry_data.get("python_path")
        if (
            not python_path or not Path(python_path).is_file()
            or not os.access(python_path, os.X_OK)
        ):
            return f"worker interpreter is missing or not executable; re-run `muse pull {model_id}`"
    except OSError:
        return (
            "model environment is inaccessible; check filesystem permissions "
            f"and mounts or re-run `muse pull {model_id}`"
        )
    return None


def load_backend(model_id: str, **kwargs) -> Any:
    """Import backend class and instantiate it.

    `backend_path` has the form "package.module:ClassName". The class
    is expected to accept (hf_repo, local_dir, **kwargs) in its constructor.

    For resolver-pulled models, manifest.capabilities are merged into the
    kwargs (caller's explicit kwargs win). This lets generic runtimes
    like LlamaCppModel pull `gguf_file`, `chat_template`, `context_length`
    out of the persisted manifest without the worker having to know
    those keys exist. `model_id` is also injected so generic runtimes
    (one class, many models) know which model they're loading.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    if not is_pulled(model_id):
        raise RuntimeError(f"model {model_id!r} not pulled; run `muse pull {model_id}`")
    entry = catalog_known[model_id]
    catalog = _read_catalog()
    entry_data = catalog[model_id]
    manifest = get_manifest(model_id)
    _validate_model_provenance(model_id, entry, entry_data, manifest)
    module_path, class_name = entry.backend_path.split(":", 1)
    module = _import_backend_module(module_path)
    cls = getattr(module, class_name)
    local_dir = entry_data["local_dir"]
    # entry.extra holds capabilities from either the bundled MANIFEST
    # (read live from source by known_models() each call) or the persisted
    # manifest in catalog.json (resolver-pulled). Reading from entry here
    # means bundled scripts' capabilities (e.g. device: cpu on kokoro) are
    # honored at load time, not just resolver-pulled ones.
    capabilities = entry.extra
    merged: dict = {"model_id": model_id, **capabilities, **kwargs}
    # Device precedence, most authoritative first:
    #   1. catalog `device_override`  (operator, via `muse models set-device`)
    #   2. manifest `capabilities.device` pin (model-author affinity, e.g.
    #      kokoro's "cpu") -- overrides the supervisor --device flag
    #   3. caller kwargs device (the --device flag), already folded into kwargs
    #   4. "auto" (runtime select_device picks cuda if available, else cpu)
    # The override beats even the manifest pin so an operator can force a
    # cpu-pinned model onto cuda (or back to cpu to save VRAM) per deployment
    # without editing the bundled script. override="auto" un-pins a model to
    # auto-detect. Other capability keys still lose to kwargs (the documented
    # contract); device is the exception because it is a placement preference.
    override = entry_data.get("device_override")
    if override:
        merged["device"] = override
    elif "device" in capabilities and capabilities["device"] != "auto":
        merged["device"] = capabilities["device"]
    # GPU-layers precedence (spec 2026-07-08), most authoritative first:
    #   1. catalog `gpu_layers_override` (operator, via
    #      `muse models set-gpu-layers`)
    #   2. manifest `capabilities.n_gpu_layers` (already in `merged` via the
    #      capabilities splat above)
    #   3. runtime default (-1 in LlamaCppModel: everything the GPU fits)
    # Applied AFTER the kwargs merge, like device_override, so the operator
    # pin also beats caller kwargs: it is a placement preference.
    gpu_layers = entry_data.get("gpu_layers_override")
    if gpu_layers is not None:
        merged["n_gpu_layers"] = gpu_layers
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **merged)


def _human_size(b: int) -> str:
    """Format bytes as 'N.N GB' / 'N MB' / 'N KB' for table display.

    Returns '-' for 0 (callers use 0 as the missing-size sentinel).
    GB uses one decimal; MB and KB are integer-rounded to keep the
    column narrow enough for table alignment.
    """
    if b == 0:
        return "-"
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


