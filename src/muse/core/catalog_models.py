"""The known-models view: discovered scripts merged with pulled manifests.

`known_models()` projects every discovered model script's MANIFEST, plus
every resolver-pulled manifest persisted in catalog.json, onto the stable
`CatalogEntry` shape; `get_manifest()` returns the matching MANIFEST dict.
Both re-apply the curated overlay and operator pins through one helper so
they can never disagree. The discovery scan is cached per process; the
merged view is keyed by the catalog file's identity.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path

from muse.core import config
from muse.core.catalog_store import (
    CatalogError,
    _CatalogFileKey,
    _catalog_file_key,
    _catalog_path,
    _read_catalog,
    _validated_persisted_manifest,
)
from muse.core.curated import find_curated, find_curated_by_uri
from muse.core.discovery import (
    WEIGHTLESS,
    DiscoveredModel,
    discover_models,
    is_weightless_manifest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """Stable catalog shape derived from a model script's MANIFEST."""
    model_id: str
    modality: str              # MIME-style: "audio/speech", "embedding/text", etc.
    backend_path: str          # "module.path:ClassName"
    hf_repo: str
    description: str = ""
    pip_extras: tuple[str, ...] = ()
    system_packages: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)  # voices, default_size, capabilities
    # Keep new optional fields after `extra`: a few integrations construct
    # CatalogEntry positionally and the original eight-field order is public.
    python_sources: tuple[dict, ...] = ()
    bundled: bool = False


def _bundled_models_dir() -> Path:
    """Path to the in-repo `src/muse/models/` directory."""
    # catalog_models.py sits at src/muse/core/catalog_models.py; parents[1] is src/muse/.
    return Path(__file__).resolve().parents[1] / "models"


def _user_models_dir() -> Path:
    """Path to the per-user `~/.muse/models/` drop-in directory.

    Users can drop `.py` model scripts here to add backends without
    modifying the muse source tree. Resolves via `Path.home()` so
    monkeypatching `$HOME` in tests Just Works.
    """
    return Path.home() / ".muse" / "models"


def _env_models_dir() -> Path | None:
    """Optional extra models dir from the `$MUSE_MODELS_DIR` env var."""
    env = config.get("paths.models_dir")
    return Path(env) if env else None


def _model_dirs() -> list[Path]:
    """Scan order for model discovery: bundled, then user dir, then env.

    First-found-wins on model_id collision, so bundled models shadow
    user and env entries with the same id. This is intentional: users
    cannot silently replace a bundled model by dropping a script with
    the same id. To override, rename or remove the bundled script.
    """
    dirs = [_bundled_models_dir(), _user_models_dir()]
    env = _env_models_dir()
    if env is not None:
        dirs.append(env)
    return dirs


def _manifest_to_catalog_entry(discovered: DiscoveredModel) -> CatalogEntry:
    """Project a DiscoveredModel onto the CatalogEntry shape.

    Manifest -> CatalogEntry mapping:
        model_id        -> model_id                 (required)
        modality        -> modality                 (required)
        hf_repo         -> hf_repo                  (required)
        description     -> description              (optional, defaults "")
        pip_extras      -> pip_extras               (tuple-coerced, defaults ())
        system_packages -> system_packages          (tuple-coerced, defaults ())
        python_sources  -> python_sources            (tuple-coerced, defaults ())
        capabilities    -> extra                    (dict-copied, defaults {})
        revision, allow_patterns, hf_artifacts -> extra (download metadata)
        weights         -> extra["weights"]         (only "none": weightless)
    backend_path is synthesized from the Model class's module and name.

    A weightless script (`weights: "none"`) has no `hf_repo`; its entry
    carries `hf_repo == ""` and `extra["weights"] == "none"`.
    """
    m = discovered.manifest
    cls = discovered.model_class
    source_path = getattr(discovered, "source_path", None)
    extra = dict(m.get("capabilities", {}))
    for key in ("revision", "allow_patterns", "hf_artifacts"):
        if key in m:
            extra[key] = copy.deepcopy(m[key])
    if is_weightless_manifest(m):
        extra["weights"] = WEIGHTLESS
        hf_repo = m.get("hf_repo") or ""
    else:
        hf_repo = m["hf_repo"]
    return CatalogEntry(
        model_id=m["model_id"],
        modality=m["modality"],
        backend_path=f"{cls.__module__}:{cls.__name__}",
        hf_repo=hf_repo,
        description=m.get("description", ""),
        pip_extras=tuple(m.get("pip_extras", ())),
        system_packages=tuple(m.get("system_packages", ())),
        python_sources=tuple(m.get("python_sources", ())),
        extra=extra,
        bundled=(
            source_path is not None
            and Path(source_path).parent.resolve()
            == _bundled_models_dir().resolve()
        ),
    )


# Discovery projection cache: the importlib-driven scan of model script
# dirs. Built once per process (script imports execute module bodies, and
# bundled/user scripts do not change under a running server); reset via
# _reset_known_models_cache(). The scripts' MANIFEST dicts are kept beside
# their projected entries for get_manifest().
_discovered_entries_cache: dict[str, CatalogEntry] | None = None
_discovered_manifests_cache: dict[str, dict] | None = None
_discovered_sources_cache: dict[str, Path] | None = None


# Merged known-models cache, keyed by the catalog.json identity
# (path_str, mtime_ns) it was built against. A catalog write from ANY
# process -- the admin pull endpoint's `muse pull` subprocess, an operator
# running `muse pull` / `muse models remove` in a shell beside a running
# supervisor -- bumps the file's mtime, so the next known_models() call
# rebuilds the merge instead of serving a frozen snapshot. Without this,
# the supervisor 404'd "unknown model" on enable/route for anything pulled
# after its cache was built, even though catalog.json and /v1/models both
# showed the entry. _MISSING_CATALOG_KEY keeps the no-catalog-file state
# cacheable (fresh install: nothing to merge, stable until a file appears).
_known_models_cache: tuple[_CatalogFileKey, dict[str, CatalogEntry]] | None = None
_MISSING_CATALOG_KEY: _CatalogFileKey = ("<no-catalog>", -2, -2, -2, -2, -2)


# Guards known_models()'s check-then-populate so two threads racing on the
# first call do not both run discover_models() (importlib imports that
# would execute user script bodies twice). Double-checked: the hot path
# checks outside the lock.
_KNOWN_MODELS_LOCK: threading.Lock = threading.Lock()


def _persisted_manifest_to_catalog_entry(
    catalog_model_id: str, manifest: object,
) -> CatalogEntry:
    """Project a catalog-persisted manifest dict onto the CatalogEntry shape.

    Resolver-pulled models persist their full synthesized MANIFEST inside
    catalog.json (under the `manifest` key) so that `known_models()` can
    surface them without rerunning discovery. The persisted dict carries
    `backend_path` directly (it was synthesized from the resolver's
    runtime class path), unlike script-discovered models where backend_path
    is computed from the Model class's `__module__:__name__`.
    """
    manifest = _validated_persisted_manifest(catalog_model_id, manifest)
    return CatalogEntry(
        model_id=manifest["model_id"],
        modality=manifest["modality"],
        backend_path=manifest["backend_path"],
        hf_repo=manifest["hf_repo"],
        description=manifest.get("description", ""),
        pip_extras=tuple(manifest.get("pip_extras", ())),
        system_packages=tuple(manifest.get("system_packages", ())),
        python_sources=tuple(manifest.get("python_sources", ())),
        extra=dict(manifest.get("capabilities", {})),
    )


def _apply_manifest_overlays(model_id: str, manifest: dict, entry_data: dict) -> dict:
    """Apply current curated pins/capabilities, then base_override.

    Operates on a persisted resolver manifest and returns a fresh dict
    (never mutates the input).

    Shared by `known_models()` (which feeds `load_backend` via
    `entry.extra`) and `get_manifest()` (which `worker.py` uses to
    register/gate the model), so a curated.yaml edit or an operator
    `--base` pin can never diverge between what CONSTRUCTS the backend
    and what ADVERTISES/GATES it. Before this helper existed, only
    known_models() re-applied the curated overlay; get_manifest()
    returned the raw persisted manifest, so a curated.yaml capability
    edit + restart left the two read paths disagreeing.

    Order: curated overlay first (curated.yaml is hand-edited and may
    postdate the persisted manifest, so it wins on capability
    collision), then base_override (an operator `--base` pin, applied
    AFTER curated so it wins over both the tag-declared base and a
    curated base_model -- mirrors `device_override`).
    """
    manifest = dict(manifest)

    curated = find_curated(model_id)
    if curated is None:
        source = entry_data.get("source")
        if source:
            curated = find_curated_by_uri(source)
    if curated is not None:
        if curated.capabilities or curated.code_revision:
            merged_caps = dict(manifest.get("capabilities") or {})
            merged_caps.update(curated.capabilities)
            if curated.code_revision is not None:
                merged_caps["code_revision"] = curated.code_revision
            manifest["capabilities"] = merged_caps
        if curated.revision is not None:
            # This is the currently reviewed target, not a claim about the
            # snapshot already on disk. load_backend compares it with the
            # persisted pull revision and requires a re-pull on mismatch.
            manifest["revision"] = curated.revision

    base_override = entry_data.get("base_override")
    if base_override:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps["base_model"] = base_override
        manifest["capabilities"] = merged_caps

    return manifest


def known_models() -> dict[str, CatalogEntry]:
    """Return {model_id: CatalogEntry} for every discovered model.

    Two sources are merged:
      1. `discover_models` over the configured dirs (script-based models,
         bundled or user-dropped).
      2. catalog.json entries with a `manifest` field (resolver-pulled
         models persisted by `_pull_via_resolver`).

    Bundled / discovered scripts win on model_id collision: a user
    cannot silently shadow a script by pulling a same-id resolver
    entry. The persisted entry is dropped from the merge with a
    debug log; the resolver entry can still be removed via
    `muse models remove`.

    Caching is two-tier. The discovery scan (importlib over script dirs)
    is cached for the process lifetime: new SCRIPTS still require a
    restart. The merged result is memoized against catalog.json's
    (path, mtime_ns), so catalog changes written by ANY process (the
    admin pull subprocess, an operator's CLI pull/remove beside a running
    supervisor) are picked up on the next call -- no restart, no manual
    cache reset. The stat key is taken BEFORE the catalog read: if the
    file changes between stat and read we cache newer content under an
    older key, which the next call's key mismatch rebuilds away; the
    reverse (stale content under a fresh key) cannot happen.

    Cache population uses double-checked locking under _KNOWN_MODELS_LOCK,
    which is always acquired before _CATALOG_WRITE_LOCK.
    """
    global _known_models_cache, _discovered_entries_cache
    global _discovered_manifests_cache, _discovered_sources_cache
    # Fast path: cache built against the current catalog file identity.
    key = _catalog_cache_key()
    cached = _known_models_cache
    if cached is not None and key is not None and cached[0] == key:
        return cached[1]
    # Slow path: acquire the lock and re-check inside it.
    with _KNOWN_MODELS_LOCK:
        # Re-key under the lock: the catalog may have changed while we
        # waited, and another thread may have already rebuilt for the
        # current key.
        key = _catalog_cache_key()
        cached = _known_models_cache
        if cached is not None and key is not None and cached[0] == key:
            return cached[1]
        if _discovered_entries_cache is None:
            discovered = discover_models(_model_dirs())
            _discovered_entries_cache = {
                model_id: _manifest_to_catalog_entry(d)
                for model_id, d in discovered.items()
            }
            _discovered_manifests_cache = {
                model_id: d.manifest for model_id, d in discovered.items()
            }
            _discovered_sources_cache = {
                model_id: Path(d.source_path)
                for model_id, d in discovered.items()
                if getattr(d, "source_path", None) is not None
            }
        entries = dict(_discovered_entries_cache)
        catalog = _read_catalog()
        for model_id, entry_data in catalog.items():
            manifest = entry_data.get("manifest")
            if not manifest:
                # Legacy entry: pulled via the bare-id path; the
                # corresponding script's discovery already populated
                # `entries`. Nothing to merge.
                continue
            if model_id in entries:
                logger.debug(
                    "skipping persisted manifest for %s: shadowed by bundled script",
                    model_id,
                )
                continue
            # Re-apply curated capabilities overlay (+ base_override) so
            # edits to curated.yaml / operator pins take effect on next
            # process restart without requiring a re-pull. See
            # _apply_manifest_overlays for the shared order/rationale
            # (also used by get_manifest() so the two never diverge).
            try:
                manifest = _apply_manifest_overlays(model_id, manifest, entry_data)
                entries[model_id] = _persisted_manifest_to_catalog_entry(
                    model_id, manifest,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "skipping invalid persisted manifest for %s: %s",
                    model_id,
                    exc,
                )
        if key is not None:
            _known_models_cache = (key, entries)
        return entries


def _catalog_cache_key() -> _CatalogFileKey | None:
    """Identity of the catalog file the known_models merge was built against.

    (path_str, mtime_ns) for an existing file; a stable sentinel when the
    file does not exist (fresh install -- cacheable until a file appears);
    None when the file exists but cannot be stat'ed (never cache: always
    rebuild rather than risk serving a snapshot we cannot validate).
    """
    p = _catalog_path()
    try:
        info = p.lstat()
    except FileNotFoundError:
        return _MISSING_CATALOG_KEY
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return None
    return _catalog_file_key(p, info)


def _reset_known_models_cache() -> None:
    """Clear both known_models caches so discovery re-runs on next call.

    Catalog-side staleness is handled automatically by the mtime key in
    known_models(); this reset exists for the DISCOVERY tier (tests that
    drop new script files into a models dir mid-process) and as a
    belt-and-braces invalidation after in-process catalog mutations.

    Takes _KNOWN_MODELS_LOCK (L9): a lock-free `cache = None` races the
    lock-guarded rebuild in known_models(). Serializing on the same lock
    forces the invalidation to order strictly before or after any in-flight
    rebuild, so the next known_models() call always rebuilds fresh.
    """
    global _known_models_cache, _discovered_entries_cache
    global _discovered_manifests_cache, _discovered_sources_cache
    with _KNOWN_MODELS_LOCK:
        _known_models_cache = None
        _discovered_entries_cache = None
        _discovered_manifests_cache = None
        _discovered_sources_cache = None


def list_known(modality: str | None = None) -> list[CatalogEntry]:
    entries = list(known_models().values())
    if modality is None:
        return entries
    return [e for e in entries if e.modality == modality]


def is_weightless(entry: CatalogEntry) -> bool:
    """True for a script-discovered model that declares `weights: "none"`."""
    return entry.extra.get("weights") == WEIGHTLESS


def script_source_sha256(model_id: str) -> str:
    """SHA-256 (hex) of the discovered model script file's bytes.

    This is the weightless models' provenance: their code is the whole
    model, so the receipt pins the script source instead of an HF commit.
    Raises KeyError for an id that is not a discovered script and OSError
    when the file cannot be read.
    """
    known_models()  # populates the discovery cache
    with _KNOWN_MODELS_LOCK:
        source = (_discovered_sources_cache or {}).get(model_id)
    if source is None:
        raise KeyError(f"{model_id!r} is not a discovered model script")
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _script_model_ids() -> set[str]:
    """Ids of discovered model scripts (bundled or user-dropped)."""
    known_models()  # populates the discovery cache
    return set(_discovered_entries_cache or ())


def get_manifest(model_id: str) -> dict:
    """Return the MANIFEST dict for a known model.

    Two sources, in order of preference:
      1. catalog.json's persisted manifest (resolver-pulled models). The
         resolver synthesized this dict at pull time; it's the source of
         truth for that entry. The curated-capabilities overlay and any
         operator base_override are re-applied here via
         `_apply_manifest_overlays`, the SAME helper `known_models()`
         uses, so a curated.yaml edit affects gating (this function) and
         construction (known_models() -> load_backend) identically --
         see `_apply_manifest_overlays` for why that matters.
      2. The model script's module-level MANIFEST, cached by discovery.

    Returns a copy so callers can mutate without affecting the source.

    Raises KeyError if the model is not in `known_models()`.
    """
    catalog_known = known_models()
    catalog = _read_catalog()
    entry_data = catalog.get(model_id, {})
    persisted = entry_data.get("manifest")
    with _KNOWN_MODELS_LOCK:
        discovered_manifest = (_discovered_manifests_cache or {}).get(model_id)
    if model_id not in catalog_known:
        if persisted is not None:
            try:
                _validated_persisted_manifest(model_id, persisted)
            except (TypeError, ValueError) as exc:
                raise CatalogError(
                    f"catalog entry for {model_id!r} has an invalid manifest: {exc}"
                ) from exc
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    if persisted is not None and discovered_manifest is None:
        try:
            overlaid = _apply_manifest_overlays(model_id, persisted, entry_data)
            return _validated_persisted_manifest(model_id, overlaid)
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError(
                f"catalog entry for {model_id!r} has an invalid manifest: {exc}"
            ) from exc
    if discovered_manifest is None:
        # The discovery cache was reset after known_models() built it.
        known_models()
        with _KNOWN_MODELS_LOCK:
            discovered_manifest = (_discovered_manifests_cache or {}).get(model_id)
    # The script's own MANIFEST, from discovery: a script may alias its Model
    # to a shared runtime class, so backend_path's module need not carry it.
    return dict(discovered_manifest or {})
