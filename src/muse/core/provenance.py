"""Model revision and artifact components, for reproducibility headers.

A model's primary revision is the exact HF commit its weights were pulled
at. Models that combine several artifacts (a LoRA over a base model,
multi-repo pulls, pinned code) also carry a component list and a short
digest over it, so a downstream pipeline can pin the full weight set.
A weightless model (`weights: "none"`) has no revision; its one `code`
component's digest is the first 16 hex characters of its script SHA-256.

Stdlib only. `lookup_revision` never raises: unknown or broken metadata
yields a revision of None.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SNAP = re.compile(r"/snapshots/([0-9a-f]{40})(?:/|$)")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactComponent:
    role: str  # "weights" | "base" | "code"
    repo: str
    revision: str | None


@dataclass(frozen=True)
class ModelRevision:
    model_id: str
    revision: str | None
    components: tuple[ArtifactComponent, ...]
    artifacts_digest: str | None


def _hex40(value: Any) -> str | None:
    return value if isinstance(value, str) and _HEX40.match(value) else None


def _primary(entry: Mapping, manifest: Mapping) -> str | None:
    rev = _hex40(entry.get("revision"))
    if rev:
        return rev
    local_dir = entry.get("local_dir")
    if isinstance(local_dir, str):
        m = _SNAP.search(local_dir)
        if m:
            return m.group(1)
    return _hex40(manifest.get("revision"))


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _base_component(entry, manifest, catalog) -> ArtifactComponent | None:
    caps = manifest.get("capabilities")
    if not isinstance(caps, Mapping) or not caps.get("lora_adapter"):
        return None
    base = _str(entry.get("base_override")) or _str(caps.get("base_model"))
    if base is None:
        return None
    base_entry = (catalog or {}).get(base)
    if not isinstance(base_entry, Mapping):
        base_entry = next(
            (e for e in (catalog or {}).values()
             if isinstance(e, Mapping) and e.get("hf_repo") == base),
            None,
        )
    if isinstance(base_entry, Mapping):
        stored = base_entry.get("manifest")
        base_manifest = stored if isinstance(stored, Mapping) else {}
        repo = _str(base_entry.get("hf_repo")) or _str(base_manifest.get("hf_repo")) or base
        return ArtifactComponent("base", repo, _primary(base_entry, base_manifest))
    return ArtifactComponent("base", base, None)


def _weightless_revision(model_id: str, entry: Mapping) -> ModelRevision:
    """A weightless model: no HF revision, one code component."""
    digest = None
    prov = entry.get("artifact_provenance")
    if isinstance(prov, list):
        for item in prov:
            if (isinstance(item, Mapping) and item.get("kind") == "code"
                    and isinstance(item.get("source_sha256"), str)
                    and _SHA256.match(item["source_sha256"])):
                digest = item["source_sha256"][:16]
                break
    component = ArtifactComponent("code", f"muse/{model_id}", None)
    return ModelRevision(model_id, None, (component,), digest)


def model_revision(model_id: str, entry: Mapping, manifest: Mapping,
                   catalog: Mapping[str, Mapping] | None = None) -> ModelRevision:
    entry = entry if isinstance(entry, Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    if entry.get("weights") == "none":
        return _weightless_revision(model_id, entry)
    primary = _primary(entry, manifest)
    repo = _str(entry.get("hf_repo")) or _str(manifest.get("hf_repo")) or ""
    components: list[ArtifactComponent] = []
    prov = entry.get("artifact_provenance")
    if isinstance(prov, list):
        for item in prov:
            if isinstance(item, Mapping) and _str(item.get("repo_id")):
                components.append(ArtifactComponent(
                    "weights", item["repo_id"], _hex40(item.get("revision"))))
    if not components and repo:
        components.append(ArtifactComponent("weights", repo, primary))
    code = _hex40(manifest.get("code_revision"))
    if code:
        components.append(ArtifactComponent("code", repo, code))
    base = _base_component(entry, manifest, catalog)
    if base is not None:
        components.append(base)
    digest = None
    if len(components) > 1:
        payload = json.dumps(sorted([c.role, c.repo, c.revision] for c in components))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return ModelRevision(model_id, primary, tuple(components), digest)


def revision_headers(rev: ModelRevision) -> list[tuple[bytes, bytes]]:
    out = [(b"x-muse-model", rev.model_id.encode("latin-1", "replace"))]
    if rev.revision:
        out.append((b"x-muse-model-revision", rev.revision.encode()))
    if rev.artifacts_digest:
        out.append((b"x-muse-model-artifacts", rev.artifacts_digest.encode()))
    return out


def system_fingerprint(rev: ModelRevision) -> str | None:
    return f"{rev.model_id}@{rev.revision[:12]}" if rev.revision else None


_cache: dict[str, ModelRevision] = {}
_cache_key: tuple | None = None  # catalog_store._catalog_file_key
_cache_lock = threading.Lock()


def lookup_revision(model_id: str) -> ModelRevision:
    """Resolve a model's revision from the catalog. Never raises."""
    global _cache, _cache_key
    try:
        from muse.core import catalog_store
        try:
            path = catalog_store._catalog_path()
            key = catalog_store._catalog_file_key(path, path.lstat())
        except OSError:
            key = None
        with _cache_lock:
            if key != _cache_key or key is None:
                _cache = {}
                _cache_key = key
            hit = _cache.get(model_id)
        if hit is not None:
            return hit
        catalog = catalog_store._read_catalog()
        entry = catalog.get(model_id)
        if not isinstance(entry, Mapping):
            return ModelRevision(model_id, None, (), None)
        try:
            from muse.core.catalog_models import get_manifest
            manifest = get_manifest(model_id)
        except Exception:  # noqa: BLE001
            stored = entry.get("manifest")
            manifest = stored if isinstance(stored, Mapping) else {}
        rev = model_revision(model_id, entry, manifest, catalog)
        with _cache_lock:
            # Only store if the catalog has not changed since we sampled it.
            if key is not None and key == _cache_key:
                _cache[model_id] = rev
        return rev
    except Exception:  # noqa: BLE001
        return ModelRevision(model_id, None, (), None)
