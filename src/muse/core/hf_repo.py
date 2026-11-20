"""HF repo id matching shared by provenance and admission sizing.

A LoRA's `base_model` may name its base by muse catalog id or by HF repo id
(a raw `hf://` LoRA pull takes it from the Hub tag). Repo ids are matched
case-insensitively, ignoring an `hf://` scheme and any `@revision` suffix,
against both an entry's top-level `hf_repo` and its persisted manifest's.

Stdlib only.
"""
from __future__ import annotations

from typing import Any, Mapping


def repo_key(repo: Any) -> str | None:
    """An HF repo id normalized for matching: lowercase, without an
    `hf://` scheme or an `@revision` suffix. None when it is not a repo id."""
    if not isinstance(repo, str) or "/" not in repo:
        return None
    key = repo.strip().lower()
    if key.startswith("hf://"):
        key = key[len("hf://"):]
    return key.split("@", 1)[0].strip("/") or None


def catalog_entries_for_repo(
    repo: str, catalog: Mapping[str, Any],
) -> list[tuple[str, Mapping]]:
    """Catalog entries (sorted by model id) whose `hf_repo`, top-level or in
    the persisted manifest, is `repo`. Empty when `repo` is not a repo id."""
    want = repo_key(repo)
    if want is None:
        return []
    out = []
    for model_id, entry in sorted(catalog.items()):
        if not isinstance(entry, Mapping):
            continue
        manifest = entry.get("manifest")
        repos = {
            repo_key(entry.get("hf_repo")),
            repo_key(manifest.get("hf_repo") if isinstance(manifest, Mapping) else None),
        }
        if want in repos:
            out.append((model_id, entry))
    return out
