"""Shared scaffolding for modality HF resolver plugins (`modalities/*/hf.py`).

Plugins are loaded single-file (no relative imports), but absolute
imports of `muse.core.*` work, so the boilerplate every plugin repeated
lives here: the repo-id -> catalog-id rule, the model-card license read,
the plain "list_models(filter=TAG) -> SearchResult" search, and the
transformers-shape download allow-list.

Stdlib-only at import time (SearchResult comes from muse.core.resolvers,
itself stdlib-only), so plugin import cost is unchanged.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from muse.core.resolvers import SearchResult


def hf_model_id(repo_id: str) -> str:
    """Catalog id for an HF repo: its basename, lowercased."""
    return repo_id.split("/", 1)[-1].lower()


def hf_repo_license(info: Any) -> str | None:
    """License declared on the repo's model card, or None."""
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def hf_tag_search(
    api: Any,
    query: str,
    *,
    sort: str,
    limit: int,
    tag: str,
    modality: str,
    keep: Callable[[str], bool] | None = None,
) -> Iterable[SearchResult]:
    """Yield one SearchResult per repo carrying HF task tag `tag`.

    `keep(repo_id)` optionally post-filters (e.g. a repo-name allowlist).
    """
    for repo in api.list_models(search=query, filter=tag, sort=sort, limit=limit):
        repo_id = getattr(repo, "id", "") or ""
        if keep is not None and not keep(repo_id):
            continue
        yield SearchResult(
            uri=f"hf://{repo_id}",
            model_id=hf_model_id(repo_id),
            modality=modality,
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo_id,
        )


def transformers_allow_patterns(info: Any) -> list[str]:
    """`snapshot_download` allow-list for a transformers-shape repo.

    Pulls fp16 safetensors when the repo ships them (else all
    safetensors) plus json/txt/md metadata, and skips tf/flax/onnx
    duplicates. Pickle weights (`pytorch_model*.bin`) are a fallback
    only: allowed when the repo has no safetensors at all, so a repo
    shipping both encodings is not downloaded twice. Callers append
    their own tokenizer / processor / `*.py` patterns.
    """
    siblings = [s.rfilename for s in getattr(info, "siblings", None) or []]
    if any(".fp16." in f for f in siblings):
        patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
    else:
        patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
    if not any(f.endswith(".safetensors") for f in siblings):
        patterns.append("pytorch_model*.bin")
    return patterns
