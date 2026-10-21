"""HF resolver plugin for cross-encoder rerankers.

Sniffs HF repos with the `cross-encoder` tag, OR repos with both the
`text-classification` tag and `rerank` in the repo name (case-insensitive).
This second branch catches BAAI/bge-reranker-v2-m3 and similar repos
that ship under text-classification without the dedicated cross-encoder
tag.

Priority 115: more specific than embedding/text (110); wins over
text/classification (200, catch-all).

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision
from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
    transformers_allow_patterns,
)


_RUNTIME_PATH = (
    "muse.modalities.text_rerank.runtimes.cross_encoder:CrossEncoderRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "sentence-transformers>=5.0.0")


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "cross-encoder" in tags:
        return True
    if "text-classification" in tags:
        repo_id = (getattr(info, "id", "") or "").lower()
        return "rerank" in repo_id
    return False


def _max_length_for(repo_id: str) -> int:
    """Heuristic max_length per known reranker lineage.

    Most rerankers default to 512. The bge-m3 lineage ships an 8K
    context. Jina v2 defaults to 1024. Override via curated capabilities
    when needed.
    """
    name = repo_id.lower()
    if "bge-reranker-v2-m3" in name or "bge-m3" in name:
        return 8192
    if "jina-reranker-v2" in name:
        return 1024
    return 512


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "text/rerank",
        "hf_repo": repo_id,
        "description": f"Cross-encoder reranker: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {
            "device": "auto",
            "max_length": _max_length_for(repo_id),
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        # Tokenizer files, including any SentencePiece model (spiece.model,
        # XLM-R's sentencepiece.bpe.model for bge-reranker-v2-m3 et al).
        allow_patterns.extend(["tokenizer*", "*.model", "*.py"])
        download_kwargs = {
            "repo_id": repo_id,
            "allow_patterns": allow_patterns,
            "cache_dir": str(cache_root) if cache_root else None,
        }
        if revision is not None:
            download_kwargs["revision"] = revision
        return Path(snapshot_download(**download_kwargs))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HuggingFace for cross-encoder rerankers.

    Filter: cross-encoder tag (when available). Some reranker repos
    ship under text-classification only and won't show up via the tag
    filter; we surface those via the `query` term `rerank` when the
    user passes one.
    """
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="cross-encoder", modality="text/rerank",
    )


HF_PLUGIN = {
    "modality": "text/rerank",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 115: more specific than embedding/text (110); wins over
    # text/classification (200, catch-all). Loses to file-pattern
    # plugins at 100 (GGUF, faster-whisper, diffusers).
    "priority": 115,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
