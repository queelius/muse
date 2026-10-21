"""HF resolver plugin for sentence-transformers embedding/text models.

Sniffs HF repos for the `sentence-transformers` tag or the
`sentence_transformers_config.json` sibling file, and synthesizes a
manifest that targets SentenceTransformerModel. No variants; one
manifest per repo.

Loaded via single-file import; must not use relative imports.
See docs/HF_PLUGINS.md for the authoring rules.
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


_RUNTIME_PATH = "muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel"
_PIP_EXTRAS = ("torch>=2.1.0", "sentence-transformers>=5.0.0")


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "sentence-transformers" in tags:
        return True
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(Path(f).name == "sentence_transformers_config.json" for f in siblings)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "embedding/text",
        "hf_repo": repo_id,
        "description": f"Sentence-Transformers: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {},
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        # SentencePiece tokenizers (XLM-R sentencepiece.bpe.model,
        # Llama-style tokenizer.model, T5 spiece.model).
        allow_patterns.append("*.model")
        # Inert unless trust_remote_code is enabled by a curated overlay,
        # but required locally when it is (Nomic/Jina-style repositories).
        allow_patterns.append("*.py")
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
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="sentence-transformers", modality="embedding/text",
    )


HF_PLUGIN = {
    "modality": "embedding/text",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
