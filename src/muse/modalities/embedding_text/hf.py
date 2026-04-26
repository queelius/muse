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

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = "muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel"
_PIP_EXTRAS = ("torch>=2.1.0", "sentence-transformers>=2.2.0")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "sentence-transformers" in tags:
        return True
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(Path(f).name == "sentence_transformers_config.json" for f in siblings)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "embedding/text",
        "hf_repo": repo_id,
        "description": f"Sentence-Transformers: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {},
    }

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="sentence-transformers", sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="embedding/text",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
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
