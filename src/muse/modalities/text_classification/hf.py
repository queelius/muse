"""HF resolver plugin for HF text-classification models.

Tag-only sniff: any repo with the `text-classification` tag is claimed.
Priority 200 so this plugin runs LAST after more specific shapes
(GGUF file pattern, faster-whisper CT2 shape, sentence-transformers
config) have had their chance.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.text_classification.runtimes.hf_text_classifier"
    ":HFTextClassifier"
)
_PIP_EXTRAS = ("transformers>=4.36.0", "torch>=2.1.0")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "text-classification" in tags


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "text/classification",
        "hf_repo": repo_id,
        "description": f"Text classifier: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {},
    }

    def _download(cache_root: Path) -> Path:
        # text-classification repos commonly ship pytorch_model.bin +
        # tf_model.h5 + flax + safetensors. We only need safetensors (or
        # pytorch_model.bin as fallback for older repos). Mirror the
        # fp16 detection from image_generation/hf.py so distilled fp16
        # variants don't pull fp32 too.
        siblings = [s.rfilename for s in getattr(info, "siblings", [])]
        has_fp16 = any(".fp16." in f for f in siblings)
        if has_fp16:
            allow_patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
        else:
            allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
        allow_patterns.append("pytorch_model.bin")
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="text-classification",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="text/classification",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "text/classification",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 200,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
