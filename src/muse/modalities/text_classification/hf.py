"""HF resolver plugin for HF text-classification models.

Tag-based sniff with two-way dispatch:

  - Repos tagged `zero-shot-classification` (or whose name suggests
    NLI: 'zero-shot', 'mnli', 'nli', 'xnli') resolve to the
    HFZeroShotPipeline runtime with supports_zero_shot=True.
  - Repos tagged `text-classification` (or whose name suggests
    sentiment / classification heads, after the zero-shot check
    fails) resolve to HFTextClassifier with
    supports_classification=True.

Priority 200 so this plugin runs LAST after more specific shapes
(GGUF file pattern, faster-whisper CT2 shape, sentence-transformers
config) have had their chance.

Loaded via single-file import; no relative imports.
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


_CLASSIFIER_RUNTIME = (
    "muse.modalities.text_classification.runtimes.hf_text_classifier"
    ":HFTextClassifier"
)
_ZERO_SHOT_RUNTIME = (
    "muse.modalities.text_classification.runtimes.hf_zero_shot"
    ":HFZeroShotPipeline"
)
_PIP_EXTRAS = ("transformers>=4.36.0", "torch>=2.1.0")


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "text-classification" in tags or "zero-shot-classification" in tags:
        return True
    # Some NLI checkpoints ship under generic 'text-classification' or
    # without our preferred tags but the repo name makes intent clear.
    repo_id = (getattr(info, "id", "") or "").lower()
    if any(s in repo_id for s in ("zero-shot", "mnli", "nli", "xnli")):
        return True
    return False


def _is_zero_shot(info) -> bool:
    """Return True if this repo should resolve to HFZeroShotPipeline.

    Zero-shot-classification tag wins. Fallback: repo name pattern,
    since some NLI checkpoints carry only the generic
    `text-classification` tag.
    """
    tags = getattr(info, "tags", None) or []
    if "zero-shot-classification" in tags:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(s in repo_id for s in ("zero-shot", "mnli", "nli", "xnli"))


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    is_zs = _is_zero_shot(info)
    runtime_path = _ZERO_SHOT_RUNTIME if is_zs else _CLASSIFIER_RUNTIME
    capabilities: dict = {
        "supports_classification": not is_zs,
        "supports_zero_shot": is_zs,
    }
    description = (
        f"Zero-shot classifier (NLI): {repo_id}" if is_zs
        else f"Text classifier: {repo_id}"
    )
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "text/classification",
        "hf_repo": repo_id,
        "description": description,
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        # SentencePiece tokenizers (DeBERTa-v3 spm.model, XLM-R
        # sentencepiece.bpe.model) often ship without a tokenizer.json.
        allow_patterns.append("*.model")
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=runtime_path,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="text-classification", modality="text/classification",
    )


HF_PLUGIN = {
    "modality": "text/classification",
    # Top-level runtime_path metadata is the catch-all classifier;
    # _resolve() picks the zero-shot variant per-repo when appropriate.
    "runtime_path": _CLASSIFIER_RUNTIME,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 200,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
