"""HF resolver plugin for audio/classification.

Sniffs `audio-classification` tag with repo-name fallback for known
audio classifier families. Priority 110 (same slot as audio/embedding,
image/cv).
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
    "muse.modalities.audio_classification.runtimes.hf_audio_classifier"
    ":HFAudioClassifier"
)
_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.40.0", "librosa>=0.10.0")


_NAME_HINTS = (
    "wav2vec2-emotion",
    "wav2vec2-superb",
    "hubert-superb",
    "ast-",
    "audio-classification",
    "mms-lid",
    "panns",
    "audioset",
    "yamnet",
)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "audio-classification" in tags:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(h in repo_id for h in _NAME_HINTS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "audio/classification",
        "hf_repo": repo_id,
        "description": f"Audio classifier: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {"device": "auto"},
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        # Feature extractor config + tokenizer (some models have both).
        allow_patterns.extend([
            "preprocessor_config.json", "feature_extractor_config.json",
            "tokenizer*", "vocab.json",
        ])
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="audio-classification", modality="audio/classification",
    )


HF_PLUGIN = {
    "modality": "audio/classification",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
