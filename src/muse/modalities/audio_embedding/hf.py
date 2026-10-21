"""HF resolver plugin for audio embedders.

Sniffs HF repos that carry the `feature-extraction` task tag AND a
repo-name pattern matching `clap`, `mert`, `audio-encoder`, `wav2vec`,
or `audio-embedding`. The dual check (tag plus name pattern) stops
text-only feature extractors from being picked up.

Priority 105: between embedding/text (110) and image-generation
file-pattern (100). Loses to file-pattern plugins (GGUF,
faster-whisper, diffusers) so a multi-purpose repo that also ships,
say, a CT2 ASR model still resolves as audio/transcription.

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    SearchResult,
    hf_commit_revision,
)
from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
    transformers_allow_patterns,
)


_RUNTIME_PATH = (
    "muse.modalities.audio_embedding.runtimes.transformers_audio"
    ":AudioEmbeddingRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "librosa>=0.10.0",
)


_NAME_PATTERNS = (
    "clap",
    "mert",
    "audio-encoder",
    "wav2vec",
    "audio-embedding",
)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "feature-extraction" not in tags:
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(s in repo_id for s in _NAME_PATTERNS)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Per-pattern capability defaults inferred from repo name.

    The size buckets (small / base / large) follow each family's
    naming convention; manual override via curated `capabilities:`
    overlay or per-call request fields wins.
    """
    rid = repo_id.lower()
    # CLAP family: cross-modal aligned audio + text.
    if "clap" in rid:
        return {
            "supports_text_embeddings_too": True,
            "dimensions": 512,
            "sample_rate": 48000,
        }
    # MERT v1: music understanding, mean-pool over time. Needs
    # trust_remote_code=True (the repo ships a custom feature
    # extractor); _download already pulls *.py for this reason, so
    # the capability must actually be set or the runtime's
    # trust_remote_code=False default raises transformers' error at
    # load. Only the MERT family gets this flag; every other pattern
    # stays security-conservative (no trust_remote_code key at all).
    if "mert" in rid:
        return {
            "supports_text_embeddings_too": False,
            "dimensions": 768,
            "sample_rate": 24000,
            "trust_remote_code": True,
        }
    # wav2vec / wav2vec2: speech foundation models, hidden_size pool.
    if "wav2vec" in rid:
        if "large" in rid:
            return {
                "supports_text_embeddings_too": False,
                "dimensions": 1024,
                "sample_rate": 16000,
            }
        return {
            "supports_text_embeddings_too": False,
            "dimensions": 768,
            "sample_rate": 16000,
        }
    # Generic audio-encoder / audio-embedding: leave dimensions absent
    # so the runtime auto-detects.
    return {
        "supports_text_embeddings_too": False,
        "sample_rate": 16000,
    }


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    defaults = _infer_defaults(repo_id)
    capabilities: dict[str, Any] = {
        "device": "auto",
        "memory_gb": 1.0,
        "max_duration_seconds": 60.0,
    }
    capabilities.update(defaults)
    revision = hf_commit_revision(info)
    if capabilities.get("trust_remote_code") and revision is None:
        raise ResolverError(
            f"refusing to enable trust_remote_code for {repo_id!r} without "
            "an immutable Hugging Face commit"
        )

    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "audio/embedding",
        "hf_repo": repo_id,
        "description": f"Audio embedder: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        # Prefer safetensors. Pull preprocessor_config.json (mandatory
        # for AutoFeatureExtractor load). Include *.py so trust_remote_code
        # paths can read the custom feature extractor (e.g. MERT).
        allow_patterns = transformers_allow_patterns(info)
        allow_patterns.extend([
            "preprocessor_config.json",
            "*.py",
            "tokenizer*", "vocab*", "merges*", "spiece.model",
        ])
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
    """Search HuggingFace for feature-extraction repos matching audio
    name patterns.

    Filter: feature-extraction task tag. Returns one row per repo
    whose id contains a recognized audio-embedding family marker.
    """
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="feature-extraction", modality="audio/embedding",
        keep=lambda rid: any(s in rid.lower() for s in _NAME_PATTERNS),
    )


HF_PLUGIN = {
    "modality": "audio/embedding",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 105: between embedding/text (110) and image-generation file-pattern (100).
    # Wins over text/classification (200, catch-all). Loses to file-pattern
    # plugins at 100 so a multi-purpose repo with model_index.json still
    # resolves as image/generation.
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
