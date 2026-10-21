"""HF resolver plugin for image embedders.

Sniffs HF repos that ship an image processor (preprocessor_config.json
sibling) AND carry one of the relevant task tags
(`image-feature-extraction`, `feature-extraction`, or
`image-classification`). The dual check stops text-only feature
extractors and pure-classifier repos that don't ship image processors
from being picked up.

Priority 105: between embedding/text (110) and image-generation
file-pattern (100). Loses to file-pattern plugins (GGUF,
faster-whisper, diffusers) so a multi-purpose repo that also ships a
diffusers pipeline still resolves as image/generation.

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision
from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
    transformers_allow_patterns,
)


_RUNTIME_PATH = (
    "muse.modalities.image_embedding.runtimes.transformers_image"
    ":ImageEmbeddingRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    # Transformers 5 image processors use torchvision-backed operations.
    "torchvision>=0.16.0",
    "transformers>=4.36.0",
    "Pillow>=9.1.0",
)


_RELEVANT_TAGS = (
    "image-feature-extraction",
    "feature-extraction",
    "image-classification",
)


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_image_processor_config = any(
        Path(s).name == "preprocessor_config.json" for s in siblings
    )
    has_relevant_tag = any(t in tags for t in _RELEVANT_TAGS)
    return has_image_processor_config and has_relevant_tag


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Per-pattern capability defaults inferred from repo name.

    The ranges (small / base / large) are heuristic but honest: each
    family's published checkpoints follow a naming convention we can
    pattern-match. Manual override via curated `capabilities:` overlay
    or per-call request fields wins.
    """
    rid = repo_id.lower()
    # CLIP family: cross-modal aligned, projection_dim is the public spec.
    if "clip" in rid:
        if "large" in rid or "/clip-vit-l" in rid:
            return {
                "supports_text_embeddings_too": True,
                "dimensions": 768,
                "image_size": 224,
            }
        return {
            "supports_text_embeddings_too": True,
            "dimensions": 512,
            "image_size": 224,
        }
    # SigLIP / SigLIP2: similar to CLIP, slightly larger projection.
    if "siglip" in rid:
        if "large" in rid:
            return {
                "supports_text_embeddings_too": True,
                "dimensions": 1024,
                "image_size": 256,
            }
        return {
            "supports_text_embeddings_too": True,
            "dimensions": 768,
            "image_size": 256,
        }
    # DINOv2 / DINOv3: pure image, hidden_size is the embedding dim.
    if "dinov" in rid:
        if "small" in rid:
            return {
                "supports_text_embeddings_too": False,
                "dimensions": 384,
                "image_size": 224,
            }
        if "large" in rid:
            return {
                "supports_text_embeddings_too": False,
                "dimensions": 1024,
                "image_size": 224,
            }
        # base / unspecified
        return {
            "supports_text_embeddings_too": False,
            "dimensions": 768,
            "image_size": 224,
        }
    # Generic ViT / vit-base / vit-large.
    if "vit" in rid:
        if "large" in rid:
            return {
                "supports_text_embeddings_too": False,
                "dimensions": 1024,
                "image_size": 224,
            }
        return {
            "supports_text_embeddings_too": False,
            "dimensions": 768,
            "image_size": 224,
        }
    # Fallback: unknown architecture. Leave dimensions unset so the
    # runtime's own _detect_dimensions auto-fills it from model.config.
    return {
        "supports_text_embeddings_too": False,
        "dimensions": None,
        "image_size": 224,
    }


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    defaults = _infer_defaults(repo_id)
    capabilities: dict[str, Any] = {
        "device": "auto",
        "memory_gb": 1.0,
    }
    capabilities.update({k: v for k, v in defaults.items() if v is not None})
    # If the inferred dimensions is None we keep it absent so the
    # runtime auto-detects it; the rest of the caps still flow through.

    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "image/embedding",
        "hf_repo": repo_id,
        "description": f"Image embedder: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        # Prefer safetensors. Pull the preprocessor_config.json (mandatory
        # for AutoImageProcessor / AutoProcessor load) plus any tokenizer
        # files in case the repo is a CLIP-shaped composite that ships a
        # text tokenizer alongside the image processor.
        allow_patterns = transformers_allow_patterns(info)
        allow_patterns.extend([
            "preprocessor_config.json",
            "tokenizer*", "vocab*", "merges*", "spiece.model",
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
    """Search HuggingFace for image-feature-extraction-tagged repos.

    Filter: image-feature-extraction task tag. Returns one row per repo.
    """
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="image-feature-extraction", modality="image/embedding",
    )


HF_PLUGIN = {
    "modality": "image/embedding",
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
