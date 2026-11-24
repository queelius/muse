"""HF resolver plugin for neural foreground matting / background removal models
(BiRefNet, BiRefNet_lite, RMBG, BEN2).

Loaded via single-file import; no relative imports.
Priority 95: below promptable segmenters (110) and i2i upscalers (105),
matching specifically by BiRefNet / RMBG / BEN2 repo-name tokens so it never
captures SAM / SegFormer / Mask2Former repos.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
    transformers_allow_patterns,
)
from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision

_RUNTIME_PATH = (
    "muse.modalities.image_matting.runtimes.birefnet:BiRefNetRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "Pillow>=10.0",
    "numpy",
    "einops",
)

_MATTING_NAME_RE = re.compile(
    r"(?:^|[-_.])(birefnet|rmbg|ben2)(?:$|[-_.0-9])",
    re.IGNORECASE,
)

_NON_VISION_PIPELINES = frozenset({
    "text-generation",
    "text2text-generation",
    "automatic-speech-recognition",
    "text-to-speech",
    "text-to-audio",
    "text-to-video",
    "feature-extraction",
    "sentence-similarity",
    "translation",
    "summarization",
    "text-classification",
    "fill-mask",
    "token-classification",
    "question-answering",
})


def _is_matting_repo_name(repo_id: str) -> bool:
    if not isinstance(repo_id, str) or not repo_id:
        return False
    name = repo_id.rsplit("/", 1)[-1].lower()
    return _MATTING_NAME_RE.search(name) is not None


def _infer_capabilities(repo_id: str) -> dict[str, Any]:
    name = (repo_id or "").rsplit("/", 1)[-1].lower()
    if "birefnet" in name and "lite" in name:
        return {"memory_gb": 1.5, "input_size": 1024}
    if "birefnet" in name and (
        "_hr" in name or "-hr" in name or name.endswith("hr")
    ):
        return {"memory_gb": 4.0, "input_size": 2048}
    if "birefnet" in name:
        return {"memory_gb": 2.5, "input_size": 1024}
    if "rmbg" in name:
        return {"memory_gb": 2.0, "input_size": 1024}
    if "ben2" in name:
        return {"memory_gb": 2.0, "input_size": 1024}
    return {"memory_gb": 2.0, "input_size": 1024}


def _sniff(info: Any) -> bool:
    tags = [str(t).lower() for t in (getattr(info, "tags", None) or [])]
    if "gguf" in tags:
        return False
    pipeline_tag = getattr(info, "pipeline_tag", None)
    if isinstance(pipeline_tag, str) and pipeline_tag.lower() in _NON_VISION_PIPELINES:
        return False
    repo_id = getattr(info, "id", "")
    return _is_matting_repo_name(repo_id)


def _resolve(repo_id: str, variant: str | None, info: Any) -> ResolvedModel:
    capabilities = {
        "device": "auto",
        "supports_defringe": True,
        "supports_hard_alpha": True,
        "supports_crop": True,
        **_infer_capabilities(repo_id),
    }
    revision = hf_commit_revision(info)
    manifest: dict[str, Any] = {
        "model_id": hf_model_id(repo_id),
        "modality": "image/matting",
        "hf_repo": repo_id,
        "description": f"Foreground matting / background removal: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        if "*.py" not in allow_patterns:
            allow_patterns.append("*.py")
        return Path(
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root) if cache_root else None,
                allow_patterns=allow_patterns,
                revision=revision,
            )
        )

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(
    api: HfApi,
    query: str,
    *,
    sort: str,
    limit: int,
) -> Iterable[SearchResult]:
    return hf_tag_search(
        api,
        query,
        sort=sort,
        limit=limit,
        tag="image-segmentation",
        modality="image/matting",
        keep=_is_matting_repo_name,
    )


HF_PLUGIN = {
    "modality": "image/matting",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 95,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
