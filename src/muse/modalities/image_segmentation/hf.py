"""HF resolver plugin for promptable segmenters (SAM / SAM-2 family).

The one runtime, ``SAM2Runtime``, loads ``AutoModelForMaskGeneration``,
so the sniff claims only repos that model class can serve: any repo
tagged ``mask-generation``, plus ``image-segmentation``-tagged repos
whose name is a SAM-family token (``sam``, ``sam2``, ``sam-...``).
Semantic/panoptic segmenters (SegFormer, Mask2Former, OneFormer,
BiRefNet, DETR-panoptic) and CLIPSeg are not claimed: they would
install and then fail to load. Capabilities by repo-name pattern:

  - ``*sam2-hiera-tiny*``: small (40MB), points + boxes + auto.
  - ``*sam2-hiera-base*``: balanced (150MB).
  - ``*sam2-hiera-large*``: best quality (225MB).
  - ``*sam-*``: original SAM family.
  - fallback: conservative defaults preferring point + box prompts.

Priority 110: equal to image/animation; the tag-set is unique to
segmenters so collisions with i2i upscalers (priority 105) do not
arise in practice.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision
from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
)


_RUNTIME_PATH = (
    "muse.modalities.image_segmentation.runtimes.sam2_runtime"
    ":SAM2Runtime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "transformers>=4.43.0",
    "Pillow>=9.1.0",
    "numpy",
)


def _infer_capabilities(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults by repo-id substring match."""
    rid = (repo_id or "").lower()
    if "sam2-hiera-tiny" in rid:
        return {
            "max_masks": 64,
            "memory_gb": 0.8,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    if "sam2-hiera-base" in rid:
        return {
            "max_masks": 64,
            "memory_gb": 1.5,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    if "sam2-hiera-large" in rid:
        return {
            "max_masks": 64,
            "memory_gb": 2.5,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    if "sam-" in rid or rid.endswith("/sam"):
        return {
            "max_masks": 64,
            "memory_gb": 1.5,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    return {
        "max_masks": 16,
        "memory_gb": 1.0,
        "supports_text_prompts": False,
        "supports_point_prompts": True,
        "supports_box_prompts": True,
        "supports_automatic": True,
    }


_SAM_NAME_RE = re.compile(r"(?:^|[-_.])sam2?(?=$|[-_.])")


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "mask-generation" in tags:
        return True
    if "image-segmentation" not in tags:
        return False
    name = (getattr(info, "id", "") or "").rsplit("/", 1)[-1].lower()
    return _SAM_NAME_RE.search(name) is not None


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    capabilities = {
        "device": "auto",
        **_infer_capabilities(repo_id),
    }
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "image/segmentation",
        "hf_repo": repo_id,
        "description": f"Promptable segmenter: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        # SAM-2 repos ship a small set of files: weights, config,
        # processor config. snapshot_download with a generous allowlist
        # picks them all up without dragging in unnecessary extras.
        allow_patterns = [
            "*.safetensors",
            "*.json",
            "*.txt",
            "preprocessor_config.json",
        ]
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
            allow_patterns=allow_patterns,
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
        tag="mask-generation", modality="image/segmentation",
    )


HF_PLUGIN = {
    "modality": "image/segmentation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
