"""HF resolver plugin for diffusers text-to-image models.

Sniffs HF repos for `model_index.json` (the diffusers pipeline config)
plus the `text-to-image` tag. Synthesizes a manifest with capabilities
inferred from repo name (turbo/flux/sdxl/sd3 patterns set sensible
default steps/guidance/size).

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.image_generation.runtimes.diffusers"
    ":DiffusersText2ImageModel"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow",
    "safetensors",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults so each model lands with reasonable
    steps/guidance/size without users having to tweak. Override per-call
    via request fields or per-model via curated capabilities overlay."""
    rid = repo_id.lower()
    if "flux" in rid and "schnell" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 4,
            "default_guidance": 0.0,
        }
    if "flux" in rid and "dev" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 28,
            "default_guidance": 3.5,
        }
    if "turbo" in rid:
        return {
            "default_size": [512, 512],
            "default_steps": 1,
            "default_guidance": 0.0,
        }
    if "sdxl" in rid or "stable-diffusion-xl" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 25,
            "default_guidance": 7.5,
        }
    if "stable-diffusion-3" in rid or "sd3" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 28,
            "default_guidance": 4.5,
        }
    return {
        "default_size": [512, 512],
        "default_steps": 25,
        "default_guidance": 7.5,
    }


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    has_t2i_tag = "text-to-image" in tags
    return has_pipeline_config and has_t2i_tag


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    defaults = _infer_defaults(repo_id)
    capabilities = {
        **defaults,
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
    }
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/generation",
        "hf_repo": repo_id,
        "description": f"Diffusers text-to-image: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
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
        search=query, filter="text-to-image",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/generation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/generation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
