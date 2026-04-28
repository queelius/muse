"""HF resolver plugin for fused-checkpoint AnimateDiff variants.

Sniffs HF repos that ship a complete diffusers pipeline (model_index.json
sibling), advertise the `text-to-video` tag, and have `animate` or
`motion` in the repo name. AnimateLCM is the canonical match.

Limitation: the plugin pairs every match with a default SD 1.5 base
(`emilianJR/epiCRealism`) so resolver-pulled models work out of the box.
Pre-curated configs that need a different base should use the bundled
`animatediff-motion-v3` script or override via curated capabilities.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.image_animation.runtimes.animatediff"
    ":AnimateDiffRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow>=9.1.0",
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
    """Sensible per-pattern defaults for fused AnimateDiff variants.

    Resolver-pulled models advertise `supports_text_to_animation: True`
    and `supports_image_to_animation: False`. Base model defaults to a
    SD 1.5 checkpoint; users can override via curated capabilities or
    by editing the persisted manifest.
    """
    rid = repo_id.lower()
    base = {
        "default_size": [512, 512],
        "default_frames": 16,
        "default_fps": 8,
        "base_model": "emilianJR/epiCRealism",
        "supports_text_to_animation": True,
        "supports_image_to_animation": False,
        "min_frames": 8,
        "max_frames": 24,
    }
    if "animatelcm" in rid:
        return {**base, "default_steps": 4, "default_guidance": 1.0}
    return {**base, "default_steps": 25, "default_guidance": 7.5}


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    repo_id = getattr(info, "id", "") or ""
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    has_t2v_tag = "text-to-video" in tags
    rid = repo_id.lower()
    name_matches = ("animate" in rid) or ("motion" in rid)
    return has_pipeline_config and has_t2v_tag and name_matches


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    capabilities = _infer_defaults(repo_id)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/animation",
        "hf_repo": repo_id,
        "description": f"AnimateDiff fused checkpoint: {repo_id}",
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
        search=query, filter="text-to-video",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/animation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/animation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
