"""HF resolver plugin for 3d/generation.

Sniffs HF repos for image-to-3d / text-to-3d shape and synthesizes a
manifest pointing at the right runtime. Priority 110, same slot as the
other modality-specific plugins (audio_classification, audio_embedding,
image_segmentation, image_ocr, image_cv).

v0.41.0 routes ALL matched repos through TripoSRRuntime (the only
runtime currently implemented). v0.43.0 adds per-family dispatch via
_runtime_path_for(): Shap-E repos route to ShapERuntime; all other
repos fall through to TripoSRRuntime until their dedicated runtimes
ship in v0.44.0+ (Wonder3D), v0.45.0+ (TRELLIS), v0.46.0+ (Hunyuan3D-2).

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_TRIPOSR_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.triposr:TripoSRRuntime"
)
_SHAPE_E_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.shape_e:ShapERuntime"
)

# pip_extras audit philosophy: declare every direct + transitive import
# the runtime can hit at load time. tsr is the canonical TripoSR pip
# package; trimesh handles GLB serialization; omegaconf + einops are
# tsr's transitive deps that AutoModel.from_pretrained triggers.
_TRIPOSR_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.40.0",
    "trimesh>=4.0",
    "tsr",
    "Pillow",
    "numpy",
    "omegaconf",
    "einops",
    "huggingface_hub",
)
_SHAPE_E_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers",
    "trimesh",
)


def _runtime_path_for(repo_id: str) -> str:
    """Pick the right runtime backend for an HF repo by family.

    Extension point: when v0.44.0 adds Wonder3D, add one branch here
    plus one runtime file. Curated entries get their runtime via this
    dispatch; ad-hoc HF URI pulls without a curated entry fall through
    to TripoSR.
    """
    name = repo_id.lower()
    if "shap-e" in name or "shape-e" in name:
        return _SHAPE_E_RUNTIME_PATH
    return _TRIPOSR_RUNTIME_PATH


def _pip_extras_for(runtime_path: str) -> tuple[str, ...]:
    if runtime_path == _SHAPE_E_RUNTIME_PATH:
        return _SHAPE_E_PIP_EXTRAS
    return _TRIPOSR_PIP_EXTRAS


# Repo-name allowlist: the canonical 3D generation repos. These match
# regardless of HF tagging (some repos are sloppy with tags). Highest
# precedence.
_NAME_HINTS = (
    "triposr",
    "trellis",
    "wonder3d",
    "hunyuan3d",
    "shap-e",
    "instantmesh",
    "stable-3d",
)
# Tag-based fallback. The canonical 3D generation tags on HF.
_TAG_HINTS = ("image-to-3d", "text-to-3d")
# Repo-name substrings whose model declares supports_text_to_3d=True.
# These are the families that natively accept text prompts in 2026;
# TripoSR / Wonder3D / InstantMesh are image-only.
_TEXT_CAPABLE_NAME_HINTS = ("trellis", "hunyuan3d", "shap-e")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    repo_id = (getattr(info, "id", "") or "").lower()
    # Repo-name allowlist dominates: well-known 3D generators always
    # claim, even when their HF tags are sloppy or absent.
    if any(s in repo_id for s in _NAME_HINTS):
        return True
    # Tag-based fallback.
    tags = getattr(info, "tags", None) or []
    return any(t in tags for t in _TAG_HINTS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    """Per-family dispatch: Shap-E -> ShapERuntime; all others -> TripoSRRuntime.

    Ad-hoc HF URI pulls that don't have a curated entry fall through to
    TripoSR. Curated entries for TRELLIS, Wonder3D, and Hunyuan3D-2
    also land here on pull; they route to TripoSRRuntime until v0.44.0+
    ships their dedicated runtimes.
    """
    name = repo_id.lower()
    runtime_path = _runtime_path_for(repo_id)
    pip_extras = _pip_extras_for(runtime_path)

    capabilities: dict = {
        "device": "cuda",
        "supports_image_to_3d": True,
        "supports_text_to_3d": False,
        "output_format": "glb",
    }

    # Shap-E is text-only (no image-to-3D in the base model).
    if runtime_path == _SHAPE_E_RUNTIME_PATH:
        capabilities["supports_image_to_3d"] = False
        capabilities["supports_text_to_3d"] = True
    elif any(s in name for s in _TEXT_CAPABLE_NAME_HINTS):
        # Repo-name hints for other text-capable families (TRELLIS,
        # Hunyuan3D-2). Image direction stays True for dual-direction repos.
        capabilities["supports_text_to_3d"] = True

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "3d/generation",
        "hf_repo": repo_id,
        "description": f"3D generation: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(pip_extras),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        # snapshot_download returns the local cache directory. Without
        # explicit allow_patterns the full repo lands; 3d generators
        # tend to ship config.json + model.safetensors plus optional
        # decoder/triplane files, so taking everything is safer than
        # guessing patterns per family.
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=runtime_path,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HF for 3d-tagged repos.

    Iterates both `image-to-3d` and `text-to-3d` tags and dedupes by
    repo id so a multi-tagged repo (TRELLIS, Hunyuan3D-2) yields one
    SearchResult, not two.
    """
    seen: set[str] = set()
    for tag in _TAG_HINTS:
        repos = api.list_models(
            search=query, filter=tag,
            sort=sort, limit=limit,
        )
        for repo in repos:
            repo_id = getattr(repo, "id", None)
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            yield SearchResult(
                uri=f"hf://{repo_id}",
                model_id=_model_id(repo_id),
                modality="3d/generation",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo_id,
            )


HF_PLUGIN = {
    "modality": "3d/generation",
    "runtime_path": _TRIPOSR_RUNTIME_PATH,
    "pip_extras": _TRIPOSR_PIP_EXTRAS,
    "system_packages": (),
    # 110: tag-based, more specific than text-classification (200) but
    # loses to file-pattern plugins (100). Same slot as
    # image_segmentation, audio_embedding, image_ocr, audio_classification,
    # image_cv.
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
