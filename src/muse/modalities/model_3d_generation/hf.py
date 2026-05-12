"""HF resolver plugin for 3d/generation.

Sniffs HF repos for image-to-3d / text-to-3d shape and synthesizes a
manifest pointing at the right runtime. Priority 110, same slot as the
other modality-specific plugins (audio_classification, audio_embedding,
image_segmentation, image_ocr, image_cv).

Per-family dispatch via _family_for(): Shap-E repos route to
ShapERuntime; TRELLIS repos route to TRELLISRuntime (v0.44.0+); all
other repos fall through to TripoSRRuntime until their dedicated
runtimes ship in v0.45.0+ (Hunyuan3D-2); Wonder3D is deferred
indefinitely. Adding a new family means appending one _Family entry
to _FAMILIES plus shipping a runtime file; no new dispatch functions
and no new conditional branches in _resolve.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
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
_TRELLIS_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.trellis:TRELLISRuntime"
)
# TRELLIS uses Microsoft's standalone SDK (NOT a transformers/diffusers
# AutoPipeline). Verified at v0.44.0 implementation time against the
# real downloaded SDK. The TRELLIS GitHub repo has native-build deps
# (kaolin, xformers, flash-attn, nvdiffrast) that pip cannot install
# cleanly on all systems; Microsoft's setup.sh script is the official
# install path. Users may need to follow that script manually after
# `muse pull trellis-image` if the git pip-install below fails. See:
#   https://github.com/microsoft/TRELLIS/blob/main/setup.sh
_TRELLIS_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.46.0",
    "diffusers>=0.27.0",
    "trimesh",
    "accelerate",
    "Pillow",
    "numpy",
    # TRELLIS standalone SDK from Microsoft. May fail to pip-install
    # on hosts without CUDA toolchain or compatible NVIDIA driver. If
    # so, fall back to following Microsoft's setup.sh inside the
    # per-model venv at ~/.muse/venvs/trellis-image/.
    "trellis @ git+https://github.com/microsoft/TRELLIS.git",
)
_HUNYUAN3D_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.hunyuan3d:Hunyuan3DRuntime"
)
# Hunyuan3D-2 uses Tencent's standalone SDK (NOT a transformers/diffusers
# AutoPipeline). Verified at v0.45.0 implementation time. Tencent's
# GitHub repo has native-build deps (kaolin, xformers, custom CUDA
# extensions) that pip cannot install cleanly on all systems; Tencent's
# setup script is the official install path. Users may need to follow
# that script manually after `muse pull hunyuan3d-2` if the git
# pip-install below fails. See: https://github.com/Tencent/Hunyuan3D-2/
_HUNYUAN3D_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.46.0",
    "diffusers>=0.27.0",
    "trimesh",
    "accelerate",
    "Pillow",
    "numpy",
    # Tencent Hunyuan3D-2 SDK from GitHub. May fail on hosts without
    # CUDA toolchain or compatible NVIDIA driver. Fallback: clone +
    # follow setup.sh inside the per-model venv at ~/.muse/venvs/hunyuan3d-2/.
    "hy3dgen @ git+https://github.com/Tencent/Hunyuan3D-2.git",
)


@dataclass(frozen=True)
class _Family:
    """One 3D-generation model family: how to detect it and how to load it.

    Adding a new family in v0.44.0+ means appending one _Family entry to
    _FAMILIES below, plus shipping the named runtime file. No new
    dispatch functions, no new conditional branches in _resolve.

    Fields:
      name_hints: repo-name substrings (lowercased) that identify this family.
      runtime_path: backend_path written into the catalog manifest.
      pip_extras: per-model pip dependencies installed into the model venv.
      capability_overrides: dict merged over the default capabilities block;
        values here win. Shap-E uses this to flip image/text support flags.
      trust_remote_code: when True, forwarded into capabilities so the
        runtime can pass trust_remote_code=True to from_pretrained. Used
        by TRELLIS (v0.45.0) and Hunyuan3D-2 (v0.46.0).
      system_packages: OS-level packages (e.g. libGL) needed by the runtime.
        Empty for most families; Wonder3D / TRELLIS may declare libGL.
    """

    name_hints: tuple[str, ...]
    runtime_path: str
    pip_extras: tuple[str, ...]
    capability_overrides: dict = field(default_factory=dict)
    trust_remote_code: bool = False
    system_packages: tuple[str, ...] = ()


_FAMILIES: tuple[_Family, ...] = (
    _Family(  # Shap-E (unchanged from v0.43.x)
        name_hints=("shap-e", "shape-e"),
        runtime_path=_SHAPE_E_RUNTIME_PATH,
        pip_extras=_SHAPE_E_PIP_EXTRAS,
        capability_overrides={
            "supports_image_to_3d": False,
            "supports_text_to_3d": True,
        },
    ),
    _Family(  # TRELLIS (unchanged from v0.44.0)
        name_hints=("trellis",),
        runtime_path=_TRELLIS_RUNTIME_PATH,
        pip_extras=_TRELLIS_PIP_EXTRAS,
        capability_overrides={
            "supports_image_to_3d": True,
            "supports_text_to_3d": False,
        },
        trust_remote_code=True,
    ),
    _Family(  # NEW v0.45.0: Hunyuan3D-2, dual-direction
        name_hints=("hunyuan3d",),
        runtime_path=_HUNYUAN3D_RUNTIME_PATH,
        pip_extras=_HUNYUAN3D_PIP_EXTRAS,
        capability_overrides={
            "supports_image_to_3d": True,
            "supports_text_to_3d": True,
        },
        trust_remote_code=True,
    ),
    # Wonder3D: deferred indefinitely (v0.44.0 decision).
)

_DEFAULT_FAMILY = _Family(
    name_hints=(),
    runtime_path=_TRIPOSR_RUNTIME_PATH,
    pip_extras=_TRIPOSR_PIP_EXTRAS,
)

# Precomputed tuple used by _pip_extras_for to avoid recomputation per call.
_ALL_FAMILIES: tuple[_Family, ...] = _FAMILIES + (_DEFAULT_FAMILY,)


def _matches_hint(name: str, hint: str) -> bool:
    """Word-boundary substring match.

    The hint matches `name` only when it appears with non-alphanumeric
    chars on both sides (or at string boundaries). Prevents false positives
    like `my-reshape-enhancer` matching `shape-e`.
    """
    pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])",
        re.IGNORECASE,
    )
    return bool(pattern.search(name))


def _family_for(repo_id: str) -> _Family:
    """Pick the family by name-hint match; fall back to TripoSR."""
    name = repo_id.lower()
    return next(
        (f for f in _FAMILIES if any(_matches_hint(name, h) for h in f.name_hints)),
        _DEFAULT_FAMILY,
    )


def _runtime_path_for(repo_id: str) -> str:
    """Thin alias preserved for test stability. Real dispatch via _family_for."""
    return _family_for(repo_id).runtime_path


def _pip_extras_for(runtime_path: str) -> tuple[str, ...]:
    """Thin alias preserved for test stability. Real dispatch via _family_for."""
    for family in _ALL_FAMILIES:
        if family.runtime_path == runtime_path:
            return family.pip_extras
    return _DEFAULT_FAMILY.pip_extras  # unreachable in practice but type-safe


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
# Reserved tuple. All known text-capable 3D families now declare their
# capability via _Family.capability_overrides; the hint-list path
# is preserved as a fallback for ad-hoc URIs whose family is not yet
# registered. Add a hint here only when a new ad-hoc text-capable
# repo lands without a corresponding _Family entry.
# NOTE: Shap-E, TRELLIS, and Hunyuan3D-2 are intentionally absent here.
# All get their supports_text_to_3d value via capability_overrides in
# _FAMILIES; the guard in _resolve skips this check when
# capability_overrides has already set the flag. The invariant is
# tested by test_text_capable_hints_does_not_overlap_with_family_overrides.
_TEXT_CAPABLE_NAME_HINTS: tuple[str, ...] = ()


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
    """Single-dispatch via _family_for: routes each repo to the right runtime.

    Ad-hoc HF URI pulls that don't have a curated entry fall through to
    TripoSR via _DEFAULT_FAMILY. Curated entries for TRELLIS, Wonder3D,
    and Hunyuan3D-2 also land here on pull; they route to TripoSRRuntime
    until v0.44.0+ ships their dedicated runtimes.
    """
    family = _family_for(repo_id)

    capabilities: dict = {
        "device": "cuda",
        "supports_image_to_3d": True,
        "supports_text_to_3d": False,
        "output_format": "glb",
    }
    capabilities.update(family.capability_overrides)

    # Apply text-capable hints ONLY when capability_overrides has not
    # already set the flag (avoids double-application for Shap-E).
    if "supports_text_to_3d" not in family.capability_overrides:
        name = repo_id.lower()
        if any(s in name for s in _TEXT_CAPABLE_NAME_HINTS):
            capabilities["supports_text_to_3d"] = True

    if family.trust_remote_code:
        capabilities["trust_remote_code"] = True

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "3d/generation",
        "hf_repo": repo_id,
        "description": f"3D generation: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(family.pip_extras),
        "system_packages": list(family.system_packages),
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
        backend_path=family.runtime_path,
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
    "scheme": "hf",
    "modality": "3d/generation",
    # Framework-required top-level keys. The values here are TripoSR-specific
    # placeholders that satisfy the plugin contract (REQUIRED_HF_PLUGIN_KEYS in
    # core/discovery.py); the actual per-resolve runtime_path and pip_extras
    # come from _family_for(repo_id) inside _resolve below.
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
