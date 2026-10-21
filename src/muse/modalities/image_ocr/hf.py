"""HF resolver plugin for image/ocr.

Sniffs the canonical `image-to-text` tag for vision-encoder + text-
decoder OCR models. Repo-name fallback for `trocr` / `nougat` /
`texteller` substrings catches checkpoints that ship without the
canonical tag.

Does NOT claim `image-text-to-text` tag (those are VLMs, future #97
image/description modality). The exclusion is explicit because some
VLM repos still carry the older `image-to-text` tag too; we want to
leave VLM dispatch to the VLM modality when it lands.

Priority 110: tag-based, more specific than the text-classification
catch-all (200) but loses to file-pattern plugins (100). Same slot as
image_segmentation, audio_embedding.

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


_RUNTIME_PATH = (
    "muse.modalities.image_ocr.runtimes.hf_vision2seq:HFVision2SeqRuntime"
)
# torchvision is required by transformers.AutoImageProcessor on the
# split-component path (repos that lack preprocessor_config.json, like
# TexTeller and other older vision-encoder-decoder OCR repos). Without
# it, AutoImageProcessor.from_pretrained raises ImportError. The
# transformers 4.x AutoProcessor unified path (TrOCR, Nougat) doesn't
# need torchvision, but listing it unconditionally matches the
# pip_extras audit philosophy (declare every direct + transitive
# import the runtime can hit at load time).
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.40.0",
    "Pillow",
)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    repo_id = (getattr(info, "id", "") or "").lower()
    # Repo-name allowlist: well-known OCR repos always claim, even
    # when their HF tags are sloppy. Explicit list dominates the
    # tag-based fallback below; this is what makes `muse pull texteller`
    # work even if the upstream re-tags later.
    if any(s in repo_id for s in ("trocr", "nougat", "texteller")):
        return True
    # Tag-based dispatch:
    # - `image-to-text` alone -> OCR shape (vision-encoder-decoder).
    # - `image-text-to-text` alone -> VLM shape (Llava / Qwen-VL
    #   style; takes image + text input). Defer to the future
    #   image/description modality (#97).
    # - BOTH tags -> the model accepts both paths but the
    #   image-to-text presence proves it has an OCR-compatible mode.
    #   Claim it for image/ocr; the user can disambiguate via the
    #   curated YAML if a real VLM ends up here.
    has_i2t = "image-to-text" in tags
    has_it2t = "image-text-to-text" in tags
    if has_i2t:
        return True
    if has_it2t:
        return False
    return False


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    name = repo_id.lower()
    capabilities: dict = {"device": "auto"}
    # Auto-detect from repo name; users can override via curated.yaml.
    if "handwritten" in name:
        capabilities["supports_handwritten"] = True
    if any(s in name for s in ("nougat", "texteller", "math", "latex")):
        capabilities["supports_math"] = True

    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "image/ocr",
        "hf_repo": repo_id,
        "description": f"OCR model: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        # Tokenizer + processor files. AutoProcessor reads
        # preprocessor_config.json to dispatch to TrOCRProcessor /
        # NougatProcessor / etc.
        allow_patterns.extend([
            "tokenizer*", "spiece.model", "preprocessor_config.json",
            "vocab.json", "merges.txt", "special_tokens_map.json",
            "added_tokens.json", "generation_config.json",
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
        tag="image-to-text", modality="image/ocr",
    )


HF_PLUGIN = {
    "modality": "image/ocr",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 110: tag-based, more specific than text-classification (200)
    # but loses to file-pattern plugins (100). Same slot as
    # image_segmentation, audio_embedding.
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
