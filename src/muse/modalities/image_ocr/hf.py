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

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.image_ocr.runtimes.hf_vision2seq:HFVision2SeqRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.40.0", "Pillow")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    # Explicit VLM exclusion: image-text-to-text repos belong to a
    # future image/description modality (#97) even if they also carry
    # the older image-to-text tag.
    if "image-text-to-text" in tags:
        return False
    if "image-to-text" in tags:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(s in repo_id for s in ("trocr", "nougat", "texteller"))


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    name = repo_id.lower()
    capabilities: dict = {"device": "auto"}
    # Auto-detect from repo name; users can override via curated.yaml.
    if "handwritten" in name:
        capabilities["supports_handwritten"] = True
    if any(s in name for s in ("nougat", "texteller", "math", "latex")):
        capabilities["supports_math"] = True

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/ocr",
        "hf_repo": repo_id,
        "description": f"OCR model: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        siblings = [s.rfilename for s in getattr(info, "siblings", [])]
        has_fp16 = any(".fp16." in f for f in siblings)
        if has_fp16:
            allow_patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
        else:
            allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
        # Older OCR repos still ship pytorch_model.bin (no safetensors).
        allow_patterns.append("pytorch_model.bin")
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
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="image-to-text",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/ocr",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
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
