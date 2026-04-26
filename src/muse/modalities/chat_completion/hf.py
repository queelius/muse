"""HF resolver plugin for GGUF chat/completion models.

Sniffs HuggingFace repos for `.gguf` siblings and synthesizes a manifest
that targets the LlamaCppModel generic runtime. Variant (quant tag) is
required: a single GGUF repo often publishes 5+ quants and there is no
defensible default. `muse search foo --modality chat/completion`
enumerates each variant as a separate row.

This plugin is loaded by `discover_hf_plugins` via single-file import,
so it must NOT use relative imports or import from sibling modality
modules. See docs/HF_PLUGINS.md for the authoring rules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from muse.core.chat_formats import lookup_chat_format
from muse.core.resolvers import ResolvedModel, ResolverError, SearchResult


_VARIANT_RE = re.compile(
    r"(q\d+_[a-z0-9_]+|iq\d+_[a-z0-9]+|f16|bf16|f32)", re.IGNORECASE,
)

_RUNTIME_PATH = "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
_PIP_EXTRAS = ("llama-cpp-python>=0.2.90",)


def _extract_variant(gguf_filename: str) -> str:
    stem = Path(gguf_filename).stem
    m = _VARIANT_RE.search(stem)
    return (m.group(1).lower() if m else stem).replace(".", "_")


def _match_gguf_variant(files: list[str], variant: str) -> str | None:
    norm = variant.lower()
    for f in files:
        if _extract_variant(f) == norm:
            return f
    return None


def _gguf_model_id(repo_id: str, variant: str) -> str:
    base = repo_id.split("/", 1)[-1].lower()
    if not base.endswith("-gguf"):
        base = f"{base}-gguf"
    return f"{base}-{variant.lower().replace('_', '-')}"


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff_supports_tools(chat_template: str | None) -> bool:
    if not chat_template or not isinstance(chat_template, str):
        return False
    return bool(re.search(r"(\bif\s+tools\b|\{\{\s*tools|tool_calls)", chat_template))


def _try_sniff_tools_from_repo(repo_id: str) -> bool | None:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
    except Exception:
        return None
    try:
        cfg = json.loads(Path(path).read_text())
    except Exception:
        return None
    return _sniff_supports_tools(cfg.get("chat_template"))


def _try_sniff_context_length_from_repo(repo_id: str) -> int | None:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
        cfg = json.loads(Path(path).read_text())
        return int(cfg.get("max_position_embeddings") or 0) or None
    except Exception:
        return None


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(f.endswith(".gguf") for f in siblings)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    gguf_files = [f for f in siblings if f.endswith(".gguf")]
    if not gguf_files:
        raise ResolverError(f"no .gguf files in {repo_id}")
    if variant is None:
        variants = sorted({_extract_variant(f) for f in gguf_files})
        raise ResolverError(
            f"variant required for GGUF repo {repo_id}; available: {variants}"
        )
    matched = _match_gguf_variant(gguf_files, variant)
    if matched is None:
        variants = sorted({_extract_variant(f) for f in gguf_files})
        raise ResolverError(
            f"variant {variant!r} not found in {repo_id}; available: {variants}"
        )

    supports_tools = _try_sniff_tools_from_repo(repo_id)
    ctx_length = _try_sniff_context_length_from_repo(repo_id)

    hints = lookup_chat_format(repo_id) or {}

    model_id = _gguf_model_id(repo_id, variant)
    capabilities: dict[str, Any] = {
        "gguf_file": matched,
        "supports_tools": hints.get("supports_tools", supports_tools),
    }
    if "chat_format" in hints:
        capabilities["chat_format"] = hints["chat_format"]
    if ctx_length:
        capabilities["context_length"] = ctx_length

    manifest = {
        "model_id": model_id,
        "modality": "chat/completion",
        "hf_repo": repo_id,
        "description": f"GGUF model: {repo_id} ({variant})",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        allow_patterns = [matched, "tokenizer*", "config.json", "*.md"]
        return Path(snapshot_download(
            repo_id=repo_id, allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(search=query, filter="gguf", sort=sort, limit=limit)
    for repo in repos:
        siblings = getattr(repo, "siblings", None) or []
        if not siblings:
            try:
                info = api.repo_info(repo.id, files_metadata=True)
                siblings = info.siblings
            except Exception:
                continue
        variant_to_size: dict[str, float] = {}
        for s in siblings:
            if not s.rfilename.endswith(".gguf"):
                continue
            variant = _extract_variant(s.rfilename)
            size_bytes = getattr(s, "size", None) or 0
            variant_to_size[variant] = variant_to_size.get(variant, 0) + size_bytes
        for variant, total_bytes in variant_to_size.items():
            yield SearchResult(
                uri=f"hf://{repo.id}@{variant}",
                model_id=_gguf_model_id(repo.id, variant),
                modality="chat/completion",
                size_gb=(total_bytes / 1e9) if total_bytes else None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=f"{repo.id} ({variant})",
            )


HF_PLUGIN = {
    "modality": "chat/completion",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
