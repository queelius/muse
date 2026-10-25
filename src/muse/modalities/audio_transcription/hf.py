"""HF resolver for CT2 faster-whisper and the native Qwen3-ASR checkpoint.

Sniffs HF repos for the CT2 file shape (model.bin + config.json +
tokenizer.json or vocabulary.txt) plus the ASR tag. Synthesizes a
manifest that targets FasterWhisperModel. The exact Qwen3-ASR repo dispatches
to its own runtime with a pinned download allow-list. ffmpeg is declared as a
system package so the catalog warns when it's missing on PATH.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, ResolverError, SearchResult, hf_commit_revision
from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
)


_RUNTIME_PATH = "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel"
_PIP_EXTRAS = ("faster-whisper>=1.0.0",)
_SYSTEM_PACKAGES = ("ffmpeg",)


_QWEN_REPO = "qwen/qwen3-asr-1.7b-hf"
_QWEN_RUNTIME = "muse.modalities.audio_transcription.runtimes.qwen3_asr:Qwen3ASRRuntime"
_QWEN_EXTRAS = ("torch>=2.11", "torchcodec>=0.12", "transformers>=5.13.0")
_QWEN_REQUIRED = (
    "config.json", "model.safetensors", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "generation_config.json",
    "chat_template.jinja",
)


def _is_qwen(repo_id: str) -> bool:
    return isinstance(repo_id, str) and repo_id.lower() == _QWEN_REPO


def _qwen_shape(info) -> bool:
    names = {s.rfilename for s in (getattr(info, "siblings", None) or [])}
    return all(name in names for name in _QWEN_REQUIRED)


def _resolve_qwen(repo_id: str, info) -> ResolvedModel:
    if not _qwen_shape(info):
        raise ResolverError(f"unsupported Qwen3-ASR checkpoint shape for {repo_id!r}")
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "audio/transcription",
        "hf_repo": repo_id,
        "description": f"Qwen3 ASR 1.7B transcription: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_QWEN_EXTRAS),
        "system_packages": ["ffmpeg"],
        "capabilities": {
            "device": "auto", "memory_gb": 5.0,
            "supports_word_timestamps": False,
            "supports_segment_timestamps": False,
            "supports_translation": False,
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id, revision=revision,
            allow_patterns=[*_QWEN_REQUIRED, "README.md"],
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(manifest=manifest, backend_path=_QWEN_RUNTIME, download=_download)


def _looks_like_ct2(siblings: list[str], tags: list[str]) -> bool:
    names = {Path(f).name for f in siblings}
    has_ct2_shape = (
        "model.bin" in names
        and "config.json" in names
        and ("vocabulary.txt" in names or "tokenizer.json" in names)
    )
    has_asr_tag = "automatic-speech-recognition" in tags
    return has_ct2_shape and has_asr_tag


def _sniff(info) -> bool:
    if _is_qwen(getattr(info, "id", "")):
        return _qwen_shape(info)
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    return _looks_like_ct2(siblings, tags)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    if _is_qwen(repo_id):
        return _resolve_qwen(repo_id, info)
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "audio/transcription",
        "hf_repo": repo_id,
        "description": f"Faster-Whisper: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": list(_SYSTEM_PACKAGES),
        "capabilities": {},
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
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
        tag="automatic-speech-recognition", modality="audio/transcription",
    )


HF_PLUGIN = {
    "modality": "audio/transcription",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": _SYSTEM_PACKAGES,
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
