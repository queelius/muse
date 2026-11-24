"""HF resolver plugin for symbolic ABC / MIDI causal-LM repos.

Sniff: HF transformers repos (with ``config.json``) that either carry
symbolic-music tags (``symbolic-music``, ``abc-notation``,
``midi-generation``, ``text-to-midi``) or match known ABC/MIDI LM repo
lineages (``text-to-music``, ``notagen``, ``abc-gpt``, ``tunesformer``,
``midi-lm``, ``abc-music``).

Priority 95: evaluated before generic ``text-generation`` catch-alls so
symbolic ABC causal LMs resolve to ``music/midi`` rather than
``chat/completion``.

Loaded via single-file import; no relative imports and no sibling
``muse.modalities.*`` imports. See ``docs/HF_PLUGINS.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.hf_plugin_helpers import (
    hf_model_id,
    hf_repo_license,
    hf_tag_search,
    transformers_allow_patterns,
)
from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = "muse.modalities.music_midi.runtimes.abc_lm:AbcMidiLMRuntime"
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "accelerate",
)

_SYMBOLIC_TAGS: frozenset[str] = frozenset({
    "symbolic-music",
    "abc-notation",
    "midi-generation",
    "text-to-midi",
})

_REPO_PATTERNS: tuple[str, ...] = (
    "text-to-music",
    "notagen",
    "abc-gpt",
    "abc_gpt",
    "tunesformer",
    "midi-lm",
    "abc-music",
)


def _has_config_json(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(Path(f).name == "config.json" for f in siblings)


def _sniff(info) -> bool:
    """Match HuggingFace symbolic ABC/MIDI causal-LM repos."""
    if not _has_config_json(info):
        return False
    tags = {t.lower() for t in (getattr(info, "tags", None) or []) if isinstance(t, str)}
    if tags & _SYMBOLIC_TAGS:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    if any(pat in repo_id for pat in _REPO_PATTERNS):
        return True
    if ("music" in tags or "text-generation" in tags) and (
        "midi" in repo_id or "abc" in repo_id
    ):
        return True
    return False


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "music/midi",
        "hf_repo": repo_id,
        "description": f"Symbolic ABC/MIDI LM: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {
            "device": "auto",
            "memory_gb": 2.0,
            "max_new_tokens": 512,
            "format": "abc",
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    allow_patterns = transformers_allow_patterns(info)

    def _download(cache_root: Path) -> Path:
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
        allow_patterns=tuple(allow_patterns),
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HuggingFace for symbolic ABC / MIDI music models."""
    return hf_tag_search(
        api,
        query,
        sort=sort,
        limit=limit,
        tag="text-generation",
        modality="music/midi",
        keep=lambda rid: (
            any(pat in rid.lower() for pat in _REPO_PATTERNS)
            or "midi" in rid.lower()
            or "abc" in rid.lower()
        ),
    )


HF_PLUGIN = {
    "modality": "music/midi",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 95,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
