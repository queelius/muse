"""HF resolver plugin for seq2seq summarizers.

Sniffs HF repos with the `summarization` task tag. The plugin only
covers BART/PEGASUS/T5-shape summarizers via AutoModelForSeq2SeqLM.
Chat-style summarizers (Qwen-summarize, Llama-summarize) belong in the
chat/completion modality.

Priority 110: matches embedding/text. The `summarization` tag is
specific enough that no other plugin should claim it; we don't need
extra specificity. Wins over text/classification (200, catch-all),
loses to file-pattern plugins at 100 (GGUF, faster-whisper, diffusers).

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
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
    "muse.modalities.text_summarization.runtimes.bart_seq2seq:BartSeq2SeqRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.36.0")


# Repo-name substrings that suggest the repo was fine-tuned on dialog
# data (meeting/chat transcripts). Used for the
# supports_dialog_summarization capability hint.
_DIALOG_HINTS = ("samsum", "dialog", "chat", "meeting")


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "summarization" in tags


def _supports_dialog(repo_id: str) -> bool:
    """True if the repo name suggests dialog-tuned summarization."""
    name = repo_id.lower()
    return any(hint in name for hint in _DIALOG_HINTS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "text/summarization",
        "hf_repo": repo_id,
        "description": f"Seq2seq summarizer: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {
            "device": "auto",
            "default_length": "medium",
            "default_format": "paragraph",
            "supports_dialog_summarization": _supports_dialog(repo_id),
            "memory_gb": 1.5,
            "max_input_tokens": 1024,
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = transformers_allow_patterns(info)
        # BART tokenizer needs vocab.json + merges.txt; PEGASUS/T5 need
        # spiece.model. Pull them all defensively so the tokenizer load
        # never blows up on a missing vocab artifact.
        allow_patterns.extend([
            "tokenizer*", "spiece.model", "merges.txt", "vocab.json",
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
    """Search HuggingFace for summarization-tagged repos.

    Filter: summarization task tag. Returns one row per matching repo.
    """
    return hf_tag_search(
        api, query, sort=sort, limit=limit,
        tag="summarization", modality="text/summarization",
    )


HF_PLUGIN = {
    "modality": "text/summarization",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 110: matches embedding/text's tier. Wins over text/classification
    # (200, catch-all). Loses to file-pattern plugins at 100 (GGUF,
    # faster-whisper, diffusers).
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
