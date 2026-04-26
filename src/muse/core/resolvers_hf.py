"""HuggingFace Hub resolver.

URI shapes:
  hf://org/repo                  # sentence-transformers (embedding/text)
  hf://org/repo-GGUF@<variant>   # GGUF (chat/completion); variant required
  hf://org/faster-whisper-tiny   # CT2 faster-whisper (audio/transcription)
  hf://org/Text-Moderation       # text-classification (text/classification)

Sniff logic (see `_sniff_repo_shape`):
  - any .gguf sibling          -> gguf
  - sentence-transformers tag  -> sentence-transformers
  - sentence_transformers_config.json sibling -> sentence-transformers
  - model.bin + config.json + (vocabulary.txt|tokenizer.json) + ASR tag -> faster-whisper
  - text-classification tag    -> text-classification
  - else                       -> unknown (raises on resolve)

Search:
  - modality="chat/completion": HfApi.list_models(filter="gguf") +
    enumerate each repo's .gguf files as separate variants.
  - modality="embedding/text": HfApi.list_models(filter="sentence-transformers")
  - modality="audio/transcription": HfApi.list_models(filter="automatic-speech-recognition")
  - modality="text/classification": HfApi.list_models(filter="text-classification")

Capability sniffing:
  - supports_tools: loads tokenizer_config.json's chat_template (when
    present) and regex-matches for `{% if tools %}` / `{{ tools` markers.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from muse.core.discovery import discover_hf_plugins, _default_hf_plugin_dirs
from muse.core.resolvers import (
    Resolver,
    ResolvedModel,
    ResolverError,
    SearchResult,
    parse_uri,
    register_resolver,
)


logger = logging.getLogger(__name__)

LLAMA_CPP_RUNTIME_PATH = (
    "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
)
SENTENCE_TRANSFORMER_RUNTIME_PATH = (
    "muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel"
)

LLAMA_CPP_PIP_EXTRAS = ("llama-cpp-python>=0.2.90",)
SENTENCE_TRANSFORMER_PIP_EXTRAS = (
    "torch>=2.1.0",
    "sentence-transformers>=2.2.0",
)
FASTER_WHISPER_RUNTIME_PATH = (
    "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel"
)
FASTER_WHISPER_PIP_EXTRAS = ("faster-whisper>=1.0.0",)
FASTER_WHISPER_SYSTEM_PACKAGES = ("ffmpeg",)

TEXT_CLASSIFIER_RUNTIME_PATH = (
    "muse.modalities.text_classification.runtimes.hf_text_classifier"
    ":HFTextClassifier"
)
TEXT_CLASSIFIER_PIP_EXTRAS = ("transformers>=4.36.0", "torch>=2.1.0")
TEXT_CLASSIFIER_SYSTEM_PACKAGES = ()


class HFResolver(Resolver):
    """Resolver for hf:// URIs.

    Plugin-based dispatch: each modality contributes a hf.py exporting an
    HF_PLUGIN dict (see docs/HF_PLUGINS.md). On resolve, plugins are
    iterated in (priority, modality) order; first sniff to return True
    wins. On search, plugins are filtered by modality (or all consulted
    when no filter).

    During the migration window the legacy `_sniff_repo_shape` cascade
    runs as a fallback for modalities that have not yet shipped a
    plugin file. The fallback is removed in Task 7 once all four
    bundled modalities have migrated.
    """

    scheme = "hf"

    def __init__(self, plugins: list[dict] | None = None) -> None:
        self._api = HfApi()
        self._plugins = plugins if plugins is not None else discover_hf_plugins(
            _default_hf_plugin_dirs()
        )

    def resolve(self, uri: str) -> ResolvedModel:
        scheme, repo_id, variant = parse_uri(uri)
        if scheme != "hf":
            raise ResolverError(f"HFResolver cannot resolve scheme {scheme!r}")

        info = self._api.repo_info(repo_id)
        for plugin in self._plugins:
            if plugin["sniff"](info):
                return plugin["resolve"](repo_id, variant, info)

        # Legacy fallback: removed in Task 7 once all bundled modalities
        # have migrated.
        legacy = self._legacy_resolve(repo_id, variant, info)
        if legacy is not None:
            return legacy

        tags = getattr(info, "tags", None) or []
        siblings = [s.rfilename for s in getattr(info, "siblings", [])][:5]
        raise ResolverError(
            f"no HF plugin matched {repo_id!r}; tags={tags}, "
            f"siblings={siblings}..."
        )

    def search(self, query: str, **filters) -> Iterable[SearchResult]:
        modality = filters.get("modality")
        sort = filters.get("sort", "downloads")
        limit = filters.get("limit", 20)

        if modality is not None:
            matched = [p for p in self._plugins if p["modality"] == modality]
            if not matched:
                # Legacy fallback for not-yet-migrated modalities.
                legacy = self._legacy_search(query, modality, sort, limit)
                if legacy is not None:
                    yield from legacy
                    return
                supported = sorted(p["modality"] for p in self._plugins)
                raise ResolverError(
                    f"HFResolver.search does not support modality {modality!r}; "
                    f"supported: {supported}"
                )
        else:
            matched = self._plugins

        for plugin in matched:
            yield from plugin["search"](self._api, query, sort=sort, limit=limit)

    def _legacy_resolve(self, repo_id, variant, info):
        """Old 4-branch dispatch. Removed in Task 7."""
        shape = _sniff_repo_shape(info)
        if shape == "gguf":
            return self._resolve_gguf(repo_id, variant, info)
        if shape == "sentence-transformers":
            return self._resolve_sentence_transformer(repo_id, info)
        if shape == "faster-whisper":
            return self._resolve_faster_whisper(repo_id, info)
        if shape == "text-classification":
            return self._resolve_text_classifier(repo_id, info)
        return None

    def _legacy_search(self, query, modality, sort, limit):
        """Old per-modality search. Removed in Task 7."""
        if modality == "chat/completion":
            return self._search_gguf(query, sort=sort, limit=limit)
        if modality == "embedding/text":
            return self._search_sentence_transformers(query, sort=sort, limit=limit)
        if modality == "audio/transcription":
            return self._search_faster_whisper(query, sort=sort, limit=limit)
        if modality == "text/classification":
            return self._search_text_classifier(query, sort=sort, limit=limit)
        return None

    # --- GGUF branch ---

    def _resolve_gguf(self, repo_id: str, variant: str | None, info) -> ResolvedModel:
        gguf_files = [
            s.rfilename for s in info.siblings
            if s.rfilename.endswith(".gguf")
        ]
        if not gguf_files:
            raise ResolverError(f"no .gguf files in {repo_id}")
        if variant is None:
            variants = [_extract_variant(f) for f in gguf_files]
            raise ResolverError(
                f"variant required for GGUF repo {repo_id}; "
                f"available: {sorted(set(variants))}"
            )
        matched = _match_gguf_variant(gguf_files, variant)
        if matched is None:
            variants = [_extract_variant(f) for f in gguf_files]
            raise ResolverError(
                f"variant {variant!r} not found in {repo_id}; "
                f"available: {sorted(set(variants))}"
            )

        supports_tools = _try_sniff_tools_from_repo(self._api, repo_id)
        ctx_length = _try_sniff_context_length_from_repo(self._api, repo_id)

        # Look up curated chat-format hints (chat_formats.yaml). The
        # lookup table is the source of truth for "this model family
        # works with this llama.cpp chat handler" and lets users get
        # working tool calls without hand-editing manifests. The hints
        # also override the supports_tools sniff result when they
        # disagree (the YAML is curated; the sniff is heuristic).
        from muse.core.chat_formats import lookup_chat_format
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
            "pip_extras": list(LLAMA_CPP_PIP_EXTRAS),
            "system_packages": [],
            "capabilities": capabilities,
        }

        def _download(cache_root: Path) -> Path:
            allow_patterns = [matched, "tokenizer*", "config.json", "*.md"]
            return Path(snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=LLAMA_CPP_RUNTIME_PATH,
            download=_download,
        )

    def _search_gguf(self, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
        repos = self._api.list_models(
            search=query, filter="gguf", sort=sort, limit=limit,
        )
        for repo in repos:
            siblings = getattr(repo, "siblings", None) or []
            if not siblings:
                try:
                    # `files_metadata=True` is what makes RepoSibling.size populated.
                    # Without it, .size is always None and --max-size-gb is meaningless.
                    info = self._api.repo_info(repo.id, files_metadata=True)
                    siblings = info.siblings
                except Exception:
                    continue
            # Per-repo deduplication by variant: sharded GGUFs (model-q4_k_m-00001-of-00003.gguf)
            # and repos that publish duplicate quants emit the same @variant tag for multiple
            # files. We sum sizes across files sharing a variant (so a sharded model reports
            # its true total size) and emit one row per variant per repo.
            variant_to_size: dict[str, float] = {}
            variant_to_first_file: dict[str, str] = {}
            for s in siblings:
                if not s.rfilename.endswith(".gguf"):
                    continue
                variant = _extract_variant(s.rfilename)
                size_bytes = getattr(s, "size", None) or 0
                variant_to_size[variant] = variant_to_size.get(variant, 0) + size_bytes
                variant_to_first_file.setdefault(variant, s.rfilename)
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

    # --- Sentence-Transformers branch ---

    def _resolve_sentence_transformer(self, repo_id: str, info) -> ResolvedModel:
        manifest = {
            "model_id": _sentence_transformer_model_id(repo_id),
            "modality": "embedding/text",
            "hf_repo": repo_id,
            "description": f"Sentence-Transformers: {repo_id}",
            "license": _repo_license(info),
            "pip_extras": list(SENTENCE_TRANSFORMER_PIP_EXTRAS),
            "system_packages": [],
            "capabilities": {},
        }

        def _download(cache_root: Path) -> Path:
            return Path(snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=SENTENCE_TRANSFORMER_RUNTIME_PATH,
            download=_download,
        )

    def _search_sentence_transformers(self, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
        repos = self._api.list_models(
            search=query, filter="sentence-transformers",
            sort=sort, limit=limit,
        )
        for repo in repos:
            yield SearchResult(
                uri=f"hf://{repo.id}",
                model_id=_sentence_transformer_model_id(repo.id),
                modality="embedding/text",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo.id,
            )

    # --- Faster-Whisper branch ---

    def _resolve_faster_whisper(self, repo_id: str, info) -> ResolvedModel:
        manifest = {
            "model_id": repo_id.split("/", 1)[-1].lower(),
            "modality": "audio/transcription",
            "hf_repo": repo_id,
            "description": f"Faster-Whisper: {repo_id}",
            "license": _repo_license(info),
            "pip_extras": list(FASTER_WHISPER_PIP_EXTRAS),
            "system_packages": list(FASTER_WHISPER_SYSTEM_PACKAGES),
            "capabilities": {},
        }

        def _download(cache_root: Path) -> Path:
            return Path(snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=FASTER_WHISPER_RUNTIME_PATH,
            download=_download,
        )

    def _search_faster_whisper(self, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
        repos = self._api.list_models(
            search=query, filter="automatic-speech-recognition",
            sort=sort, limit=limit,
        )
        for repo in repos:
            yield SearchResult(
                uri=f"hf://{repo.id}",
                model_id=repo.id.split("/", 1)[-1].lower(),
                modality="audio/transcription",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo.id,
            )

    # --- Text-Classifier branch ---

    def _resolve_text_classifier(self, repo_id: str, info) -> ResolvedModel:
        manifest = {
            "model_id": repo_id.split("/", 1)[-1].lower(),
            "modality": "text/classification",
            "hf_repo": repo_id,
            "description": f"Text classifier: {repo_id}",
            "license": _repo_license(info),
            "pip_extras": list(TEXT_CLASSIFIER_PIP_EXTRAS),
            "system_packages": list(TEXT_CLASSIFIER_SYSTEM_PACKAGES),
            "capabilities": {},
        }

        def _download(cache_root: Path) -> Path:
            return Path(snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=TEXT_CLASSIFIER_RUNTIME_PATH,
            download=_download,
        )

    def _search_text_classifier(self, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
        repos = self._api.list_models(
            search=query, filter="text-classification",
            sort=sort, limit=limit,
        )
        for repo in repos:
            yield SearchResult(
                uri=f"hf://{repo.id}",
                model_id=repo.id.split("/", 1)[-1].lower(),
                modality="text/classification",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo.id,
            )


# --- sniff helpers (module-level, pytest-friendly) ---

def _looks_like_faster_whisper(siblings: list[str], tags: list[str]) -> bool:
    """CT2 faster-whisper repos have model.bin + config.json +
    (vocabulary.txt or tokenizer.json), plus the ASR tag."""
    names = {Path(f).name for f in siblings}
    has_ct2_shape = (
        "model.bin" in names
        and "config.json" in names
        and ("vocabulary.txt" in names or "tokenizer.json" in names)
    )
    has_asr_tag = "automatic-speech-recognition" in tags
    return has_ct2_shape and has_asr_tag


def _looks_like_text_classifier(siblings: list[str], tags: list[str]) -> bool:
    """HF text-classification repos carry the `text-classification` tag.
    Sibling shape varies (PyTorch / safetensors / older bin formats); we
    don't gate on file presence, only on the tag, since transformers
    handles the loading ambiguity for us at AutoModelForSequenceClassification
    time.
    """
    return "text-classification" in tags


def _sniff_repo_shape(info) -> str:
    """Return one of: 'gguf' | 'sentence-transformers' | 'faster-whisper' | 'text-classification' | 'unknown'."""
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    if any(f.endswith(".gguf") for f in siblings):
        return "gguf"
    if "sentence-transformers" in tags:
        return "sentence-transformers"
    if any(Path(f).name == "sentence_transformers_config.json" for f in siblings):
        return "sentence-transformers"
    if _looks_like_faster_whisper(siblings, tags):
        return "faster-whisper"
    if _looks_like_text_classifier(siblings, tags):
        return "text-classification"
    return "unknown"


_VARIANT_RE = re.compile(r"(q\d+_[a-z0-9_]+|iq\d+_[a-z0-9]+|f16|bf16|f32)", re.IGNORECASE)


def _extract_variant(gguf_filename: str) -> str:
    """Extract a quant tag like `q4_k_m` from e.g. `qwen3-8b-q4_k_m.gguf`."""
    stem = Path(gguf_filename).stem
    m = _VARIANT_RE.search(stem)
    return (m.group(1).lower() if m else stem).replace(".", "_")


def _match_gguf_variant(files: list[str], variant: str) -> str | None:
    """Find the file whose quant tag matches `variant` (case-insensitive)."""
    norm = variant.lower()
    for f in files:
        if _extract_variant(f) == norm:
            return f
    return None


def _gguf_model_id(repo_id: str, variant: str) -> str:
    """Synthesize a model_id like 'qwen3-8b-gguf-q4-k-m'."""
    base = repo_id.split("/", 1)[-1].lower()
    if not base.endswith("-gguf"):
        base = f"{base}-gguf"
    return f"{base}-{variant.lower().replace('_', '-')}"


def _sentence_transformer_model_id(repo_id: str) -> str:
    """Synthesize a model_id from the repo name (lowercased)."""
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _try_sniff_tools_from_repo(api: HfApi, repo_id: str) -> bool | None:
    """Try to read tokenizer_config.json and check for tool-calling template.

    Returns True / False when the file is present; None when it isn't.
    Any network / parse error returns None silently.
    """
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer_config.json",
        )
    except Exception:
        return None
    try:
        cfg = json.loads(Path(path).read_text())
    except Exception:
        return None
    return _sniff_supports_tools(cfg.get("chat_template"))


def _sniff_supports_tools(chat_template: str | None) -> bool:
    if not chat_template or not isinstance(chat_template, str):
        return False
    return bool(re.search(r"(\bif\s+tools\b|\{\{\s*tools|tool_calls)", chat_template))


def _try_sniff_context_length_from_repo(api: HfApi, repo_id: str) -> int | None:
    """Best-effort: read config.json's `max_position_embeddings`.

    GGUF files carry their own context length in header metadata, which
    llama-cpp-python respects at load time. This sniff is just for
    display in /v1/models; runtime truth is in the GGUF.
    """
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
        cfg = json.loads(Path(path).read_text())
        return int(cfg.get("max_position_embeddings") or 0) or None
    except Exception:
        return None


# Register on import so `from muse.core import resolvers_hf` is enough.
register_resolver(HFResolver())
