"""HuggingFace Hub resolver.

URI shapes:
  hf://org/repo                  # sentence-transformers (embedding/text)
  hf://org/repo-GGUF@<variant>   # GGUF (chat/completion); variant required
  hf://org/faster-whisper-tiny   # CT2 faster-whisper (audio/transcription)
  hf://org/Text-Moderation       # text-classification (text/classification)

Sniff logic (see `_sniff_repo_shape`):
  - sentence-transformers tag  -> sentence-transformers
  - sentence_transformers_config.json sibling -> sentence-transformers
  - model.bin + config.json + (vocabulary.txt|tokenizer.json) + ASR tag -> faster-whisper
  - text-classification tag    -> text-classification
  - else                       -> unknown (raises on resolve)

Note: GGUF dispatch was moved to `muse.modalities.chat_completion.hf`
as part of the per-modality HF plugin refactor. The plugin sniffs
.gguf siblings before this legacy fallback runs.

Search:
  - modality="embedding/text": HfApi.list_models(filter="sentence-transformers")
  - modality="audio/transcription": HfApi.list_models(filter="automatic-speech-recognition")
  - modality="text/classification": HfApi.list_models(filter="text-classification")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

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

SENTENCE_TRANSFORMER_RUNTIME_PATH = (
    "muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel"
)

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
        """Legacy per-shape dispatch. Removed in Task 7."""
        shape = _sniff_repo_shape(info)
        if shape == "sentence-transformers":
            return self._resolve_sentence_transformer(repo_id, info)
        if shape == "faster-whisper":
            return self._resolve_faster_whisper(repo_id, info)
        if shape == "text-classification":
            return self._resolve_text_classifier(repo_id, info)
        return None

    def _legacy_search(self, query, modality, sort, limit):
        """Old per-modality search. Removed in Task 7."""
        if modality == "embedding/text":
            return self._search_sentence_transformers(query, sort=sort, limit=limit)
        if modality == "audio/transcription":
            return self._search_faster_whisper(query, sort=sort, limit=limit)
        if modality == "text/classification":
            return self._search_text_classifier(query, sort=sort, limit=limit)
        return None

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
    """Return one of: 'sentence-transformers' | 'faster-whisper' | 'text-classification' | 'unknown'."""
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    if "sentence-transformers" in tags:
        return "sentence-transformers"
    if any(Path(f).name == "sentence_transformers_config.json" for f in siblings):
        return "sentence-transformers"
    if _looks_like_faster_whisper(siblings, tags):
        return "faster-whisper"
    if _looks_like_text_classifier(siblings, tags):
        return "text-classification"
    return "unknown"


def _sentence_transformer_model_id(repo_id: str) -> str:
    """Synthesize a model_id from the repo name (lowercased)."""
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


# Register on import so `from muse.core import resolvers_hf` is enough.
register_resolver(HFResolver())
