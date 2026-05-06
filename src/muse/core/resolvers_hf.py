"""HuggingFace Hub resolver.

URI shapes:
  hf://org/repo                  # sentence-transformers (embedding/text)
  hf://org/repo-GGUF@<variant>   # GGUF (chat/completion); variant required
  hf://org/faster-whisper-tiny   # CT2 faster-whisper (audio/transcription)
  hf://org/Text-Moderation       # text-classification (text/classification)

All four bundled modalities ship per-modality hf.py plugins. The resolver
itself is a thin dispatcher: it sniffs each plugin in (priority, modality)
order on resolve, and filters by modality on search.
"""
from __future__ import annotations

from typing import Iterable

from huggingface_hub import HfApi

from muse.core.discovery import discover_hf_plugins, _default_hf_plugin_dirs
from muse.core.resolvers import (
    Resolver,
    ResolvedModel,
    ResolverError,
    SearchResult,
    parse_uri,
    register_resolver,
)


class HFResolver(Resolver):
    """Resolver for hf:// URIs.

    Plugin-based dispatch: each modality contributes a hf.py exporting an
    HF_PLUGIN dict (see docs/HF_PLUGINS.md). On resolve, plugins are
    iterated in (priority, modality) order; first sniff to return True
    wins. On search, plugins are filtered by modality (or all consulted
    when no filter).
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

        tags = getattr(info, "tags", None) or []
        siblings = [s.rfilename for s in getattr(info, "siblings", [])][:5]
        raise ResolverError(
            f"no HF plugin matched {repo_id!r}; tags={tags}, "
            f"siblings={siblings}..."
        )

    def resolve_via_modality(self, uri: str, modality: str) -> ResolvedModel:
        """Resolve a URI through the plugin for the named modality,
        bypassing priority-based sniff dispatch.

        Used when curated.yaml declares a `modality:` field for a URI
        that the priority-based resolve would otherwise misclassify.
        Reranker repos (BAAI/bge-reranker-base) are sentence-transformers
        models so the embedding/text plugin's sniff returns True; the
        text/rerank plugin needs to win when the curated entry says so.

        Returns the chosen plugin's resolved model. Raises ResolverError
        when no plugin claims the named modality.
        """
        scheme, repo_id, variant = parse_uri(uri)
        if scheme != "hf":
            raise ResolverError(f"HFResolver cannot resolve scheme {scheme!r}")

        for plugin in self._plugins:
            if plugin["modality"] == modality:
                info = self._api.repo_info(repo_id)
                return plugin["resolve"](repo_id, variant, info)

        supported = sorted({p["modality"] for p in self._plugins})
        raise ResolverError(
            f"no HF plugin for modality {modality!r}; "
            f"registered: {supported}"
        )

    def search(self, query: str, **filters) -> Iterable[SearchResult]:
        modality = filters.get("modality")
        sort = filters.get("sort", "downloads")
        limit = filters.get("limit", 20)

        if modality is not None:
            matched = [p for p in self._plugins if p["modality"] == modality]
            if not matched:
                supported = sorted(p["modality"] for p in self._plugins)
                raise ResolverError(
                    f"HFResolver.search does not support modality {modality!r}; "
                    f"supported: {supported}"
                )
        else:
            matched = self._plugins

        for plugin in matched:
            yield from plugin["search"](self._api, query, sort=sort, limit=limit)


# Register on import so `from muse.core import resolvers_hf` is enough.
register_resolver(HFResolver())
