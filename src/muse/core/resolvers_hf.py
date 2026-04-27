"""HuggingFace Hub resolver.

URI shapes:
  hf://org/repo                  # sentence-transformers (embedding/text)
  hf://org/repo-GGUF@<variant>   # GGUF (chat/completion); variant required
  hf://org/faster-whisper-tiny   # CT2 faster-whisper (audio/transcription)
  hf://org/Text-Moderation       # text-classification (text/classification)

All four bundled modalities now ship per-modality hf.py plugins. The
legacy fallback methods (`_legacy_resolve`, `_legacy_search`,
`_sniff_repo_shape`) are empty no-op dispatchers and are removed in
Task 7.
"""
from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


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
        return None

    def _legacy_search(self, query, modality, sort, limit):
        """Old per-modality search. Removed in Task 7."""
        return None


# --- sniff helpers (module-level, pytest-friendly) ---

def _sniff_repo_shape(info) -> str:
    """Return 'unknown' for everything; all shapes now match via plugins.
    Removed in Task 7."""
    return "unknown"


# Register on import so `from muse.core import resolvers_hf` is enough.
register_resolver(HFResolver())
