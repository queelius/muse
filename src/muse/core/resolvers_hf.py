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

import logging
import time
from typing import Iterable

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

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

# Transient-failure retry for Hub metadata fetches. repo_info() can fail
# under rapid calls (rate-limit 429, 5xx, flaky socket) or return a
# partial/malformed response that makes huggingface_hub itself raise an
# arbitrary low-level error (observed: TypeError from formatting a None
# field). These are transient and worth a bounded retry; a missing/gated
# repo is NOT and surfaces immediately.
_REPO_INFO_MAX_ATTEMPTS = 3
_REPO_INFO_BACKOFF_BASE = 0.5  # seconds; exponential (0.5, 1.0, ...)


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

    def _repo_info(self, repo_id: str):
        """Fetch Hub repo metadata, resilient to transient failures.

        repo_info() raises for two very different reasons:
          - deterministic + meaningful: the repo is missing or gated
            (RepositoryNotFoundError, which GatedRepoError subclasses).
            Surface immediately so the caller sees the real reason.
          - transient: rate-limit/429, 5xx, a flaky socket, or a malformed
            partial response that makes huggingface_hub raise an arbitrary
            low-level error internally (e.g. TypeError from formatting a
            None field). Retry a bounded number of times with backoff,
            then raise a clear, retryable ResolverError instead of leaking
            the raw exception to the user.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _REPO_INFO_MAX_ATTEMPTS + 1):
            try:
                return self._api.repo_info(repo_id)
            except RepositoryNotFoundError:
                raise  # missing/gated: meaningful, deterministic, do not mask
            except Exception as exc:  # noqa: BLE001 - transient/unexpected
                last_exc = exc
                if attempt < _REPO_INFO_MAX_ATTEMPTS:
                    delay = _REPO_INFO_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.debug(
                        "repo_info(%s) attempt %d/%d failed (%s); retrying in %.1fs",
                        repo_id, attempt, _REPO_INFO_MAX_ATTEMPTS,
                        type(exc).__name__, delay,
                    )
                    time.sleep(delay)
        raise ResolverError(
            f"failed to fetch Hub metadata for {repo_id!r} after "
            f"{_REPO_INFO_MAX_ATTEMPTS} attempts; the Hub may be rate-limiting "
            f"or temporarily unavailable, retry shortly "
            f"({type(last_exc).__name__}: {last_exc})"
        ) from last_exc

    def resolve(self, uri: str) -> ResolvedModel:
        scheme, repo_id, variant = parse_uri(uri)
        if scheme != "hf":
            raise ResolverError(f"HFResolver cannot resolve scheme {scheme!r}")

        info = self._repo_info(repo_id)
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
                info = self._repo_info(repo_id)
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
