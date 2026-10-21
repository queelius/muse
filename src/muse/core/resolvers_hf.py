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

from collections.abc import Mapping, Sequence
from dataclasses import replace
import inspect
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
    COMMIT_RE,
    hf_commit_revision,
    parse_uri,
    register_resolver,
)

logger = logging.getLogger(__name__)


def _accepts_kwarg(method, name: str) -> bool:
    """Whether ``method`` declares parameter ``name`` or a **kwargs catch-all.

    Lets optional kwargs like ``base_override`` reach only the plugins that
    opted in, while plain ``resolve(repo_id, variant, info)`` callables keep
    working.
    """
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name
        for param in sig.parameters.values()
    )


def _validated_hf_result(
    resolved: ResolvedModel,
    *,
    repo_id: str,
    revision: str,
) -> ResolvedModel:
    """Bind a plugin result to the exact Hub metadata commit and receipt."""
    if not isinstance(resolved, ResolvedModel):
        raise ResolverError("HF plugin returned an invalid resolved-model object")
    manifest = resolved.manifest
    if not isinstance(manifest, dict):
        raise ResolverError("HF plugin returned a non-object manifest")
    if manifest.get("hf_repo") != repo_id:
        raise ResolverError(
            f"HF plugin resolved an unexpected repository: expected {repo_id!r}, "
            f"got {manifest.get('hf_repo')!r}"
        )
    if manifest.get("revision") != revision:
        raise ResolverError(
            f"HF plugin did not preserve the resolved immutable commit for "
            f"{repo_id!r}: expected {revision}, got {manifest.get('revision')!r}"
        )

    raw_receipt: object = resolved.artifact_provenance or ({
        "repo_id": repo_id,
        "revision": revision,
        "subdir": ".",
    },)
    if isinstance(raw_receipt, (str, bytes)) or not isinstance(raw_receipt, Sequence):
        raise ResolverError("HF plugin artifact provenance must be a sequence")
    receipt: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(raw_receipt):
        if not isinstance(raw, Mapping):
            raise ResolverError(f"HF artifact provenance item {index} is not an object")
        unknown = set(raw) - {
            "repo_id",
            "revision",
            "subdir",
            "allow_patterns",
            "required_patterns",
        }
        if unknown:
            raise ResolverError(
                f"HF artifact provenance item {index} has unknown keys: "
                f"{sorted(unknown)}"
            )
        item_repo = raw.get("repo_id")
        item_revision = raw.get("revision")
        subdir = raw.get("subdir", ".")
        if (
            not isinstance(item_repo, str)
            or item_repo.count("/") != 1
            or any(char.isspace() for char in item_repo)
        ):
            raise ResolverError(f"HF artifact provenance item {index} has invalid repo_id")
        if (
            not isinstance(item_revision, str)
            or COMMIT_RE.fullmatch(item_revision) is None
        ):
            raise ResolverError(
                f"HF artifact provenance item {index} lacks an immutable commit"
            )
        if (
            not isinstance(subdir, str)
            or not subdir
            or "/" in subdir
            or "\\" in subdir
            or subdir == ".."
        ):
            raise ResolverError(f"HF artifact provenance item {index} has invalid subdir")
        canonical: dict[str, object] = {
            "repo_id": item_repo,
            "revision": item_revision,
            "subdir": subdir,
        }
        for field_name in ("allow_patterns", "required_patterns"):
            patterns = raw.get(field_name)
            if patterns is None:
                continue
            if (
                isinstance(patterns, (str, bytes))
                or not isinstance(patterns, Sequence)
                or not all(isinstance(value, str) and value for value in patterns)
            ):
                raise ResolverError(
                    f"HF artifact provenance item {index} has invalid {field_name}"
                )
            canonical[field_name] = list(patterns)
        identity = (item_repo, item_revision, subdir)
        if identity in seen:
            raise ResolverError(f"duplicate HF artifact provenance item {index}")
        seen.add(identity)
        receipt.append(canonical)
    if not any(
        item["repo_id"] == repo_id and item["revision"] == revision
        for item in receipt
    ):
        raise ResolverError("HF artifact provenance omits the resolved primary repository")
    allow_patterns = resolved.allow_patterns
    if allow_patterns is not None:
        if (
            isinstance(allow_patterns, (str, bytes))
            or not isinstance(allow_patterns, Sequence)
            or not all(isinstance(value, str) and value for value in allow_patterns)
        ):
            raise ResolverError("HF plugin returned invalid allow_patterns")
        allow_patterns = tuple(allow_patterns)
    return replace(
        resolved,
        artifact_provenance=tuple(receipt),
        allow_patterns=allow_patterns,
    )

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

    def _repo_info(self, repo_id: str, *, revision: str | None = None):
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
                if revision is None:
                    return self._api.repo_info(repo_id)
                return self._api.repo_info(repo_id, revision=revision)
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

    def resolve(
        self,
        uri: str,
        *,
        modality: str | None = None,
        base_override: str | None = None,
        revision: str | None = None,
    ) -> ResolvedModel:
        """Resolve ``uri`` through the first plugin whose sniff matches.

        A ``modality`` (from a curated entry) selects that modality's
        plugin directly, bypassing sniff: reranker repos such as
        BAAI/bge-reranker-base are sentence-transformers models that the
        embedding/text plugin would otherwise claim. ``base_override`` is
        forwarded only to plugin resolve callables that accept it.
        """
        scheme, repo_id, variant = parse_uri(uri)
        if scheme != "hf":
            raise ResolverError(f"HFResolver cannot resolve scheme {scheme!r}")
        candidates = [
            plugin for plugin in self._plugins
            if modality is None or plugin["modality"] == modality
        ]
        if not candidates:
            supported = sorted({p["modality"] for p in self._plugins})
            raise ResolverError(
                f"no HF plugin for modality {modality!r}; "
                f"registered: {supported}"
            )

        info = self._repo_info(repo_id, revision=revision)
        resolved_revision = hf_commit_revision(info)
        if resolved_revision is None:
            raise ResolverError(
                f"Hugging Face did not return an immutable 40-character "
                f"commit for {repo_id!r}"
            )
        if revision is not None and resolved_revision != revision:
            raise ResolverError(
                f"Hugging Face resolved {repo_id!r} to an unexpected commit; "
                f"requested {revision}, got {resolved_revision!r}"
            )
        plugin = next(
            (p for p in candidates if modality is not None or p["sniff"](info)),
            None,
        )
        if plugin is None:
            tags = getattr(info, "tags", None) or []
            siblings = [s.rfilename for s in getattr(info, "siblings", [])][:5]
            raise ResolverError(
                f"no HF plugin matched {repo_id!r}; tags={tags}, "
                f"siblings={siblings}..."
            )

        resolve_fn = plugin["resolve"]
        if base_override is not None and _accepts_kwarg(resolve_fn, "base_override"):
            resolved = resolve_fn(repo_id, variant, info, base_override=base_override)
        else:
            resolved = resolve_fn(repo_id, variant, info)
        return _validated_hf_result(
            resolved,
            repo_id=repo_id,
            revision=resolved_revision,
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
