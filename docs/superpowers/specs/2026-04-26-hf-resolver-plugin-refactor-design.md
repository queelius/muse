# Per-modality HF resolver plugins (#129)

**Date:** 2026-04-26
**Closes:** task #129
**Driver:** the next sprint adds 6+ modalities (audio/generation, image/description, image/ocr, image/segmentation, text/translation, text/summarization, etc.). Today's HF resolver gates each modality through a hardcoded if/elif cascade in `src/muse/core/resolvers_hf.py`; expanding it linearly would add ~3 methods + 1 sniff branch + 1 search branch per modality, in a file that already knows about every other modality. Refactor first; modalities second.

## Goal

Replace `resolvers_hf.py`'s centralized 4-branch dispatch with a discovery-driven plugin pattern: each modality contributes a `hf.py` next to its `__init__.py`; the resolver iterates plugins by priority. Adding a new modality with HF support becomes "drop one file"; the resolver itself is closed for modification.

## Non-goals

- Generic resolver-plugin framework supporting non-HF resolvers (ollama, replicate, civitai). The refactor keeps `resolvers_hf.py` HF-specific. Future schemes get sibling plugin files (`ollama.py`, etc.) with their own discovery; cross-resolver abstraction is **not** in scope here.
- Changing the public `Resolver` ABC, the `parse_uri` contract, or the `ResolvedModel` / `SearchResult` dataclasses.
- Behavior changes for end users beyond one explicit case (search-without-modality-filter; see "Dispatch" below).

## Constraints

1. **Bare-install contract.** `muse pull hf://...` must work on `pip install muse` with no `[server]` extras (no fastapi, no torch). This forces plugin loading to bypass the modality package's `__init__.py` (which transitively imports fastapi via `routes.py`). Solved with single-file `spec_from_file_location` loading, mirroring `_load_script` for external user model scripts.
2. **Discovery never raises.** Malformed plugins log + skip. The resolver must always start, even if a plugin file is broken.
3. **First-found-wins on collisions.** Matches `discover_modalities` semantics. Same-priority plugins resolve by sorted modality dir name (deterministic tiebreak).
4. **No new abstract base classes** as plugin contracts. Muse's instinct is "data + callables, not objects" (see `MANIFEST` in model scripts; `MODALITY` + `build_router` in modality packages). The HF plugin follows that shape: a single dict with required keys.

## Architecture

```
muse pull hf://Org/Repo[@variant]
    |
    v
catalog._pull_via_resolver
    |
    v
resolvers.resolve(uri)
    |
    v
HFResolver.resolve(uri)   <-- refactored: thin dispatcher
    |
    | iterates self._plugins (sorted by priority)
    | first plugin whose sniff(info) returns True wins
    v
plugin["resolve"](repo_id, variant, info)  <-- lives in modalities/<name>/hf.py
    |
    v
ResolvedModel(manifest, backend_path, download)
```

`HFResolver._plugins` is populated once at construction by `discover_hf_plugins(default_dirs())`. Discovery walks the bundled `modalities/` tree plus `$MUSE_MODALITIES_DIR`, loads each `hf.py` as a single-file module, validates the `HF_PLUGIN` dict, and returns the sorted list.

## Plugin contract

Each modality's `src/muse/modalities/<name>/hf.py` exports a top-level `HF_PLUGIN: dict` with these keys:

| Key | Type | Required | Purpose |
|---|---|---|---|
| `modality` | `str` | yes | MIME tag (must match the modality's `MODALITY` constant) |
| `runtime_path` | `str` | yes | `"muse.modalities.X.runtimes.Y:Cls"`; consumed by catalog at `load_backend` time |
| `pip_extras` | `tuple[str, ...]` | yes | Per-model venv `pip install` args |
| `system_packages` | `tuple[str, ...]` | yes (may be empty) | Apt/brew packages logged as missing if absent from PATH |
| `priority` | `int` | yes | Lower checked first; default convention is `100` for specific plugins, `200`+ for tag-only catch-alls |
| `sniff` | `Callable[[info], bool]` | yes | Pure function; True iff this plugin claims `info` |
| `resolve` | `Callable[[repo_id, variant, info], ResolvedModel]` | yes | Build the manifest + download closure |
| `search` | `Callable[[api, query, *, sort, limit], Iterable[SearchResult]]` | yes | Yield search rows for this modality |

Required-key validation at discovery time uses the same shape as `REQUIRED_MANIFEST_KEYS` in `discover_models`: a tuple, looped, missing keys → log + skip.

Authoring constraints (documented in `docs/MODEL_SCRIPTS.md`-equivalent for HF plugins):

```python
# modalities/audio_transcription/hf.py (GOOD)
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pathlib import Path
from muse.core.resolvers import ResolvedModel, SearchResult

# modalities/audio_transcription/hf.py (BAD)
from .protocol import TranscriptionResult       # relative import → needs package init
from muse.modalities.audio_transcription.codec  # absolute sibling → needs package init
import torch                                    # heavy dep; bare-install breaks
```

The author rule: `hf.py` may import stdlib, `huggingface_hub`, `muse.core.*`. Anything else risks the bare-install contract.

## Discovery

New module-level function in `muse.core.discovery`:

```python
def discover_hf_plugins(dirs: list[Path]) -> list[dict]:
    """Scan dirs in order; return sorted list of HF_PLUGIN dicts.

    Each dir is treated as a modalities-tree root. For each subdir:
      <root>/<name>/hf.py  →  load as single-file module via
                              spec_from_file_location (mangled name);
                              read HF_PLUGIN; validate required keys;
                              skip with warning on any failure.

    Plugins are returned sorted by (priority, modality), lower priority
    checked first, deterministic tiebreak by modality tag for stable
    test snapshots.

    Discovery never raises: a malformed hf.py logs + skips, so the
    resolver still starts with the working plugins.
    """
```

Required keys validation uses a module-level constant:

```python
REQUIRED_HF_PLUGIN_KEYS = (
    "modality", "runtime_path", "pip_extras", "system_packages",
    "priority", "sniff", "resolve", "search",
)
```

Default scan dirs (used by `HFResolver.__init__` when no override):

```python
def _default_hf_plugin_dirs() -> list[Path]:
    bundled = Path(__file__).resolve().parents[1] / "modalities"
    env = os.environ.get("MUSE_MODALITIES_DIR")
    return [bundled] + ([Path(env)] if env else [])
```

Mangled module name pattern (for sys.modules disambiguation):

```python
mod_name = f"_muse_hf_plugin_{modality_dir_name}"
```

## Dispatch

```python
class HFResolver(Resolver):
    scheme = "hf"

    def __init__(self, plugins: list[dict] | None = None) -> None:
        self._api = HfApi()
        # Inject for tests; default to disk discovery.
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
        raise ResolverError(
            f"no HF plugin matched {repo_id!r}; tags={tags}, "
            f"siblings={[s.rfilename for s in info.siblings][:5]}..."
        )

    def search(self, query: str, **filters) -> Iterable[SearchResult]:
        modality = filters.get("modality")
        sort = filters.get("sort", "downloads")
        limit = filters.get("limit", 20)

        if modality is not None:
            matched = [p for p in self._plugins if p["modality"] == modality]
            if not matched:
                raise ResolverError(
                    f"HFResolver.search does not support modality {modality!r}; "
                    f"supported: {sorted(p['modality'] for p in self._plugins)}"
                )
        else:
            matched = self._plugins

        for plugin in matched:
            yield from plugin["search"](self._api, query, sort=sort, limit=limit)
```

### Behavior change: search-with-no-modality-filter

Today's `search(query)` with no `modality` filter iterates only `gguf` and `sentence-transformers`, hardcoded. After the refactor, it iterates **all** discovered plugins. Documented in CHANGELOG; tests assert the broader sweep. Justification: today's hardcoded subset was incidental (the first two modalities to land); users running `muse search foo` reasonably expect "search everything muse knows about." The volume increase is bounded by N (number of modalities), and each plugin's `search()` is one HfApi call.

## What stays in `resolvers_hf.py`

After the refactor, the file contains:

- The `HFResolver` class (constructor + dispatcher; ~50 lines).
- Module-level `register_resolver(HFResolver())` call.
- Modality-agnostic helpers used by multiple plugins:
  - `_repo_license(info)`: reads `info.card_data.license`.

That's it. Everything else moves.

## What moves out

To `modalities/chat_completion/hf.py`:
- `_resolve_gguf`, `_search_gguf`
- `_extract_variant`, `_match_gguf_variant`, `_gguf_model_id`, `_VARIANT_RE`
- `_try_sniff_tools_from_repo`, `_sniff_supports_tools`, `_try_sniff_context_length_from_repo`
- `LLAMA_CPP_RUNTIME_PATH`, `LLAMA_CPP_PIP_EXTRAS`
- The `chat_formats.yaml` lookup integration

To `modalities/embedding_text/hf.py`:
- `_resolve_sentence_transformer`, `_search_sentence_transformers`
- `_sentence_transformer_model_id`
- `SENTENCE_TRANSFORMER_RUNTIME_PATH`, `SENTENCE_TRANSFORMER_PIP_EXTRAS`

To `modalities/audio_transcription/hf.py`:
- `_resolve_faster_whisper`, `_search_faster_whisper`, `_looks_like_faster_whisper`
- `FASTER_WHISPER_RUNTIME_PATH`, `FASTER_WHISPER_PIP_EXTRAS`, `FASTER_WHISPER_SYSTEM_PACKAGES`

To `modalities/text_classification/hf.py`:
- `_resolve_text_classifier`, `_search_text_classifier`, `_looks_like_text_classifier`
- `TEXT_CLASSIFIER_RUNTIME_PATH`, `TEXT_CLASSIFIER_PIP_EXTRAS`, `TEXT_CLASSIFIER_SYSTEM_PACKAGES`

`_sniff_repo_shape` is **deleted entirely**. Its job is now handled by per-plugin `sniff` functions iterated in priority order.

## Initial plugin priorities

Encoded so that "specific shapes" win over "tag-only" catch-alls:

| Modality | Priority | Reason |
|---|---|---|
| `chat/completion` (GGUF) | 100 | File-pattern match (`.gguf` siblings); never overlaps |
| `audio/transcription` (faster-whisper) | 100 | File-pattern + ASR tag; very specific |
| `embedding/text` (sentence-transformers) | 110 | Tag OR config-file match; could in principle conflict with text-classification on multi-tag repos |
| `text/classification` | 200 | Tag-only; broadest catch-all |

Future modalities slot in: `image/segmentation`, `image/ocr`, `image/description` likely 100-110 (file-pattern + tag); `text/translation`, `text/summarization` likely 200 (tag-only).

## Migration order

Two phases. Each phase ends with a green `pytest -m "not slow"`.

**Phase 1: introduce the infrastructure with feature flag.**
1. Add `discover_hf_plugins` to `muse.core.discovery`. Tests for valid/invalid plugins, priority sort, deterministic tiebreak.
2. Document the contract in a new `docs/HF_PLUGINS.md` (similar to `docs/MODEL_SCRIPTS.md`).
3. Refactor `HFResolver.__init__` to accept `plugins=` injection (default: empty list, meaning no plugins yet).
4. Add a fallback branch: if no plugin matches, fall through to the existing `_sniff_repo_shape` cascade. **Both code paths run**; unit tests cover both. Commit.

**Phase 2: migrate modalities one at a time.**
5. `modalities/chat_completion/hf.py` (GGUF). Copy `_resolve_gguf` / `_search_gguf` / sniff / helpers. Bind into `HF_PLUGIN`. Confirm the GGUF code path now goes through plugins (not the legacy fallback) by removing the GGUF branch from `_sniff_repo_shape`. Tests green. Commit.
6. `modalities/embedding_text/hf.py`. Same pattern. Remove ST branch from `_sniff_repo_shape`. Commit.
7. `modalities/audio_transcription/hf.py`. Same. Remove faster-whisper branch. Commit.
8. `modalities/text_classification/hf.py`. Same. Remove text-classification branch. Commit.

**Phase 3: cleanup.**
9. Delete `_sniff_repo_shape` (now empty). Delete the legacy fallback branch in `HFResolver.resolve`. Delete the corresponding constants/helpers from `resolvers_hf.py`. Tests green. Commit.
10. Update CLAUDE.md, README.md to reference the plugin pattern instead of "resolvers_hf.py knows about each modality."

After each migration step, the system is fully working: only the dispatch path changes for that one modality. If a step breaks, only that step needs revision.

## Tests

New test files:

**`tests/core/test_hf_plugin_discovery.py`**: discovery semantics.
- Valid `HF_PLUGIN` dict in a temp dir → discovered with all keys.
- Missing required key → logged + skipped.
- Wrong type (e.g. `priority: "high"`) → logged + skipped.
- Two plugins, different priorities → sorted ascending.
- Two plugins, same priority → tiebreak by modality dir name (alphabetical).
- Malformed `hf.py` (SyntaxError) → logged + skipped, others still discovered.
- `$MUSE_MODALITIES_DIR` adds external plugins.

**`tests/core/test_hf_resolver_dispatch.py`**: resolver dispatch logic.
- Plugin injection via `HFResolver(plugins=[...])` for hermetic tests.
- First plugin whose `sniff` returns True wins; later plugins not consulted.
- No plugin matches → `ResolverError` with helpful message.
- `search(query, modality=X)` only consults plugins with that modality.
- `search(query)` (no filter) consults all plugins.
- `search(query, modality=unknown)` → `ResolverError`.

**`tests/modalities/<name>/test_hf_plugin.py`**: one per migrated modality.
- `HF_PLUGIN` dict has all required keys with valid types.
- `sniff(positive_info)` → True; `sniff(negative_info)` → False (synthetic info objects).
- `resolve(repo_id, variant, info)` → `ResolvedModel` with expected `manifest` shape, `backend_path`, callable `download`.
- `search(api, query, sort=..., limit=...)` yields `SearchResult` instances with the right modality tag.

Existing `tests/core/test_resolvers_hf.py` shrinks: tests of the per-modality logic move to `tests/modalities/*/test_hf_plugin.py`. What stays: `parse_uri` round-trips, `register_resolver` registration, `HFResolver` scheme matching.

## Documentation deliverables

- `docs/HF_PLUGINS.md` (new): authoring guide. Plugin contract, the import constraints (no relative imports, no fastapi, no torch), priority conventions, an end-to-end example.
- `docs/RESOLVERS.md` (update): cross-link to HF_PLUGINS.md; remove the prose describing `_sniff_repo_shape`'s 4-branch dispatch.
- `CLAUDE.md` (update): the "Adding a new modality" section now says "drop a `hf.py` next to your `__init__.py`" instead of "add a branch to `resolvers_hf.py`".

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| A plugin's `hf.py` accidentally imports fastapi (e.g., via `from .codec import ...`) → bare-install breaks at `muse pull` | CI smoke test with bare install (`pip install muse` only, no extras) running `muse pull` against a known repo; tracked separately as #124 |
| Discovery picks up a stale/orphan `hf.py` left by a removed modality | First-found-wins; explicit `priority` makes precedence inspectable; `muse search` and `muse pull` work the same regardless |
| Search-no-modality-filter fanout → N httpx calls per `muse search` | Bounded by modality count (~6 today, ~12 long-term); each call is parallel-friendly; can move to `asyncio.gather` later if needed |
| Plugin imports collide in `sys.modules` | Mangled name `_muse_hf_plugin_<dir>` per plugin; same trick `_load_script` uses for external scripts |

## Out of scope (filed for later)

- Generic resolver-plugin framework supporting non-HF schemes (ollama, replicate). Each scheme would get its own dispatcher + plugin contract; cross-scheme abstraction is premature.
- Plugin-level caching of `info` objects across `sniff` calls within one resolve (currently each plugin re-reads `info.tags`/`info.siblings`; cheap given they're already-fetched data, but a perf trace could find inefficiencies).
- Async plugins: today everything is sync. If a plugin's `search` ever needs to fan out to multiple HF endpoints, switching to `async def` would be a contract change.
