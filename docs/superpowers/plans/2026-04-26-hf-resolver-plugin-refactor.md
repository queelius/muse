# HF Resolver Plugin Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `resolvers_hf.py`'s centralized 4-branch dispatch with a discovery-driven plugin pattern. Each modality contributes a `hf.py` next to its `__init__.py` exporting an `HF_PLUGIN: dict`. The resolver iterates plugins by priority; first sniff to return True wins. Adding a modality with HF support becomes "drop one file."

**Architecture:** New `discover_hf_plugins(dirs)` in `muse.core.discovery` walks `<root>/<name>/hf.py`, loads each as a single-file module via `spec_from_file_location` (bypasses package init, preserves the bare-install contract), validates required keys, returns sorted list. `HFResolver.__init__(plugins=)` takes injection (default: disk discovery). `HFResolver.resolve` and `HFResolver.search` iterate plugins. Migration phased so each commit is green.

**Tech Stack:** Python stdlib (`importlib.util`, `ast`), pytest, `huggingface_hub` (existing).

**Spec:** `docs/superpowers/specs/2026-04-26-hf-resolver-plugin-refactor-design.md`

**Target version:** v0.15.0 (structural change paving the way for more modalities; no user-visible breaking changes; one documented behavior change: `muse search foo` with no modality filter now iterates all plugins).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/core/discovery.py` | modify | +`REQUIRED_HF_PLUGIN_KEYS`, +`discover_hf_plugins`, +`_default_hf_plugin_dirs`, +`_load_hf_plugin_script` |
| `src/muse/core/resolvers_hf.py` | modify | Refactor `HFResolver` to plugin dispatch with legacy fallback; remove sniff branches + per-modality methods one at a time |
| `src/muse/modalities/chat_completion/hf.py` | create | GGUF plugin: sniff/resolve/search + variant helpers + `chat_formats.yaml` integration |
| `src/muse/modalities/embedding_text/hf.py` | create | sentence-transformers plugin: sniff/resolve/search |
| `src/muse/modalities/audio_transcription/hf.py` | create | faster-whisper plugin: sniff/resolve/search + CT2 shape detection |
| `src/muse/modalities/text_classification/hf.py` | create | text-classification plugin: sniff/resolve/search (tag-only catch-all) |
| `tests/core/test_hf_plugin_discovery.py` | create | discover_hf_plugins semantics: validation, priority sort, tiebreak |
| `tests/core/test_hf_resolver_dispatch.py` | create | dispatcher logic with injected plugins |
| `tests/modalities/chat_completion/test_hf_plugin.py` | create | GGUF plugin contract |
| `tests/modalities/embedding_text/test_hf_plugin.py` | create | ST plugin contract |
| `tests/modalities/audio_transcription/test_hf_plugin.py` | create | faster-whisper plugin contract |
| `tests/modalities/text_classification/test_hf_plugin.py` | create | text-classifier plugin contract |
| `tests/core/test_resolvers_hf.py` | modify | Shrink: per-modality logic moves out; keep parse_uri, scheme, registration tests |
| `docs/HF_PLUGINS.md` | create | Authoring guide for HF plugins |
| `docs/RESOLVERS.md` | modify | Cross-link HF_PLUGINS.md; remove `_sniff_repo_shape` prose |
| `CLAUDE.md` | modify | "Adding a new modality" section now points at hf.py drop |
| `README.md` | modify | (light) Mention plugin pattern in the "three ways to add a model" section |
| `pyproject.toml` | modify | version bump 0.14.2 to 0.15.0 |

---

## Task 1: discover_hf_plugins infrastructure

Lays the foundation. No callers yet; feeds Task 2.

**Files:**
- Modify: `src/muse/core/discovery.py`
- Test: `tests/core/test_hf_plugin_discovery.py`

- [ ] **Step 1: Write the failing test (valid plugin discovery)**

Create `tests/core/test_hf_plugin_discovery.py`:

```python
"""Tests for discover_hf_plugins: per-modality HF plugin loader."""
from pathlib import Path

import pytest

from muse.core.discovery import discover_hf_plugins, REQUIRED_HF_PLUGIN_KEYS


def _write_plugin(modality_dir: Path, plugin_dict_literal: str) -> None:
    """Helper: write a hf.py with a given HF_PLUGIN literal into a temp dir."""
    modality_dir.mkdir(parents=True, exist_ok=True)
    (modality_dir / "__init__.py").write_text("")
    (modality_dir / "hf.py").write_text(plugin_dict_literal)


def test_required_keys_constant_complete():
    assert set(REQUIRED_HF_PLUGIN_KEYS) == {
        "modality", "runtime_path", "pip_extras", "system_packages",
        "priority", "sniff", "resolve", "search",
    }


def test_discovers_valid_plugin(tmp_path):
    _write_plugin(tmp_path / "audio_transcription", '''
HF_PLUGIN = {
    "modality": "audio/transcription",
    "runtime_path": "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel",
    "pip_extras": ("faster-whisper>=1.0.0",),
    "system_packages": ("ffmpeg",),
    "priority": 100,
    "sniff": lambda info: True,
    "resolve": lambda repo_id, variant, info: None,
    "search": lambda api, query, **kw: iter(()),
}
''')
    plugins = discover_hf_plugins([tmp_path])
    assert len(plugins) == 1
    p = plugins[0]
    assert p["modality"] == "audio/transcription"
    assert p["priority"] == 100
    assert callable(p["sniff"])
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/core/test_hf_plugin_discovery.py -v
```

Expected: `ImportError: cannot import name 'discover_hf_plugins' from 'muse.core.discovery'`.

- [ ] **Step 3: Add REQUIRED_HF_PLUGIN_KEYS and discover_hf_plugins**

Add to `src/muse/core/discovery.py` at the bottom (above the existing `_load_script`/`_load_package` helpers):

```python
REQUIRED_HF_PLUGIN_KEYS = (
    "modality", "runtime_path", "pip_extras", "system_packages",
    "priority", "sniff", "resolve", "search",
)


def discover_hf_plugins(dirs: list[Path]) -> list[dict]:
    """Scan dirs for `<dir>/<name>/hf.py` files exporting HF_PLUGIN.

    Each hf.py is loaded as a single-file module via spec_from_file_location
    (mangled name) so the modality package's __init__.py is not executed.
    This preserves the bare-install contract: `muse pull` works without
    fastapi installed because plugins don't transitively pull in routes.

    Validation: HF_PLUGIN must be a dict with all REQUIRED_HF_PLUGIN_KEYS.
    Missing keys, type mismatches, or import errors log a warning and skip
    the plugin. Discovery never raises.

    Returns plugins sorted by (priority asc, modality asc) so the dispatcher
    iterates specific shapes before catch-alls and the order is deterministic
    across machines.
    """
    found: list[dict] = []
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            hf_py = sub / "hf.py"
            if not hf_py.exists():
                continue
            try:
                module = _load_hf_plugin_script(hf_py)
            except Exception as e:
                logger.warning(
                    "skipping HF plugin %s: import failed (%s)", hf_py, e,
                )
                continue
            plugin = getattr(module, "HF_PLUGIN", None)
            if not isinstance(plugin, dict):
                logger.warning(
                    "skipping HF plugin %s: no top-level HF_PLUGIN dict", hf_py,
                )
                continue
            missing = [k for k in REQUIRED_HF_PLUGIN_KEYS if k not in plugin]
            if missing:
                logger.warning(
                    "skipping HF plugin %s: missing required keys %s",
                    hf_py, missing,
                )
                continue
            found.append(plugin)
    return sorted(found, key=lambda p: (p["priority"], p["modality"]))


def _load_hf_plugin_script(path: Path) -> Any:
    """Import hf.py as a single-file module. Bypasses package __init__.py.

    The mangled module name avoids sys.modules collisions when multiple
    modalities each ship a hf.py.
    """
    mod_name = f"_muse_hf_plugin_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/core/test_hf_plugin_discovery.py::test_required_keys_constant_complete tests/core/test_hf_plugin_discovery.py::test_discovers_valid_plugin -v
```

Expected: both PASS.

- [ ] **Step 5: Add error-path tests**

Append to `tests/core/test_hf_plugin_discovery.py`:

```python
def test_skips_plugin_with_missing_required_key(tmp_path, caplog):
    _write_plugin(tmp_path / "broken", '''
HF_PLUGIN = {
    "modality": "x/y",
    "runtime_path": "a:B",
    "pip_extras": (),
    "system_packages": (),
    # priority intentionally missing
    "sniff": lambda info: True,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    import logging
    with caplog.at_level(logging.WARNING):
        plugins = discover_hf_plugins([tmp_path])
    assert plugins == []
    assert any("missing required keys" in r.message for r in caplog.records)


def test_skips_plugin_with_no_hf_plugin_attr(tmp_path, caplog):
    _write_plugin(tmp_path / "noattr", "x = 1\n")
    import logging
    with caplog.at_level(logging.WARNING):
        plugins = discover_hf_plugins([tmp_path])
    assert plugins == []
    assert any("no top-level HF_PLUGIN" in r.message for r in caplog.records)


def test_skips_plugin_with_syntax_error(tmp_path, caplog):
    (tmp_path / "syntax").mkdir()
    (tmp_path / "syntax" / "__init__.py").write_text("")
    (tmp_path / "syntax" / "hf.py").write_text("def def def\n")
    import logging
    with caplog.at_level(logging.WARNING):
        plugins = discover_hf_plugins([tmp_path])
    assert plugins == []
    assert any("import failed" in r.message for r in caplog.records)


def test_orders_by_priority_then_modality(tmp_path):
    _write_plugin(tmp_path / "alpha", '''
HF_PLUGIN = {
    "modality": "z/last", "runtime_path": "a:B",
    "pip_extras": (), "system_packages": (),
    "priority": 200,
    "sniff": lambda info: False,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    _write_plugin(tmp_path / "beta", '''
HF_PLUGIN = {
    "modality": "a/first", "runtime_path": "a:B",
    "pip_extras": (), "system_packages": (),
    "priority": 100,
    "sniff": lambda info: False,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    _write_plugin(tmp_path / "gamma", '''
HF_PLUGIN = {
    "modality": "b/second", "runtime_path": "a:B",
    "pip_extras": (), "system_packages": (),
    "priority": 100,
    "sniff": lambda info: False,
    "resolve": lambda *a, **kw: None,
    "search": lambda *a, **kw: iter(()),
}
''')
    plugins = discover_hf_plugins([tmp_path])
    assert [p["modality"] for p in plugins] == ["a/first", "b/second", "z/last"]


def test_returns_empty_when_dirs_empty():
    assert discover_hf_plugins([]) == []


def test_skips_nonexistent_dirs(tmp_path):
    bogus = tmp_path / "does-not-exist"
    assert discover_hf_plugins([bogus]) == []
```

- [ ] **Step 6: Run all discovery tests, expect pass**

```bash
pytest tests/core/test_hf_plugin_discovery.py -v
```

Expected: 7 passed.

- [ ] **Step 7: Add `_default_hf_plugin_dirs` helper for the resolver**

Append to `src/muse/core/discovery.py`:

```python
def _default_hf_plugin_dirs() -> list[Path]:
    """Default scan paths: bundled modalities + $MUSE_MODALITIES_DIR if set.

    Mirrors `modality_tags()` and `discover_modalities` ordering so all
    discovery surfaces walk the same roots in the same precedence.
    """
    bundled = Path(__file__).resolve().parents[1] / "modalities"
    env = os.environ.get("MUSE_MODALITIES_DIR")
    return [bundled] + ([Path(env)] if env else [])
```

Add a smoke test:

```python
def test_default_dirs_includes_bundled():
    from muse.core.discovery import _default_hf_plugin_dirs
    dirs = _default_hf_plugin_dirs()
    assert dirs[0].name == "modalities"
    assert dirs[0].is_dir()
```

- [ ] **Step 8: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/muse/core/discovery.py tests/core/test_hf_plugin_discovery.py
git commit -m "feat(discovery): discover_hf_plugins for per-modality HF plugins (#129)

Lays the infrastructure for the HF resolver plugin refactor. New
function discover_hf_plugins(dirs) walks <dir>/<name>/hf.py, loads
each via spec_from_file_location (bypasses package __init__.py to
preserve the bare-install contract), validates HF_PLUGIN against
REQUIRED_HF_PLUGIN_KEYS, returns the sorted list (priority asc,
modality asc, deterministic tiebreak).

No callers yet; HFResolver wires it in next."
```

---

## Task 2: HFResolver plugin dispatch with legacy fallback

Wires `discover_hf_plugins` into the resolver. Both code paths run in parallel; behavior unchanged for users.

**Files:**
- Modify: `src/muse/core/resolvers_hf.py`
- Test: `tests/core/test_hf_resolver_dispatch.py`

- [ ] **Step 1: Write the failing test (plugin injection + dispatch)**

Create `tests/core/test_hf_resolver_dispatch.py`:

```python
"""Tests for HFResolver dispatch over per-modality plugins."""
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import ResolvedModel, ResolverError, SearchResult
from muse.core.resolvers_hf import HFResolver


def _make_plugin(modality, *, priority=100, sniff_returns=False, resolve_returns=None, search_returns=()):
    """Build an HF_PLUGIN dict with controllable callbacks for tests."""
    return {
        "modality": modality,
        "runtime_path": f"muse.modalities.{modality.replace('/', '_')}.runtimes.fake:Fake",
        "pip_extras": (),
        "system_packages": (),
        "priority": priority,
        "sniff": MagicMock(return_value=sniff_returns),
        "resolve": MagicMock(return_value=resolve_returns),
        "search": MagicMock(return_value=iter(search_returns)),
    }


def test_resolve_first_matching_plugin_wins():
    p_low = _make_plugin("a/first", priority=100, sniff_returns=False)
    p_high = _make_plugin(
        "b/second", priority=200, sniff_returns=True,
        resolve_returns=ResolvedModel(
            manifest={"model_id": "x"}, backend_path="a:B",
            download=lambda root: root,
        ),
    )
    resolver = HFResolver(plugins=[p_low, p_high])
    fake_info = MagicMock()
    with patch.object(resolver._api, "repo_info", return_value=fake_info):
        result = resolver.resolve("hf://org/repo")
    assert result.manifest["model_id"] == "x"
    p_low["sniff"].assert_called_once_with(fake_info)
    p_high["sniff"].assert_called_once_with(fake_info)
    p_high["resolve"].assert_called_once()


def test_resolve_no_plugin_matches_raises_clean_error():
    p1 = _make_plugin("x/y", sniff_returns=False)
    resolver = HFResolver(plugins=[p1])
    fake_info = MagicMock(siblings=[], tags=["random"])
    with patch.object(resolver._api, "repo_info", return_value=fake_info):
        with pytest.raises(ResolverError, match="no HF plugin matched"):
            resolver.resolve("hf://org/repo")


def test_resolve_short_circuits_on_first_match():
    """Once a plugin's sniff returns True, later plugins are not consulted."""
    p_first = _make_plugin(
        "a/x", priority=100, sniff_returns=True,
        resolve_returns=ResolvedModel(
            manifest={"model_id": "a"}, backend_path="a:B",
            download=lambda root: root,
        ),
    )
    p_second = _make_plugin("b/y", priority=200, sniff_returns=True)
    resolver = HFResolver(plugins=[p_first, p_second])
    with patch.object(resolver._api, "repo_info", return_value=MagicMock()):
        resolver.resolve("hf://org/repo")
    p_first["sniff"].assert_called_once()
    p_second["sniff"].assert_not_called()


def test_search_with_modality_filter_consults_only_matching():
    p_a = _make_plugin("a/x", search_returns=[SearchResult(
        uri="hf://a/1", model_id="m1", modality="a/x",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    p_b = _make_plugin("b/y", search_returns=[SearchResult(
        uri="hf://b/2", model_id="m2", modality="b/y",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    resolver = HFResolver(plugins=[p_a, p_b])
    rows = list(resolver.search("foo", modality="a/x"))
    assert [r.modality for r in rows] == ["a/x"]
    p_a["search"].assert_called_once()
    p_b["search"].assert_not_called()


def test_search_with_no_modality_filter_consults_all_plugins():
    p_a = _make_plugin("a/x", search_returns=[SearchResult(
        uri="hf://a/1", model_id="m1", modality="a/x",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    p_b = _make_plugin("b/y", search_returns=[SearchResult(
        uri="hf://b/2", model_id="m2", modality="b/y",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    resolver = HFResolver(plugins=[p_a, p_b])
    rows = list(resolver.search("foo"))
    assert sorted(r.modality for r in rows) == ["a/x", "b/y"]
    p_a["search"].assert_called_once()
    p_b["search"].assert_called_once()


def test_search_with_unknown_modality_raises_clean_error():
    p_a = _make_plugin("a/x")
    resolver = HFResolver(plugins=[p_a])
    with pytest.raises(ResolverError, match="does not support modality"):
        list(resolver.search("foo", modality="never/heard-of-it"))


def test_default_constructor_loads_from_disk():
    """No plugins= arg falls back to discover_hf_plugins(default_dirs)."""
    with patch("muse.core.resolvers_hf.discover_hf_plugins", return_value=[]) as mock_discover:
        HFResolver()
    mock_discover.assert_called_once()
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/core/test_hf_resolver_dispatch.py -v
```

Expected: import errors / signature mismatches (no `plugins=` arg yet).

- [ ] **Step 3: Refactor HFResolver to plugin dispatch with legacy fallback**

Edit `src/muse/core/resolvers_hf.py`:

Replace the imports block:

```python
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
```

Replace the `class HFResolver` definition (only the class itself; helpers below stay for now):

```python
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
        # have migrated. Tracked by the legacy_fallback xfail watchdog
        # below.
        legacy = self._legacy_resolve(repo_id, variant, info)
        if legacy is not None:
            return legacy

        tags = getattr(info, "tags", None) or []
        siblings = [s.rfilename for s in getattr(info, "siblings", [])][:5]
        raise ResolverError(
            f"no HF plugin matched {repo_id!r}; tags={tags}, "
            f"siblings={siblings}..."
        )

    def search(self, query: str, **filters):
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

    # --- LEGACY (kept until Task 7) ---
    # (existing _resolve_gguf, _search_gguf, _resolve_sentence_transformer,
    # _search_sentence_transformers, _resolve_faster_whisper,
    # _search_faster_whisper, _resolve_text_classifier,
    # _search_text_classifier methods stay verbatim below this line)
```

The existing `_resolve_*`, `_search_*` methods stay in place (do not delete yet). Only the dispatcher logic (`resolve`, `search`) and `__init__` change.

- [ ] **Step 4: Run dispatch tests, expect pass**

```bash
pytest tests/core/test_hf_resolver_dispatch.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run full fast lane (legacy still works)**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green. Existing `tests/core/test_resolvers_hf.py` tests still pass via the legacy fallback (no plugins discovered yet because no `hf.py` files exist).

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/resolvers_hf.py tests/core/test_hf_resolver_dispatch.py
git commit -m "feat(resolver): HFResolver plugin dispatch with legacy fallback (#129)

HFResolver.__init__ now accepts plugins= injection. resolve() and
search() iterate plugins in priority order; first sniff to return
True wins. When no plugin matches, the existing _sniff_repo_shape
cascade runs as a fallback so behaviour is unchanged for users
during the migration window.

The legacy fallback is removed in a later commit once all four
bundled modalities (gguf, ST, faster-whisper, text-classification)
ship hf.py plugin files."
```

---

## Task 3: chat_completion/hf.py (GGUF migration)

First modality migration. Most complex one (variant handling, chat_formats.yaml integration), so tackling it first surfaces any contract gaps.

**Files:**
- Create: `src/muse/modalities/chat_completion/hf.py`
- Test: `tests/modalities/chat_completion/test_hf_plugin.py`
- Modify: `src/muse/core/resolvers_hf.py` (remove `gguf` branches from `_legacy_resolve` / `_legacy_search` + delete `_resolve_gguf` / `_search_gguf` and GGUF helpers)

- [ ] **Step 1: Write the failing test (plugin contract shape)**

Create `tests/modalities/chat_completion/test_hf_plugin.py`:

```python
"""Tests for the chat_completion HF plugin (GGUF)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.chat_completion.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel, SearchResult


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN, f"missing {key!r}"


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "chat/completion"
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
    )
    assert HF_PLUGIN["priority"] == 100


def test_sniff_true_on_repo_with_gguf_files():
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_when_no_gguf():
    info = _fake_info(siblings=["model.bin", "config.json"])
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_requires_variant_for_gguf():
    info = _fake_info(siblings=["model-q4_k_m.gguf", "model-q8_0.gguf"])
    from muse.core.resolvers import ResolverError
    with pytest.raises(ResolverError, match="variant required"):
        HF_PLUGIN["resolve"]("org/Model-GGUF", None, info)


def test_resolve_returns_resolved_model_with_correct_manifest():
    info = _fake_info(siblings=["model-q4_k_m.gguf"])
    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=False), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        result = HF_PLUGIN["resolve"]("org/Model-GGUF", "q4_k_m", info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "chat/completion"
    assert result.manifest["hf_repo"] == "org/Model-GGUF"
    assert result.manifest["capabilities"]["gguf_file"] == "model-q4_k_m.gguf"
    assert result.backend_path.endswith(":LlamaCppModel")


def test_search_yields_search_results_with_modality_tag():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=100)
    fake_repo.siblings = [
        MagicMock(rfilename="model-q4_k_m.gguf", size=2_500_000_000),
    ]
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "qwen", sort="downloads", limit=20))
    assert len(rows) >= 1
    assert all(r.modality == "chat/completion" for r in rows)
    assert all(r.uri.startswith("hf://") for r in rows)
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/modalities/chat_completion/test_hf_plugin.py -v
```

Expected: `ModuleNotFoundError: No module named 'muse.modalities.chat_completion.hf'`.

- [ ] **Step 3: Create chat_completion/hf.py by copying GGUF logic from resolvers_hf.py**

Create `src/muse/modalities/chat_completion/hf.py`:

```python
"""HF resolver plugin for GGUF chat/completion models.

Sniffs HuggingFace repos for `.gguf` siblings and synthesizes a manifest
that targets the LlamaCppModel generic runtime. Variant (quant tag) is
required: a single GGUF repo often publishes 5+ quants and there is no
defensible default. `muse search foo --modality chat/completion`
enumerates each variant as a separate row.

This plugin is loaded by `discover_hf_plugins` via single-file import,
so it must NOT use relative imports or import from sibling modality
modules. See docs/HF_PLUGINS.md for the authoring rules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from muse.core.chat_formats import lookup_chat_format
from muse.core.resolvers import ResolvedModel, ResolverError, SearchResult


_VARIANT_RE = re.compile(
    r"(q\d+_[a-z0-9_]+|iq\d+_[a-z0-9]+|f16|bf16|f32)", re.IGNORECASE,
)


def _extract_variant(gguf_filename: str) -> str:
    stem = Path(gguf_filename).stem
    m = _VARIANT_RE.search(stem)
    return (m.group(1).lower() if m else stem).replace(".", "_")


def _match_gguf_variant(files: list[str], variant: str) -> str | None:
    norm = variant.lower()
    for f in files:
        if _extract_variant(f) == norm:
            return f
    return None


def _gguf_model_id(repo_id: str, variant: str) -> str:
    base = repo_id.split("/", 1)[-1].lower()
    if not base.endswith("-gguf"):
        base = f"{base}-gguf"
    return f"{base}-{variant.lower().replace('_', '-')}"


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff_supports_tools(chat_template: str | None) -> bool:
    if not chat_template or not isinstance(chat_template, str):
        return False
    return bool(re.search(r"(\bif\s+tools\b|\{\{\s*tools|tool_calls)", chat_template))


def _try_sniff_tools_from_repo(api: HfApi, repo_id: str) -> bool | None:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
    except Exception:
        return None
    try:
        cfg = json.loads(Path(path).read_text())
    except Exception:
        return None
    return _sniff_supports_tools(cfg.get("chat_template"))


def _try_sniff_context_length_from_repo(api: HfApi, repo_id: str) -> int | None:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
        cfg = json.loads(Path(path).read_text())
        return int(cfg.get("max_position_embeddings") or 0) or None
    except Exception:
        return None


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(f.endswith(".gguf") for f in siblings)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    gguf_files = [f for f in siblings if f.endswith(".gguf")]
    if not gguf_files:
        raise ResolverError(f"no .gguf files in {repo_id}")
    if variant is None:
        variants = sorted({_extract_variant(f) for f in gguf_files})
        raise ResolverError(
            f"variant required for GGUF repo {repo_id}; available: {variants}"
        )
    matched = _match_gguf_variant(gguf_files, variant)
    if matched is None:
        variants = sorted({_extract_variant(f) for f in gguf_files})
        raise ResolverError(
            f"variant {variant!r} not found in {repo_id}; available: {variants}"
        )

    # The shared HfApi instance is created lazily here. Tests patch it.
    api = HfApi()
    supports_tools = _try_sniff_tools_from_repo(api, repo_id)
    ctx_length = _try_sniff_context_length_from_repo(api, repo_id)

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
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        allow_patterns = [matched, "tokenizer*", "config.json", "*.md"]
        return Path(snapshot_download(
            repo_id=repo_id, allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(search=query, filter="gguf", sort=sort, limit=limit)
    for repo in repos:
        siblings = getattr(repo, "siblings", None) or []
        if not siblings:
            try:
                info = api.repo_info(repo.id, files_metadata=True)
                siblings = info.siblings
            except Exception:
                continue
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


_RUNTIME_PATH = "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
_PIP_EXTRAS = ("llama-cpp-python>=0.2.90",)


HF_PLUGIN = {
    "modality": "chat/completion",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

- [ ] **Step 4: Run plugin tests, expect pass**

```bash
pytest tests/modalities/chat_completion/test_hf_plugin.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Confirm legacy code path is now bypassed for GGUF**

The HF resolver now discovers `chat_completion/hf.py` via `discover_hf_plugins`. The plugin's sniff matches GGUF repos before the legacy fallback runs. Verify with the existing `tests/core/test_resolvers_hf.py::test_resolve_gguf*` tests.

```bash
pytest tests/core/test_resolvers_hf.py -v -k gguf
```

Expected: tests still green (the plugin path produces the same ResolvedModel as the legacy one).

- [ ] **Step 6: Remove the GGUF branch from `_legacy_resolve` and `_legacy_search`**

Edit `src/muse/core/resolvers_hf.py`. In `_legacy_resolve`:

Remove these two lines:
```python
        if shape == "gguf":
            return self._resolve_gguf(repo_id, variant, info)
```

In `_legacy_search`:

Remove these two lines:
```python
        if modality == "chat/completion":
            return self._search_gguf(query, sort=sort, limit=limit)
```

In `_sniff_repo_shape`, remove the GGUF branch:
```python
    if any(f.endswith(".gguf") for f in siblings):
        return "gguf"
```

- [ ] **Step 7: Delete the now-unused legacy GGUF helpers**

Delete from `src/muse/core/resolvers_hf.py`:
- `LLAMA_CPP_RUNTIME_PATH` constant
- `LLAMA_CPP_PIP_EXTRAS` constant
- `_resolve_gguf` method
- `_search_gguf` method
- `_extract_variant` (function-level, in modality file now)
- `_match_gguf_variant`
- `_gguf_model_id`
- `_VARIANT_RE`
- `_try_sniff_tools_from_repo`
- `_sniff_supports_tools`
- `_try_sniff_context_length_from_repo`

The `from muse.core.chat_formats import lookup_chat_format` import (line 163 in the old file) was inside `_resolve_gguf` so it gets deleted with the method.

- [ ] **Step 8: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green. The chat_completion modality is now fully served by `chat_completion/hf.py`.

- [ ] **Step 9: Commit**

```bash
git add src/muse/modalities/chat_completion/hf.py \
        tests/modalities/chat_completion/test_hf_plugin.py \
        src/muse/core/resolvers_hf.py
git commit -m "refactor(hf): migrate GGUF dispatch to chat_completion/hf.py (#129)

First of four modality migrations. The HF plugin pattern lives at
modalities/chat_completion/hf.py exporting HF_PLUGIN. resolvers_hf.py
loses _resolve_gguf, _search_gguf, and the GGUF-specific helpers
(variant regex, sniff helpers, runtime constants). The legacy
fallback's gguf branch is also removed, making this modality
fully plugin-driven.

Behaviour unchanged: same ResolvedModel shape, same SearchResult
output, same error messages."
```

---

## Task 4: embedding_text/hf.py (sentence-transformers migration)

Same pattern as Task 3, simpler (no variant, no chat_formats integration).

**Files:**
- Create: `src/muse/modalities/embedding_text/hf.py`
- Test: `tests/modalities/embedding_text/test_hf_plugin.py`
- Modify: `src/muse/core/resolvers_hf.py` (remove ST branches + helpers)

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/embedding_text/test_hf_plugin.py`:

```python
"""Tests for the embedding_text HF plugin (sentence-transformers)."""
from unittest.mock import MagicMock

import pytest

from muse.modalities.embedding_text.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "embedding/text"
    assert HF_PLUGIN["runtime_path"].endswith(":SentenceTransformerModel")
    assert HF_PLUGIN["priority"] == 110


def test_sniff_true_on_sentence_transformers_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["sentence-transformers"])) is True


def test_sniff_true_on_st_config_file():
    info = _fake_info(siblings=["sentence_transformers_config.json"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_random_repo():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-generation"])) is False


def test_resolve_returns_resolved_model():
    info = _fake_info(tags=["sentence-transformers"])
    result = HF_PLUGIN["resolve"]("sentence-transformers/all-MiniLM-L6-v2", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "embedding/text"
    assert result.manifest["model_id"] == "all-minilm-l6-v2"


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=50)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "minilm", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "embedding/text"
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/modalities/embedding_text/test_hf_plugin.py -v
```

Expected: ImportError (no `embedding_text/hf.py` yet).

- [ ] **Step 3: Create embedding_text/hf.py**

Create `src/muse/modalities/embedding_text/hf.py`:

```python
"""HF resolver plugin for sentence-transformers embedding/text models.

Sniffs HF repos for the `sentence-transformers` tag or the
`sentence_transformers_config.json` sibling file, and synthesizes a
manifest that targets SentenceTransformerModel. No variants; one
manifest per repo.

Loaded via single-file import; must not use relative imports.
See docs/HF_PLUGINS.md for the authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "sentence-transformers" in tags:
        return True
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(Path(f).name == "sentence_transformers_config.json" for f in siblings)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "embedding/text",
        "hf_repo": repo_id,
        "description": f"Sentence-Transformers: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
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
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="sentence-transformers", sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="embedding/text",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


_RUNTIME_PATH = "muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel"
_PIP_EXTRAS = ("torch>=2.1.0", "sentence-transformers>=2.2.0")


HF_PLUGIN = {
    "modality": "embedding/text",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

- [ ] **Step 4: Run plugin tests, expect pass**

```bash
pytest tests/modalities/embedding_text/test_hf_plugin.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Remove ST branches from legacy code**

Edit `src/muse/core/resolvers_hf.py`:

In `_legacy_resolve`, remove:
```python
        if shape == "sentence-transformers":
            return self._resolve_sentence_transformer(repo_id, info)
```

In `_legacy_search`, remove:
```python
        if modality == "embedding/text":
            return self._search_sentence_transformers(query, sort=sort, limit=limit)
```

In `_sniff_repo_shape`, remove:
```python
    if "sentence-transformers" in tags:
        return "sentence-transformers"
    if any(Path(f).name == "sentence_transformers_config.json" for f in siblings):
        return "sentence-transformers"
```

Delete `_resolve_sentence_transformer`, `_search_sentence_transformers`, `_sentence_transformer_model_id`. Delete `SENTENCE_TRANSFORMER_RUNTIME_PATH` and `SENTENCE_TRANSFORMER_PIP_EXTRAS`.

- [ ] **Step 6: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/muse/modalities/embedding_text/hf.py \
        tests/modalities/embedding_text/test_hf_plugin.py \
        src/muse/core/resolvers_hf.py
git commit -m "refactor(hf): migrate sentence-transformers to embedding_text/hf.py (#129)

Second migration. Behaviour unchanged."
```

---

## Task 5: audio_transcription/hf.py (faster-whisper migration)

**Files:**
- Create: `src/muse/modalities/audio_transcription/hf.py`
- Test: `tests/modalities/audio_transcription/test_hf_plugin.py`
- Modify: `src/muse/core/resolvers_hf.py`

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/audio_transcription/test_hf_plugin.py`:

```python
"""Tests for the audio_transcription HF plugin (faster-whisper)."""
from unittest.mock import MagicMock

from muse.modalities.audio_transcription.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "audio/transcription"
    assert HF_PLUGIN["runtime_path"].endswith(":FasterWhisperModel")
    assert HF_PLUGIN["priority"] == 100
    assert "ffmpeg" in HF_PLUGIN["system_packages"]


def test_sniff_true_on_ct2_shape_with_asr_tag():
    info = _fake_info(
        siblings=["model.bin", "config.json", "tokenizer.json"],
        tags=["automatic-speech-recognition"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_asr_tag():
    info = _fake_info(
        siblings=["model.bin", "config.json", "tokenizer.json"],
        tags=["text-generation"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_with_asr_tag_but_wrong_shape():
    info = _fake_info(
        siblings=["model.safetensors", "config.json"],
        tags=["automatic-speech-recognition"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_returns_resolved_model():
    info = _fake_info(
        siblings=["model.bin", "config.json", "tokenizer.json"],
        tags=["automatic-speech-recognition"],
    )
    result = HF_PLUGIN["resolve"]("Systran/faster-whisper-tiny", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "audio/transcription"
    assert result.manifest["model_id"] == "faster-whisper-tiny"


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="Systran/faster-whisper-base", downloads=200)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "whisper", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "audio/transcription"
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/modalities/audio_transcription/test_hf_plugin.py -v
```

- [ ] **Step 3: Create audio_transcription/hf.py**

Create `src/muse/modalities/audio_transcription/hf.py`:

```python
"""HF resolver plugin for CT2 faster-whisper audio/transcription models.

Sniffs HF repos for the CT2 file shape (model.bin + config.json +
tokenizer.json or vocabulary.txt) plus the ASR tag. Synthesizes a
manifest that targets FasterWhisperModel. ffmpeg is declared as a
system package so the catalog warns when it's missing on PATH.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


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
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    return _looks_like_ct2(siblings, tags)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "audio/transcription",
        "hf_repo": repo_id,
        "description": f"Faster-Whisper: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": list(_SYSTEM_PACKAGES),
        "capabilities": {},
    }

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="automatic-speech-recognition",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="audio/transcription",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


_RUNTIME_PATH = "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel"
_PIP_EXTRAS = ("faster-whisper>=1.0.0",)
_SYSTEM_PACKAGES = ("ffmpeg",)


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
```

- [ ] **Step 4: Run plugin tests, expect pass**

```bash
pytest tests/modalities/audio_transcription/test_hf_plugin.py -v
```

- [ ] **Step 5: Remove faster-whisper branches from legacy code**

Edit `src/muse/core/resolvers_hf.py`:

In `_legacy_resolve` remove:
```python
        if shape == "faster-whisper":
            return self._resolve_faster_whisper(repo_id, info)
```

In `_legacy_search` remove:
```python
        if modality == "audio/transcription":
            return self._search_faster_whisper(query, sort=sort, limit=limit)
```

In `_sniff_repo_shape` remove the `_looks_like_faster_whisper` branch.

Delete `_resolve_faster_whisper`, `_search_faster_whisper`, `_looks_like_faster_whisper`. Delete `FASTER_WHISPER_RUNTIME_PATH`, `FASTER_WHISPER_PIP_EXTRAS`, `FASTER_WHISPER_SYSTEM_PACKAGES`.

- [ ] **Step 6: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Commit**

```bash
git add src/muse/modalities/audio_transcription/hf.py \
        tests/modalities/audio_transcription/test_hf_plugin.py \
        src/muse/core/resolvers_hf.py
git commit -m "refactor(hf): migrate faster-whisper to audio_transcription/hf.py (#129)

Third migration."
```

---

## Task 6: text_classification/hf.py (catch-all migration)

Last modality. Tag-only sniff makes this the broadest plugin; priority is `200` so it's checked last.

**Files:**
- Create: `src/muse/modalities/text_classification/hf.py`
- Test: `tests/modalities/text_classification/test_hf_plugin.py`
- Modify: `src/muse/core/resolvers_hf.py`

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/text_classification/test_hf_plugin.py`:

```python
"""Tests for the text_classification HF plugin."""
from unittest.mock import MagicMock

from muse.modalities.text_classification.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "text/classification"
    assert HF_PLUGIN["runtime_path"].endswith(":HFTextClassifier")
    # priority 200: tag-only catch-all, must lose to specific plugins
    assert HF_PLUGIN["priority"] == 200


def test_sniff_true_on_text_classification_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-classification"])) is True


def test_sniff_false_on_random_repo():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-generation"])) is False


def test_resolve_returns_resolved_model():
    info = _fake_info(tags=["text-classification"])
    result = HF_PLUGIN["resolve"]("KoalaAI/Text-Moderation", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "text/classification"
    assert result.manifest["model_id"] == "text-moderation"


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/classifier", downloads=80)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "moderation", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "text/classification"
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/modalities/text_classification/test_hf_plugin.py -v
```

- [ ] **Step 3: Create text_classification/hf.py**

Create `src/muse/modalities/text_classification/hf.py`:

```python
"""HF resolver plugin for HF text-classification models.

Tag-only sniff: any repo with the `text-classification` tag is claimed.
Priority 200 so this plugin runs LAST after more specific shapes
(GGUF file pattern, faster-whisper CT2 shape, sentence-transformers
config) have had their chance.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "text-classification" in tags


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "text/classification",
        "hf_repo": repo_id,
        "description": f"Text classifier: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
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
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="text-classification",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="text/classification",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


_RUNTIME_PATH = (
    "muse.modalities.text_classification.runtimes.hf_text_classifier"
    ":HFTextClassifier"
)
_PIP_EXTRAS = ("transformers>=4.36.0", "torch>=2.1.0")


HF_PLUGIN = {
    "modality": "text/classification",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 200,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

- [ ] **Step 4: Run plugin tests, expect pass**

```bash
pytest tests/modalities/text_classification/test_hf_plugin.py -v
```

- [ ] **Step 5: Remove text-classification branches from legacy code**

Edit `src/muse/core/resolvers_hf.py`:

In `_legacy_resolve` remove:
```python
        if shape == "text-classification":
            return self._resolve_text_classifier(repo_id, info)
```

In `_legacy_search` remove:
```python
        if modality == "text/classification":
            return self._search_text_classifier(query, sort=sort, limit=limit)
```

In `_sniff_repo_shape` remove the `_looks_like_text_classifier` branch.

Delete `_resolve_text_classifier`, `_search_text_classifier`, `_looks_like_text_classifier`. Delete `TEXT_CLASSIFIER_RUNTIME_PATH`, `TEXT_CLASSIFIER_PIP_EXTRAS`, `TEXT_CLASSIFIER_SYSTEM_PACKAGES`.

- [ ] **Step 6: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Commit**

```bash
git add src/muse/modalities/text_classification/hf.py \
        tests/modalities/text_classification/test_hf_plugin.py \
        src/muse/core/resolvers_hf.py
git commit -m "refactor(hf): migrate text-classification to text_classification/hf.py (#129)

Fourth and final modality migration. All four bundled modalities
now ship hf.py; the legacy fallback path in HFResolver no longer
triggers for any known shape."
```

---

## Task 7: Delete legacy fallback

All four modalities migrated. The fallback paths (`_legacy_resolve`, `_legacy_search`, `_sniff_repo_shape`) are now dead code.

**Files:**
- Modify: `src/muse/core/resolvers_hf.py`
- Modify: `tests/core/test_resolvers_hf.py` (drop tests of deleted functions)

- [ ] **Step 1: Confirm legacy paths are empty**

After Tasks 3-6, `_sniff_repo_shape` should always return `"unknown"` (all branches removed); `_legacy_resolve` should always return None; `_legacy_search` should always return None.

```bash
grep -n "if shape ==\|if modality ==" /home/spinoza/github/repos/muse/src/muse/core/resolvers_hf.py
```

Expected: zero matches (legacy bodies are now empty after the per-task removals).

- [ ] **Step 2: Remove legacy methods from HFResolver**

Edit `src/muse/core/resolvers_hf.py`. In `HFResolver.resolve`, replace the legacy fallback block:

```python
        # Legacy fallback: removed in Task 7 once all bundled modalities
        # have migrated. Tracked by the legacy_fallback xfail watchdog
        # below.
        legacy = self._legacy_resolve(repo_id, variant, info)
        if legacy is not None:
            return legacy

        tags = getattr(info, "tags", None) or []
        siblings = [s.rfilename for s in getattr(info, "siblings", [])][:5]
        raise ResolverError(
            f"no HF plugin matched {repo_id!r}; tags={tags}, "
            f"siblings={siblings}..."
        )
```

with:

```python
        tags = getattr(info, "tags", None) or []
        siblings = [s.rfilename for s in getattr(info, "siblings", [])][:5]
        raise ResolverError(
            f"no HF plugin matched {repo_id!r}; tags={tags}, "
            f"siblings={siblings}..."
        )
```

In `HFResolver.search`, remove the legacy fallback block:

```python
            if not matched:
                # Legacy fallback for not-yet-migrated modalities.
                legacy = self._legacy_search(query, modality, sort, limit)
                if legacy is not None:
                    yield from legacy
                    return
                supported = sorted(p["modality"] for p in self._plugins)
                raise ResolverError(...)
```

Replace with:

```python
            if not matched:
                supported = sorted(p["modality"] for p in self._plugins)
                raise ResolverError(
                    f"HFResolver.search does not support modality {modality!r}; "
                    f"supported: {supported}"
                )
```

Delete the now-unused methods on `HFResolver`:
- `_legacy_resolve`
- `_legacy_search`

Delete the module-level helpers no longer used by anyone:
- `_sniff_repo_shape`

The remaining file should contain: `HFResolver` class (with `__init__`, `resolve`, `search`), the `register_resolver(HFResolver())` call at module bottom, and any module docstring. Verify the file is short:

```bash
wc -l /home/spinoza/github/repos/muse/src/muse/core/resolvers_hf.py
```

Expected: roughly 80 lines (was about 490 before the refactor).

- [ ] **Step 3: Update tests/core/test_resolvers_hf.py**

The existing file has tests for the deleted functions. Trim it. Keep only tests of:
- `parse_uri` round-trips
- `HFResolver(scheme=)` matching the right URI scheme
- `register_resolver` registration

Remove tests of `_resolve_gguf`, `_search_gguf`, `_resolve_sentence_transformer`, etc. (per-modality coverage now lives in `tests/modalities/<name>/test_hf_plugin.py`).

Use `pytest --collect-only tests/core/test_resolvers_hf.py` to enumerate, then prune.

```bash
pytest --collect-only tests/core/test_resolvers_hf.py 2>&1 | grep -E "^\s+<(Test|Function)" | head -40
```

For each test name that mentions `gguf`, `sentence_transformer`, `faster_whisper`, `text_classifier`, `_sniff_repo_shape`, `_extract_variant`, `_match_gguf_variant`, `_looks_like_*`, delete it. Anything purely about `parse_uri`, `HFResolver.scheme`, registration: keep.

- [ ] **Step 4: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/resolvers_hf.py tests/core/test_resolvers_hf.py
git commit -m "refactor(hf): drop legacy dispatch from HFResolver (#129)

All four bundled modalities ship hf.py plugins now. The
_legacy_resolve / _legacy_search / _sniff_repo_shape fallback
paths are unreachable; deleting them. resolvers_hf.py is now
~80 lines: just HFResolver (init, resolve, search) plus the
register_resolver call.

Per-modality test coverage moved to tests/modalities/<name>/
test_hf_plugin.py during the four migration commits.
test_resolvers_hf.py is trimmed accordingly."
```

---

## Task 8: docs/HF_PLUGINS.md authoring guide

**Files:**
- Create: `docs/HF_PLUGINS.md`

- [ ] **Step 1: Write the authoring guide**

Create `docs/HF_PLUGINS.md`:

```markdown
# Authoring HF resolver plugins

A plugin teaches muse's HF resolver how to recognize, resolve, and
search a particular model shape on HuggingFace. Each modality
contributes one plugin file; the resolver discovers them at startup.

## File location

`src/muse/modalities/<name>/hf.py` (bundled) or `<MUSE_MODALITIES_DIR>/<name>/hf.py` (user-contributed).

## The plugin contract

Top-level `HF_PLUGIN: dict` with these keys:

| Key | Type | Purpose |
|---|---|---|
| `modality` | `str` | MIME tag (must match the modality's `MODALITY` constant) |
| `runtime_path` | `str` | `"muse.modalities.X.runtimes.Y:Cls"` |
| `pip_extras` | `tuple[str, ...]` | per-model venv install args |
| `system_packages` | `tuple[str, ...]` | apt/brew packages required (may be empty) |
| `priority` | `int` | lower checked first; 100 for specific shapes, 200+ for tag-only catch-alls |
| `sniff` | `Callable[[info], bool]` | True iff this plugin claims `info` |
| `resolve` | `Callable[[repo_id, variant, info], ResolvedModel]` | build manifest + download closure |
| `search` | `Callable[[api, query, *, sort, limit], Iterable[SearchResult]]` | yield rows for `muse search` |

## Authoring rules

`hf.py` is loaded as a single-file module via `spec_from_file_location`,
bypassing the modality package's `__init__.py`. This keeps `muse pull`
working on a bare install (no fastapi). The cost: relative imports
(`from .protocol import ...`) and absolute sibling imports
(`from muse.modalities.X.codec import ...`) both fail because the
parent package is not initialized.

What you may import:
- stdlib
- `huggingface_hub` (base dep)
- `muse.core.*` (lightweight: resolvers, chat_formats, errors)

What you may NOT import:
- relative siblings (`.protocol`, `.codec`, `.routes`)
- absolute siblings (`muse.modalities.X.protocol`)
- heavy deps (torch, transformers, fastapi, llama_cpp)

## Priority conventions

| Range | When to use |
|---|---|
| 100 | File-pattern + tag (very specific). Examples: GGUF (`.gguf` siblings), CT2 (model.bin + config.json + ASR tag) |
| 110 | Tag OR config-file (medium-specific). Example: sentence-transformers |
| 200 | Tag-only (broad catch-all). Example: text-classification |

If two plugins ever sniff True on the same repo, lower priority wins.
Same priority resolves alphabetically by modality tag. The intent: the
narrower shape always wins.

## Example skeleton

```python
"""HF resolver plugin for <my modality>."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "my-task-tag" in tags


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "my/modality",
        "hf_repo": repo_id,
        "description": f"My modality: {repo_id}",
        "license": getattr(getattr(info, "card_data", None), "license", None),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {},
    }
    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))
    return ResolvedModel(
        manifest=manifest, backend_path=_RUNTIME_PATH, download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    for repo in api.list_models(search=query, filter="my-task-tag", sort=sort, limit=limit):
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="my/modality",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


_RUNTIME_PATH = "muse.modalities.my_modality.runtimes.my_runtime:MyModel"
_PIP_EXTRAS = ("my-dep>=1.0",)


HF_PLUGIN = {
    "modality": "my/modality",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

## Testing

Each plugin should ship a `tests/modalities/<name>/test_hf_plugin.py`
covering:
- `HF_PLUGIN` has all required keys (`REQUIRED_HF_PLUGIN_KEYS` in
  `muse.core.discovery`)
- metadata correctness (modality, runtime_path, priority)
- `sniff` returns True on a positive synthetic info; False on a negative
- `resolve` returns a `ResolvedModel` with the right manifest shape
- `search` yields `SearchResult` instances with the right modality tag

Use `unittest.mock.MagicMock` for `info` and `api`; no real network
calls in unit tests.
```

- [ ] **Step 2: Commit**

```bash
git add docs/HF_PLUGINS.md
git commit -m "docs(hf): authoring guide for per-modality HF plugins (#129)

Documents the HF_PLUGIN contract, the import constraints that fall
out of single-file loading, the priority conventions, and an
end-to-end skeleton with a matching test pattern."
```

---

## Task 9: Update CLAUDE.md, README.md, RESOLVERS.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `docs/RESOLVERS.md`

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`, find the "Adding a new modality (rare)" section. Replace:

```
1. Create `src/muse/modalities/<mime_name>/` (e.g. `audio_transcriptions/`
   for MODALITY `"audio/transcription"`). Use underscores in the dir
   name; the MIME tag has the slash.
2. Write `protocol.py` (Protocol + Result dataclass), `routes.py`
   (with `build_router(registry) -> APIRouter`), `client.py` (HTTP
   client), and `codec.py` (encoding for this modality's output).
3. Export from `__init__.py`: `MODALITY = "audio/transcription"` (the
   MIME string) and `build_router` (the router factory). Also re-export
   the Protocol + Result for user imports.
4. Add bundled model scripts under `src/muse/models/`.
5. Add tests under `tests/modalities/<mime_name>/` and
   `tests/models/test_<new_model>.py`.

No edits to `worker.py`, `catalog.py`, `registry.py`, or `server.py`
are needed: discovery handles the wiring.
```

with:

```
1. Create `src/muse/modalities/<mime_name>/` (e.g. `audio_transcriptions/`
   for MODALITY `"audio/transcription"`). Use underscores in the dir
   name; the MIME tag has the slash.
2. Write `protocol.py` (Protocol + Result dataclass), `routes.py`
   (with `build_router(registry) -> APIRouter`), `client.py` (HTTP
   client), and `codec.py` (encoding for this modality's output).
3. Export from `__init__.py`: `MODALITY = "audio/transcription"` (the
   MIME string) and `build_router` (the router factory). Also re-export
   the Protocol + Result for user imports.
4. (HF support) write `hf.py` exporting `HF_PLUGIN: dict` (sniff/
   resolve/search + metadata). See `docs/HF_PLUGINS.md` for the
   contract and authoring rules. Loaded via single-file import,
   so no relative imports.
5. Add bundled model scripts under `src/muse/models/` (or rely on
   the resolver alone for uniform-shape modalities).
6. Add tests under `tests/modalities/<mime_name>/` (route + plugin)
   and `tests/models/test_<new_model>.py`.

No edits to `worker.py`, `catalog.py`, `registry.py`, `server.py`,
or `resolvers_hf.py` are needed: discovery handles the wiring.
```

- [ ] **Step 2: Update README.md**

In `README.md`, the "three ways to add a model" section already mentions modalities. Locate the third bullet ("Add a whole new modality") and append:

```
3. **Add a whole new modality** (rare) by dropping a subpackage into
   `src/muse/modalities/` or `$MUSE_MODALITIES_DIR`. The subpackage
   exports `MODALITY` + `build_router` and discovery picks it up.
   Optional: drop a `hf.py` next to `__init__.py` exporting an
   `HF_PLUGIN` dict; muse's HF resolver picks it up the same way and
   `muse search`/`muse pull hf://...` work for the new modality.
```

(replace the existing third bullet with the version above; the rest of the README is unchanged).

- [ ] **Step 3: Update docs/RESOLVERS.md**

In `docs/RESOLVERS.md`, find the section that describes `_sniff_repo_shape` (or any prose claiming the resolver knows about specific modalities directly). Replace the prose with a short note:

```markdown
## How HF resolution works

The HF resolver discovers per-modality plugins at startup. Each
modality contributes a `hf.py` file next to its `__init__.py`
exporting an `HF_PLUGIN: dict` (sniff/resolve/search + metadata).

On `muse pull hf://Org/Repo[@variant]`:
1. The resolver's `repo_info(repo_id)` fetches HuggingFace metadata.
2. Plugins are iterated in (priority asc, modality asc) order.
3. The first plugin whose `sniff(info)` returns True wins; its
   `resolve(repo_id, variant, info)` synthesizes the manifest.

On `muse search foo --modality M`: only plugins with `modality == M`
are consulted. With no modality filter: all plugins run.

See `docs/HF_PLUGINS.md` for how to author a plugin.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md docs/RESOLVERS.md
git commit -m "docs: point at hf.py plugin pattern instead of resolvers_hf.py (#129)

CLAUDE.md's 'Adding a new modality' section now lists hf.py as
optional step 4. README.md mentions the plugin in the modality
bullet. docs/RESOLVERS.md drops the _sniff_repo_shape prose and
cross-links HF_PLUGINS.md."
```

---

## Task 10: v0.15.0 release

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/muse/__init__.py` (docstring "as of v0.14.2" -> "as of v0.15.0")

- [ ] **Step 1: Bump version**

Edit `pyproject.toml`:

```toml
version = "0.15.0"
```

Edit `src/muse/__init__.py`. Find the line:

```
the bundled modalities are:
```

(it's preceded by "As of v0.14.2"). Change to "As of v0.15.0".

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -q --timeout=300
```

Expected: all green (slow lane included).

- [ ] **Step 3: Smoke test discovery on the running tree**

```bash
python -c "
from muse.core.discovery import discover_hf_plugins, _default_hf_plugin_dirs
plugins = discover_hf_plugins(_default_hf_plugin_dirs())
print(f'Discovered {len(plugins)} HF plugins:')
for p in plugins:
    print(f'  {p[\"priority\"]:3d}  {p[\"modality\"]}  -> {p[\"runtime_path\"]}')
"
```

Expected output:
```
Discovered 4 HF plugins:
  100  audio/transcription  -> muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel
  100  chat/completion  -> muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel
  110  embedding/text  -> muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel
  200  text/classification  -> muse.modalities.text_classification.runtimes.hf_text_classifier:HFTextClassifier
```

- [ ] **Step 4: Smoke test bare-install contract**

Confirm `muse --help` and `muse pull --help` work without fastapi:

```bash
muse --help | head -3
muse pull --help | head -3
```

Expected: usage strings, no ImportError.

- [ ] **Step 5: Commit + tag**

```bash
git add pyproject.toml src/muse/__init__.py
git commit -m "chore(release): v0.15.0

Per-modality HF resolver plugin refactor (#129).

resolvers_hf.py used to grow ~3 methods + 1 sniff branch + 1 search
branch per modality, in a file that already knew about every other
modality. After this refactor, it is a thin dispatcher (~80 lines)
that iterates plugins discovered from disk. Each modality now ships
a hf.py exporting an HF_PLUGIN dict with sniff/resolve/search
callbacks plus metadata.

Adding a new modality with HF support is now a one-file drop.

Documented behaviour change: \`muse search foo\` (no modality filter)
iterates ALL plugins, not just gguf+sentence-transformers as before.
The volume increase is bounded by N (modality count); each call is
one HfApi list_models per plugin.

Closes #129."

git tag -a v0.15.0 -m "v0.15.0: per-modality HF resolver plugins"
```

- [ ] **Step 6: Push (ask the user; don't auto-push)**

Ask the user: "v0.15.0 committed and tagged locally. Push to origin?"

---

## Self-review checklist

After implementation:

1. **Spec coverage:** the spec's "What stays / what moves" section is realized in Tasks 3-6 (each migration moves the right helpers). Tasks 7-9 implement the cleanup and docs sections. ✓

2. **Placeholder scan:** `grep -nE "TBD|TODO|XXX|FIXME|placeholder" docs/superpowers/plans/2026-04-26-hf-resolver-plugin-refactor.md` should return nothing. ✓

3. **Type consistency:** `HF_PLUGIN` keys (modality, runtime_path, pip_extras, system_packages, priority, sniff, resolve, search) are consistent across discovery validation, dispatch consumption, and per-plugin authoring. `REQUIRED_HF_PLUGIN_KEYS` matches. ✓

4. **TDD discipline:** every implementation step is preceded by a failing test step, then the run-fail step, then the implementation, then run-pass, then commit. ✓

5. **Migration safety:** Tasks 3-6 each leave the system fully working (legacy fallback covers anything not yet migrated). Task 7 only triggers after all four bundled modalities have plugins. Each commit is independently green. ✓

6. **No silent behavior change:** the one user-visible change (`muse search foo` with no modality filter consults all plugins) is documented in the v0.15.0 commit message, the test file, and the spec's "behavior change" section. ✓
