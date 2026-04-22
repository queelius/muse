# Migrate MiniLM + Qwen3-Embedding to resolver (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete two redundant bundled embedding scripts and route their catalog ids through the generic `SentenceTransformerModel` runtime via curated resolver entries, adding a curated capability overlay so Qwen3-Embedding can set `trust_remote_code=True` without a bundled script.

**Architecture:** Adds one optional `capabilities:` field to curated.yaml entries. The field merges into the resolver-synthesized manifest's capabilities block during `catalog._pull_via_resolver`. Then rewrites two curated entries (`all-minilm-l6-v2`, `qwen3-embedding-0.6b`) to point at their HF repos and drops the two bundled scripts. No resolver or runtime code changes; the merged capabilities already flow into the runtime constructor via the existing `load_backend` path.

**Tech Stack:** Python 3.10+, pytest, pyyaml, existing `HFResolver` + `SentenceTransformerModel`.

**Spec:** `docs/superpowers/specs/2026-04-22-migrate-st-embeddings-to-resolver-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/core/curated.py` | modify | add `capabilities` field to `CuratedEntry`, accept it in `_entry_from_dict` |
| `src/muse/core/catalog.py` | modify | `_pull_via_resolver` accepts `capabilities_overlay`, `pull()` forwards `curated.capabilities` |
| `src/muse/curated.yaml` | modify | rename `all-minilm-l6-v2-st` to `all-minilm-l6-v2`, convert `qwen3-embedding-0.6b` from bundled to URI + overlay |
| `src/muse/models/all_minilm_l6_v2.py` | delete | generic runtime covers it |
| `src/muse/models/qwen3_embedding_0_6b.py` | delete | generic runtime + overlay cover it |
| `tests/models/test_all_minilm_l6_v2.py` | delete | script gone |
| `tests/models/test_qwen3_embedding_0_6b.py` | delete | script gone |
| `tests/core/test_curated.py` | modify | new test: capabilities field parses and round-trips |
| `tests/core/test_catalog.py` | modify | drop two ids from the `test_known_models_includes_bundled_scripts` assertion; new test for overlay merge into persisted manifest |
| `tests/core/test_discovery.py` | modify | drop two ids from the bundled-ids expected set |
| `README.md` | modify | update the bundled-scripts listing |
| `pyproject.toml` | modify | version bump 0.11.8 to 0.12.0 |

---

### Task 1: Add `capabilities` field to `CuratedEntry`

**Files:**
- Modify: `src/muse/core/curated.py`
- Test: `tests/core/test_curated.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_curated.py`:

```python
def test_load_curated_parses_capabilities_overlay():
    yaml_text = """
- id: q3e
  uri: hf://Qwen/Qwen3-Embedding-0.6B
  modality: embedding/text
  size_gb: 0.6
  description: Qwen3-Embedding 0.6B
  capabilities:
    trust_remote_code: true
    matryoshka: true
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert len(entries) == 1
    e = entries[0]
    assert e.id == "q3e"
    assert e.capabilities == {"trust_remote_code": True, "matryoshka": True}


def test_load_curated_capabilities_defaults_to_empty_dict():
    yaml_text = """
- id: minimal
  uri: hf://x/y
  modality: chat/completion
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries[0].capabilities == {}


def test_load_curated_capabilities_non_dict_is_rejected():
    yaml_text = """
- id: bad
  uri: hf://x/y
  modality: chat/completion
  capabilities: "not a dict"
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_curated.py -v -k capabilities`
Expected: FAIL with AttributeError (no `capabilities` field on CuratedEntry) and the third test FAIL because non-dict is accepted today.

- [ ] **Step 3: Add the field to `CuratedEntry`**

In `src/muse/core/curated.py`, change the dataclass:

```python
@dataclass(frozen=True)
class CuratedEntry:
    """One row in the curated recommendations YAML."""
    id: str
    bundled: bool
    uri: str | None
    modality: str | None
    size_gb: float | None
    description: str | None
    tags: tuple[str, ...]
    capabilities: dict
```

- [ ] **Step 4: Validate + project the field in `_entry_from_dict`**

In `src/muse/core/curated.py`, update `_entry_from_dict` to validate and carry the new field:

```python
def _entry_from_dict(d: dict) -> CuratedEntry:
    """Validate + project a dict from YAML onto CuratedEntry."""
    if "id" not in d:
        raise ValueError("missing required key 'id'")
    bundled = bool(d.get("bundled", False))
    uri = d.get("uri")
    if not bundled and not uri:
        raise ValueError(
            f"entry {d['id']!r}: must set either 'uri' (resolver) "
            "or 'bundled: true' (script alias)"
        )
    if bundled and uri:
        raise ValueError(
            f"entry {d['id']!r}: cannot set both 'uri' and 'bundled: true'"
        )
    caps = d.get("capabilities", {})
    if not isinstance(caps, dict):
        raise ValueError(
            f"entry {d['id']!r}: 'capabilities' must be a mapping, got {type(caps).__name__}"
        )
    return CuratedEntry(
        id=d["id"],
        bundled=bundled,
        uri=uri,
        modality=d.get("modality"),
        size_gb=d.get("size_gb"),
        description=d.get("description"),
        tags=tuple(d.get("tags", ())),
        capabilities=dict(caps),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/core/test_curated.py -v`
Expected: PASS for all curated tests (original + new).

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/curated.py tests/core/test_curated.py
git commit -m "feat(curated): capabilities overlay field on CuratedEntry"
```

---

### Task 2: Wire the overlay through `pull()` into persisted manifest

**Files:**
- Modify: `src/muse/core/catalog.py:253-320` (the `pull` function) and `src/muse/core/catalog.py:363-430` (the `_pull_via_resolver` function)
- Test: `tests/core/test_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_catalog.py`:

```python
def test_pull_via_resolver_merges_curated_capabilities_overlay(tmp_catalog):
    """Curated entries may carry a capabilities dict; it merges into the
    persisted manifest's capabilities so the runtime gets the overlay."""
    from unittest.mock import MagicMock, patch
    from muse.core.catalog import pull, _read_catalog
    from muse.core.curated import CuratedEntry
    from muse.core.resolvers import ResolvedModel

    fake_curated = CuratedEntry(
        id="my-model",
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=0.5,
        description="custom",
        tags=(),
        capabilities={"trust_remote_code": True, "custom_flag": 42},
    )

    fake_resolved = ResolvedModel(
        manifest={
            "model_id": "repo",
            "modality": "embedding/text",
            "hf_repo": "org/repo",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": {"base_caps_key": "base_val"},
        },
        backend_path="fake.mod:Cls",
        download=lambda cache_root: cache_root / "weights" / "my-model",
    )

    # resolve() is imported locally inside _pull_via_resolver
    # (`from muse.core.resolvers import resolve`), so the patch must
    # target the source module, not muse.core.catalog.
    with patch("muse.core.catalog.find_curated", return_value=fake_curated), \
         patch("muse.core.resolvers.resolve", return_value=fake_resolved), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]), \
         patch("muse.core.catalog.venv_python", return_value="/fake/py"):
        pull("my-model")

    catalog = _read_catalog()
    assert "my-model" in catalog
    persisted = catalog["my-model"]["manifest"]
    assert persisted["capabilities"] == {
        "base_caps_key": "base_val",
        "trust_remote_code": True,
        "custom_flag": 42,
    }


def test_pull_via_resolver_overlay_wins_on_collision(tmp_catalog):
    """On key collision, curated capabilities win (curated is hand-edited
    source of truth; resolver output is heuristic)."""
    from unittest.mock import patch
    from muse.core.catalog import pull, _read_catalog
    from muse.core.curated import CuratedEntry
    from muse.core.resolvers import ResolvedModel

    fake_curated = CuratedEntry(
        id="collide",
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=None,
        description=None,
        tags=(),
        capabilities={"shared_key": "curated_wins"},
    )
    fake_resolved = ResolvedModel(
        manifest={
            "model_id": "repo",
            "modality": "embedding/text",
            "hf_repo": "org/repo",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": {"shared_key": "resolver_loses"},
        },
        backend_path="fake.mod:Cls",
        download=lambda cache_root: cache_root / "weights" / "collide",
    )
    with patch("muse.core.catalog.find_curated", return_value=fake_curated), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]), \
         patch("muse.core.catalog.venv_python", return_value="/fake/py"), \
         patch("muse.core.resolvers.resolve", return_value=fake_resolved):
        pull("collide")

    persisted = _read_catalog()["collide"]["manifest"]
    assert persisted["capabilities"]["shared_key"] == "curated_wins"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_catalog.py -v -k overlay`
Expected: FAIL (overlay not merged yet; persisted capabilities lack the curated keys).

- [ ] **Step 3: Thread `capabilities_overlay` through `_pull_via_resolver`**

In `src/muse/core/catalog.py`, change `_pull_via_resolver` signature and merge capabilities into the manifest (insert the merge between `resolve(uri)` and the `model_id` assignment):

```python
def _pull_via_resolver(
    uri: str,
    *,
    model_id_override: str | None = None,
    capabilities_overlay: dict | None = None,
) -> None:
    """Pull a model via a resolver URI (e.g. hf://Qwen/Qwen3-8B-GGUF@q4_k_m).

    Looks up the resolver for the URI's scheme, calls `resolve(uri)` to
    get a synthesized ResolvedModel (manifest + backend_path + download
    callable), creates the per-model venv, installs deps, downloads the
    weights via `resolved.download()`, persists the synthesized manifest
    plus a `source: <uri>` field into catalog.json, and invalidates the
    known_models cache so the next call sees the new entry.

    `model_id_override` is set when the URI was reached via a curated
    alias (e.g. user typed `qwen3-8b-q4` which expands to
    `hf://Qwen/Qwen3-8B-Instruct-GGUF@q4_k_m`). The override replaces
    the resolver's synthesized model_id so the catalog stores the
    friendly curated id.

    `capabilities_overlay` is set when the URI was reached via a curated
    alias that declared its own `capabilities:` block. It merges into
    the resolver-synthesized manifest's `capabilities` (shallow merge;
    overlay wins on key collision). The merged block ends up in the
    persisted manifest and flows into the runtime constructor via
    `load_backend`.
    """
    from muse.core.resolvers import resolve

    resolved = resolve(uri)
    manifest = dict(resolved.manifest)
    # Resolver may put backend_path in the manifest itself, or only on
    # the ResolvedModel. Persist it consistently so load_backend can
    # find it without consulting the resolver again.
    manifest.setdefault("backend_path", resolved.backend_path)

    if capabilities_overlay:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps.update(capabilities_overlay)
        manifest["capabilities"] = merged_caps

    if model_id_override:
        manifest["model_id"] = model_id_override
    model_id = manifest["model_id"]
    # ... rest of the function unchanged (venv + download + persist)
```

(The rest of the function body is unchanged.)

- [ ] **Step 4: Pass `curated.capabilities` from `pull()` into `_pull_via_resolver`**

In `src/muse/core/catalog.py`, update the curated branch of `pull()`:

```python
    curated = find_curated(identifier)
    if curated is not None:
        if curated.uri:
            # Resolver-pulled curated entry. Override the synthesized id
            # so the catalog stores the friendly curated id (e.g.
            # qwen3-8b-q4) instead of qwen3-8b-instruct-gguf-q4-k-m.
            # Also forward the curated capabilities overlay so any
            # runtime-specific settings (trust_remote_code, chat_format,
            # context_length) land in the persisted manifest.
            _pull_via_resolver(
                curated.uri,
                model_id_override=curated.id,
                capabilities_overlay=curated.capabilities or None,
            )
            return
        # Bundled curated entry: id equals an existing bundled script's
        # model_id. Fall through to the bundled path with that id.
        _pull_bundled(curated.id)
        return
```

- [ ] **Step 5: Run the overlay tests and the whole catalog suite to verify**

Run: `pytest tests/core/test_catalog.py -v`
Expected: 50+ passing (48 prior + 2 new overlay tests).

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/catalog.py tests/core/test_catalog.py
git commit -m "feat(catalog): merge curated capabilities overlay into persisted manifest"
```

---

### Task 3: Rewrite the two curated entries in `curated.yaml`

**Files:**
- Modify: `src/muse/curated.yaml`

- [ ] **Step 1: Rewrite the MiniLM entry (rename to drop the `-st` suffix)**

In `src/muse/curated.yaml`, replace the existing MiniLM block:

```yaml
- id: all-minilm-l6-v2-st
  uri: hf://sentence-transformers/all-MiniLM-L6-v2
  modality: embedding/text
  size_gb: 0.1
  description: "MiniLM 384 dims, 22MB: CPU-friendly default embedder"
```

with:

```yaml
- id: all-minilm-l6-v2
  uri: hf://sentence-transformers/all-MiniLM-L6-v2
  modality: embedding/text
  size_gb: 0.1
  description: "MiniLM 384 dims, 22MB: CPU-friendly default embedder"
```

- [ ] **Step 2: Rewrite the Qwen3-Embedding entry to URI + overlay**

In `src/muse/curated.yaml`, replace the existing Qwen3-Embedding block:

```yaml
- id: qwen3-embedding-0.6b
  bundled: true
```

with:

```yaml
- id: qwen3-embedding-0.6b
  uri: hf://Qwen/Qwen3-Embedding-0.6B
  modality: embedding/text
  size_gb: 0.6
  description: "Qwen3-Embedding 0.6B: 1024 dims (matryoshka), 32K context, Apache 2.0"
  capabilities:
    trust_remote_code: true
```

- [ ] **Step 3: Verify the YAML parses cleanly**

Run:
```bash
python -c "from muse.core.curated import load_curated, _reset_curated_cache_for_tests; \
  _reset_curated_cache_for_tests(); \
  ents = load_curated(); \
  print([(e.id, bool(e.uri), e.capabilities) for e in ents if 'minilm' in e.id or 'qwen3-embedding' in e.id])"
```

Expected output:
```
[('all-minilm-l6-v2', True, {}), ('qwen3-embedding-0.6b', True, {'trust_remote_code': True})]
```

- [ ] **Step 4: Commit**

```bash
git add src/muse/curated.yaml
git commit -m "feat(curated): route MiniLM + Qwen3-Embedding through resolver"
```

---

### Task 4: Delete the two bundled scripts and their test files

**Files:**
- Delete: `src/muse/models/all_minilm_l6_v2.py`
- Delete: `src/muse/models/qwen3_embedding_0_6b.py`
- Delete: `tests/models/test_all_minilm_l6_v2.py`
- Delete: `tests/models/test_qwen3_embedding_0_6b.py`

- [ ] **Step 1: Delete the scripts and their tests**

```bash
git rm src/muse/models/all_minilm_l6_v2.py \
       src/muse/models/qwen3_embedding_0_6b.py \
       tests/models/test_all_minilm_l6_v2.py \
       tests/models/test_qwen3_embedding_0_6b.py
```

- [ ] **Step 2: Verify discovery no longer returns those ids**

Run:
```bash
python -c "from pathlib import Path; \
  import muse.models as m; \
  from muse.core.discovery import discover_models; \
  ids = sorted(discover_models([Path(m.__file__).parent]).keys()); \
  print(ids); \
  assert 'all-minilm-l6-v2' not in ids; \
  assert 'qwen3-embedding-0.6b' not in ids; \
  assert 'nv-embed-v2' in ids"
```

Expected output (list does not contain the two removed ids, does contain `nv-embed-v2`):
```
['bark-small', 'kokoro-82m', 'nv-embed-v2', 'sd-turbo', 'soprano-80m']
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove redundant bundled embedding scripts (generic runtime covers)"
```

---

### Task 5: Update test expectations and README

**Files:**
- Modify: `tests/core/test_discovery.py:339`
- Modify: `tests/core/test_catalog.py:54-55`
- Modify: `README.md:143`

- [ ] **Step 1: Run the full non-slow suite to see expected failures**

Run: `pytest -m "not slow" -q 2>&1 | tail -40`
Expected: Failures in `tests/core/test_discovery.py` and `tests/core/test_catalog.py` asserting those two ids are still present.

- [ ] **Step 2: Drop the two ids from `test_discovery.py` expected set**

In `tests/core/test_discovery.py` around line 339, change:

```python
            "all-minilm-l6-v2", "qwen3-embedding-0.6b", "nv-embed-v2",
```

to:

```python
            "nv-embed-v2",
```

(Remove only the two migrated ids; keep `nv-embed-v2` and any other bundled ids in the set.)

- [ ] **Step 3: Drop the two ids from `test_catalog.py` assertion**

In `tests/core/test_catalog.py` around lines 54-55, remove:

```python
    assert "all-minilm-l6-v2" in catalog
    assert "qwen3-embedding-0.6b" in catalog
```

If the surrounding test had a count assertion (like `assert len(catalog) == N`), adjust `N` down by 2 here as well. Verify by reading lines 45-75 of `tests/core/test_catalog.py` before editing.

- [ ] **Step 4: Update `README.md` bundled-scripts listing**

In `README.md` around line 143, change:

```
  - `all_minilm_l6_v2.py`, `qwen3_embedding_0_6b.py`, `nv_embed_v2.py` (embedding/text)
```

to:

```
  - `nv_embed_v2.py` (embedding/text; MiniLM and Qwen3-Embedding are now resolver-pulled via the generic runtime, see `curated.yaml`)
```

- [ ] **Step 5: Run the full non-slow suite to verify green**

Run: `pytest -m "not slow" -q 2>&1 | tail -10`
Expected: `N passed, 12 deselected` (N should equal the prior count minus however many tests were in the two deleted `tests/models/test_*` files).

- [ ] **Step 6: Commit**

```bash
git add tests/core/test_discovery.py tests/core/test_catalog.py README.md
git commit -m "test+docs: update expectations for removed embedding scripts"
```

---

### Task 6: Version bump, final sweep, tag

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change:

```toml
version = "0.11.8"
```

to:

```toml
version = "0.12.0"
```

- [ ] **Step 2: Full non-slow test sweep**

Run: `pytest -m "not slow" -q 2>&1 | tail -5`
Expected: all passing, no failures, no errors.

- [ ] **Step 3: Stage and commit the version bump**

```bash
git add pyproject.toml
git commit -m "chore(release): v0.12.0"
```

- [ ] **Step 4: Tag v0.12.0 annotated**

```bash
git tag -a v0.12.0 -m "$(cat <<'EOF'
v0.12.0: MiniLM + Qwen3-Embedding routed through resolver

Breaking (embedding/text backend_path only; same HTTP API):
  all-minilm-l6-v2 and qwen3-embedding-0.6b are now pulled via the
  generic SentenceTransformerModel runtime. Users who pulled either
  under v0.11.x must re-pull:

    muse models remove all-minilm-l6-v2
    muse models remove qwen3-embedding-0.6b
    muse pull all-minilm-l6-v2
    muse pull qwen3-embedding-0.6b

New: curated.yaml entries support an optional `capabilities:` mapping
that merges into the resolver-synthesized manifest. Used to set
`trust_remote_code: true` for Qwen3-Embedding; usable by any future
curated entry that needs runtime-specific overrides.
EOF
)"
```

- [ ] **Step 5: Verify tag**

Run: `git tag -l v0.12.0`
Expected: `v0.12.0`

---

## Success criteria

- Bundled scripts `all_minilm_l6_v2.py` and `qwen3_embedding_0_6b.py` are gone.
- `muse pull all-minilm-l6-v2` and `muse pull qwen3-embedding-0.6b` both resolve to `hf://` URIs and use `SentenceTransformerModel`.
- `~/.muse/catalog.json` entry for `qwen3-embedding-0.6b` has `manifest.capabilities.trust_remote_code == True`.
- Full non-slow test suite is green.
- Tag `v0.12.0` exists locally (not pushed).

## Verification (manual, optional)

After merge, the user can verify end-to-end on frodo:

```bash
# on frodo, after pulling v0.12.0
muse models remove all-minilm-l6-v2 qwen3-embedding-0.6b || true
muse pull all-minilm-l6-v2
muse pull qwen3-embedding-0.6b
muse serve &
# from a client:
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8000/v1', api_key='x')
r = c.embeddings.create(model='qwen3-embedding-0.6b', input=['hello world'])
print('qwen3-embedding-0.6b dims:', len(r.data[0].embedding))
r = c.embeddings.create(model='all-minilm-l6-v2', input=['hello world'])
print('all-minilm-l6-v2 dims:', len(r.data[0].embedding))
"
# Expected: dims 1024 and 384 respectively
```
