# Voice Catalog (AI-friendly discovery, phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Saved voices become a catalog that an agent can browse, search, audition, annotate and organize into categories over HTTP. No read or metadata edit ever loads a model.

**Architecture:**
- **Voice files and locking:** a stdlib-only `voice_catalog` module owns the voice files' metadata, reads and metadata writes, under one cross-process advisory lock. The existing numpy `VoiceStore` builds on it for saves.
- **Light routes:** a new generic seam, `build_light_router(catalog)` exported by a modality package, serves model-free routes. Both the worker and the gateway mount it; the gateway mounts it ahead of its catch-all proxy, so these routes answer from disk without a worker.
- **Discovery:** the capabilities document gains `options.voices` on saved-voice models.
- **MCP:** the generated MCP tools learn path parameters, `PATCH` and `DELETE`.

**Tech Stack:** Python 3.10+, FastAPI and pydantic v2, stdlib `fcntl` advisory locks through `muse.core.securefs`, and pytest with FastAPI `TestClient`.

**Spec:** `docs/superpowers/specs/2026-11-24-ai-friendly-discovery-design.md` (phase 1 only).

## Global Constraints

- **Discovery never loads a model:** discovery and the voice routes `GET /voices`, `GET /voices/{name}`, `GET /voices/{name}/audio`, `PATCH /voices/{name}` and `DELETE /voices/{name}` never load a model or spawn a worker. Only `POST /voices` (decode and save) goes to the worker.
- **Light modules stay light:** `voice_errors.py`, `voice_catalog.py` and `light.py` in `audio_speech/` import only the stdlib, fastapi, starlette, pydantic, `muse.core.*` and each other at module level. No numpy, torch or runtimes.
- **`category`:** a slash path of 1-4 segments, each `[a-z0-9][a-z0-9_-]{0,31}`, or null.
- **`tags`:** at most 16, each `[a-z0-9][a-z0-9_-]{0,31}`, deduplicated and sorted.
- **`description`:** at most 500 characters.
- **`notes`:** at most 2000 characters.
- **`design`:** null or `{model, instructions}`. `instructions` is at most 4096 characters; `model` is at most 256 characters (my call).
- **`updated_at`:** ISO 8601 UTC, set by the server on every save and metadata change.
- **Errors:**
  - A metadata rule violation is 400 `invalid_voice_metadata`, and the message names the field and the rule.
  - A wrong JSON type is the existing 422 `invalid_request` envelope.
  - `PATCH` or `DELETE` on a bundled voice is 409 `voice_read_only`.
- **Old voice files still load:** `voice.json` files without the new fields load unchanged; missing fields read as null or `[]`.
- **Bundled voices are read-only:** `narrator` gains `category: "narrator/neutral"`, tags, `description` and `design`.
- **Compatibility:** `GET /voices` keeps the `voices` name list. Existing MCP tool names stay; new ones are `muse_get_voice`, `muse_get_voice_audio`, `muse_update_voice` and `muse_delete_voice` (the last is destructive).
- **No auth on voice writes:** anyone who can reach the server may save, edit and delete voices; there is no admin token (owner decision).
- **Isolated tests:** tests use a tmp `MUSE_HOME` or `paths.home` and never touch `~/.muse`.
- **Commits:** Conventional Commits with a scope (`feat(audio-speech): ...`), each ending with the session's `Co-Authored-By` line.
- **Worktree:** work in a git worktree. Run tests with `PYTHONPATH=$PWD/src`, and run the fast lane via `python scripts/preflight.py` before the final review.

## Review Focus

1. **Voices saved before this change:** `brineknife_captain` and the 10 epic narrators have no new fields. They list, get and PATCH fine, and PATCH adds only the fields sent. Tested in Task 2.
2. **No voices root yet:** on a fresh install the root does not exist. A list returns only the bundled `narrator`, and no read creates a directory or lock file. Tested in Task 2.
3. **Hostile or odd names in the URL:** `GET /voices/..%2Fetc`, uppercase, or 65 characters. These return 404 `voice_not_found` and never touch a path outside the root. Tested in Task 6.
4. **Human-typed metadata:** `"Narrator/Epic/"`, `" Male , GRAVELLY "`, or an empty string to clear a field. Case and spaces are normalized (lowercased, trimmed, outer slashes stripped), and an empty string means null. Tested in Task 1.
5. **A save racing a read across processes:** a list or get running in the gateway while the worker overwrites the same voice sees either the old voice or the new one, never a mix. Tested in Task 2 (the lock blocks the reader).

---

### Task 1: Voice errors and metadata rules (pure)

**Files:**
- Create: `src/muse/modalities/audio_speech/voice_errors.py`
- Create: `src/muse/modalities/audio_speech/voice_catalog.py` (rules section only in this task)
- Modify: `src/muse/modalities/audio_speech/protocol.py`: move `VoiceError`, `UnknownVoiceError` and `VoiceUnreadableError` into `voice_errors.py` and re-import them, so existing import paths keep working.
- Test: `tests/modalities/audio_speech/test_voice_catalog_rules.py`

**Interfaces:**
- Produces:
  - `normalize_metadata(fields: Mapping[str, Any], *, partial: bool) -> dict`
    - Returns only the known keys present: `category`, `tags`, `description`, `notes`, `design`.
    - Raises `VoiceError("invalid_voice_metadata", msg, 400)`.
    - With `partial=False`, the absent keys are filled with `None` (and `[]` for `tags`).
  - `VoiceEntry` (frozen dataclass), fields:
    - `name: str`
    - `transcript: str`
    - `sample_rate: int`
    - `duration_seconds: float`
    - `sha256: str | None`
    - `read_only: bool`
    - `category: str | None`
    - `tags: tuple[str, ...]`
    - `description: str | None`
    - `notes: str | None`
    - `design: dict | None`
    - `created_at: str | None`
    - `updated_at: str | None`
  - `VoiceEntry.to_dict(model_id: str) -> dict` returns the wire shape: `object: "audio.voice"`, `model`, every field above, and `audio_url: f"/v1/audio/speech/voices/{name}/audio?model={model_id}"`.
  - `entry_from_meta(name: str, meta: Mapping, *, read_only: bool) -> VoiceEntry`. This is lenient: a malformed optional field reads as null or `[]`; a missing required field raises `VoiceUnreadableError`.
  - `filter_entries(entries, *, category: str | None, tags: Sequence[str], q: str | None) -> list[VoiceEntry]`
  - `category_counts(entries) -> list[dict]`, giving `[{"path", "count"}]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_normalize_lowercases_trims_and_strips_slashes():
    out = normalize_metadata({"category": " Narrator/Epic/ ", "tags": " Male , GRAVELLY ,male"}, partial=True)
    assert out == {"category": "narrator/epic", "tags": ["gravelly", "male"]}

def test_tags_accept_list_or_comma_string():
    assert normalize_metadata({"tags": ["b", "a"]}, partial=True)["tags"] == ["a", "b"]

def test_empty_string_and_none_clear():
    assert normalize_metadata({"category": "", "notes": None}, partial=True) == {"category": None, "notes": None}

def test_full_fills_absent_keys():
    assert normalize_metadata({}, partial=False) == {
        "category": None, "tags": [], "description": None, "notes": None, "design": None}

@pytest.mark.parametrize("fields,needle", [
    ({"category": "a/b/c/d/e"}, "category"),          # 5 segments
    ({"category": "bad seg"}, "category"),
    ({"tags": [f"t{i}" for i in range(17)]}, "tags"),
    ({"description": "x" * 501}, "description"),
    ({"notes": "x" * 2001}, "notes"),
    ({"design": {"model": "m", "instructions": "x" * 4097}}, "design"),
    ({"design": {"model": "m", "extra": 1}}, "design"),
])
def test_rule_violations_raise_invalid_voice_metadata(fields, needle):
    with pytest.raises(VoiceError) as e:
        normalize_metadata(fields, partial=True)
    assert e.value.code == "invalid_voice_metadata" and e.value.status == 400 and needle in str(e.value)

def test_entry_from_old_meta_has_null_metadata():
    e = entry_from_meta("old", {"name": "old", "transcript": "t", "sample_rate": 24000,
                                "duration_seconds": 2.0, "sha256": "ab"}, read_only=False)
    assert (e.category, e.tags, e.description, e.design, e.updated_at) == (None, (), None, None, None)

def test_filter_category_matches_subtree_tags_and_q():
    es = [_e("a", category="narrator/epic", tags=("deep",)), _e("b", category="narrator"),
          _e("c", category="npc/guard", description="Gruff city guard")]
    assert [e.name for e in filter_entries(es, category="narrator", tags=[], q=None)] == ["a", "b"]
    assert [e.name for e in filter_entries(es, category=None, tags=["deep"], q=None)] == ["a"]
    assert [e.name for e in filter_entries(es, category=None, tags=[], q="GRUFF")] == ["c"]

def test_category_counts_include_ancestors():
    es = [_e("a", category="narrator/epic"), _e("b", category="narrator/epic"), _e("c", category="narrator")]
    assert category_counts(es) == [{"path": "narrator", "count": 3}, {"path": "narrator/epic", "count": 2}]
```

`_e` is a local helper that builds a `VoiceEntry` with defaults.

- [ ] **Step 2: Run the tests and see them fail**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech/test_voice_catalog_rules.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement it**

- Move the three exceptions to `voice_errors.py` unchanged. `protocol.py` then does `from muse.modalities.audio_speech.voice_errors import VoiceError, UnknownVoiceError, VoiceUnreadableError`.
- Write the `voice_catalog.py` rules using the limits in Global Constraints.
- `q` matches as a case-insensitive substring of name, category, description, notes and tags.

- [ ] **Step 4: Run the new tests plus the existing speech tests**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech -q`
Expected: PASS.

- [ ] **Step 5: Commit** with `feat(audio-speech): voice metadata rules and stdlib voice errors`.

---

### Task 2: VoiceCatalog storage with a cross-process lock

**Files:**
- Modify: `src/muse/modalities/audio_speech/voice_catalog.py` (add the storage part)
- Test: `tests/modalities/audio_speech/test_voice_catalog_store.py`
- Test: `tests/modalities/audio_speech/test_light_imports.py` (the import allowlist)

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `DATA_DIR: Path` points at the package's `data/` directory.
  - `BUNDLED_VOICES: dict[str, tuple[Path, Path]]` is `{"narrator": (DATA_DIR / "narrator.wav", DATA_DIR / "narrator.json")}`.
  - `class VoiceCatalog`:
    - `@classmethod for_root(cls, root: Path, *, bundled: Mapping[str, tuple[Path, Path]] = BUNDLED_VOICES) -> VoiceCatalog` is a process-wide memo keyed by (resolved root, bundled items). Every caller in a process shares one instance, and so one in-process lock.
    - `@classmethod for_model(cls, model_id: str) -> VoiceCatalog` is shorthand for `for_root(voice_store_root(model_id))`. Move `voice_store_root` from `voices.py` to here and re-export it from `voices.py`.
    - `lock(*, exclusive: bool) -> ContextManager[None]`
    - `names() -> list[str]` lists bundled voices first, then saved ones sorted (same order as today's `VoiceStore.names`).
    - `entries() -> list[VoiceEntry]`
    - `entry(name: str) -> VoiceEntry` raises `UnknownVoiceError` (404 `voice_not_found`).
    - `audio_bytes(name: str) -> bytes`
    - `update(name: str, patch: Mapping[str, Any]) -> VoiceEntry` raises `VoiceError("voice_read_only", ..., 409)` for a bundled voice.
    - `delete(name: str) -> None` raises the same 409 for a bundled voice.
    - `saved_dir_unlocked(name: str) -> Path` and `saved_names_unlocked() -> list[str]` are for `VoiceStore` (Task 3), which calls them while holding `lock(exclusive=True)`.
  - `NAME_PATTERN` moves here; `voices.py` re-exports it.

**The lock:**
- The lock file is `<root>/.lock`, opened with `securefs.open_lock_file(..., create=True)` and taken with `securefs.lock_file(handle, exclusive=...)`.
- `lock()` holds the instance's `threading.RLock` for the whole block. At depth 0 it takes the file lock on a fresh handle; nested calls reuse it. A nested exclusive request under a shared outer lock raises `RuntimeError("voice lock upgrade")`.
- A shared lock when `root` does not exist yields without creating anything. An exclusive lock first runs `securefs.ensure_directory(root, error=VoiceUnreadableError, label="voices directory")`.
- Reads (`names`, `entries`, `entry`, `audio_bytes`) take the shared lock. `update` and `delete` take the exclusive lock.
- Directory scanning keeps today's `_saved_names` rules: skip names that fail `NAME_PATTERN` (so `.lock`, `.tmp-*` and `.old-*` are skipped), symlinks, and directories without a readable `reference.wav` and `voice.json`.
- `entries()` memoizes each parsed `voice.json` by `(name, securefs.file_identity(...))`, so repeat listings only stat the files.

**Writes:**
- `update` merges `normalize_metadata(patch, partial=True)` into the stored meta, sets `updated_at` to now in UTC ISO format, and replaces `voice.json` with `securefs.atomic_write_text(..., error=VoiceUnreadableError)`. It never touches `reference.wav`, `transcript` or `created_at`.
- `delete` keeps today's rename-to-`.old-*`-then-rmtree.

- [ ] **Step 1: Write the failing tests**

Each test sets `paths.home` to `tmp_path` via `monkeypatch.setenv("MUSE_HOME", ...)` and `config.reset_config()`.

```python
def test_fresh_install_lists_only_bundled_and_creates_nothing(tmp_path):
    cat = VoiceCatalog.for_root(tmp_path / "voices" / "m")
    assert cat.names() == ["narrator"]
    assert [e.name for e in cat.entries()] == ["narrator"]
    assert not (tmp_path / "voices").exists()

def test_old_voice_json_lists_and_patch_adds_only_sent_fields(tmp_path):
    root = _write_voice(tmp_path, "brineknife_captain", meta_extra={})   # today's fields only
    cat = VoiceCatalog.for_root(root)
    e = cat.update("brineknife_captain", {"category": "npc/pirate/captain"})
    assert e.category == "npc/pirate/captain" and e.tags == () and e.updated_at
    meta = json.loads((root / "brineknife_captain" / "voice.json").read_text())
    assert meta["transcript"] == "Ahoy." and meta["created_at"] == "2026-11-20T00:00:00+00:00"

def test_update_and_delete_bundled_are_read_only(tmp_path):
    cat = VoiceCatalog.for_root(tmp_path / "v")
    for call in (lambda: cat.update("narrator", {"notes": "x"}), lambda: cat.delete("narrator")):
        with pytest.raises(VoiceError) as e:
            call()
        assert (e.value.code, e.value.status) == ("voice_read_only", 409)

def test_unknown_and_bad_names_are_voice_not_found(tmp_path):
    cat = VoiceCatalog.for_root(_write_voice(tmp_path, "a"))
    for bad in ("nope", "../etc", "A", "x" * 65):
        with pytest.raises(UnknownVoiceError):
            cat.entry(bad)

def test_corrupt_voice_json_is_skipped_in_list_and_unreadable_on_get(tmp_path):
    root = _write_voice(tmp_path, "bad"); (root / "bad" / "voice.json").write_text("{")
    cat = VoiceCatalog.for_root(root)
    assert "bad" not in cat.names()
    with pytest.raises(VoiceUnreadableError):     # today's behaviour: the directory exists, so 500, not 404
        cat.entry("bad")

def test_reader_in_another_process_waits_for_exclusive_lock(tmp_path):
    root = _write_voice(tmp_path, "a")
    child = _spawn_lock_holder(root, hold_seconds=0.6)   # subprocess: for_root(root).lock(exclusive=True), print "held", sleep, exit
    child.stdout.readline()                               # "held"
    t0 = time.monotonic(); VoiceCatalog.for_root(root).entries(); waited = time.monotonic() - t0
    child.wait(timeout=10)
    assert waited >= 0.4

def test_nested_lock_reuses_and_upgrade_raises(tmp_path):
    cat = VoiceCatalog.for_root(_write_voice(tmp_path, "a"))
    with cat.lock(exclusive=True):
        assert cat.names()                                # nested shared under exclusive: fine
    with cat.lock(exclusive=False):
        with pytest.raises(RuntimeError, match="upgrade"):
            with cat.lock(exclusive=True):
                pass
```

`_write_voice(tmp_path, name, meta_extra=None)` writes `<tmp>/voices/m/<name>/` with a 2 s silent 24 kHz `reference.wav` and today's `voice.json` fields (`transcript: "Ahoy."`, `created_at: "2026-11-20T00:00:00+00:00"`, `sha256`, `sample_rate`, `duration_seconds`, `source_filename: null`) plus `meta_extra`. It returns the root.

`test_light_imports.py` parses each of `voice_errors.py`, `voice_catalog.py` and `light.py` (skipped while that file is missing) with `ast`. It asserts every top-level `import` / `from` names a module in `sys.stdlib_module_names`, or starts with `fastapi`, `starlette`, `pydantic`, `muse.core` or `muse.modalities.audio_speech.voice_`.

- [ ] **Step 2: Run the tests and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech/test_voice_catalog_store.py tests/modalities/audio_speech/test_light_imports.py -q`

- [ ] **Step 3: Implement it** following the interface and lock rules above.

- [ ] **Step 4: Run the tests and see them pass.** Same command, then all of `tests/modalities/audio_speech`.

- [ ] **Step 5: Commit** with `feat(audio-speech): stdlib voice catalog with cross-process lock`.

---

### Task 3: VoiceStore on the catalog; metadata on save; bundled narrator metadata

**Files:**
- Modify: `src/muse/modalities/audio_speech/voices.py`
- Modify: `src/muse/modalities/audio_speech/runtimes/qwen3_tts.py` (`bundled_voices()` and `save_voice`)
- Modify: `src/muse/modalities/audio_speech/data/narrator.json`
- Test: `tests/modalities/audio_speech/test_voice_store.py` (extend)
- Test: `tests/modalities/audio_speech/test_qwen3_tts*.py` and `tests/models/test_qwen3_tts_0_6b_base.py`, if present (extend)

**Interfaces:**
- Consumes: `VoiceCatalog.for_root`, `lock`, `saved_names_unlocked`, `saved_dir_unlocked`, `normalize_metadata`, `BUNDLED_VOICES` (Task 2).
- Produces:
  - `VoiceStore.save(name, audio, sample_rate, transcript, *, overwrite=False, source_filename=None, metadata: Mapping[str, Any] | None = None) -> StoredVoice`
  - `Qwen3TTSRuntime.save_voice(..., metadata: Mapping[str, Any] | None = None)` passes it through.
  - `VoiceStore.catalog: VoiceCatalog`

- [ ] **Step 1: Write the failing tests**

```python
def test_save_writes_metadata_and_updated_at(tmp_path):
    store = VoiceStore(tmp_path / "v")
    store.save("grimbold", _audio(2.0), 24000, "Welcome, traveler.",
               metadata={"category": "npc/dwarf/blacksmith", "tags": "male,gravelly",
                         "design": {"model": "qwen3-tts-1.7b-voicedesign", "instructions": "gruff dwarf"}})
    e = store.catalog.entry("grimbold")
    assert (e.category, e.tags, e.design["model"]) == ("npc/dwarf/blacksmith", ("gravelly", "male"), "qwen3-tts-1.7b-voicedesign")
    assert e.updated_at and e.created_at

def test_save_bad_metadata_is_400_and_writes_nothing(tmp_path):
    store = VoiceStore(tmp_path / "v")
    with pytest.raises(VoiceError) as ex:
        store.save("g", _audio(2.0), 24000, "t", metadata={"category": "Bad Cat!"})
    assert ex.value.code == "invalid_voice_metadata"
    assert store.names() == []

def test_store_and_catalog_share_one_lock(tmp_path):
    store = VoiceStore(tmp_path / "v")
    assert store.catalog is VoiceCatalog.for_root(tmp_path / "v", bundled={})

def test_bundled_narrator_has_metadata():
    e = VoiceCatalog.for_root(Path("/nonexistent"), bundled=BUNDLED_VOICES).entry("narrator")
    assert e.category == "narrator/neutral" and e.read_only and e.description and e.design["model"] == "qwen3-tts-1.7b-voicedesign"
```

The existing `test_voice_store.py` cases (overwrite, voice limit, bundled read-only, corrupt voices, half-written directories) must still pass unchanged.

- [ ] **Step 2: Run them and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech/test_voice_store.py -q`

- [ ] **Step 3: Implement it**

**`VoiceStore`:**
- It keeps its constructor signature.
- It builds `self.catalog = VoiceCatalog.for_root(root, bundled={n: (sv.audio_path, sv.meta_path) for n, sv in bundled.items()})`.
- It replaces `self._lock` with `self.catalog.lock(...)`:
  - `get`, `get_with_audio` and `names` take the shared lock.
  - `save` and `delete` take the exclusive lock.
  - Inside an exclusive block, use the `*_unlocked` helpers, never a public locked method that would take a new file handle.
- `save` validates metadata with `normalize_metadata(metadata or {}, partial=False)` before writing anything. It writes the normalized fields plus `updated_at` (equal to `created_at` on a save) into `voice.json`.
- `get` stays as today: it returns `StoredVoice`, and the synthesis path is unchanged.

**Runtime:** `bundled_voices()` iterates `BUNDLED_VOICES`.

**`narrator.json`:** add `"category": "narrator/neutral"`, `"tags": ["neutral", "clear", "adult"]` and `"design": {"model": "qwen3-tts-1.7b-voicedesign", "instructions": <the runtime's DEFAULT_INSTRUCTIONS text, verbatim>}`. Keep `description` and `generated_by`.

- [ ] **Step 4: Run all speech tests and the Qwen3-TTS model tests.**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech tests/models -q -k "voice or qwen3 or speech"`
Expected: PASS.

- [ ] **Step 5: Commit** with `feat(audio-speech): VoiceStore uses the voice catalog; metadata on save`.

---

### Task 4: Metadata form fields on POST /v1/audio/speech/voices

**Files:**
- Modify: `src/muse/modalities/audio_speech/routes.py` (`save_voice`)
- Test: `tests/modalities/audio_speech/test_routes_saved_voices.py` (extend)

**Interfaces:**
- Consumes: `backend.save_voice(..., metadata=...)` (Task 3).
- Produces:
  - Optional `Form` fields: `category: str | None`, `tags: str | None` (comma-separated), `description: str | None`, `notes: str | None`, `design_model: str | None`, `design_instructions: str | None`. Each has a `description=` that states its rule from Global Constraints.
  - `design` is `{"model": design_model, "instructions": design_instructions}` when either is given, else omitted.
  - The response keeps `object`, `model`, `name`, `duration_seconds` and `sha256`, and adds `category`, `tags`, `description`, `notes` and `design`, read back from `backend.voice_entry(name)` when the backend has one. Add `Qwen3TTSRuntime.voice_entry(name) -> VoiceEntry` (a shared-lock catalog read). For a backend without it (the test fakes), read them from the normalized metadata.

- [ ] **Step 1: Write the failing tests**

```python
def test_save_passes_metadata_fields():
    fake = FakeTTS(); c = _client(fake, caps={"supports_saved_voices": True})
    r = c.post("/v1/audio/speech/voices", files={"audio": ("a.wav", _wav(2.0), "audio/wav")},
               data={"name": "grimbold", "transcript": "Hi.", "category": "npc/dwarf",
                     "tags": "male, gravelly", "design_instructions": "gruff"})
    assert r.status_code == 200
    md = fake.saved[0]["metadata"]
    assert md["category"] == "npc/dwarf" and md["tags"] == "male, gravelly" and md["design"] == {"model": None, "instructions": "gruff"}

def test_save_invalid_metadata_is_400_envelope():
    fake = FakeTTS(save_raises=VoiceError("invalid_voice_metadata", "category: bad segment", 400))
    r = _client(fake, caps={"supports_saved_voices": True}).post(... same upload ...)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_voice_metadata"
```

Update `FakeTTS.save_voice` to accept and record `metadata`. `design` keeps `None` for an absent half; `normalize_metadata` then rejects or keeps it. Make the rule explicit in `normalize_metadata`: a `design` with `model: None` is allowed, but at least one of the two must be a non-empty string.

- [ ] **Step 2: Run the test and see it fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech/test_routes_saved_voices.py -q`

- [ ] **Step 3: Implement it**

- Add the form fields and pass `metadata={k: v for k, v in {...}.items() if v is not None}`.
- In `src/muse/modalities/audio_speech/__init__.py`, extend the `POST /v1/audio/speech/voices` DISCOVERY `example` with realistic `category`, `tags`, `description`, `design_model` and `design_instructions` values. The existing meta-test checks those names exist on the route.

- [ ] **Step 4: Run the tests and see them pass**, plus `tests/core/test_discovery_meta.py`.

- [ ] **Step 5: Commit** with `feat(audio-speech): voice metadata fields on save`.

---

### Task 5: The light-route seam (core, worker, discovery validator, meta-test)

**Files:**
- Create: `src/muse/core/light.py`
- Modify: `src/muse/core/discovery.py`
  - Add `light_hooks(dirs)`.
  - `_valid_discovery_route` accepts `PATCH` and the optional `light` key.
  - Refactor the package loop in `discover_modalities` into a shared `_modality_packages(dirs) -> Iterator[tuple[str, Path, ModuleType]]` that keeps today's first-found-wins and logging behaviour.
- Modify: `src/muse/core/errors.py`: add `validation_error_response(exc: RequestValidationError) -> JSONResponse`, moved from `create_app`'s handler body, which then calls it.
- Modify: `src/muse/cli_impl/worker.py`: mount each light router after the modality routers.
- Modify: `tests/core/test_discovery_meta.py`
- Test: `tests/core/test_light_seam.py`

**Interfaces:**
- Produces (`muse.core.light`):
  - `@dataclass(frozen=True) class LightModel` with:
    - `model_id: str`
    - `capabilities: Mapping[str, Any]`
    - `backend: Any = None` (the in-process model in a worker; `None` in the gateway)
  - `class LightRequestError(Exception)` takes `(status: int, code: str, message: str)`.
  - `class LightCatalog(Protocol)` declares:
    - `resolve(self, modality: str, model_id: str | None) -> LightModel`, which raises `LightRequestError`.
    - `async forward(self, request: Request, model_id: str) -> Response`
  - `class RegistryLightCatalog(LightCatalog)` wraps a `ModalityRegistry`:
    - `resolve` uses `registry.get_or_404` and turns its `ModelNotFoundError` into `LightRequestError(404, "model_not_found", ...)`. `None` picks the registry default.
    - `forward` raises `RuntimeError`, because a worker always has the backend.
  - `async def resolve_light(catalog, modality, model_id) -> LightModel | JSONResponse` runs `resolve` in `asyncio.to_thread` and maps `LightRequestError` to `error_response`.
- Produces (`muse.core.discovery`):
  - `LightHooks` (NamedTuple) holds `build_light_router: Callable[[LightCatalog], APIRouter] | None` and `model_options: Callable[[str, Mapping], dict | None] | None`.
  - `light_hooks(dirs: list[Path] | None = None) -> dict[str, LightHooks]` covers the packages that export either name from `__init__`.
- DISCOVERY route keys:
  - Optional `"light": bool`.
  - `"method"` may be `"PATCH"`, which requires `request_content_type == "application/json"`.

- [ ] **Step 1: Write the failing tests**

In `tests/core/test_light_seam.py`, use a toy modality package written under `tmp_path`, the same way the existing discovery tests do. It exports `MODALITY = "toy/x"`, `build_router`, `build_light_router` (a `GET /v1/toy/ping` that calls `resolve_light` and returns `{"model": m.model_id}`) and a DISCOVERY with that route marked `"light": True`.

```python
def test_light_hooks_found(tmp_path): assert "toy/x" in light_hooks([_toy(tmp_path)])
def test_worker_style_mount_serves_light_route(tmp_path):
    reg = ModalityRegistry(); reg.register("toy/x", _Fake("t1"), manifest={...})
    hooks = light_hooks([_toy(tmp_path)])["toy/x"]
    app = create_app(registry=reg, routers={"toy/x:light": hooks.build_light_router(RegistryLightCatalog(reg))})
    assert TestClient(app).get("/v1/toy/ping").json() == {"model": "t1"}
    assert TestClient(app).get("/v1/toy/ping?model=nope").json()["error"]["code"] == "model_not_found"
def test_validator_accepts_patch_json_and_light_flag(): ...
def test_validator_rejects_patch_without_json(): ...
```

In `tests/core/test_discovery_meta.py`:
- `_api_routes(tag)` also includes `light_hooks()[tag].build_light_router(RegistryLightCatalog(ModalityRegistry()))` routes when present.
- Add `test_light_flag_matches_light_router`: a declared route has `light: True` exactly when its `(method, path)` is served by the light router.
- `test_examples_match_route_request_models` partitions each example's keys into path parameters (`{name}` in the path), query parameters and body. It validates the JSON body part with path and query keys removed, and accepts `PATCH`.

- [ ] **Step 2: Run the tests and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/core/test_light_seam.py tests/core/test_discovery_meta.py tests/core/test_discovery.py -q`

- [ ] **Step 3: Implement it**

- Write `muse.core.light` as specified above.
- Add the discovery changes.
- In `worker.py`, for each `(tag, hooks)` in `light_hooks(modality_dirs())` with a router, set `routers[f"{tag}:light"] = hooks.build_light_router(RegistryLightCatalog(registry))`.
- Move the validation-envelope body into `muse.core.errors.validation_error_response`.

- [ ] **Step 4: Run the tests and see them pass**, plus `tests/cli_impl/test_worker*.py` and `tests/core`.

- [ ] **Step 5: Commit** with `feat(discovery): light routes that answer without a model`.

---

### Task 6: Speech light router (list, get, audio, PATCH, DELETE) and its DISCOVERY

**Files:**
- Create: `src/muse/modalities/audio_speech/light.py`
- Modify: `src/muse/modalities/audio_speech/routes.py`: remove `GET /speech/voices` and `DELETE /speech/voices/{name}`, now served by `light.py`.
- Modify: `src/muse/modalities/audio_speech/__init__.py`:
  - Export `build_light_router`.
  - DISCOVERY: add the three new routes, set `light: true` on the five light routes, and name MCP tools on GET one, audio, PATCH and DELETE.
  - Give every route a realistic example, with `"<model>"` for the model as today.
  - Add the recipe to `summary` / `use_for`: design a take on `qwen3-tts-1.7b-voicedesign` with `instructions`, save it with `POST /voices` including `category`, `description` and `design_instructions`, then speak with `voice=<name>` on `qwen3-tts-0.6b-base`. Browse with `?category=` and audition with `/voices/{name}/audio`.
- Modify: `docs/MODALITIES.md` (the audio/speech voices section) and the CLAUDE.md audio/speech bullet.
- Test: `tests/modalities/audio_speech/test_light_routes.py`
- Modify: `tests/modalities/audio_speech/test_routes_saved_voices.py`: move the DELETE tests here, against the light router.

**Interfaces:**
- Consumes: `resolve_light`, `LightModel` (Task 5), and `VoiceCatalog.for_model`, `filter_entries` and `category_counts` (Tasks 1-2).
- Produces: `build_light_router(catalog: LightCatalog) -> APIRouter` with prefix `/v1/audio`. Every handler is `async`, and all disk work runs in `asyncio.to_thread`. `light.py` uses the literal `"audio/speech"` rather than importing the package `__init__`, which keeps it inside the import allowlist.

| Route | Parameters | 200 response |
|---|---|---|
| `GET /speech/voices` | `model`, `category`, `tag` (repeatable), `q` (query) | `{"object": "list", "model", "voices": [names of returned entries], "data": [entry dicts], "categories": category_counts(all entries)}` |
| `GET /speech/voices/{name}` | `model` (query) | entry dict |
| `GET /speech/voices/{name}/audio` | `model` (query) | `audio/wav` bytes, with `Content-Disposition: inline; filename="<name>.wav"` |
| `PATCH /speech/voices/{name}` | JSON body `VoicePatch` | entry dict |
| `DELETE /speech/voices/{name}` | `model` (query) | `{"object": "audio.voice.deleted", "model", "name", "deleted": true}` (unchanged) |

`VoicePatch` is a pydantic model with `extra="forbid"`. Every field below has a `description`:
- `model: str | None`
- `category: str | None`
- `tags: list[str] | str | None`
- `description: str | None`
- `notes: str | None`
- `design: dict | None`

Only the fields actually sent (`model_fields_set`, minus `model`) go to `catalog.update`.

Behaviour by model:
- **Saved-voice model** (`capabilities.supports_saved_voices`): every route uses `VoiceCatalog.for_model(model_id)`.
- **`GET /speech/voices` on another model with a `backend`** (worker): list `backend.voices` as minimal entries `{"object": "audio.voice", "model", "name", "read_only": true}`, with `categories: []` and the same filters applied.
- **`GET /speech/voices` on another model without a `backend`** (gateway): `return await catalog.forward(request, model_id)`.
- **Any other route on a model without saved voices:** 400 `voice_saving_not_supported` (today's message).

Errors:
- `UnknownVoiceError` is 404 `voice_not_found`.
- `VoiceError` returns its own status and code.
- `VoiceUnreadableError` is 500 `voice_unreadable`, logged.

- [ ] **Step 1: Write the failing tests**

Use a worker-style app (`create_app` plus `routes.build_router` plus `build_light_router(RegistryLightCatalog(reg))`), with tmp `MUSE_HOME` and a fake backend whose capabilities set `supports_saved_voices: True`.

```python
def test_list_has_data_categories_and_filters(): ...   # save two voices via VoiceCatalog fixtures; ?category=narrator&tag=deep
def test_get_one_and_audio_url():
    e = c.get("/v1/audio/speech/voices/narrator?model=fake-tts").json()
    assert e["read_only"] and e["audio_url"] == "/v1/audio/speech/voices/narrator/audio?model=fake-tts"
def test_audio_returns_reference_wav():
    r = c.get("/v1/audio/speech/voices/narrator/audio?model=fake-tts")
    assert r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
def test_patch_sets_and_clears_fields():
    c.patch("/v1/audio/speech/voices/g?model=fake-tts" ...)  # body {"model": "fake-tts", "category": "npc/guard", "notes": "x"}, then {"notes": None}
def test_patch_unknown_field_is_422_envelope(): ...      # {"colour": "red"} -> 422 invalid_request
def test_patch_bundled_is_409(): ...
@pytest.mark.parametrize("bad", ["..%2Fetc", "NARRATOR", "x" * 65, "nope"])
def test_bad_names_are_404_voice_not_found(bad): ...
def test_preset_model_lists_backend_voices():           # caps without supports_saved_voices, backend.voices = ["af_heart"]
    assert c.get("/v1/audio/speech/voices?model=kokoro").json()["voices"] == ["af_heart"]
def test_preset_model_patch_is_voice_saving_not_supported(): ...
def test_routes_never_touch_backend_for_saved_voice_model(): ...   # backend is a sentinel object() with no methods
```

- [ ] **Step 2: Run them and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech -q`

- [ ] **Step 3: Implement** `light.py`, the route removals in `routes.py` and the DISCOVERY edits. Then update the docs.

- [ ] **Step 4: Run the speech tests, the discovery meta-tests and the light-import test**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech tests/core/test_discovery_meta.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** with `feat(audio-speech): voice catalog routes served without a model`.

---

### Task 7: Gateway serves light routes from disk

**Files:**
- Create: `src/muse/cli_impl/light_catalog.py` (`GatewayLightCatalog`)
- Modify: `src/muse/cli_impl/gateway.py`:
  - Mount light routers after `install_agent_routes` and before `proxy`.
  - `extract_model_from_request` treats `PATCH` like `POST`.
  - Non-admin `RequestValidationError` uses `validation_error_response`: in `muse/admin/errors.py` `_validation_handler`, the non-admin branch calls it.
- Modify: `src/muse/cli_impl/agent_routes.py`: `_modality_schema` also includes the modality's light router, built over `RegistryLightCatalog(ModalityRegistry())`, so `/openapi.json` documents the light routes.
- Modify: `docs/AGENT_DISCOVERY.md` (the light-routes paragraph) and CLAUDE.md "Agent discovery" (one bullet).
- Test: `tests/cli_impl/test_gateway_light_routes.py`

**Interfaces:**
- Consumes: `LightCatalog`, `LightModel`, `LightRequestError`, `light_hooks` (Task 5).
- Produces: `GatewayLightCatalog(state, timeout: float, cancellable_paths: frozenset[str])`.
  - **`resolve(modality, model_id)`:**
    - `None` is 400 `model_required` (no phase-1 light route is model-optional).
    - An id failing `_valid_model_id` is 400 `invalid_model`, with the proxy's message.
    - A catalog read `CatalogError` is 503 `catalog_unavailable`.
    - An id not in `_read_catalog()` / `known_models()` is 404 `model_not_found`.
    - `enabled: false` is 409 `model_disabled`, with the proxy's message.
    - A manifest `modality` that is not `modality` is 404 `model_not_found`.
    - Otherwise it returns `LightModel(model_id, capabilities=get_manifest(model_id)["capabilities"] or {})`.
  - **`forward(request, model_id)`:** `await _route_via_director(request, request.url.path.lstrip("/"), model_id, state, timeout, cancellable_paths=cancellable_paths)`.

- [ ] **Step 1: Write the failing tests**

Use `build_gateway` with a fake state whose `director.acquire` records calls and raises if called on a saved-voice route. Isolate the catalog (`MUSE_CATALOG_DIR` = tmp) and pull nothing: write the catalog entries for `qwen3-tts-0.6b-base` (enabled), `kokoro-82m` (enabled) and a disabled id.

```python
def test_list_saved_voices_never_acquires():
    r = client.get("/v1/audio/speech/voices?model=qwen3-tts-0.6b-base")
    assert r.status_code == 200 and "narrator" in r.json()["voices"] and director.acquire_calls == []
def test_patch_and_delete_never_acquire(): ...
def test_missing_model_is_400_model_required(): ...
def test_disabled_model_is_409_model_disabled(): ...
def test_unknown_model_is_404(): ...
def test_preset_model_list_forwards_through_director(): ...   # patch _route_via_director, assert called with model "kokoro-82m"
def test_patch_validation_error_is_envelope_not_detail(): ...
def test_openapi_lists_voice_light_routes():
    paths = client.get("/openapi.json").json()["paths"]
    assert "patch" in paths["/v1/audio/speech/voices/{name}"] and "/v1/audio/speech/voices/{name}/audio" in paths
```

- [ ] **Step 2: Run them and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/cli_impl/test_gateway_light_routes.py -q`

- [ ] **Step 3: Implement it**

Mount in `build_gateway` with:

```python
for tag, hooks in light_hooks().items():
    if hooks.build_light_router:
        app.include_router(hooks.build_light_router(GatewayLightCatalog(state, timeout, app.state.disconnect_cancellable_paths)))
```

Check that `app.state.disconnect_cancellable_paths` is set before this point, and move its assignment earlier if it isn't.

- [ ] **Step 4: Run the tests and see them pass**, plus `tests/cli_impl/test_gateway*.py` and `tests/admin`.

- [ ] **Step 5: Commit** with `feat(gateway): serve light routes from disk ahead of the proxy`.

---

### Task 8: `options.voices` in the capabilities document

**Files:**
- Modify: `src/muse/modalities/audio_speech/light.py`: add `model_options`, exported from `__init__`.
- Modify: `src/muse/core/capabilities.py`: `build_document(..., options: Mapping[str, dict] | None = None)`. Every model entry gains `"options"`: `{}` when there are none.
- Modify: `src/muse/cli_impl/gateway.py`: `_capabilities_document` gathers options.
- Modify: `docs/AGENT_DISCOVERY.md` (a short `options` section)
- Test: `tests/core/test_capabilities.py` (extend), `tests/modalities/audio_speech/test_light_routes.py` (extend), and `tests/cli_impl/test_gateway_capabilities*.py` (extend whichever file covers `_capabilities_document`)

**Interfaces:**
- Produces:
  - `model_options(model_id: str, capabilities: Mapping) -> dict | None`. It returns `None` unless `supports_saved_voices`; otherwise:
    `{"voices": {"count": N, "categories": category_counts(entries), "items": [{"name", "category", "tags", "description"} ...], "list_route": f"/v1/audio/speech/voices?model={model_id}"}}`
  - In `build_document`, the unfiltered document (`modality is None and model is None`) replaces each option's `items` with `sample` (its first 10). A filtered document keeps `items`.

- [ ] **Step 1: Write the failing tests**

```python
def test_unfiltered_options_are_capped():
    opts = {"m": {"voices": {"count": 12, "categories": [], "items": [{"name": f"v{i}"} for i in range(12)], "list_route": "/x"}}}
    doc = build_document(server=S, discovery={}, models=[(facts("m", "audio/speech"), None)], options=opts)
    v = doc["models"][0]["options"]["voices"]
    assert "items" not in v and len(v["sample"]) == 10 and v["count"] == 12
def test_filtered_options_are_complete(): ...   # model="m" -> items has 12
def test_model_without_options_has_empty_options(): ...
def test_model_options_lists_bundled_narrator(tmp_home):
    o = model_options("qwen3-tts-0.6b-base", {"supports_saved_voices": True})["voices"]
    assert o["count"] >= 1 and o["items"][0]["name"] == "narrator"
def test_gateway_capabilities_include_voice_options_without_acquire(): ...
```

- [ ] **Step 2: Run them and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/core/test_capabilities.py tests/modalities/audio_speech/test_light_routes.py -q`

- [ ] **Step 3: Implement it**

- In `_capabilities_document`, after `fresh` facts are merged, build `options = {mid: hooks.model_options(mid, facts.capabilities)}` for each model whose modality has `light_hooks()[modality].model_options`.
- Catch and log any exception per model, and skip that model's options.
- Pass `options` to `build_document`.

- [ ] **Step 4: Run the tests and see them pass**, plus `tests/core tests/cli_impl -q -k capabilit`.

- [ ] **Step 5: Commit** with `feat(discovery): saved voices in the capabilities document`.

---

### Task 9: MCP tools for path parameters, PATCH and DELETE

**Files:**
- Modify: `src/muse/mcp/generated.py`:
  - `_request_schema` handles `DELETE` like `GET` (query parameters) and `PATCH` like `POST` (JSON body).
  - Path parameters become required string properties.
  - `call_generated` substitutes path parameters, URL-quoted with `safe=""`, and sends `DELETE` with params and `PATCH` with JSON.
  - `_annotations` sets `destructiveHint=True` for `DELETE`.
  - Update the `_annotations` docstring.
- Modify: `src/muse/mcp/client.py`: `call(route, ..., path_params: dict | None = None)` formats `route["path"]`.
- Modify: the CLAUDE.md MCP section: one line, "generated tools support path parameters, `PATCH` and `DELETE`; `DELETE` tools are destructive".
- Test: `tests/mcp/test_generated.py` (extend)

**Interfaces:**
- Consumes: the DISCOVERY routes from Task 6 (`muse_get_voice`, `muse_get_voice_audio`, `muse_update_voice`, `muse_delete_voice`) and the OpenAPI from Task 7.

- [ ] **Step 1: Write the failing tests**, using the existing `_gen(caps_doc, openapi_doc, name)` and `fake_http` helpers.

```python
def test_path_param_is_required_and_substituted(caps_doc, openapi_doc, fake_http):
    gen = _gen(caps_doc, openapi_doc, "muse_get_voice")
    assert "name" in gen.tool.inputSchema["required"]
    seen = {}
    client = fake_http(lambda r: (seen.setdefault("url", str(r.url)), httpx.Response(200, json={"name": "a b"}))[1])
    call_generated(client, gen, {"model": "m", "name": "a b"})
    assert "/v1/audio/speech/voices/a%20b?model=m" in seen["url"]
def test_patch_sends_json_body(): ...      # method PATCH, json {"model": "m", "category": "npc"}
def test_delete_is_destructive_and_sends_query(): ...
def test_get_voice_audio_writes_wav_file(): ...   # response audio/wav -> file entry with sha256
def test_voice_tools_generated_from_real_discovery(): ...   # all four new names present
```

- [ ] **Step 2: Run them and see them fail.**

Run: `PYTHONPATH=$PWD/src pytest tests/mcp -q`

- [ ] **Step 3: Implement it.**

- [ ] **Step 4: Run all of `tests/mcp`** and see it pass.

- [ ] **Step 5: Commit** with `feat(mcp): path parameters, PATCH and DELETE in generated tools`.

---

## After the plan (controller, with the owner's OK)

- **Fast lane:** run `python scripts/preflight.py`. Then run the slow lane: `pytest tests -q -m slow`.
- **Server:** ask before restarting the owner's server (`systemctl --user restart muse-local.service`).
- **Backfill (operational):** PATCH metadata onto the voices already saved on `qwen3-tts-0.6b-base`.
  - `narrator_{sage,bard,chronicler,queen,trailer}_{1,2}` get `category: "narrator/epic"`, plus `description` and `design.instructions` from `<scratchpad>/epic-narrator/meta.json`.
  - `brineknife_captain` gets `category: "npc/pirate/captain"`.
