# NPC Voices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Designed voices (Qwen3-TTS VoiceDesign with `instructions`) and saved, cloned voices (Qwen3-TTS 0.6B Base) on muse's audio/speech modality.

**Architecture:** A `VoiceStore` helper owns the on-disk voice library. The speech route gains an `instructions` field and a typed unknown-voice error; two new routes save and delete voices through model methods. One shared Qwen3-TTS runtime serves two bundled scripts (design mode and clone mode).

**Tech Stack:** FastAPI, numpy, soundfile, `qwen-tts==0.1.1` (pins transformers 4.57.3), torch; pytest with fakes and a mocked `qwen_tts`.

**Spec:** `docs/superpowers/specs/2026-11-20-npc-voices-design.md`

## Global Constraints

- Work only in this worktree; run tests with `PYTHONPATH=$PWD/src pytest ...`.
- Never touch `~/.muse` or the owner's server on :8000; tests use `tmp_path` and `MUSE_HOME` / `MUSE_CATALOG_DIR` overrides.
- Heavy imports (torch, qwen_tts, soundfile) are deferred via an `_ensure_deps()` helper with module-level sentinels, per CLAUDE.md.
- Errors use `muse.core.errors.error_response` (OpenAI envelope); never leak paths or exception text in bodies.
- Existing speech models behave exactly as before for requests without the new fields.
- Model ids: `qwen3-tts-1.7b-voicedesign` (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb67327fd37bb2e042aab12ff7391903235d3`) and `qwen3-tts-0.6b-base` (`Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc`). License `Apache-2.0`.
- Voice name regex `^[a-z0-9][a-z0-9_-]{0,63}$`; reference 1-30 s; transcript 1-1000 chars (after strip); `instructions` max 4096 chars.
- Reference clips are decoded to mono at 24000 Hz (Qwen3-TTS codec rate) and stored once as PCM16 WAV.
- Commits: Conventional Commits with a scope, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Rulings against the spec

- Reference audio is stored at 24000 Hz rather than "original rate, capped at 48 kHz": `decode_audio` needs a target rate, and 24 kHz is what the model consumes, so nothing is lost that the model would use. Cost if wrong: re-save voices.
- The delete route gets no MCP tool (`mcp_tool: None`): the MCP generator has no DELETE or path-parameter support, and adding it is out of scope. Cost if wrong: a small follow-up in `muse.mcp.generated`.
- The save route returns 200, not 201, like every other muse POST.

## Review Focus

1. A request sending `voice: "alloy"` (OpenAI default) to the Base model: expect 400 `voice_not_found` listing known voices, not 500. (Task 2, Task 4)
2. A reference upload in a non-WAV container (mp3/ogg) or over 30 s: expect a stored 24 kHz WAV, or 400 `invalid_reference_audio`; never a partial directory left behind. (Task 1, Task 3)
3. Overwriting a voice that is in the Base model's clone-prompt cache: the next line must use the new reference. (Task 4)
4. A name like `../x`, `Grimbold` or `narrator`: 400 `invalid_voice_name` / 400 `voice_read_only`, nothing written. (Task 1, Task 3)
5. `instructions` sent to Kokoro: 400 `instructions_not_supported`, and Kokoro's `synthesize` never receives an `instructions` kwarg when the field is absent. (Task 2)

---

### Task 1: Voice store, voice errors, settings

**Files:**
- Create: `src/muse/modalities/audio_speech/voices.py`
- Modify: `src/muse/modalities/audio_speech/protocol.py` (errors), `src/muse/modalities/audio_speech/__init__.py` (re-export errors)
- Modify: `src/muse/core/config.py` (two `limits.*` settings), `docs/CONFIG.md` (if it lists settings by hand)
- Test: `tests/modalities/audio_speech/test_voice_store.py`

**Interfaces:**
- Produces, in `protocol.py`:
  - `class VoiceError(ValueError)`: `__init__(self, code: str, message: str, status: int = 400)`; attributes `code`, `status`.
  - `class UnknownVoiceError(VoiceError)`: `__init__(self, name: str | None, known: list[str])`; `code="voice_not_found"`, `status=404`; message names the voice and up to 5 known names.
  - `class VoiceUnreadableError(RuntimeError)`: stored voice exists but cannot be read.
- Produces, in `voices.py`:
  - `VOICE_SAMPLE_RATE = 24000`, `NAME_PATTERN`, `MIN_SECONDS = 1.0`, `MAX_SECONDS = 30.0`, `MAX_TRANSCRIPT_CHARS = 1000`.
  - `@dataclass(frozen=True) class StoredVoice`: `name: str, audio_path: Path, transcript: str, sample_rate: int, duration_seconds: float, sha256: str, read_only: bool, identity: tuple`.
  - `class VoiceStore`: `__init__(self, root: Path, *, bundled: Mapping[str, StoredVoice] | None = None, max_voices: int | None = None)`; `names() -> list[str]` (bundled first, then saved, sorted); `get(name: str) -> StoredVoice`; `save(name: str, audio: np.ndarray, sample_rate: int, transcript: str, *, overwrite: bool = False, source_filename: str | None = None) -> StoredVoice`; `delete(name: str) -> None`.
  - `load_bundled_voice(name: str, wav_path: Path, json_path: Path) -> StoredVoice`.
- Settings: `limits.max_saved_voices` (env `MUSE_MAX_SAVED_VOICES`, type like other positive ints, default 200) and `limits.voice_reference_max_bytes` (env `MUSE_VOICE_REFERENCE_MAX_BYTES`, default 20 MB).

Layout per saved voice: `root/<name>/reference.wav` + `root/<name>/voice.json` with keys `name, transcript, sample_rate, duration_seconds, sha256, created_at (UTC ISO-8601), source_filename`. `identity` is the `(st_mtime_ns, st_size, st_ino)` tuple of both files. `save` validates (name, audio 1-D finite, duration, transcript), refuses bundled names (`voice_read_only`), checks `voice_exists` (409) unless `overwrite`, checks `voice_limit_reached` (409) for new names only, writes PCM16 WAV + json into `root/.tmp-<random>/` then `os.replace`s the directory into place (overwrite: rename old aside, replace, remove old). A store-wide `threading.Lock` serialises save/delete. Directories found without both files, or with unparsable json, raise `VoiceUnreadableError` from `get` and are skipped by `names()`. Use `muse.core.securefs` checks where they apply (owned root, no-follow open).

- [ ] **Step 1: Write the failing tests** in `test_voice_store.py`:

```python
def test_save_then_get_round_trips(tmp_path):
    store = VoiceStore(tmp_path)
    audio = np.sin(np.linspace(0, 100, 24000 * 2)).astype("float32") * 0.5
    v = store.save("grimbold", audio, 24000, "  Welcome, traveler.  ")
    assert v.transcript == "Welcome, traveler."
    assert v.sample_rate == 24000 and abs(v.duration_seconds - 2.0) < 1e-3
    assert (tmp_path / "grimbold" / "reference.wav").is_file()
    assert store.get("grimbold").sha256 == v.sha256 == hashlib.sha256(v.audio_path.read_bytes()).hexdigest()
    assert store.names() == ["grimbold"]

@pytest.mark.parametrize("name", ["../x", "Grimbold", "", "-a", "a" * 65, "a/b", "a.b"])
def test_invalid_names_rejected_and_nothing_written(tmp_path, name):
    with pytest.raises(VoiceError) as e:
        VoiceStore(tmp_path).save(name, _two_seconds(), 24000, "hi")
    assert e.value.code == "invalid_voice_name" and list(tmp_path.iterdir()) == []

@pytest.mark.parametrize("seconds", [0.5, 30.5])
def test_duration_bounds(tmp_path, seconds): ...  # code == "invalid_reference_audio"

@pytest.mark.parametrize("text", ["", "   ", "x" * 1001])
def test_transcript_bounds(tmp_path, text): ...  # code == "invalid_transcript"

def test_existing_name_needs_overwrite(tmp_path):  # 409 voice_exists; overwrite=True changes sha256
def test_limit_counts_new_names_only(tmp_path):  # max_voices=1: second name -> 409 voice_limit_reached; overwrite of first still ok
def test_bundled_voice_is_read_only(tmp_path):  # save/delete "narrator" -> 400 voice_read_only; names() starts with "narrator"
def test_delete_and_unknown(tmp_path):  # delete removes dir; get/delete unknown -> UnknownVoiceError(status 404), message lists known names
def test_half_written_directory_is_unreadable_not_listed(tmp_path):  # dir with only voice.json -> get raises VoiceUnreadableError; names() == []
def test_non_finite_audio_rejected(tmp_path):  # NaN in audio -> invalid_reference_audio
```

- [ ] **Step 2: Run** `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech/test_voice_store.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** the errors, `voices.py` and the two settings as specified above. soundfile is a speech-venv dependency; import it lazily inside `save`/`load_bundled_voice` (`muse --help` must not import it). If soundfile is unavailable in the dev venv, write the WAV with the stdlib `wave` module instead and drop the soundfile dependency entirely (preferred: `wave` + numpy only).
- [ ] **Step 4: Run** the test file plus `PYTHONPATH=$PWD/src pytest tests/core/test_config*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(speech): voice store for saved reference voices`.

### Task 2: `instructions` on /v1/audio/speech and typed unknown-voice errors

**Files:**
- Modify: `src/muse/modalities/audio_speech/routes.py`
- Test: `tests/modalities/audio_speech/test_routes_instructions.py`

**Interfaces:**
- Consumes: `UnknownVoiceError`, `VoiceUnreadableError` (Task 1).
- Produces: `SpeechRequest.instructions: str | None = Field(None, max_length=4096)`; capability `supports_instructions` (bool). When set, `instructions=req.instructions` is added to the `synthesize` and `synthesize_stream` kwargs; when unset, the kwarg is absent.

Rules: `instructions` (after strip, empty treated as unset) on a model without `supports_instructions` → 400 `instructions_not_supported`. `UnknownVoiceError` from synthesis (stream: raised before the first chunk, i.e. from `synthesize_stream`'s first `next`) → 400 `voice_not_found` with its message. `VoiceUnreadableError` → 500 `voice_unreadable`, generic message, logged. `_voice()` returns `metadata["voice"]` whenever the key is present, even when it is None (so a model that ignores `voice` does not echo the client's value in `X-Muse-Voice`); otherwise it falls back as today. Add a test: fake returning `metadata={"voice": None}` with `voice: "alloy"` → no `X-Muse-Voice` header. For streaming, resolve the voice failure before the SSE response starts: call a backend hook `getattr(model, "check_voice", None)` with `req.voice` before streaming if present (Task 4 implements it on the Base model).

- [ ] **Step 1: Write failing tests** using a `FakeTTS` like `test_routes.py`'s, registered with capabilities:

```python
def test_instructions_forwarded_when_supported(): # caps {"supports_instructions": True}; fake records kwargs["instructions"] == "gruff dwarf"
def test_instructions_absent_means_no_kwarg():    # default caps; "instructions" not in recorded kwargs
def test_instructions_rejected_when_unsupported(): # 400, body["error"]["code"] == "instructions_not_supported"
def test_blank_instructions_treated_as_unset():   # "   " on unsupported model -> 200
def test_instructions_too_long():                 # 4097 chars -> 422 or 400 (whatever FastAPI validation already returns for max_length)
def test_unknown_voice_is_400():                  # fake raises UnknownVoiceError("alloy", ["narrator"]) -> 400 voice_not_found, "narrator" in message
def test_unknown_voice_stream_is_400_before_sse(): # fake.check_voice raises -> 400 JSON, not text/event-stream
def test_unreadable_voice_is_500_without_paths():  # raises VoiceUnreadableError("/secret/path") -> 500 voice_unreadable, "/secret" not in body
```

- [ ] **Step 2: Run** the new file. Expected: FAIL.
- [ ] **Step 3: Implement** in `routes.py`.
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech -q`. Expected: all PASS (existing tests unchanged).
- [ ] **Step 5: Commit** `feat(speech): instructions field and typed voice errors`.

### Task 3: Save and delete routes, discovery, gateway DELETE

**Files:**
- Modify: `src/muse/modalities/audio_speech/routes.py`, `src/muse/modalities/audio_speech/__init__.py` (DISCOVERY), `src/muse/cli_impl/gateway.py` (`extract_model_from_request`)
- Test: `tests/modalities/audio_speech/test_routes_saved_voices.py`, add a case to `tests/cli_impl/test_gateway.py`

**Interfaces:**
- Consumes: `VoiceError`, `UnknownVoiceError`, `VOICE_SAMPLE_RATE`, `MAX_SECONDS` (Task 1); model methods `save_voice(name, audio, sample_rate, transcript, *, overwrite=False, source_filename=None) -> StoredVoice` and `delete_voice(name) -> None` (implemented in Task 4; fakes here); capability `supports_saved_voices`.
- Produces:
  - `POST /v1/audio/speech/voices` multipart: `model` (Form, optional), `name` (Form), `transcript` (Form), `overwrite` (Form bool, default false), `audio` (File). 200 `{"object": "audio.voice", "model", "name", "duration_seconds", "sha256"}`.
  - `DELETE /v1/audio/speech/voices/{name}?model=...`: 200 `{"object": "audio.voice.deleted", "model", "name", "deleted": true}`.
  - DISCOVERY routes: POST with `mcp_tool: "muse_save_voice"`, `request_content_type: "multipart/form-data"`, `file_fields: ["audio"]`, example `{"model": "<model>", "name": "grimbold", "transcript": "Welcome, traveler."}`; DELETE with `mcp_tool: None`.

Flow (POST): model lookup → 400 `voice_saving_not_supported` without the capability → `read_upload(audio, max_bytes=limits.voice_reference_max_bytes, limit="MUSE_VOICE_REFERENCE_MAX_BYTES=<n>", label="reference audio")` → `run_on_spooled_upload` calling `decode_audio(path, sample_rate=VOICE_SAMPLE_RATE, max_duration_seconds=MAX_SECONDS)` then `model.save_voice(...)` off-loop. Map `AudioDurationExceededError` → 400 `invalid_reference_audio`; `AudioDecodeError` → 415 `unsupported_media_type` (as alignment does); `AudioDecoderUnavailableError` → 500 generic; `VoiceError` → its `status` and `code`. DELETE: capability gate, then `delete_voice`; `UnknownVoiceError` → 404 `voice_not_found`; read-only → 400. Gateway: `DELETE` reads `request.query_params.get("model")` like `GET`.

- [ ] **Step 1: Write failing tests** (patch `decode_audio` in the routes module to return a `DecodedAudio` so no TorchCodec is needed, plus one test with a real tiny WAV if TorchCodec is importable, `pytest.importorskip`):

```python
def test_save_voice_ok():                 # fake.save_voice called with name="grimbold", transcript, sample_rate=24000, overwrite=False; 200 body keys as above
def test_save_requires_capability():      # 400 voice_saving_not_supported
def test_save_maps_voice_errors():        # fake raises VoiceError("voice_exists", "...", 409) -> 409 voice_exists
def test_save_too_long_reference():       # decode raises AudioDurationExceededError -> 400 invalid_reference_audio
def test_save_undecodable():              # AudioDecodeError -> 415 unsupported_media_type
def test_save_oversize_upload():          # setting to 10 bytes -> 413 payload_too_large
def test_delete_ok_and_unknown():         # 200 deleted true; UnknownVoiceError -> 404 voice_not_found
def test_discovery_lists_both_routes():   # DISCOVERY paths include POST and DELETE; DELETE mcp_tool is None
# test_gateway.py
async def test_extract_model_reads_delete_query(): # DELETE ...?model=qwen3-tts-0.6b-base -> "qwen3-tts-0.6b-base"
```

- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src pytest tests/modalities/audio_speech tests/cli_impl/test_gateway.py tests/core/test_capabilities*.py tests/mcp -q`. Expected: PASS (capabilities/MCP meta-tests accept the new routes; fix any DISCOVERY schema complaints they raise).
- [ ] **Step 5: Commit** `feat(speech): save and delete reference voices over HTTP`.

### Task 4: Qwen3-TTS runtime, two bundled scripts, bundled narrator voice

**Files:**
- Create: `src/muse/modalities/audio_speech/runtimes/__init__.py`, `src/muse/modalities/audio_speech/runtimes/qwen3_tts.py`
- Create: `src/muse/models/qwen3_tts_1_7b_voicedesign.py`, `src/muse/models/qwen3_tts_0_6b_base.py`
- Create: `src/muse/modalities/audio_speech/data/narrator.wav`, `src/muse/modalities/audio_speech/data/narrator.json`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]` adds `"*.wav"`), `scripts/validate_installed_distribution.py` if it checks package data
- Test: `tests/models/test_qwen3_tts.py`

**Interfaces:**
- Consumes: `VoiceStore`, `StoredVoice`, `load_bundled_voice`, `VOICE_SAMPLE_RATE` (Task 1); `UnknownVoiceError`; `voices_dir(model_id)` from `audio_speech/backends/base.py` (import lazily: that module imports torch/transformers at top level; prefer moving `voices_dir` and `_muse_home` into `voices.py` and re-exporting from `backends/base.py`); `select_device`, `dtype_for_name`, `set_inference_mode`, `LoadTimer` from `muse.core.runtime_helpers`.
- Produces: `class Qwen3TTSModel` (muse runtime; name it `Qwen3TTSRuntime` to avoid clashing with qwen_tts's class) with constructor `(*, hf_repo, local_dir=None, device="auto", model_id, mode: Literal["design", "clone"], dtype="bf16", default_instructions: str | None = None, **_)`, properties `model_id`, `sample_rate` (24000), `voices` (design: `[]`; clone: `store.names()`), `default_voice` (clone: `"narrator"`, design: None), methods `synthesize(text, **kw) -> AudioResult`, `synthesize_stream(text, **kw) -> Iterator[AudioChunk]` (one chunk), `check_voice(voice)` (clone only), `save_voice(...)`, `delete_voice(name)`, and `_inference_lock` (threading.Lock).

Behaviour:
- Load: `qwen_tts.Qwen3TTSModel.from_pretrained(local_dir or hf_repo, device_map=<device str>, dtype=<torch dtype>)`; bf16 on cuda, float32 on cpu.
- Design mode: `instruct = kw.get("instructions") or default_instructions`; `generate_voice_design(text=text, instruct=instruct, language=None)` (None → qwen's "Auto"). `metadata["voice"] = None`. Default description: `"A clear, neutral adult narrator with an even, friendly tone, speaking at a moderate pace."`
- Clone mode: `voice = kw.get("voice") or "narrator"`; `stored = store.get(voice)` (UnknownVoiceError passes through); clone prompt from `create_voice_clone_prompt(ref_audio=(waveform_float32, 24000), ref_text=stored.transcript)` — always the `(array, sr)` tuple, never a string (qwen-tts treats strings as paths, URLs or base64). Cache prompts in an `OrderedDict` LRU of 32 keyed by `(name, stored.identity)`; `save_voice`/`delete_voice` evict that name. `generate_voice_clone(text=text, language=None, voice_clone_prompt=prompt)`. `metadata["voice"] = voice`.
- Output: first wav as float32 numpy, clipped to [-1, 1]; `AudioResult(audio, sr)`.
- `speed` other than 1.0 is ignored (qwen-tts has no speed control); document in `limits`.
- Manifests (both): `modality: "audio/speech"`, pinned `revision`, `license: "Apache-2.0"`, `pip_extras: ("qwen-tts==0.1.1", "numpy")` plus torch as other GPU speech scripts declare it, `system_packages: ()`, capabilities `{"sample_rate": 24000, "device": "auto", "memory_gb": <design 5.0 / base 3.0>, "mode": ..., ...}` — classify any new non-`supports_` key in `core/capability_keys.py` (`mode`, `default_instructions` → HIDDEN). Design: `supports_instructions: True`, `voices: []`. Base: `supports_saved_voices: True`, `voice_cloning: True`. Fill `description`, `summary`, `strengths`, `limits` (no word timestamps, no speed control, ~10 s per short line on a 12 GB GPU without flash-attn, English and 9 other languages auto-detected), `quality_tier: "good"` (match the vocabulary other scripts use).
- Narrator clip: generate once with the spike venv at `/tmp/claude-1000/-home-spinoza-github-muse/715aa660-bcda-4b0b-b0c0-93b3639bedad/scratchpad/qwen-tts-spike/.venv` (VoiceDesign, the default description, text `"Welcome, traveler. The road ahead is long, but the inn is warm."`), resample/encode to 24 kHz mono PCM16 WAV, about 4 s; `narrator.json` holds `{"name": "narrator", "transcript": <text>, "sample_rate": 24000, "description": <default description>, "generated_by": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb673"}`. Keep the generation script in the scratchpad, not the repo. If the GPU is busy with the owner's server, run it anyway only if `nvidia-smi` shows ≥ 6 GB free; otherwise run on CPU.

- [ ] **Step 1: Write failing tests** with a fake `qwen_tts` module injected via the module-level sentinel (record calls, return `([np.zeros(2400, "float32")], 24000)`):

```python
def test_design_uses_instructions_and_auto_language():  # instruct == "gruff dwarf", language is None
def test_design_default_instructions_when_absent():
def test_clone_defaults_to_narrator_and_passes_tuple(): # ref_audio is a tuple (ndarray, 24000); ref_text == narrator transcript
def test_clone_unknown_voice_raises():                  # UnknownVoiceError for "alloy"; message mentions "narrator"
def test_clone_prompt_cached_and_evicted_on_overwrite():# two calls -> create_voice_clone_prompt once; save_voice(overwrite) -> next call recomputes
def test_save_and_delete_go_through_store(tmp_path, monkeypatch):  # MUSE_HOME=tmp_path; voices lists "narrator", "grimbold"
def test_output_is_clipped_float32():
def test_manifests_are_well_formed():                   # both MANIFESTs: ids, revisions (40 hex), capability flags as above
def test_bundled_narrator_is_packaged():               # importlib.resources path exists, 24 kHz mono, 2-8 s
def test_module_import_is_light():                     # importing the scripts does not import torch or qwen_tts (subprocess check, as other tests do)
```

- [ ] **Step 2: Run** `PYTHONPATH=$PWD/src pytest tests/models/test_qwen3_tts.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the runtime and scripts; generate and commit the narrator files; update package data.
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src pytest tests/models/test_qwen3_tts.py tests/core/test_runtime_helpers_meta.py tests/core -q -m "not slow"`. Expected: PASS (meta-tests accept the new runtime and capability keys).
- [ ] **Step 5: Commit** `feat(speech): Qwen3-TTS VoiceDesign and Base models with saved voices`.

### Task 5: Docs, full lane, real-model check

**Files:**
- Modify: `docs/MODALITIES.md` (audio/speech section), `CLAUDE.md` (the audio/speech bullet: mention Qwen3-TTS designed + saved voices), `docs/CONFIG.md` if not done in Task 1
- Modify: the spec's storage line to say 24 kHz, and its MCP line to say delete has no MCP tool (per Rulings)

- [ ] **Step 1: Docs.** In MODALITIES.md document `instructions`, `supports_instructions`, `supports_saved_voices`, both voice routes with their error codes, the storage layout, the `narrator` voice, and the design-then-clone workflow in four lines.
- [ ] **Step 2: Fast lane.** `PYTHONPATH=$PWD/src python scripts/preflight.py` (or `pytest tests/ -q -m "not slow"` if preflight refuses the worktree). Expected: all pass. Then `PYTHONPATH=$PWD/src pytest tests/ -q` for the slow lane if time allows; record the result.
- [ ] **Step 3: Real-model check (isolated).** With `MUSE_CATALOG_DIR=<scratchpad>/muse-voices-catalog` and `MUSE_HOME=<scratchpad>/muse-voices-home`, from the worktree: `muse pull qwen3-tts-0.6b-base` and `muse pull qwen3-tts-1.7b-voicedesign` (probes must pass), `muse serve --port 8011`, then: design a line with `instructions`, save it as `grimbold` via multipart, speak two lines with `voice: "grimbold"`, list voices, delete `grimbold`, confirm `voice: "alloy"` gives 400. Use the worktree's code (`PYTHONPATH` or install the worktree editable into the per-model venvs per CLAUDE.md's worktree note; check `muse models info` that the venv imports the worktree). Stop the server afterwards. Do not use port 8000.
- [ ] **Step 4: Commit** `docs(speech): designed and saved voices`.
