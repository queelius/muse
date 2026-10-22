# Agent Discovery and Model Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A muse server describes itself to AI agents over HTTP (`/v1/capabilities`, full OpenAPI, `/`, `/llms.txt`), stamps every inference response with the weights revision that produced it, and a thin local `muse mcp` generates its tools from that description and exchanges media as local files.

**Architecture:** Modality packages export a literal `DISCOVERY` dict (AST-read like `CONCURRENCY_CEILING`); bundled manifests and `curated.yaml` carry agent-facing model notes. `muse.core.capabilities` builds one document from those plus catalog, provenance, director and telemetry state; the gateway, workers and the federation coordinator serve it. The gateway adds revision headers in `_forward`. `muse mcp` fetches the document plus `/openapi.json` from `MUSE_SERVER` and generates tools; the hand-written inference tool modules are deleted.

**Tech Stack:** Python 3.11, FastAPI 0.141, httpx, `mcp` SDK 1.x (lowlevel `Server`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-21-agent-discovery-design.md`

## Global Constraints

- Header names, exactly: `X-Muse-Model`, `X-Muse-Model-Revision`, `X-Muse-Model-Artifacts`.
- `system_fingerprint` value: `f"{model_id}@{revision[:12]}"`; omitted when revision is unknown.
- `quality_tier` ∈ `{"best", "good", "draft"}`; absolute within a modality across bundled + curated models.
- Model note fields: `summary` (str, one line), `strengths` (list[str]), `limits` (list[str]), `quality_tier`.
- Image preview: one inline PNG, long side 512 px, only when the tool call passes `preview: true`.
- MCP output dir setting: `mcp.output_dir`, env `MUSE_MCP_OUTPUT_DIR`, default `./muse-output/` (relative to the MCP process cwd).
- PyPI distribution is `museq`; new extra `mcp = ["httpx", "mcp>=1.8.0,<2"]`. Setup line: `claude mcp add muse -e MUSE_SERVER=http://host:8000 -- uvx --from "museq[mcp]" muse mcp`.
- Existing MCP tool names stay (`muse_speak`, `muse_transcribe`, `muse_generate_image`, `muse_edit_image`, ...); admin tools unchanged.
- Additive only: existing `/v1/models` fields and every inference route's request/response shape are unchanged (new headers and a new `revision` field only).
- Deferred imports: `muse --help` must not import fastapi; `muse mcp` (stdio) must not import fastapi or `muse.cli_impl.gateway`.
- Public endpoints never include admin token material, job records, or worker pids.
- Discovery reads run off the event loop (`asyncio.to_thread`) and never trigger a model load.

## Review Focus

1. A catalog entry with no `revision`, no `artifact_provenance`, and a `local_dir` that is not an HF snapshot path (user script, local weights) → revision `null`, no revision header, no crash. Test in Task 5.
2. A worker or node that answers `/v1/capabilities` with garbage or times out → the aggregate still returns 200 with that node's models missing and the node listed as unreachable. Test in Task 10.
3. An MCP `*_path` input that is a symlink or `..` escaping the allowed prefixes → refused before any read. Test in Task 11.
4. A streamed (SSE) or large (>prefetch ceiling) forwarded response still carries the revision headers. Test in Task 6.
5. A model manifest with a capability key nobody classified as public or hidden → the meta-test fails naming the key, rather than the key silently leaking or vanishing. Test in Task 3.

---

### Task 1: The `DISCOVERY` seam, with the four pipeline modalities

**Files:**
- Modify: `src/muse/core/discovery.py` (add `modality_discovery`, next to `concurrency_ceilings`)
- Modify: `src/muse/modalities/{audio_speech,audio_alignment,audio_transcription,audio_quality}/__init__.py` (add `DISCOVERY`)
- Test: `tests/core/test_discovery_seam.py`

**Interfaces:**
- Produces: `modality_discovery(dirs: list[Path] | None = None) -> dict[str, dict]` mapping modality tag → validated `DISCOVERY` dict; malformed declarations logged and skipped (same policy as `concurrency_ceilings`); default scan memoized via `_declarations`.
- Produces the `DISCOVERY` schema every later task reads:
  ```python
  DISCOVERY = {
      "summary": str,                      # one paragraph
      "use_for": [str, ...],
      "not_for": [str, ...],
      "routes": [{
          "method": "POST",
          "path": "/v1/audio/speech",
          "request_content_type": "application/json" | "multipart/form-data",
          "response_content_type": "audio/wav" | "application/json" | ...,
          "compat": "openai:audio.speech" | "cohere:rerank" | "libretranslate" | None,
          "mcp_tool": "muse_speak",        # stable MCP tool name
          "file_fields": ["file"],         # inputs that carry a file (multipart parts, or JSON data-URL strings)
          "media_fields": [],              # JSON paths of base64 media in the response, e.g. ["data[].b64_json"]
          "example": {...},                # minimal request body / form fields, model id "<model>"
      }, ...],
  }
  ```
  Binary responses (raw audio) have `response_content_type` other than `application/json` and empty `media_fields`.

- [ ] **Step 1: Write the failing tests**

```python
def test_modality_discovery_reads_literal(tmp_path):
    pkg = tmp_path / "audio_x"; pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'MODALITY = "audio/x"\n'
        'DISCOVERY = {"summary": "s", "use_for": ["a"], "not_for": [], "routes": ['
        '{"method": "POST", "path": "/v1/x", "request_content_type": "application/json",'
        ' "response_content_type": "application/json", "compat": None, "mcp_tool": "muse_x",'
        ' "file_fields": [], "media_fields": [], "example": {"model": "<model>"}}]}\n'
    )
    assert modality_discovery([tmp_path])["audio/x"]["routes"][0]["mcp_tool"] == "muse_x"

def test_malformed_discovery_is_skipped(tmp_path, caplog):
    # missing "routes" key -> skipped with a warning naming the modality
    ...
    assert "audio/x" not in modality_discovery([tmp_path])
    assert "DISCOVERY" in caplog.text

def test_pipeline_modalities_declare_discovery():
    d = modality_discovery()
    for tag, tool in [("audio/speech", "muse_speak"), ("audio/alignment", "muse_align"),
                      ("audio/transcription", "muse_transcribe"),
                      ("audio/quality", "muse_assess_audio_quality")]:
        assert tool in {r["mcp_tool"] for r in d[tag]["routes"]}

def test_examples_match_route_request_models():
    # For each declared route of the four modalities: build the modality's router
    # against an empty ModalityRegistry, find the APIRoute with that path+method,
    # and check the example: JSON routes -> route.body_field's model validates it
    # (with "<model>" substituted); multipart routes -> example keys are a subset
    # of the route's form/file parameter names.
    ...
```

- [ ] **Step 2: Run** `pytest tests/core/test_discovery_seam.py -v` — Expected: FAIL (`modality_discovery` not defined).
- [ ] **Step 3: Implement `modality_discovery`** in `core/discovery.py` using `_declarations("DISCOVERY", dirs)`; a validator `_valid_discovery(value) -> bool` checks the keys and types above (unknown route keys rejected).
- [ ] **Step 4: Write `DISCOVERY` for the four modalities.** Speech covers `POST /v1/audio/speech` (`compat: "openai:audio.speech"`, `muse_speak`, response `audio/wav`) and `GET /v1/audio/speech/voices`; transcription covers `/v1/audio/transcriptions` (`muse_transcribe`) and `/v1/audio/translations` (`muse_translate_audio`); alignment `muse_align`; quality `muse_assess_audio_quality`. Summaries say what the route is for in the terms a video pipeline uses (e.g. alignment: "word start/end times for a trusted transcript; use for caption and animation cue timing; not speech recognition").
- [ ] **Step 5: Run** `pytest tests/core/test_discovery_seam.py -v` — Expected: PASS.
- [ ] **Step 6: Commit** `feat(discovery): DISCOVERY modality seam with the audio pipeline modalities`

### Task 2: `DISCOVERY` for the remaining modalities, enforced by a meta-test

**Files:**
- Modify: every other `src/muse/modalities/*/__init__.py` (20 packages)
- Modify: `src/muse/modalities/model_3d_generation/routes.py` (make the PIL import lazy so the router builds under `muse[server]` alone)
- Test: `tests/core/test_discovery_meta.py`

**Interfaces:**
- Consumes: `modality_discovery()` (Task 1).
- Tool names that must match today's MCP names: `muse_chat`, `muse_summarize`, `muse_rerank`, `muse_classify`, `muse_embed_text`, `muse_translate`, `muse_decide`, `muse_embed_image`, `muse_generate_image`, `muse_edit_image`, `muse_vary_image`, `muse_upscale_image`, `muse_segment_image`, `muse_vectorize_image`, `muse_generate_animation`, `muse_generate_video`, `muse_generate_music`, `muse_generate_sfx`, `muse_embed_audio`. New: `muse_ocr`, `muse_estimate_depth`, `muse_detect_keypoints`, `muse_detect_objects`, `muse_classify_audio`, `muse_generate_3d`, `muse_generate_3d_from_image`, `muse_moderate`, `muse_list_languages`.

- [ ] **Step 1: Write the failing meta-test**

```python
def test_every_mounted_route_is_declared():
    routers = discover_modalities(modality_dirs())
    decl = modality_discovery()
    for tag, build in routers.items():
        mounted = {(m, r.path) for r in build(ModalityRegistry()).routes for m in r.methods}
        declared = {(r["method"], r["path"]) for r in decl[tag]["routes"]}
        assert mounted == declared, tag

def test_mcp_tool_names_unique_across_modalities(): ...

def test_existing_mcp_tool_names_preserved():
    names = {r["mcp_tool"] for d in modality_discovery().values() for r in d["routes"]}
    assert {"muse_chat", "muse_speak", "muse_generate_image", "muse_edit_image",
            "muse_generate_video", "muse_decide"} <= names
```
Extend Task 1's example-validation test to iterate all modalities (move it here, parametrized by tag).

- [ ] **Step 2: Run** `pytest tests/core/test_discovery_meta.py -v` — Expected: FAIL listing modalities without `DISCOVERY`.
- [ ] **Step 3: Write each package's `DISCOVERY`** following CLAUDE.md's modality descriptions for `compat` (OpenAI paths `openai:<resource>`, rerank/summarize `cohere:*`, translate `libretranslate`, systemone `typesafe:systemone`, the rest `None`). Chat route example sets `"stream": false`. Image generation `media_fields: ["data[].b64_json"]`; video, animation, 3D likewise per their envelopes; edits/variations/upscale/segment/ocr/vectorize/cv list their multipart file part names in `file_fields`; JSON routes taking image data URLs list them (`image` on generations, `input[]` on image embeddings).
- [ ] **Step 4: Run** `pytest tests/core/test_discovery_meta.py tests/core/test_discovery_seam.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(discovery): describe every modality route`

### Task 3: Public versus internal capability keys

**Files:**
- Create: `src/muse/core/capability_keys.py`
- Test: `tests/core/test_capability_keys.py`

**Interfaces:**
- Produces: `public_capabilities(caps: Mapping[str, Any]) -> dict` (keeps public keys, drops hidden ones, drops unclassified ones with a debug log); `classify(key: str) -> Literal["public", "hidden"] | None`.
- Public: keys matching `supports_*`, `max_*`, `min_*`, `default_*`, plus `voices`, `sample_rate`, `dimensions`, `supported_languages`, `source_language`, `target_language`, `context_length`, `device`, `memory_gb`, `quality_axes`, `primary_score`, `question_types`, `input_modalities`, `image_size`, `metric_depth`, `word_timestamps`, `calibrated`, `lora_adapter`, `base_model`, `supported_scales`, `safe_labels`.
- Hidden (the one other list): `gguf_file`, `chat_format`, `n_gpu_layers`, `n_threads`, `n_threads_batch`, `n_batch`, `flash_attn`, `use_mlock`, `type_k`, `type_v`, `allow_patterns`, `torch_compile`, `cpu_offload`, `vae_tiling`, `overlapped_decode`, `trust_remote_code`, `precision`, `option_cache_size`, `idle_timeout_seconds`, `max_concurrency`, `lora_scale`, `image_processor_overrides`, `window_seconds`, `num_beams`, `temperature`, `probabilities`, `calibration`, `confidence`. (`max_concurrency` is hidden despite the `max_` prefix: hidden wins.)

- [ ] **Step 1: Write the failing tests**

```python
def test_public_and_hidden():
    assert public_capabilities({"supports_vision": True, "gguf_file": "x.gguf",
                                "voices": ["a"], "max_concurrency": 1}) == \
        {"supports_vision": True, "voices": ["a"]}

def test_every_known_capability_key_is_classified():
    keys = set()
    for entry in known_models().values():      # bundled scripts
        keys |= set((entry.extra.get("capabilities") or {}))
    for c in load_curated():
        keys |= set(c.capabilities)
    keys |= {"chat_format", "gguf_file", "supports_tools", "context_length",
             "max_context_length", "supports_vision", "supports_multi_image"}  # resolver-synthesized
    unclassified = sorted(k for k in keys if classify(k) is None)
    assert not unclassified, f"classify these in capability_keys.py: {unclassified}"
```
(Use whatever accessor `catalog_models.known_models()` exposes for a bundled manifest's capabilities.)

- [ ] **Step 2: Run** `pytest tests/core/test_capability_keys.py -v` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `capability_keys.py` (two frozensets + one prefix tuple; hidden checked first).
- [ ] **Step 4: Run** the test — Expected: PASS (add any newly surfaced key to the right list).
- [ ] **Step 5: Commit** `feat(core): classify public capability keys`

### Task 4: Agent-facing model notes

**Files:**
- Create: `src/muse/core/model_notes.py`
- Modify: `src/muse/core/curated.py` (`CuratedEntry` gains `summary`, `strengths`, `limits`, `quality_tier`)
- Modify: `src/muse/curated.yaml` (notes on every URI entry), every `src/muse/models/*.py` `MANIFEST` (top-level notes fields)
- Test: `tests/core/test_model_notes.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class ModelNotes: summary: str | None; strengths: tuple[str, ...]; limits: tuple[str, ...]; quality_tier: str | None`; `model_notes(model_id: str, manifest: Mapping) -> ModelNotes`. Source precedence: manifest top-level fields (bundled scripts; resolver manifests normally lack them) → `find_curated(model_id)` fields → `ModelNotes(summary=manifest.get("description"), (), (), None)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_manifest_notes_win(): ...
def test_curated_notes_used_for_uri_models(): ...
def test_description_fallback():
    n = model_notes("x", {"description": "d"})
    assert n == ModelNotes("d", (), (), None)

def test_every_bundled_and_curated_model_has_notes():
    for entry in known_models().values():   # bundled scripts
        n = model_notes(entry.model_id, get_manifest(entry.model_id))
        assert n.summary and n.quality_tier in {"best", "good", "draft"}, entry.model_id
    for c in load_curated():
        if not c.bundled:
            assert c.summary and c.quality_tier in {"best", "good", "draft"}, c.id

def test_bad_tier_rejected_at_load():
    # curated.yaml with quality_tier: "great" -> load_curated raises naming the id
    ...
```

- [ ] **Step 2: Run** `pytest tests/core/test_model_notes.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `model_notes.py` and the `CuratedEntry` fields (validated in the curated loader).
- [ ] **Step 4: Draft the notes** for all 28 bundled scripts and every curated URI entry: `summary` one line of what it is for; 1-3 `strengths`, 1-3 `limits` (including license limits like CC-BY-NC); tiers judged per modality from model cards and the research already gathered (e.g. image/generation: sd-turbo and sdxl-turbo `draft`; audio/transcription whisper-base `draft`, large-v3 family `best`). Kokoro's limits must mention that it does not return word timings at synthesis.
- [ ] **Step 5: Run** `pytest tests/core/test_model_notes.py tests/core/test_curated.py -v` — Expected: PASS.
- [ ] **Step 6: Print the review table** for the owner: `python -c` one-liner over `known_models()` + `load_curated()` emitting `| id | modality | tier | summary | strengths | limits |` markdown; paste it into the final report for this plan. No file is committed for it.
- [ ] **Step 7: Commit** `feat(core): agent-facing model notes for bundled and curated models`

### Task 5: Model revision and artifact components

**Files:**
- Create: `src/muse/core/provenance.py`
- Modify: `src/muse/core/server.py` (`build_model_entry` gains `revision: str | None = None` keyword, written after the splat)
- Modify: `src/muse/cli_impl/gateway.py` (`_unloaded_catalog_entries` and `_enrich_loaded_at` path fill `revision`)
- Test: `tests/core/test_provenance.py`, extend `tests/core/test_server.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ArtifactComponent: role: str; repo: str; revision: str | None   # role: "weights" | "base" | "code"
  @dataclass(frozen=True)
  class ModelRevision:
      model_id: str
      revision: str | None                  # primary 40-hex, or None
      components: tuple[ArtifactComponent, ...]
      artifacts_digest: str | None          # set only when len(components) > 1
  def model_revision(model_id: str, entry: Mapping, manifest: Mapping,
                     catalog: Mapping[str, Mapping] | None = None) -> ModelRevision
  def revision_headers(rev: ModelRevision) -> list[tuple[bytes, bytes]]
  def system_fingerprint(rev: ModelRevision) -> str | None
  def lookup_revision(model_id: str) -> ModelRevision   # reads catalog via catalog_store._read_catalog + get_manifest; cached on (model_id, catalog mtime_ns); never raises (unknown -> revision None)
  ```
- Primary revision ladder: `entry["revision"]` → 40-hex segment after `/snapshots/` in `entry["local_dir"]` → `manifest["revision"]` → `None`.
- Components: one `weights` component per `entry["artifact_provenance"]` item (`repo_id`, `revision`), else one from `hf_repo` + primary revision; a `code` component when `manifest.get("code_revision")`; a `base` component (base model's primary, via `catalog`) when capabilities declare `lora_adapter` and a base model (`entry.get("base_override")` else `capabilities["base_model"]`).
- `artifacts_digest`: first 16 hex chars of sha256 over `json.dumps(sorted([c.role, c.repo, c.revision] for c in components))`.
- `revision_headers`: always `X-Muse-Model`; `X-Muse-Model-Revision` when revision known; `X-Muse-Model-Artifacts` when digest set.

- [ ] **Step 1: Write the failing tests**

```python
SNAP = "/home/u/.muse/weights/models--Systran--faster-whisper-base/snapshots/" + "e" * 40

def test_recorded_revision_wins():
    r = model_revision("m", {"revision": "a" * 40, "local_dir": SNAP, "hf_repo": "o/r"}, {})
    assert r.revision == "a" * 40

def test_snapshot_path_fallback_for_old_pulls():
    assert model_revision("whisper-base", {"local_dir": SNAP, "hf_repo": "Systran/faster-whisper-base"}, {}).revision == "e" * 40

def test_unknown_revision_is_none_not_error():           # Review Focus 1
    r = model_revision("mine", {"local_dir": "/data/my-weights"}, {})
    assert r.revision is None and r.artifacts_digest is None
    assert revision_headers(r) == [(b"x-muse-model", b"mine")]
    assert system_fingerprint(r) is None

def test_lora_adds_base_component_and_digest():
    catalog = {"sdxl-turbo": {"revision": "b" * 40, "hf_repo": "stabilityai/sdxl-turbo"}}
    r = model_revision("pixel-art-xl",
        {"local_dir": "/w/models--nerijs--pixel-art-xl/snapshots/" + "c" * 40, "hf_repo": "nerijs/pixel-art-xl"},
        {"capabilities": {"lora_adapter": True, "base_model": "sdxl-turbo"}}, catalog)
    assert [c.role for c in r.components] == ["weights", "base"]
    assert len(r.artifacts_digest) == 16

def test_fingerprint_format():
    assert system_fingerprint(model_revision("q", {"revision": "0123456789abcdef" + "0" * 24}, {})) == "q@0123456789ab"

def test_build_model_entry_revision_cannot_be_clobbered():
    e = build_model_entry("m", "audio/speech", {"capabilities": {"revision": "evil"}},
                          loaded=False, last_loaded_at=None, unservable_reason=None, revision="a" * 40)
    assert e["revision"] == "a" * 40
```

- [ ] **Step 2: Run** `pytest tests/core/test_provenance.py tests/core/test_server.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `provenance.py`; `build_model_entry(..., revision=None)`; gateway `/v1/models` fills `revision` for unloaded rows (from `lookup_revision`) and for resident rows that lack it.
- [ ] **Step 4: Run** the tests — Expected: PASS; then `pytest tests/cli_impl/test_gateway*.py -q` stays green.
- [ ] **Step 5: Commit** `feat(core): resolve model revision and artifact components`

### Task 6: Revision headers on every forwarded response, and `system_fingerprint`

**Files:**
- Modify: `src/muse/cli_impl/gateway.py` (`_forward` gains `extra_headers: list[tuple[bytes, bytes]] | None = None`, installed on buffered, prefetched-streaming and SSE responses; `_route_via_director` passes `revision_headers(await asyncio.to_thread(lookup_revision, model_id))`)
- Modify: `src/muse/modalities/chat_completion/codec.py` (`result_to_openai_dict(result, *, system_fingerprint=None)`, `chunk_to_openai_dict(chunk, *, system_fingerprint=None)`; `chunk_to_sse_data` forwards the keyword) and `routes.py` (pass `system_fingerprint(lookup_revision(model_id))`, computed once per request, off-loop)
- Test: `tests/cli_impl/test_gateway_revision_headers.py`, extend `tests/modalities/chat_completion/test_codec.py`

**Interfaces:**
- Consumes: `lookup_revision`, `revision_headers`, `system_fingerprint` (Task 5).
- Federation keeps calling `_forward` without `extra_headers`; the node's headers pass through `_proxy_headers` unchanged.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("kind", ["buffered", "large_streamed", "sse"])   # Review Focus 4
def test_forwarded_response_carries_revision_headers(kind, fake_worker, monkeypatch):
    monkeypatch.setattr(gateway, "lookup_revision", lambda m: ModelRevision(m, "a" * 40, (), None))
    r = client.post("/v1/audio/speech", json={"model": "kokoro-82m", "input": "hi"})
    assert r.headers["x-muse-model"] == "kokoro-82m"
    assert r.headers["x-muse-model-revision"] == "a" * 40

def test_worker_supplied_header_does_not_override_gateway(): ...  # gateway value wins

def test_chat_dict_has_fingerprint():
    assert result_to_openai_dict(res, system_fingerprint="q@0123456789ab")["system_fingerprint"] == "q@0123456789ab"

def test_chat_dict_omits_fingerprint_when_unknown():
    assert "system_fingerprint" not in result_to_openai_dict(res)
```
(Use the existing fake-worker fixtures in `tests/cli_impl/` for the three response kinds; the large case exceeds the prefetch ceiling.)

- [ ] **Step 2: Run** them — Expected: FAIL.
- [ ] **Step 3: Implement**: install `extra_headers` after `_proxy_headers` filtering so they replace any same-named worker header.
- [ ] **Step 4: Run** the new tests plus `pytest tests/cli_impl tests/modalities/chat_completion -q -m "not slow"` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(gateway): stamp inference responses with the model revision`

### Task 7: Last cold-load time from telemetry

**Files:**
- Modify: `src/muse/observability/store.py`
- Test: extend `tests/observability/test_store.py`

**Interfaces:**
- Produces: `TelemetryStore.last_cold_load_seconds() -> dict[str, float]` — for each `model_id`, the `cold_load_seconds` of its newest `model_load` event with a non-null value; takes the store's lock; empty dict on a closed store.

- [ ] **Step 1: Write the failing test** inserting three `model_load` rows for two models (one with `cold_load_seconds` NULL as the newest for model A) and asserting `{"a": <older value>, "b": <value>}`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** with one SQL query (`ORDER BY ts DESC`, first non-null per model).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(telemetry): last cold-load seconds per model`

### Task 8: The capabilities document and `GET /v1/capabilities`

**Files:**
- Create: `src/muse/core/capabilities.py`
- Modify: `src/muse/cli_impl/gateway.py` (route, before the catch-all), `src/muse/core/server.py` (`create_app` mounts a worker `/v1/capabilities` for its registry's models)
- Test: `tests/core/test_capabilities.py`, `tests/cli_impl/test_gateway_capabilities.py`

**Interfaces:**
- Consumes: `modality_discovery` (T1), `public_capabilities` (T3), `model_notes` (T4), `model_revision` (T5), `last_cold_load_seconds` (T7), `backfill_manifest_memory`, `state.director.loaded`, `state.concurrency_gate.depths()`, `revalidate_servability`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ModelFacts:            # static, cacheable on catalog mtime
      id: str; modality: str; hf_repo: str | None; license: str | None
      revision: ModelRevision; notes: ModelNotes; capabilities: dict; memory_gb: float | None
  @dataclass(frozen=True)
  class ModelState:            # live, recomputed per call
      loaded: bool; last_cold_load_seconds: float | None; queue_depth: int | None; unservable_reason: str | None
  def build_document(*, server: dict, discovery: Mapping[str, dict],
                     models: list[tuple[ModelFacts, ModelState | None]],
                     modality: str | None = None, model: str | None = None) -> dict
  def render_llms_txt(doc: dict) -> str
  ```
- Document shape (exact top-level keys): `{"object": "muse.capabilities", "server": {...}, "modalities": [...], "models": [...]}`.
  - `server`: `version`, `node`, `links: {openapi, models, capabilities, llms_txt}`, `mcp: {"install": "uvx --from \"museq[mcp]\" muse mcp", "env": {"MUSE_SERVER": <this server's base URL from the request>}}`.
  - each modality: `{"modality", "summary", "use_for", "not_for", "routes", "models": [ids]}`.
  - each model: `{"id", "modality", "hf_repo", "license", "revision", "components": [{role, repo, revision}], "summary", "strengths", "limits", "quality_tier", "capabilities", "loaded", "memory_gb", "last_cold_load_seconds", "queue_depth", "unservable_reason"}`. With `ModelState=None` (worker app) the four state keys are present as `null` except `loaded: true`.
- Filters: `?modality=` keeps that modality and its models; `?model=` keeps one model and its modality; unknown value → 404 OpenAI-shape error `{"error": {"code": "not_found", ...}}`.
- Gateway: enabled catalog models plus resident ones; static facts cached keyed by catalog `mtime_ns`; whole build in `asyncio.to_thread`.

- [ ] **Step 1: Write the failing tests**

```python
def test_document_shape_and_filters(): ...           # pure build_document with fixtures
def test_hidden_keys_never_appear():
    doc = build_document(..., models=[(facts_with_caps({"gguf_file": "x", "supports_tools": True}), state)])
    assert doc["models"][0]["capabilities"] == {"supports_tools": True}
def test_llms_txt_lists_modalities_and_links(): ...
def test_gateway_capabilities_lists_unloaded_enabled_models(gateway_client): ...
def test_gateway_capabilities_excludes_disabled_and_admin_data(gateway_client):
    body = gateway_client.get("/v1/capabilities").text
    assert "pid" not in body and "admin" not in body.lower().replace("/v1/admin", "")
def test_queue_depth_null_without_gate(): ...
def test_capabilities_does_not_trigger_load(gateway_client, director_spy):
    gateway_client.get("/v1/capabilities"); assert director_spy.acquire_calls == 0
def test_unknown_filter_404(gateway_client): ...
def test_worker_app_serves_capabilities(): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** `capabilities.py` (pure) and the two routes.
- [ ] **Step 4: Run** the new tests and `pytest tests/core tests/cli_impl -q -m "not slow"` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: /v1/capabilities discovery document`

### Task 9: Complete OpenAPI and the entry points

**Files:**
- Create: `src/muse/cli_impl/agent_routes.py`
- Modify: `src/muse/cli_impl/gateway.py` (call `install_agent_routes(app, ...)` before the catch-all)
- Test: `tests/cli_impl/test_agent_routes.py`

**Interfaces:**
- Consumes: `discover_modalities`, `modality_discovery`, `render_llms_txt` (T8), the capabilities builder.
- Produces: `install_agent_routes(app: FastAPI, *, capabilities: Callable[[Request], Awaitable[dict]]) -> None` mounting `GET /` and `GET /.well-known/muse.json` (same pointer JSON: `{"object": "muse.server", "capabilities": "/v1/capabilities", "openapi": "/openapi.json", "llms_txt": "/llms.txt", "models": "/v1/models"}`), `GET /llms.txt` (`text/markdown`), and replaces `app.openapi` with a function that merges the gateway schema with a schema built from every modality router mounted on a throwaway `FastAPI()` (each built against an empty `ModalityRegistry`; a router that fails to build is logged and skipped); each merged operation's `description` is the route's `DISCOVERY` summary; the catch-all stays `include_in_schema=False`.

- [ ] **Step 1: Write the failing tests**

```python
def test_openapi_lists_every_declared_route(gateway_client):
    paths = gateway_client.get("/openapi.json").json()["paths"]
    for d in modality_discovery().values():
        for r in d["routes"]:
            assert r["method"].lower() in paths[r["path"]]
    assert "/{full_path}" not in paths

def test_speech_request_schema_present(gateway_client):
    op = gateway_client.get("/openapi.json").json()["paths"]["/v1/audio/speech"]["post"]
    assert "requestBody" in op

def test_root_and_well_known_point_at_capabilities(gateway_client): ...
def test_llms_txt_is_markdown(gateway_client): ...
def test_broken_router_is_skipped(monkeypatch, gateway_client): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** `agent_routes.py`; the merged schema is computed once and cached on the app.
- [ ] **Step 4: Run** the tests — Expected: PASS; `muse --help` import guard tests stay green.
- [ ] **Step 5: Commit** `feat(gateway): full OpenAPI, llms.txt and discovery entry points`

### Task 10: Federation coordinator discovery

**Files:**
- Create: `src/muse/federation/capabilities.py`
- Modify: `src/muse/cli_impl/federation.py` (routes `/v1/capabilities`, `/`, `/.well-known/muse.json`, `/llms.txt`, `/openapi.json`, before the catch-all)
- Test: `tests/federation/test_capabilities_merge.py`, extend `tests/cli_impl/test_federation_app.py`

**Interfaces:**
- Produces: `merge_capabilities(node_docs: list[tuple[str, dict | None]]) -> dict` — `None` means unreachable. Models are unioned by id, and each model gets `"nodes": [{"name", "loaded", "queue_depth", "revision"}]`. Top-level `revision` is the shared value, or `null` when nodes disagree (both stay visible in `nodes`). Modalities are unioned by tag with their model lists merged. The result also carries `"unreachable_nodes": [names]`.
- The coordinator fetches each reachable node's `/v1/capabilities` concurrently on request (timeout `server.aggregation_timeout_seconds`). Any exception, non-200, or non-dict body counts as unreachable for this call. `/openapi.json` is the first reachable node's schema, with the coordinator's own paths added. `/llms.txt` is rendered from the merged doc.

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_unions_models_with_nodes(): ...
def test_disagreeing_revisions_both_shown():
    doc = merge_capabilities([("a", doc_with("m", rev="1" * 40)), ("b", doc_with("m", rev="2" * 40))])
    m = doc["models"][0]
    assert m["revision"] is None and {n["revision"] for n in m["nodes"]} == {"1" * 40, "2" * 40}

def test_bad_node_degrades_not_fails(coordinator_client):          # Review Focus 2
    # node b returns 200 with "not json"; node c times out
    r = coordinator_client.get("/v1/capabilities")
    assert r.status_code == 200 and set(r.json()["unreachable_nodes"]) == {"b", "c"}
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/federation tests/cli_impl/test_federation_app.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(federation): merged discovery across nodes`

### Task 11: MCP local file I/O

**Files:**
- Create: `src/muse/mcp/files.py`
- Modify: `src/muse/core/config.py` (new `Setting("mcp.output_dir", "MUSE_MCP_OUTPUT_DIR", "str", "./muse-output", "mcp", "Directory muse mcp writes media outputs to when a tool call gives no out_path.")`; `fetch.mcp_allowed_path_prefixes` help text says an empty value means the MCP process's working directory plus `mcp.output_dir`)
- Test: `tests/mcp/test_files.py`

**Interfaces:**
- Produces:
  ```python
  def allowed_prefixes() -> list[str]            # realpaths; empty setting -> [cwd, output_dir]
  def read_input(path: str, *, field: str, max_bytes: int) -> tuple[bytes, str, str]   # (data, filename, mime)
  def as_data_url(data: bytes, mime: str) -> str
  @dataclass(frozen=True)
  class WrittenFile: path: str; sha256: str; mime: str; bytes: int
  def write_output(data: bytes, mime: str, *, out_path: str | None) -> WrittenFile
  def media_facts(data: bytes, mime: str) -> dict   # {"duration_seconds"} for wav/flac via soundfile-free WAV header parse, {"width","height"} for images via PIL if importable else {}
  def preview_png(data: bytes) -> bytes | None      # long side 512; None if PIL unavailable or not an image
  ```
- Default filename when no `out_path`: `<output_dir>/<sha256[:16]><ext>` with the extension taken from `mimetypes.guess_extension`. Writes go through a temp file and an atomic replace. Parent dirs are created. `out_path` must also lie within `allowed_prefixes()`.
- This replaces `muse/mcp/binary_io.py`'s path/b64 resolution. The task deletes `binary_io.py` once Task 12 no longer imports it. Keep the existing per-request byte cap (`_default_max_bytes` value) as `max_bytes`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reads_file_under_cwd(tmp_path, monkeypatch): ...
def test_symlink_escape_refused(tmp_path, monkeypatch):             # Review Focus 3
    monkeypatch.chdir(tmp_path)
    (tmp_path / "link").symlink_to("/etc/hostname")
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        read_input("link", field="file", max_bytes=10**6)
def test_dotdot_escape_refused(tmp_path, monkeypatch): ...
def test_write_output_content_named(tmp_path, monkeypatch):
    w = write_output(b"RIFF....", "audio/wav", out_path=None)
    assert w.path.endswith(".wav") and Path(w.path).read_bytes() == b"RIFF...."
def test_out_path_outside_prefixes_refused(tmp_path, monkeypatch): ...
def test_preview_long_side_512(): ...
def test_wav_duration(): ...
```

- [ ] **Step 2: Run** `pytest tests/mcp/test_files.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `files.py` and the setting.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(mcp): local file inputs and content-named outputs`

### Task 12: Generated MCP tools, `muse_describe`, resources, and the light install

**Files:**
- Create: `src/muse/mcp/generated.py`
- Modify: `src/muse/mcp/server.py` (dynamic `list_tools`, `list_resources`/`read_resource`), `src/muse/mcp/client.py` (shrinks to `capabilities()`, `openapi()`, `call(route, ...)` plus the admin delegation), `src/muse/mcp/tools/__init__.py` (exports only `ADMIN_TOOLS`), `src/muse/cli_impl/mcp_server.py` (install hint `museq[mcp]`), `pyproject.toml` (`mcp` extra)
- Delete: `src/muse/mcp/tools/inference_{text,image,audio,video,decision}.py`, `src/muse/mcp/binary_io.py`, and the tests that cover only them (replace with the tests below)
- Test: `tests/mcp/test_generated.py`, `tests/mcp/test_server_dynamic.py`, `tests/mcp/test_import_hygiene.py`

**Interfaces:**
- Consumes: `/v1/capabilities` + `/openapi.json` documents (T8, T9), `files.*` (T11), `DISCOVERY` route fields (T1).
- Produces:
  ```python
  @dataclass(frozen=True)
  class GeneratedTool: name: str; tool: "mcp.types.Tool"; route: dict   # route = the DISCOVERY entry
  def generate_tools(capabilities: dict, openapi: dict) -> list[GeneratedTool]
  def call_generated(client: MuseClient, gen: GeneratedTool, args: dict) -> list[dict]   # MCP content blocks
  def describe_tool() -> "mcp.types.Tool"                                                # muse_describe
  ```
- **Input schema.** The schema is the route's request schema from OpenAPI, with these changes:
  - each `file_fields` entry becomes `<name>_path` (local file) or `<name>_url` (http(s) or data URL), exactly one of them required when the field is required;
  - optional `out_path` (string);
  - for image-producing routes only, optional `preview` (boolean);
  - `model` gets `enum` = the ids of models serving that modality;
  - chat forces `stream: false`.
- **Description.** The modality summary, then the route `compat`, then one line per model (`id — quality_tier — summary`).
- **Result.** A short text block, a `structuredContent` dict (the tool declares an `outputSchema` for it), and a `resource_link` per written file.
  - Text block: model, revision, then the path, size and duration of each file.
  - `structuredContent`: `{"model", "revision", "artifacts", "files": [{path, sha256, mime, bytes, ...media_facts}], "response": <JSON body with media fields replaced by file paths>}`.
  - Binary responses are written to a file. Each `media_fields` base64 value is decoded and written.
  - JSON results whose serialized size exceeds 20,000 characters are written to a `.json` file and summarized.
  - Revision comes from the response headers (T6).
- **Server.**
  - `list_tools` returns `muse_describe`, the generated tools and the admin tools, filtered by `--filter`.
  - The capabilities document is cached and refetched when older than 30 s.
  - If the server is unreachable, it returns only `muse_describe` (which reports the error when called) plus the admin tools.
  - Resources: `muse://capabilities` and `muse://models/{id}` (`application/json`).

- [ ] **Step 1: Write the failing tests**

```python
def test_generates_one_tool_per_declared_route(caps_doc, openapi_doc):
    names = {g.name for g in generate_tools(caps_doc, openapi_doc)}
    assert {"muse_speak", "muse_align", "muse_transcribe", "muse_assess_audio_quality"} <= names

def test_file_fields_become_path_and_url(caps_doc, openapi_doc):
    props = tool("muse_align").inputSchema["properties"]
    assert "file_path" in props and "file_url" in props and "file_b64" not in props

def test_model_enum_lists_serving_models(caps_doc, openapi_doc): ...

def test_speak_writes_wav_and_returns_link(fake_http, tmp_path, monkeypatch):
    blocks = call_generated(client, tool_gen("muse_speak"), {"model": "kokoro-82m", "input": "hi"})
    assert any(b["type"] == "resource_link" for b in blocks)
    assert not any(b["type"] == "audio" for b in blocks)

def test_image_preview_only_on_request(fake_http): ...
def test_b64_media_fields_written_to_files(fake_http): ...   # generations envelope -> response.data[0].b64_json replaced by path
def test_unreachable_server_lists_only_describe_and_admin(): ...
def test_existing_admin_tools_unchanged(): ...

def test_stdio_import_is_light():                            # test_import_hygiene.py
    out = subprocess.run([sys.executable, "-c",
        "import sys, muse.cli_impl.mcp_server, muse.mcp.server, muse.mcp.generated;"
        "print('fastapi' in sys.modules, 'muse.cli_impl.gateway' in sys.modules)"],
        capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]
```

- [ ] **Step 2: Run** `pytest tests/mcp -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `generated.py` and the server changes, then delete the old inference modules and their tests. Admin tools keep their handlers on `AdminClient`.
- [ ] **Step 4: Run** `pytest tests/mcp -q` and the fast lane `pytest tests/ -q -m "not slow"`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(mcp): tools generated from the server's discovery document`

### Task 13: Docs and live verification

**Files:**
- Create: `docs/AGENT_DISCOVERY.md`
- Modify: `CLAUDE.md` (new "Agent discovery" section; MCP section rewritten: thin local client, generated tools, file outputs, `museq[mcp]`; tool counts removed instead of updated), `docs/CONFIG.md` (`mcp.output_dir`), `src/muse/cli.py` help strings that state tool counts
- Test: none new. This task verifies against the live dev server.

- [ ] **Step 1: Write `docs/AGENT_DISCOVERY.md`.** Cover:
  - what each endpoint returns;
  - the "first three calls" for HTTP (`GET /`, `GET /v1/capabilities?modality=audio/speech`, a speech POST that reads `X-Muse-Model-Revision`);
  - the "first three calls" for MCP (`claude mcp add ...`, `muse_describe`, `muse_speak` with `out_path`);
  - how a pipeline records provenance from the headers;
  - federation behavior.
- [ ] **Step 2: Update `CLAUDE.md`, `docs/CONFIG.md` and the CLI help.**
- [ ] **Step 3: Restart `muse-local` and verify live.** Each check must hold:
  - `curl -s localhost:8000/v1/capabilities | jq '.models | length'` returns 34, the enabled-model count;
  - `curl -s 'localhost:8000/v1/capabilities?model=whisper-base' | jq .models[0].revision` returns a 40-hex string, recovered from the snapshot path;
  - `curl -sD- -o /dev/null localhost:8000/v1/audio/speech -H 'content-type: application/json' -d '{"model":"kokoro-82m","input":"hello","response_format":"wav"}'` shows `x-muse-model-revision`;
  - `curl -s localhost:8000/openapi.json | jq '.paths | length'` returns more than 21 paths and includes `/v1/audio/alignments`.
- [ ] **Step 4: Run an MCP smoke test.**
  - Write a throwaway script in the scratchpad that uses `mcp.client.stdio` to spawn `muse mcp` with `MUSE_SERVER=http://localhost:8000`.
  - Call `muse_describe`, then `muse_speak` with `out_path` in the scratchpad, then `muse_align` with `file_path` pointing at that WAV.
  - Assert the WAV exists, the alignment returns words, and both results carry a revision.
- [ ] **Step 5: Run the full verification.** `python scripts/preflight.py` must pass the fast lane, and `pytest tests/ -q -m slow` the slow lane.
- [ ] **Step 6: Commit** `docs: agent discovery guide; MCP docs for the generated client`
