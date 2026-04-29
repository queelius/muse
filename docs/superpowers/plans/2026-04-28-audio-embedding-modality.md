# `audio/embedding` Modality Implementation Plan (#146)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `audio/embedding` modality with `POST /v1/audio/embeddings` (multipart upload, OpenAI-shape envelope mirroring `/v1/embeddings`). Bundled `mert-v1-95m` script (m-a-p/MERT-v1-95M, MIT, 95M params, 768-dim) and curated `clap-htsat-fused` (laion, BSD-3-Clause, 512-dim). Generic `AudioEmbeddingRuntime` over `transformers.AutoModel` + `AutoFeatureExtractor`/`AutoProcessor` plus `librosa` decoding with per-architecture extraction dispatch. HF plugin sniffing audio feature-extraction repos at priority 105.

**Architecture:** Mirror existing modalities. New `audio_embedding/` package with protocol, codec, routes, client, `hf.py`, and `runtimes/transformers_audio.py`. New bundled `mert_v1_95m.py`. Plugin priority 105 (between embedding/text at 110 and image-generation file-pattern at 100).

**Spec:** `docs/superpowers/specs/2026-04-28-audio-embedding-modality-design.md`

**Target version:** v0.24.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/audio_embedding/__init__.py` | create | exports `MODALITY`, `build_router`, Protocol, Result, Client, PROBE_DEFAULTS |
| `src/muse/modalities/audio_embedding/protocol.py` | create | `AudioEmbeddingResult` dataclass, `AudioEmbeddingModel` Protocol |
| `src/muse/modalities/audio_embedding/codec.py` | create | re-exports `embedding_to_base64` / `base64_to_embedding` |
| `src/muse/modalities/audio_embedding/routes.py` | create | `POST /v1/audio/embeddings`, multipart, validation, decode (in runtime) |
| `src/muse/modalities/audio_embedding/client.py` | create | `AudioEmbeddingsClient` (multipart HTTP) |
| `src/muse/modalities/audio_embedding/runtimes/__init__.py` | create | empty marker |
| `src/muse/modalities/audio_embedding/runtimes/transformers_audio.py` | create | `AudioEmbeddingRuntime` generic runtime |
| `src/muse/modalities/audio_embedding/hf.py` | create | HF plugin for audio feature-extraction repos (priority 105) |
| `src/muse/models/mert_v1_95m.py` | create | bundled script (m-a-p/MERT-v1-95M) |
| `src/muse/curated.yaml` | modify | +2 entries: `mert-v1-95m` (bundled), `clap-htsat-fused` (URI) |
| `pyproject.toml` | modify | bump 0.23.0 to 0.24.0 |
| `src/muse/__init__.py` | modify | docstring v0.24.0; add `audio/embedding` to bundled modalities list |
| `CLAUDE.md` | modify | document new modality |
| `README.md` | modify | add `audio/embedding` to route list + curl example |
| `tests/modalities/audio_embedding/` (full tree) | create | protocol, codec, routes, client, hf_plugin, runtime |
| `tests/models/test_mert_v1_95m.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_audio_embedding.py` | create | slow e2e test |
| `tests/integration/test_remote_audio_embedding.py` | create | opt-in integration tests |
| `tests/integration/conftest.py` | modify | `audio_embedding_model` fixture |

---

## Task A: Protocol + Codec + skeleton

Smallest, most isolated. No callers. Foundation for everything else.

**Files:**
- Create: `src/muse/modalities/audio_embedding/__init__.py`
- Create: `src/muse/modalities/audio_embedding/protocol.py`
- Create: `src/muse/modalities/audio_embedding/codec.py`
- Create: `src/muse/modalities/audio_embedding/routes.py` (stub returning empty APIRouter; replaced in Task C)
- Create: `src/muse/modalities/audio_embedding/runtimes/__init__.py` (empty)
- Create: `tests/modalities/audio_embedding/__init__.py` (empty)
- Create: `tests/modalities/audio_embedding/test_protocol.py`
- Create: `tests/modalities/audio_embedding/test_codec.py`

- [ ] **Step 1:** Write failing protocol + codec tests
- [ ] **Step 2:** Implement protocol.py
- [ ] **Step 3:** Implement codec.py
- [ ] **Step 4:** Stub routes.py and write `__init__.py`
- [ ] **Step 5:** Run tests; verify
- [ ] **Step 6:** Commit `feat(audio-embed): audio/embedding modality skeleton + codec`

---

## Task B: AudioEmbeddingRuntime generic runtime

Wraps `transformers.AutoModel` + `AutoFeatureExtractor`/`AutoProcessor` plus `librosa` decoding. Lazy imports. Honors `device="auto"`, `dtype`, `sample_rate`, `max_duration_seconds`, `trust_remote_code` from manifest capabilities. Per-architecture `_extract_embeddings` dispatch is the single source of truth. Tests cover CLAP-style outputs, MERT-style outputs, generic outputs separately.

**Files:**
- Create: `src/muse/modalities/audio_embedding/runtimes/transformers_audio.py`
- Create: `tests/modalities/audio_embedding/runtimes/__init__.py` (empty)
- Create: `tests/modalities/audio_embedding/runtimes/test_transformers_audio.py`

- [ ] **Step 1:** Write failing runtime tests (per-architecture extraction)
- [ ] **Step 2:** Implement runtime
- [ ] **Step 3:** Run tests
- [ ] **Step 4:** Commit `feat(audio-embed): AudioEmbeddingRuntime generic runtime`

---

## Task C: Routes (POST /v1/audio/embeddings)

Replaces the Task A stub with the real endpoint. Multipart/form-data upload mirroring `/v1/audio/transcriptions`. Reads file bytes, enforces size cap, hands raw bytes to `backend.embed(...)` (offloaded to a thread; transformers forward is sync), encodes the response as either float arrays or base64 strings.

**Files:**
- Modify (replace stub): `src/muse/modalities/audio_embedding/routes.py`
- Create: `tests/modalities/audio_embedding/test_routes.py`

- [ ] **Step 1:** Write failing route tests
- [ ] **Step 2:** Implement routes
- [ ] **Step 3:** Run tests
- [ ] **Step 4:** Commit `feat(audio-embed): POST /v1/audio/embeddings route`

---

## Task D: AudioEmbeddingsClient

Mirror `TranscriptionClient`. Server URL public attribute, MUSE_SERVER env fallback, requests + raise_for_status. Multipart upload. Helper `embed(audio: bytes | list[bytes], ...)` returning `list[list[float]]` and `embed_envelope(...)` returning the full OpenAI-shape envelope.

**Files:**
- Create: `src/muse/modalities/audio_embedding/client.py`
- Modify: `src/muse/modalities/audio_embedding/__init__.py` (re-export client)
- Create: `tests/modalities/audio_embedding/test_client.py`

- [ ] **Step 1:** Write failing client tests
- [ ] **Step 2:** Implement client
- [ ] **Step 3:** Re-export in `__init__.py`
- [ ] **Step 4:** Run tests
- [ ] **Step 5:** Commit `feat(audio-embed): AudioEmbeddingsClient HTTP wrapper`

---

## Task E: Bundled mert_v1_95m script

Hand-written script for `m-a-p/MERT-v1-95M`. Wraps `transformers.AutoModel` + `AutoFeatureExtractor` directly (not via the generic runtime). Lazy imports. `trust_remote_code=True` honored. Uses the `importlib.import_module` test pattern established in v0.22.0.

**Files:**
- Create: `src/muse/models/mert_v1_95m.py`
- Create: `tests/models/test_mert_v1_95m.py`

- [ ] **Step 1:** Write failing bundled-script tests
- [ ] **Step 2:** Implement bundled script
- [ ] **Step 3:** Run tests
- [ ] **Step 4:** Commit `feat(audio-embed): bundled mert-v1-95m script`

---

## Task F: HF plugin for audio feature-extraction repos

`src/muse/modalities/audio_embedding/hf.py`. Sniff: feature-extraction tag + repo-name pattern (`clap`, `mert`, `audio-encoder`, `wav2vec`, `audio-embedding`). Priority **105** so it cleanly wins over text/classification (200 catch-all) and is below embedding/text (110).

**Files:**
- Create: `src/muse/modalities/audio_embedding/hf.py`
- Create: `tests/modalities/audio_embedding/test_hf_plugin.py`

- [ ] **Step 1:** Write failing plugin tests
- [ ] **Step 2:** Implement plugin
- [ ] **Step 3:** Run tests
- [ ] **Step 4:** Commit `feat(audio-embed): HF plugin for audio feature-extraction repos (priority 105)`

---

## Task G: Curated entries + slow e2e + integration tests

Add curated entries (2); add slow e2e via in-process supervisor; add opt-in integration suite.

**Files:**
- Modify: `src/muse/curated.yaml` (+2 entries)
- Create: `tests/cli_impl/test_e2e_audio_embedding.py`
- Modify: `tests/integration/conftest.py` (`audio_embedding_model` fixture)
- Create: `tests/integration/test_remote_audio_embedding.py`

- [ ] **Step 1:** Add curated entries
- [ ] **Step 2:** Write slow e2e test
- [ ] **Step 3:** Add integration fixture
- [ ] **Step 4:** Write integration tests
- [ ] **Step 5:** Run fast + slow lanes; integration suite skipped without env var
- [ ] **Step 6:** Commit `test(audio-embed): slow e2e + opt-in integration + curated entries`

---

## Task H: v0.24.0 release

Final wrap-up: docs, version bump, GitHub release notes, tag.

**Files:**
- Modify: `pyproject.toml` (version 0.23.0 to 0.24.0)
- Modify: `src/muse/__init__.py` (docstring; bundled-modalities list)
- Modify: `CLAUDE.md` (modality list)
- Modify: `README.md` (modality list + curl example)

- [ ] **Step 1:** Run full test suite (fast + slow)
- [ ] **Step 2:** Bump version
- [ ] **Step 3:** Update src/muse/__init__.py docstring
- [ ] **Step 4:** Update CLAUDE.md
- [ ] **Step 5:** Update README.md
- [ ] **Step 6:** Em-dash check
- [ ] **Step 7:** Commit `chore(release): v0.24.0`
- [ ] **Step 8:** Tag and push
- [ ] **Step 9:** Create GitHub release
- [ ] **Step 10:** Verify release

---

## Self-review checklist (after Task H)

- [ ] 12 modalities total now? audio/embedding discovered via discover_modalities?
- [ ] 11 of 12 with HF plugin coverage? (audio/speech remains bundled-only)
- [ ] POST /v1/audio/embeddings works end-to-end with /v1/embeddings envelope?
- [ ] Per-architecture _extract_embeddings tested for CLAP, MERT, generic separately?
- [ ] mert-v1-95m bundled and curated; clap-htsat-fused curated?
- [ ] Plugin priority correctly placed at 105 (between embedding/text and image-generation)?
- [ ] All tests pass (fast + slow lanes)?
- [ ] v0.24.0 tagged + pushed; GitHub release published?
- [ ] No em-dashes anywhere?
- [ ] No literal inference-switch-method token in any new file (use `_set_inference_mode` helper)?
