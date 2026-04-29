# `image/embedding` Modality Implementation Plan (#145)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `image/embedding` modality with `POST /v1/images/embeddings` (OpenAI-shape, mirroring `/v1/embeddings`). Bundled `dinov2-small` script (facebook/dinov2-small) and HF plugin sniffing image-feature-extraction repos. Generic `ImageEmbeddingRuntime` over `transformers.AutoModel` + `AutoProcessor` with per-architecture extraction dispatch.

**Architecture:** Mirror existing modalities. New `image_embedding/` package with protocol, codec, routes, client, `hf.py`, and `runtimes/transformers_image.py`. New bundled `dinov2_small.py`. Plugin priority 105 (between embedding/text at 110 and image-generation file-pattern at 100).

**Spec:** `docs/superpowers/specs/2026-04-28-image-embedding-modality-design.md`

**Target version:** v0.23.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/image_embedding/__init__.py` | create | exports `MODALITY`, `build_router`, Protocol, Result, Client, PROBE_DEFAULTS |
| `src/muse/modalities/image_embedding/protocol.py` | create | `ImageEmbeddingResult` dataclass, `ImageEmbeddingModel` Protocol |
| `src/muse/modalities/image_embedding/codec.py` | create | re-exports `embedding_to_base64` / `base64_to_embedding` |
| `src/muse/modalities/image_embedding/routes.py` | create | `POST /v1/images/embeddings`, request validation, image decoding |
| `src/muse/modalities/image_embedding/client.py` | create | `ImageEmbeddingsClient` (HTTP) |
| `src/muse/modalities/image_embedding/runtimes/__init__.py` | create | empty marker |
| `src/muse/modalities/image_embedding/runtimes/transformers_image.py` | create | `ImageEmbeddingRuntime` generic runtime |
| `src/muse/modalities/image_embedding/hf.py` | create | HF plugin for image-feature-extraction repos (priority 105) |
| `src/muse/models/dinov2_small.py` | create | bundled script (facebook/dinov2-small) |
| `src/muse/curated.yaml` | modify | +3 entries: `dinov2-small` (bundled), `siglip2-base` (URI), `clip-vit-base-patch32` (URI) |
| `pyproject.toml` | modify | bump 0.22.0 to 0.23.0 |
| `src/muse/__init__.py` | modify | docstring v0.23.0; add `image/embedding` to bundled modalities list |
| `CLAUDE.md` | modify | document new modality |
| `README.md` | modify | add `image/embedding` to route list + curl example |
| `tests/modalities/image_embedding/` (full tree) | create | protocol, codec, routes, client, hf_plugin, runtime |
| `tests/models/test_dinov2_small.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_image_embedding.py` | create | slow e2e test |
| `tests/integration/test_remote_image_embedding.py` | create | opt-in integration tests |
| `tests/integration/conftest.py` | modify | `image_embedding_model` fixture |

---

## Task A: Protocol + Codec + skeleton

Smallest, most isolated. No callers. Foundation for everything else.

**Files:**
- Create: `src/muse/modalities/image_embedding/__init__.py` (with re-exports; build_router stubbed in routes)
- Create: `src/muse/modalities/image_embedding/protocol.py`
- Create: `src/muse/modalities/image_embedding/codec.py`
- Create: `src/muse/modalities/image_embedding/routes.py` (stub returning empty APIRouter; replaced in Task C)
- Create: `src/muse/modalities/image_embedding/runtimes/__init__.py` (empty)
- Create: `tests/modalities/image_embedding/__init__.py` (empty)
- Create: `tests/modalities/image_embedding/test_protocol.py`
- Create: `tests/modalities/image_embedding/test_codec.py`

- [ ] **Step 1: Write the failing protocol + codec tests**
- [ ] **Step 2: Implement protocol.py**
- [ ] **Step 3: Implement codec.py**
- [ ] **Step 4: Stub routes.py and write `__init__.py`**
- [ ] **Step 5: Run tests; verify they pass**
- [ ] **Step 6: Commit `feat(image-embed): image/embedding modality skeleton + codec`**

---

## Task B: ImageEmbeddingRuntime generic runtime

Wraps `transformers.AutoModel` + `AutoProcessor`. Lazy imports. Honors
`device="auto"`, `dtype`, `image_size` from manifest capabilities.
Per-architecture `_extract_embeddings` dispatch is the runtime's
single source of truth. Tests cover CLIP-style outputs, SigLIP-style
outputs, and DINOv2-style outputs separately.

**Files:**
- Create: `src/muse/modalities/image_embedding/runtimes/transformers_image.py`
- Create: `tests/modalities/image_embedding/runtimes/__init__.py` (empty)
- Create: `tests/modalities/image_embedding/runtimes/test_transformers_image.py`

- [ ] **Step 1: Write the failing runtime tests (per-architecture extraction)**
- [ ] **Step 2: Implement runtime**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(image-embed): ImageEmbeddingRuntime generic runtime`**

---

## Task C: Routes (POST /v1/images/embeddings)

Replaces the Task A stub with the real endpoint. Validates request,
decodes input data URLs / http URLs into PIL.Image via the
`decode_image_input` helper, resolves the registered backend, calls
`backend.embed(...)` (offloaded to a thread; transformers forward is
sync), encodes the response as either float arrays or base64 strings.

**Files:**
- Modify (replace stub): `src/muse/modalities/image_embedding/routes.py`
- Create: `tests/modalities/image_embedding/test_routes.py`

- [ ] **Step 1: Write the failing route tests**
- [ ] **Step 2: Implement routes**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(image-embed): POST /v1/images/embeddings route`**

---

## Task D: ImageEmbeddingsClient

Mirror `EmbeddingsClient`. Server URL public attribute, MUSE_SERVER env
fallback, requests + raise_for_status. Helper to convert raw bytes
into a data URL.

**Files:**
- Create: `src/muse/modalities/image_embedding/client.py`
- Modify: `src/muse/modalities/image_embedding/__init__.py` (re-export client)
- Create: `tests/modalities/image_embedding/test_client.py`

- [ ] **Step 1: Write the failing client tests**
- [ ] **Step 2: Implement client**
- [ ] **Step 3: Re-export in `__init__.py`**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit `feat(image-embed): ImageEmbeddingsClient HTTP wrapper`**

---

## Task E: Bundled dinov2_small script

Hand-written script for `facebook/dinov2-small`. Wraps
`transformers.AutoModel` + `AutoImageProcessor` directly (not via the
generic runtime). Lazy imports. Uses the `importlib.import_module`
test pattern established in v0.22.0.

**Files:**
- Create: `src/muse/models/dinov2_small.py`
- Create: `tests/models/test_dinov2_small.py`

- [ ] **Step 1: Write the failing bundled-script tests**
- [ ] **Step 2: Implement bundled script**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(image-embed): bundled dinov2-small script`**

---

## Task F: HF plugin for image-feature-extraction repos

`src/muse/modalities/image_embedding/hf.py`. Sniff: relevant tag +
preprocessor_config.json sibling. Priority **105** so it cleanly wins
over text/classification (200 catch-all) and is below embedding/text
(110).

**Files:**
- Create: `src/muse/modalities/image_embedding/hf.py`
- Create: `tests/modalities/image_embedding/test_hf_plugin.py`

- [ ] **Step 1: Write the failing plugin tests**
- [ ] **Step 2: Implement plugin**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(image-embed): HF plugin for image-feature-extraction repos (priority 105)`**

---

## Task G: Curated entries + slow e2e + integration tests

Add curated entries (3); add slow e2e via in-process supervisor; add
opt-in integration suite.

**Files:**
- Modify: `src/muse/curated.yaml` (+3 entries)
- Create: `tests/cli_impl/test_e2e_image_embedding.py`
- Modify: `tests/integration/conftest.py` (`image_embedding_model` fixture)
- Create: `tests/integration/test_remote_image_embedding.py`

- [ ] **Step 1: Add curated entries**
- [ ] **Step 2: Write slow e2e test**
- [ ] **Step 3: Add integration fixture**
- [ ] **Step 4: Write integration tests**
- [ ] **Step 5: Run fast + slow lanes; integration suite skipped without env var**
- [ ] **Step 6: Commit `test(image-embed): slow e2e + opt-in integration + curated entries`**

---

## Task H: v0.23.0 release

Final wrap-up: docs, version bump, GitHub release notes, tag.

**Files:**
- Modify: `pyproject.toml` (version 0.22.0 to 0.23.0)
- Modify: `src/muse/__init__.py` (docstring; bundled-modalities list)
- Modify: `CLAUDE.md` (modality list)
- Modify: `README.md` (modality list + curl example)

- [ ] **Step 1: Run full test suite (fast + slow)**
- [ ] **Step 2: Bump version**
- [ ] **Step 3: Update src/muse/__init__.py docstring**
- [ ] **Step 4: Update CLAUDE.md**
- [ ] **Step 5: Update README.md**
- [ ] **Step 6: Em-dash check**
- [ ] **Step 7: Commit `chore(release): v0.23.0`**
- [ ] **Step 8: Tag and push**
- [ ] **Step 9: Create GitHub release**
- [ ] **Step 10: Verify release**

---

## Self-review checklist (after Task H)

- [ ] 11 modalities total now? image/embedding discovered via discover_modalities?
- [ ] 10 of 11 with HF plugin coverage? (audio/speech remains bundled-only)
- [ ] POST /v1/images/embeddings works end-to-end with /v1/embeddings envelope?
- [ ] Per-architecture _extract_embeddings tested for CLIP, SigLIP, DINOv2 separately?
- [ ] dinov2-small bundled and curated; siglip2-base + clip-vit-base-patch32 curated?
- [ ] Plugin priority correctly placed at 105 (between embedding/text and image-generation)?
- [ ] All tests pass (fast + slow lanes)?
- [ ] v0.23.0 tagged + pushed; GitHub release published?
- [ ] No em-dashes anywhere?
