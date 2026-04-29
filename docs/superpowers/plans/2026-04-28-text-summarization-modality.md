# `text/summarization` Modality Implementation Plan (#104)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `text/summarization` modality with `POST /v1/summarize` (Cohere-shape). Bundled `bart-large-cnn` script (facebook/bart-large-cnn) and HF plugin sniffing summarization-tagged repos. Generic `BartSeq2SeqRuntime` over `transformers.AutoModelForSeq2SeqLM`.

**Architecture:** Mirror existing modalities. New `text_summarization/` package with protocol, codec, routes, client, `hf.py`, and `runtimes/bart_seq2seq.py`. New bundled `bart_large_cnn.py`. Plugin priority 110 (well-defined `summarization` tag, not catch-all).

**Spec:** `docs/superpowers/specs/2026-04-28-text-summarization-modality-design.md`

**Target version:** v0.22.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/text_summarization/__init__.py` | create | exports `MODALITY`, `build_router`, Protocol, Result, Client, PROBE_DEFAULTS |
| `src/muse/modalities/text_summarization/protocol.py` | create | `SummarizationResult` dataclass, `SummarizationModel` Protocol |
| `src/muse/modalities/text_summarization/codec.py` | create | `encode_summarization_response` |
| `src/muse/modalities/text_summarization/routes.py` | create | `POST /v1/summarize`, request validation |
| `src/muse/modalities/text_summarization/client.py` | create | `SummarizationClient` (HTTP) |
| `src/muse/modalities/text_summarization/runtimes/__init__.py` | create | empty marker |
| `src/muse/modalities/text_summarization/runtimes/bart_seq2seq.py` | create | `BartSeq2SeqRuntime` generic runtime |
| `src/muse/modalities/text_summarization/hf.py` | create | HF plugin for summarization-tagged repos (priority 110) |
| `src/muse/models/bart_large_cnn.py` | create | bundled script (facebook/bart-large-cnn) |
| `src/muse/curated.yaml` | modify | +2 entries: `bart-large-cnn` (bundled), `bart-cnn-samsum` (URI) |
| `pyproject.toml` | modify | bump 0.21.0 to 0.22.0 |
| `src/muse/__init__.py` | modify | docstring v0.22.0; add `text/summarization` to bundled modalities list |
| `CLAUDE.md` | modify | document new modality |
| `README.md` | modify | add `text/summarization` to route list + curl example |
| `tests/modalities/text_summarization/` (full tree) | create | protocol, codec, routes, client, hf_plugin, runtime |
| `tests/models/test_bart_large_cnn.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_summarization.py` | create | slow e2e test |
| `tests/integration/test_remote_summarization.py` | create | opt-in integration tests |
| `tests/integration/conftest.py` | modify | `summarization_model` fixture |

---

## Task A: Protocol + Codec + skeleton

Smallest, most isolated. No callers. Foundation for everything else.

**Files:**
- Create: `src/muse/modalities/text_summarization/__init__.py` (with re-exports; build_router stubbed in routes)
- Create: `src/muse/modalities/text_summarization/protocol.py`
- Create: `src/muse/modalities/text_summarization/codec.py`
- Create: `src/muse/modalities/text_summarization/routes.py` (stub returning empty APIRouter; replaced in Task C)
- Create: `src/muse/modalities/text_summarization/runtimes/__init__.py` (empty)
- Create: `tests/modalities/text_summarization/__init__.py` (empty)
- Create: `tests/modalities/text_summarization/test_protocol.py`
- Create: `tests/modalities/text_summarization/test_codec.py`

- [ ] **Step 1: Write the failing protocol + codec tests**
- [ ] **Step 2: Implement protocol.py**
- [ ] **Step 3: Implement codec.py**
- [ ] **Step 4: Stub routes.py and write `__init__.py`**
- [ ] **Step 5: Run tests; verify they pass**
- [ ] **Step 6: Commit `feat(summarize): text/summarization modality skeleton + codec`**

---

## Task B: BartSeq2SeqRuntime generic runtime

Wraps `transformers.AutoModelForSeq2SeqLM`. Lazy imports. Honors
`device="auto"`, `dtype`, `max_input_tokens`, `default_length`,
`default_format` from manifest capabilities. The runtime owns the
length-to-max_new_tokens mapping; the route layer doesn't touch it.

**Files:**
- Create: `src/muse/modalities/text_summarization/runtimes/bart_seq2seq.py`
- Create: `tests/modalities/text_summarization/runtimes/__init__.py` (empty)
- Create: `tests/modalities/text_summarization/runtimes/test_bart_seq2seq.py`

- [ ] **Step 1: Write the failing runtime tests**
- [ ] **Step 2: Implement runtime**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(summarize): BartSeq2SeqRuntime generic runtime`**

---

## Task C: Routes (POST /v1/summarize)

Replaces the Task A stub with the real endpoint. Validates request,
resolves the registered backend, calls `backend.summarize(...)`
(offloaded to a thread; transformers generate is sync), encodes the
response.

**Files:**
- Modify (replace stub): `src/muse/modalities/text_summarization/routes.py`
- Create: `tests/modalities/text_summarization/test_routes.py`

- [ ] **Step 1: Write the failing route tests**
- [ ] **Step 2: Implement routes**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(summarize): POST /v1/summarize route (Cohere-compat)`**

---

## Task D: SummarizationClient

Mirror `RerankClient`. Server URL public attribute, MUSE_SERVER env
fallback, requests + raise_for_status.

**Files:**
- Create: `src/muse/modalities/text_summarization/client.py`
- Modify: `src/muse/modalities/text_summarization/__init__.py` (re-export client)
- Create: `tests/modalities/text_summarization/test_client.py`

- [ ] **Step 1: Write the failing client tests**
- [ ] **Step 2: Implement client**
- [ ] **Step 3: Re-export in `__init__.py`**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit `feat(summarize): SummarizationClient HTTP wrapper`**

---

## Task E: Bundled bart_large_cnn script

Hand-written script for `facebook/bart-large-cnn`. Wraps
`AutoModelForSeq2SeqLM` directly (not via the generic runtime) so the
script demonstrates the pattern muse uses for other bundled models.
Lazy imports.

**Files:**
- Create: `src/muse/models/bart_large_cnn.py`
- Create: `tests/models/test_bart_large_cnn.py`

- [ ] **Step 1: Write the failing bundled-script tests**
- [ ] **Step 2: Implement bundled script**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(summarize): bundled bart-large-cnn script`**

---

## Task F: HF plugin for summarization-tagged repos

`src/muse/modalities/text_summarization/hf.py`. Sniff: `summarization`
tag. Priority **110** so it cleanly wins over text/classification's
200 catch-all (and matches the embedding/text tier's specificity).

**Files:**
- Create: `src/muse/modalities/text_summarization/hf.py`
- Create: `tests/modalities/text_summarization/test_hf_plugin.py`

- [ ] **Step 1: Write the failing plugin tests**
- [ ] **Step 2: Implement plugin**
- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit `feat(summarize): HF plugin for summarization repos (priority 110)`**

---

## Task G: Curated entries + slow e2e + integration tests

Add curated entries; add slow e2e via in-process supervisor; add
opt-in integration suite.

**Files:**
- Modify: `src/muse/curated.yaml` (+2 entries)
- Create: `tests/cli_impl/test_e2e_summarization.py`
- Modify: `tests/integration/conftest.py` (`summarization_model` fixture)
- Create: `tests/integration/test_remote_summarization.py`

- [ ] **Step 1: Add curated entries**
- [ ] **Step 2: Write slow e2e test**
- [ ] **Step 3: Add integration fixture**
- [ ] **Step 4: Write integration tests**
- [ ] **Step 5: Run fast + slow lanes; integration suite skipped without env var**
- [ ] **Step 6: Commit `test(summarize): slow e2e + opt-in integration + curated entries`**

---

## Task H: v0.22.0 release

Final wrap-up: docs, version bump, GitHub release notes, tag.

**Files:**
- Modify: `pyproject.toml` (version 0.21.0 to 0.22.0)
- Modify: `src/muse/__init__.py` (docstring; bundled-modalities list)
- Modify: `CLAUDE.md` (modality list)
- Modify: `README.md` (modality list + curl example)

- [ ] **Step 1: Run full test suite (fast + slow)**
- [ ] **Step 2: Bump version**
- [ ] **Step 3: Update src/muse/__init__.py docstring**
- [ ] **Step 4: Update CLAUDE.md**
- [ ] **Step 5: Update README.md**
- [ ] **Step 6: Em-dash check**
- [ ] **Step 7: Commit `chore(release): v0.22.0`**
- [ ] **Step 8: Tag and push**
- [ ] **Step 9: Create GitHub release**
- [ ] **Step 10: Verify release**

---

## Self-review checklist (after Task H)

- [ ] 10 modalities total now? text/summarization discovered via discover_modalities?
- [ ] 9 of 10 with HF plugin coverage? (audio/speech remains bundled-only)
- [ ] POST /v1/summarize works end-to-end with Cohere envelope?
- [ ] Length parameter actually changes max_new_tokens (not just metadata)?
- [ ] bart-large-cnn bundled and curated; bart-cnn-samsum curated?
- [ ] Plugin priority correctly placed at 110 (well-defined `summarization` tag, not catch-all)?
- [ ] All tests pass (fast + slow lanes)?
- [ ] v0.22.0 tagged + pushed; GitHub release published?
- [ ] No em-dashes anywhere?
