# I2: Migrate MiniLM + Qwen3-Embedding scripts to the resolver (design)

**Date:** 2026-04-22
**Status:** approved
**Target release:** v0.12.0

## Goal

Delete the two bundled embedding scripts whose logic is now fully covered
by the generic `SentenceTransformerModel` runtime, while preserving the
newbie-friendly catalog ids (`all-minilm-l6-v2`, `qwen3-embedding-0.6b`)
via curated aliases that route through the HF resolver.

## Motivation

After the resolver refactor (v0.10) and the generic sentence-transformers
runtime (E1), the two bundled embedding scripts are duplicates of the
runtime's logic. They differ from the generic runtime only in:

| Script                      | Only real difference               |
|-----------------------------|-------------------------------------|
| `all_minilm_l6_v2.py`       | Hardcoded 384 dims, Apache-2.0 label |
| `qwen3_embedding_0_6b.py`   | `trust_remote_code=True`; 1024 dims |

Dimensions auto-detect from `get_sentence_embedding_dimension()` in the
generic runtime. License comes from the HF repo card. The only real
blocker for a full cutover is that Qwen3-Embedding requires
`trust_remote_code=True`, which the HF resolver's sentence-transformers
branch does not currently synthesize.

`nv_embed_v2.py` stays bundled because it has custom
`.encode(task_type=...)` semantics that do not fit the generic ST
interface.

## Design

### Layer 1: Curated capability overlay

Add one optional field to each `curated.yaml` entry: a
`capabilities:` mapping that merges into the resolver-synthesized
manifest's capabilities block before persist. This is the only schema
addition.

```yaml
- id: qwen3-embedding-0.6b
  uri: hf://Qwen/Qwen3-Embedding-0.6B
  modality: embedding/text
  size_gb: 0.6
  description: "Qwen3-Embedding 0.6B: 1024 dims, 32K context, Apache 2.0"
  capabilities:
    trust_remote_code: true
```

At pull time, `catalog._pull_via_resolver` merges
`curated.capabilities` into `manifest["capabilities"]` (shallow merge;
curated wins on collisions). At `load_backend` time, the merged
capabilities flow into the runtime constructor as kwargs (this path
already exists).

The overlay is general-purpose: future curated entries can override
`context_length`, `chat_format`, `dimensions`, `supports_tools`, or any
other field the runtime accepts. It replaces the only mechanism we had
for this (edit the bundled script), keeping the curated file as the
sole hand-editable configuration surface.

Why not teach the HF resolver to auto-sniff `trust_remote_code`?
Reasonable follow-up but orthogonal. The overlay is strictly more
general (any capability, not just one), and the user consent signal for
`trust_remote_code` lives better in a curated file than in a heuristic.

### Layer 2: Curated entries point at the resolver

- `all-minilm-l6-v2-st` (existing URI entry) renames to `all-minilm-l6-v2`.
  Drop the `-st` suffix; there is no longer a naming collision with a
  bundled script.
- `qwen3-embedding-0.6b` (currently `bundled: true`) converts to a URI
  entry pointing at `hf://Qwen/Qwen3-Embedding-0.6B` with
  `capabilities: {trust_remote_code: true}`.

### Layer 3: Delete the bundled scripts

- `src/muse/models/all_minilm_l6_v2.py`
- `src/muse/models/qwen3_embedding_0_6b.py`
- `tests/models/test_all_minilm_l6_v2.py`
- `tests/models/test_qwen3_embedding_0_6b.py`

### Layer 4: Update references

- `tests/core/test_discovery.py:339`: drop `"all-minilm-l6-v2"` and
  `"qwen3-embedding-0.6b"` from the expected bundled-ids set.
- `tests/core/test_catalog.py:54-55`: same two ids.
- `README.md:143`: update the bundled-scripts listing.
- `tests/core/test_curated.py`: add a test covering the capability
  overlay (load + merge semantics; collision wins for curated).

Left untouched:
- `tests/integration/conftest.py:87`: `require_model_fixture
  ("qwen3-embedding-0.6b")` works regardless of where the id resolves;
  the fixture asks the live server.
- `tests/modalities/embedding_text/test_client.py:87-89`: client-side
  unit test with mocked HTTP; the string is a label.
- `tests/modalities/embedding_text/runtimes/test_sentence_transformers.py:55`:
  runtime unit test; the string is a label.

## Backward compatibility

Anyone who already has `~/.muse/catalog.json` with the old bundled
`backend_path` for these two ids will hit
`ModuleNotFoundError` on `muse serve` after upgrading. Mitigation:
release notes will say:

```
Breaking (embedding/text): `all-minilm-l6-v2` and `qwen3-embedding-0.6b`
are now resolver-pulled via the generic SentenceTransformerModel runtime.
If you pulled either under v0.11.x, re-pull under v0.12.0:

  muse models remove all-minilm-l6-v2
  muse models remove qwen3-embedding-0.6b
  muse pull all-minilm-l6-v2
  muse pull qwen3-embedding-0.6b

No API changes. Same endpoints, same request/response shapes.
```

## Test strategy

- Unit: `test_curated.py` gets one new test for the capability overlay
  (load + merge + collision wins). `test_catalog.py::_pull_via_resolver`
  tests get one new case covering overlay merge into persisted manifest.
- Integration: existing `qwen3_embedding` fixture in
  `tests/integration/conftest.py` covers the end-to-end path; it runs
  against a live server, so it exercises the full load path.
- Full suite must stay green with `-m "not slow"`.

## Out of scope

- Adding `trust_remote_code` auto-sniff to the HF resolver.
- Touching `nv_embed_v2.py`.
- Generalizing the overlay to resolver-URI pulls that do NOT go through
  curated (those can live with resolver-synthesized defaults; custom
  config goes through a curated entry).
