# text/translation Modality Implementation Plan (v0.58.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship muse's twentieth modality, `text/translation`
(LibreTranslate-compatible machine translation), per the approved spec
`docs/superpowers/specs/2026-07-09-text-translation-design.md`. Read that
spec FIRST; it is the authoritative wire contract.

**Architecture:** Standard modality package
(`src/muse/modalities/text_translation/`) + one generic
`TranslationRuntime` with per-family dispatch (m2m100 / nllb / opus-mt /
madlad) + bundled `m2m100-418m` script + HF resolver plugin + a NEW
generic gateway seam (`MODEL_OPTIONAL_PATHS`) so model-less
LibreTranslate requests route to the modality's default model.

**Tech Stack:** FastAPI, transformers AutoModelForSeq2SeqLM +
AutoTokenizer, sentencepiece; tests are fully mocked (FakeModel pattern,
no weights). Mirror the `text_summarization` modality throughout -- it is
the structural sibling (same backbone, Cohere-style single route); copy
its file skeletons and test organization, then apply this spec's deltas.

## Global Constraints

- MODALITY tag is exactly `"text/translation"`; package dir
  `src/muse/modalities/text_translation/`.
- Wire codes are ISO 639-1 at the boundary. Errors use the OpenAI
  envelope via muse.core.errors patterns (never HTTPException detail
  strings).
- Deferred imports: no torch/transformers at module top-level in any
  runtime or bundled script; `_ensure_deps()` + module-level sentinels;
  tests patch the module-level names.
- Use `muse.core.runtime_helpers` (`select_device`, `dtype_for_name`,
  `set_inference_mode`, `LoadTimer`); the meta-test
  `tests/core/test_runtime_helpers_meta.py` AST-flags re-implementations.
- All new config knobs are rows in `muse.core.config.SETTINGS` (no bare
  os.environ reads).
- ASCII only in all files (repo hook rejects non-ASCII).
- Commit after each green task; message style `feat(translation): ...`
  with the repo's standard trailers.
- Fast lane must stay green: `MUSE_CATALOG_DIR=$(mktemp -d) python -m
  pytest tests/ -q -m "not slow"`.

---

### Task 1: Package skeleton -- protocol, codec, client, __init__

**Files:**
- Create: `src/muse/modalities/text_translation/__init__.py`
- Create: `src/muse/modalities/text_translation/protocol.py`
- Create: `src/muse/modalities/text_translation/codec.py`
- Create: `src/muse/modalities/text_translation/client.py`
- Create: `src/muse/modalities/text_translation/lang_names.py`
- Test: `tests/modalities/text_translation/test_protocol.py`
- Test: `tests/modalities/text_translation/test_codec.py`
- Test: `tests/modalities/text_translation/test_client.py`

**Interfaces (Produces -- later tasks rely on these EXACT signatures):**

```python
# protocol.py
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

@dataclass
class TranslationResult:
    texts: list[str]          # one translated string per input string

@runtime_checkable
class TranslationBackend(Protocol):
    def translate(self, texts: list[str], *, source: str,
                  target: str) -> TranslationResult: ...
    def supported_languages(self) -> dict[str, list[str]]:
        """{iso_code: [target iso codes]}"""
        ...
```

```python
# codec.py
def shape_response(texts: list[str], *, scalar: bool) -> dict:
    """{"translatedText": texts[0]} when scalar else {"translatedText": texts}"""

def normalize_q(q) -> tuple[list[str], bool]:
    """str -> ([q], True); list[str] -> (q, False). Raises ValueError on
    other types or non-str list items (route maps to 422/400)."""

def languages_payload(supported: dict[str, list[str]]) -> list[dict]:
    """LibreTranslate /languages shape: [{code, name, targets}], sorted by
    code; name from lang_names.ISO_639_1_NAMES with fallback to the code
    itself."""
```

`lang_names.py`: a module-level `ISO_639_1_NAMES: dict[str, str]` table
covering at least the ~100 m2m100 codes (en -> "English", etc.). Static
data, no logic.

```python
# client.py -- mirror text_summarization/client.py's structure exactly
class TranslateClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0): ...
    def translate(self, q, source: str, target: str,
                  model: str | None = None): ...   # returns str or list[str]
    def languages(self) -> list[dict]: ...
```

`__init__.py` exports: `MODALITY = "text/translation"`, `build_router`
(imported lazily in Task 3; for THIS task create a stub `build_router`
that raises NotImplementedError so discovery tests do not break --
Task 3 replaces it), `MODEL_OPTIONAL_PATHS = ("/v1/translate",
"/translate", "/languages")`, `PROBE_DEFAULTS = {"shape":
"q='The weather is nice today.' en->es", "call": lambda m:
m.translate(["The weather is nice today."], source="en", target="es")}`,
plus re-exports of `TranslationBackend`, `TranslationResult`,
`TranslateClient`.

- [ ] Step 1: write failing tests: codec scalar/list symmetry
  (`normalize_q("hi") == (["hi"], True)`, round-trip through
  `shape_response`), ValueError on `normalize_q(5)` and
  `normalize_q(["a", 3])`, `languages_payload({"en": ["es"]})` ==
  `[{"code": "en", "name": "English", "targets": ["es"]}]`, protocol
  runtime_checkable satisfaction by a fake, client request/response
  shaping with a mocked httpx (mirror the summarization client tests).
- [ ] Step 2: run, verify FAIL (module missing).
- [ ] Step 3: implement the five files.
- [ ] Step 4: run tests -> PASS; run fast lane -> green.
- [ ] Step 5: commit `feat(translation): protocol + codec + client skeleton`.

### Task 2: TranslationRuntime with per-family dispatch

**Files:**
- Create: `src/muse/modalities/text_translation/runtimes/__init__.py`
- Create: `src/muse/modalities/text_translation/runtimes/hf_translation.py`
- Create: `src/muse/modalities/text_translation/runtimes/nllb_codes.py`
- Test: `tests/modalities/text_translation/test_runtime.py`

**Interfaces:**
- Consumes: `TranslationResult` from Task 1.
- Produces: `TranslationRuntime(hf_repo=..., local_dir=..., device=...,
  model_id=..., num_beams=4, source_language=None, target_language=None,
  **_)` satisfying `TranslationBackend`. Class path referenced by
  manifests as
  `muse.modalities.text_translation.runtimes.hf_translation:TranslationRuntime`.

Family dispatch: module-level `_family_for(hf_repo: str) -> str`
returning one of `"m2m100" | "nllb" | "opus_mt" | "madlad"` by
case-insensitive substring on the repo id (`m2m100`, `nllb`, `opus-mt`,
`madlad`); unknown repos default to `"opus_mt"` semantics ONLY if
`source_language`/`target_language` kwargs are set, else raise
ValueError at construction ("unknown translation family").

Per-family generate contract (see spec section "TranslationRuntime" --
implement exactly):
- m2m100: `self._tok.src_lang = source`; `forced_bos_token_id =
  self._tok.get_lang_id(target)`. `supported_languages()` derives codes
  from `self._tok.lang_code_to_id` (every code targets every other).
- nllb: map wire ISO codes through `nllb_codes.ISO_TO_FLORES`
  (module-level dict, ~200 entries: `{"en": "eng_Latn", "es":
  "spa_Latn", ...}` -- cover the full NLLB-200 list); unmapped code ->
  raise `UnsupportedLanguageError(code, supported)` (define in
  protocol.py; route maps it to 400 invalid_language).
  `self._tok.src_lang = flores_src`; `forced_bos_token_id =
  self._tok.convert_tokens_to_ids(flores_tgt)`.
- opus_mt: no language tokens; constructor stores the declared pair;
  `translate` raises `UnsupportedLanguageError` unless (source, target)
  == the declared pair. `supported_languages()` returns
  `{src: [tgt]}`.
- madlad: prepend `f"<2{target}> "` to each input; no source token;
  `supported_languages()` returns a permissive mapping derived from the
  tokenizer's `<2xx>` tokens.

Generation: single padded batch, `num_beams` from kwarg (default 4),
`max_new_tokens = min(1024, 2 * input_token_len + 16)` per batch (use
the longest input), inside `set_inference_mode()`. Deferred imports via
`_ensure_deps()` sentinels (`torch`, `AutoModelForSeq2SeqLM`,
`AutoTokenizer`).

- [ ] Step 1: failing tests (transformers/torch fully mocked at the
  module path): family dispatch for four repo names + ValueError on
  unknown; m2m100 sets src_lang and forced_bos via get_lang_id; nllb
  maps en->eng_Latn and raises UnsupportedLanguageError on "xx";
  opus_mt refuses a non-declared pair; madlad prepends `<2es> `;
  batch returns one string per input; num_beams forwarded.
- [ ] Step 2: verify FAIL.
- [ ] Step 3: implement.
- [ ] Step 4: PASS + fast lane green (meta-test must not flag
  re-implemented helpers -- use runtime_helpers).
- [ ] Step 5: commit `feat(translation): TranslationRuntime with per-family dispatch`.

### Task 3: Routes + gateway MODEL_OPTIONAL_PATHS seam

**Files:**
- Create: `src/muse/modalities/text_translation/routes.py`
- Modify: `src/muse/modalities/text_translation/__init__.py` (real
  build_router export)
- Modify: `src/muse/core/discovery.py` (aggregate `MODEL_OPTIONAL_PATHS`
  from modality packages into a `{path: modality_tag}` map; export a
  `model_optional_paths()` accessor following the existing discovery
  patterns)
- Modify: `src/muse/cli_impl/gateway.py` (at the `model_id is None` 400
  site: consult the aggregated map; on a hit, resolve the first ENABLED
  catalog model whose modality matches; found -> proceed with that
  model_id; none -> 503 `no_default_model`. Miss -> today's 400
  `model_required` unchanged)
- Modify: `src/muse/core/config.py` (add row `limits.translate_max_chars`,
  env `MUSE_TRANSLATE_MAX_CHARS`, int, default 20000, group limits)
- Test: `tests/modalities/text_translation/test_routes.py`
- Test: `tests/cli_impl/test_gateway_default_model.py`

**Interfaces:**
- Consumes: Task 1 codec/protocol, Task 2's `UnsupportedLanguageError`.
- Produces: `build_router(registry) -> APIRouter` mounting
  `POST /v1/translate`, `POST /translate`, `GET /languages`.

Route behavior (spec "Wire contract" section is normative): pydantic
request model with `q` (str | list[str]), `source: str`, `target: str`,
`format: str = "text"`, `model: str | None = None`. 400s:
`source_detection_not_supported` (source == "auto"),
`unsupported_format` (format != "text"), `invalid_language` (from
UnsupportedLanguageError OR pre-dispatch pair validation against
`supported_languages()`), `input_too_long` (sum of q chars >
config `limits.translate_max_chars`, read per request). Model
resolution inside the router mirrors the other modalities
(`registry.get(MODALITY, model)`; None model -> registry default).
`/languages` uses the registry default model's `supported_languages()`
+ `codec.languages_payload`.

Gateway seam tests (test_gateway_default_model.py): with a fake catalog
holding one enabled `text/translation` model, a POST to `/translate`
with no `model` routes to it (assert the resolved model_id); with the
model disabled -> 503 `no_default_model`; a model-less POST to
`/v1/chat/completions` still 400s `model_required`.

- [ ] Step 1: failing tests (routes via FastAPI TestClient + FakeModel
  backend registered in a fresh ModalityRegistry; every 400 case;
  scalar/list symmetry end-to-end; /translate alias == /v1/translate;
  /languages shape; gateway seam tests).
- [ ] Step 2: verify FAIL.
- [ ] Step 3: implement routes + discovery aggregation + gateway seam +
  config row.
- [ ] Step 4: PASS + fast lane green.
- [ ] Step 5: commit `feat(translation): routes + gateway default-model seam`.

### Task 4: Bundled m2m100-418m script + curated entries

**Files:**
- Create: `src/muse/models/m2m100_418m.py`
- Modify: `src/muse/curated.yaml` (add `nllb-200-distilled-600m`,
  `opus-mt-en-es`, `opus-mt-en-de`)
- Test: `tests/models/test_m2m100_418m.py`

MANIFEST: model_id `m2m100-418m`, modality `text/translation`, hf_repo
`facebook/m2m100_418M`, license MIT, description mentions 100 languages
+ LibreTranslate-compat routes, pip_extras `["transformers", "torch",
"sentencepiece"]`, capabilities `{device: "auto", memory_gb: 2.5,
num_beams: 4}`. `Model` class: alias the shared runtime
(`from muse.modalities.text_translation.runtimes.hf_translation import
TranslationRuntime as Model`) -- the VLM-bundled pattern; no duplicated
construction logic.

Curated entries follow existing YAML shape; NLLB's description MUST
carry "CC-BY-NC-4.0 (non-commercial)"; opus entries set
`capabilities.source_language` / `target_language`.

- [ ] Step 1: failing tests mirroring `tests/models/test_bart_large_cnn.py`
  style: MANIFEST keys/modality/license, Model aliases TranslationRuntime,
  curated entries present with the NC warning and pair capabilities
  (parse curated.yaml directly).
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS + fast lane.
- [ ] Step 5: commit `feat(translation): bundled m2m100-418m + curated NLLB/Opus entries`.

### Task 5: HF resolver plugin

**Files:**
- Create: `src/muse/modalities/text_translation/hf.py`
- Test: `tests/modalities/text_translation/test_hf_plugin.py`

`HF_PLUGIN` dict per docs/HF_PLUGINS.md (single-file import, NO relative
imports -- import runtime class path as a string). Priority 110. Sniff:
repo tagged `translation` OR name matches a family pattern (`m2m100`,
`nllb`, `opus-mt`, `madlad`). Disambiguation rule from the spec: a repo
tagged BOTH `summarization` and `translation` resolves here only when
the name matches a translation family (else leave it to summarization's
plugin). Synthesized manifest: modality `text/translation`,
backend_path
`muse.modalities.text_translation.runtimes.hf_translation:TranslationRuntime`,
capabilities carry family-derived fields (opus pair parsed from the
repo name `opus-mt-{src}-{tgt}` when it matches that shape). Search:
implemented for the modality (mirror summarization's search).

- [ ] Step 1: failing tests: sniff accepts each family, rejects a
  bare summarization repo, both-tagged disambiguation, opus pair
  parsing from name, synthesized manifest shape.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS + fast lane.
- [ ] Step 5: commit `feat(translation): HF resolver plugin`.

### Task 6: MCP tool, smoke matrix, slow e2e, integration, docs, bump

**Files:**
- Modify: `src/muse/mcp/tools/inference_text.py` (+ registration site;
  add `muse_translate` following the existing per-route tool pattern;
  update the tool-count docs/tests if any assert 18)
- Modify: `src/muse/mcp/client.py` (translate call via TranslateClient
  or direct httpx, matching siblings)
- Modify: `.github/workflows/fresh-venv-smoke.yml` (matrix +=
  `opus-mt-en-es`)
- Create: slow e2e test in `tests/cli_impl/` following the existing
  supervisor-e2e pattern (fake translation model; POST /translate alias
  without model resolves via the seam)
- Create: `tests/integration/test_remote_translation.py` (opt-in;
  `test_protocol_translate_en_es_roundtrip` asserts 200 + nonempty
  differing text; skips when model absent)
- Modify: `CLAUDE.md` (modality count 19 -> 20, bullet in the modality
  list, a short section on the LibreTranslate compat + default-model
  seam), `README.md` (modality table/list)
- Modify: `pyproject.toml` version -> `0.58.0`

- [ ] Step 1: failing tests where testable (MCP tool registration count,
  smoke-matrix YAML contains the id, slow e2e RED).
- [ ] Step 2: FAIL. Step 3: implement. Step 4: full fast lane AND slow
  lane green.
- [ ] Step 5: commit `feat(translation): MCP tool + e2e + docs + v0.58.0 bump`.

---

## Self-review notes

- Spec coverage: wire contract -> T1/T3; runtime families -> T2; models
  -> T4; resolver -> T5; seam -> T3; rollout -> T6. `/languages`
  name-table -> T1 lang_names. Input cap config row -> T3.
- Type consistency: `TranslationResult.texts`, `translate(texts, *,
  source, target)`, `supported_languages()` used identically in T2/T3.
- Release ritual itself (tag, PyPI, deploy) is NOT a plan task -- it is
  session-driver work after the final review, per repo convention.
