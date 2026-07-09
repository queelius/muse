# text/translation Modality Design (v0.58.0)

Date: 2026-07-09
Status: approved (wire shape, default model, API surface each user-approved
during brainstorming; full design approved as one pass)
Task: #103, the last pending modality on the roadmap.

## Summary

Add muse's twentieth modality, `text/translation`: machine translation via
seq2seq translation models (M2M-100, NLLB-200, Opus-MT, MADLAD-400). The
wire contract is **LibreTranslate-compatible** -- the de-facto standard for
self-hosted translation servers -- so existing LibreTranslate clients work
against muse unmodified for the translate path.

## Decisions (user-approved)

1. **Wire shape: LibreTranslate-compat** (over DeepL-compat or a
   muse-native Cohere-style shape). muse is a self-hosted server;
   LibreTranslate is what self-hosted translation clients speak.
2. **Bundled default: `m2m100-418m`** (facebook/m2m100_418M). MIT license,
   one model covers all pairs across 100 languages, 418M params (~2 GB),
   CPU-runnable, `device: auto`. NLLB-600M is curated-only because its
   CC-BY-NC-4.0 license should not be the modality's default posture.
3. **API surface: "drop-in core"** -- `POST /v1/translate`, the bare
   `POST /translate` alias (LT clients hardcode that path), and
   `GET /languages`. Language auto-detection (`source: "auto"`) and
   `POST /detect` are explicitly deferred; v1 returns a structured 400.

## Wire contract

### POST /v1/translate and POST /translate (alias)

Request (LibreTranslate shape plus a muse extension):

```json
{
  "q": "Hello world" | ["Hello", "world"],
  "source": "en",
  "target": "es",
  "format": "text",          // optional; only "text" supported in v1
  "model": "m2m100-418m"     // OPTIONAL muse extension; LT clients omit it
}
```

- `q`: single string or list of strings (batch). Scalar in -> scalar out;
  list in -> list out.
- `source` / `target`: ISO 639-1 codes at the wire (`"en"`, `"es"`),
  matching LibreTranslate. Per-family model-code mapping is internal.
- Empty `q` string translates to an empty string (no error). An empty
  list returns an empty list.

Response (LibreTranslate shape):

```json
{ "translatedText": "Hola mundo" }        // scalar q
{ "translatedText": ["Hola", "mundo"] }   // list q
```

Errors (OpenAI envelope `{"error": {code, message, type}}`, consistent
with every other muse route; LT clients read the HTTP status):

- `source: "auto"` -> 400 `source_detection_not_supported` ("explicit
  source language required; detection is planned").
- `format` other than `"text"` -> 400 `unsupported_format`.
- A language code the resolved model does not support -> 400
  `invalid_language`; the message names the offending code and lists (or
  summarizes, if >20) the supported codes.
- `source == target` -> allowed; the model translates normally (identity
  round-trips are valid LT requests and useful for testing).
- Missing `q`, or `q` of wrong type -> 422 via pydantic validation.
- Input size cap: total `q` characters capped at
  `MUSE_TRANSLATE_MAX_CHARS` (`limits.translate_max_chars`, config
  registry row, default 20000, read per-request) -> 400 `input_too_long`.

### GET /languages

LibreTranslate shape, derived LIVE from the default translation model's
tokenizer (no hand-maintained language table):

```json
[ {"code": "en", "name": "English", "targets": ["es", "de", ...]}, ... ]
```

For m2m100/NLLB (many-to-many) every language lists every other language
as a target. For an Opus-MT pair model the list is exactly the declared
pair. Code->name mapping uses a small static ISO 639-1 name table
(data file, not code).

### Default-model resolution (gateway seam)

LT clients send no `model` field, but the muse gateway routes by `model`
and today 400s `model_required` when it is absent. New mechanism, derived
from structure rather than a hardcoded path table:

- A modality package MAY export `MODEL_OPTIONAL_PATHS: tuple[str, ...]`
  (exact request paths on which `model` is optional). `text_translation`
  exports `("/v1/translate", "/translate", "/languages")`.
- `discover_modalities` aggregates these into a `{path: modality_tag}`
  map alongside the existing router map.
- When `extract_model_from_request` returns None AND the request path is
  in the map, the gateway resolves the default model: the first
  **enabled** catalog model whose modality matches the tag (bundled
  m2m100 wins by discovery order on a fresh install). If no enabled model
  of that modality exists -> 503 `no_default_model`.
- All other paths keep today's 400 `model_required` exactly.

This is generic: a future modality (e.g. OpenAI-optional-model
/v1/moderations) can adopt it by exporting its own paths.

## Modality package

`src/muse/modalities/text_translation/`, MODALITY = `"text/translation"`.
Standard layout:

- `protocol.py`: `TranslationBackend` Protocol
  (`translate(texts: list[str], source: str, target: str) -> list[str]`,
  `supported_languages() -> dict[str, list[str]]` returning
  `{code: [target codes]}`), `TranslationResult` dataclass.
- `routes.py`: `build_router(registry)` mounting the three routes above.
  Capability gating happens here (pair validation against
  `supported_languages()` before dispatch).
- `codec.py`: trivial JSON shaping (scalar/list symmetry for `q` /
  `translatedText`), the ISO-code name table loader.
- `client.py`: `TranslateClient` (`translate(q, source, target,
  model=None)`, `languages()`), MUSE_SERVER-based like the siblings.
- `runtimes/hf_translation.py`: the generic runtime (below).
- `hf.py`: resolver plugin (below).
- `__init__.py`: exports `MODALITY`, `build_router`,
  `MODEL_OPTIONAL_PATHS`, `PROBE_DEFAULTS` (a short en->es sentence),
  re-exports protocol + client.

## TranslationRuntime (one runtime, per-family dispatch)

`TranslationRuntime` over `transformers.AutoModelForSeq2SeqLM` +
`AutoTokenizer`, deferred-imports pattern, `muse.core.runtime_helpers`
for device/dtype/inference-mode. NOT a reuse of summarization's
`BartSeq2SeqRuntime`: translation generation needs per-family language
plumbing the summarizer must not carry.

Family dispatch (`_family_for(hf_repo)`, name-pattern based, mirroring
the 3D `_family_for` precedent):

- **m2m100**: `tokenizer.src_lang = source`; generate with
  `forced_bos_token_id = tokenizer.get_lang_id(target)`. Wire ISO codes
  ARE the model codes. Supported set from `tokenizer.lang_code_to_id`.
- **nllb**: same mechanics with FLORES-200 codes (`eng_Latn`). A
  structured ISO-639-1 -> FLORES-200 table (module-level dict in a data
  module, covering NLLB's 200 languages; unmapped wire code -> 400
  `invalid_language`). `forced_bos_token_id =
  tokenizer.convert_tokens_to_ids(flores_code)`.
- **opus-mt**: no language tokens; the pair is fixed by the checkpoint.
  The manifest declares `capabilities.source_language` /
  `capabilities.target_language`; any other requested pair -> 400
  `invalid_language`. (Multiway opus-mt-mul-* checkpoints are out of
  scope for v1; the resolver still pulls them but they run pairless,
  documented limitation.)
- **madlad**: T5-style `<2xx>` target prefix prepended to the input; no
  source token (MADLAD auto-infers source). Curated-only, GPU-oriented.

Generation defaults: `max_new_tokens` proportional to input length
(`min(1024, 2 * input_tokens + 16)`), beam search `num_beams=4` (quality
matters more than speed for MT; configurable via
`capabilities.num_beams`). Batch `q` is tokenized as one padded batch,
one generate call.

## Models

- **Bundled** `src/muse/models/m2m100_418m.py`: facebook/m2m100_418M,
  MIT, ~2 GB, `memory_gb: 2.5` (conservative; probe self-heals),
  `device: auto`, pip_extras: transformers, torch, sentencepiece.
- **Curated** (`src/muse/curated.yaml`):
  - `nllb-200-distilled-600m` (facebook/nllb-200-distilled-600M,
    **CC-BY-NC-4.0**, description carries the non-commercial warning,
    200 languages).
  - `opus-mt-en-es`, `opus-mt-en-de` (Helsinki-NLP, ~300 MB each,
    permissive, best per-pair quality;
    `capabilities.source_language/target_language` set).
- **Resolver** (`hf.py`, priority 110): sniffs `translation`-tagged repos
  (and `text2text-generation` + name-pattern fallback), dispatches
  family by repo-name pattern (`m2m100`, `nllb`, `opus-mt`, `madlad`).
  Excludes repos already claimed by summarization's plugin (tag
  disambiguation: `translation` tag wins here; `summarization` tag wins
  there; both-tagged repos go to translation only if the name matches a
  translation family).

## Testing

- `tests/modalities/text_translation/`: protocol/codec/route tests with
  FakeModel-pattern backends (no weights): scalar/list symmetry, alias
  path, /languages shape, all five 400 cases, default-model fallback
  (gateway seam unit-tested with a fake catalog), input cap.
- `tests/models/test_m2m100_418m.py`: fully mocked (transformers patched
  at module path), family dispatch, forced_bos wiring, ISO passthrough.
- Runtime tests: NLLB FLORES mapping, opus pair refusal, madlad prefix.
- Slow e2e: supervisor test registering a fake translation model,
  end-to-end /translate alias + model-less request.
- Integration (opt-in, MUSE_REMOTE_SERVER): en->es round-trip sanity on
  the live box.
- CI smoke matrix: add `opus-mt-en-es` (~300 MB, CPU-friendly); m2m100
  at ~2 GB exceeds the lightweight guidance and is excluded.

## Out of scope (deferred)

- Language auto-detection (`source: "auto"`) and `POST /detect` -- needs
  a language-ID model class; lands later without changing this contract.
- `format: "html"` (tag-preserving translation).
- LibreTranslate API-key emulation and its `/frontend/*` endpoints.
- Multiway opus-mt-mul checkpoints (pairless operation documented).

## Rollout

MCP `muse_translate` inference tool (one-tool-per-route pattern, 19th
inference tool). Docs: CLAUDE.md modality list + conventions, README.
Ships as **v0.58.0** after the standard release ritual; deploy to frodo
+ local box; live-validate an en->es translation and a real
LibreTranslate client call against `/translate`.
