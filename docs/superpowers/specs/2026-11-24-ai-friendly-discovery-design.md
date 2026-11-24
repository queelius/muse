# AI-friendly discovery: voice catalog, described routes, named options

Date: 2026-11-24. Status: approach chosen (describe things where they are
defined, enforced by tests); design for review.

## Intent

The owner wants any AI agent that can reach a muse server (usually a private
LAN box) to work out on its own what the server can do and how to use it,
for every modality and model, including ones added later. It should be able
to inspect what exists, try things, and create new resources (voices in
particular) without a human explaining the API. The immediate driver is a
retro RPG: an agent on the owner's own machine will design, save, organize
and reuse dozens of NPC and narrator voices on muse over the LAN.

The owner prefers agents to use the plain HTTP API from a shell (curl, a
script) rather than MCP. MCP stays supported, and because its tools are
generated from `/openapi.json` and `/v1/capabilities`, it improves with the
same work. Combining modalities (design a voice, then clone it; generate an
image, then pixelize it) is left to the agent: muse describes each piece
well and does not build pipelines.

### What exists today (v0.67.0 agent discovery)

- `GET /v1/capabilities`, `/llms.txt`, `/openapi.json`, `/.well-known/muse.json`.
  Built by `muse.core.capabilities` from each modality's `DISCOVERY` literal
  (AST-read by `muse.core.discovery.modality_discovery`) and per-model notes.
- `tests/core/test_discovery_meta.py` checks that every mounted route is
  declared and that examples validate against the request models.
- An audit on 2026-11-24 against the live server found:
  - 303 of 306 request-schema fields in `/openapi.json` have no description;
    20 of 66 operations have no description.
  - 9 routes have placeholder examples (only `model`): `/v1/3d/from-image`,
    `/v1/audio/embeddings`, `/v1/audio/quality`, `/v1/audio/speech/voices`
    (GET and POST), `/v1/audio/speech/voices/{name}`, `/v1/images/keypoints`,
    `/v1/images/variations`, `/v1/images/ocr`, `/languages`.
  - No error codes are published anywhere; an agent learns them by failing.
  - Named options are not discoverable without trying: saved voices,
    preset voices, pixel-art palettes, LoRAs, MIDI modes, audio/process ops.
  - Voices are a flat list of names. `voice.json` holds name, transcript,
    sample_rate, duration_seconds, sha256, created_at, source_filename and
    nothing about what the voice sounds like or what it is for.
  - `GET /v1/audio/speech/voices?model=` is proxied to a worker, so listing
    voices cold-loads the speech model. Discovery must never load a model.
  - DELETE on a voice has no MCP tool.

## Goals

1. An agent given only the server URL can find every modality, model, route,
   request field (with meaning, type, range and default), realistic example,
   error code and named option, without loading any model.
2. Voices form a catalog an agent can browse, search, audition, create,
   annotate and organize into categories, entirely over HTTP.
3. A new modality or model gets the same quality of description by its
   author writing descriptions in the files they already edit, and CI fails
   when they don't.
4. Discovery output stays bounded as catalogs grow: short index first, full
   detail on request, long lists capped with a link to the full list.

## Non-goals

- Pipelines or multi-step "recipes" endpoints that chain modalities.
- New CLI verbs for voices (the CLI stays the admin surface; generation and
  voices are HTTP).
- LLM-written descriptions shipped without a person reviewing them.
- Cross-node voice sync under federation (each node keeps its own voices).
- Auth on voice writes: the owner decided (2026-11-24) that anyone on the
  LAN may save, edit and delete voices without the admin token.

## Anti-goals

- Metadata that can say something the code no longer does. A description,
  error list or option list that can drift from the code without a failing
  test is worse than none for an agent reasoning on its own.
- Any discovery call (capabilities, llms.txt, voice listing, option listing)
  that loads a model or spawns a worker.
- Breaking existing clients: the current `voices` name list, MCP tool names
  and request shapes keep working.

## Approaches considered

1. **Describe things where they are defined, with tests that enforce it
   (chosen).** Field descriptions on the pydantic request models and FastAPI
   `Form`/`Query` parameters (they flow into `/openapi.json` and from there
   into generated MCP tools); route descriptions, response examples, error
   codes and named options in the modality's `DISCOVERY` literal; a shared
   named-options mechanism; conformance tests that fail CI on gaps or drift.
   It keeps each description next to the code it describes and reuses the
   existing discovery path.
2. **One central metadata file** (YAML for every route, field, example and
   error, rendered into OpenAPI and capabilities). Easy to review in one
   place, but every change touches two places and this API grows fast, so
   it drifts.
3. **LLM-drafted descriptions.** Fast coverage, but nobody checks the
   claims. Usable at most to draft text a person reviews and lands under
   approach 1.

The owner picked approach 1 and the three-phase order below.

## Design

### Phase 1: voice catalog

#### Voice metadata

`voice.json` gains optional fields. Existing files without them load
unchanged (missing fields read as null / empty).

| Field | Type and limits | Meaning |
|---|---|---|
| `category` | slash path, 1-4 segments, each `[a-z0-9][a-z0-9_-]{0,31}`; or null | Where the voice sits, e.g. `narrator/epic`, `npc/human/guard`, `npc/pirate/captain` |
| `tags` | list of at most 16 strings, each `[a-z0-9][a-z0-9_-]{0,31}`, deduplicated, sorted | Free labels: `male`, `elderly`, `gravelly`, `generic` |
| `description` | string, at most 500 chars | What it sounds like |
| `notes` | string, at most 2000 chars | What it is for and how to use it ("intro and chapter cards; keep lines under 20 s") |
| `design` | object or null: `{model, instructions}`, `instructions` at most 4096 chars | How the reference was made, when it came from a designed take |
| `updated_at` | ISO 8601 UTC, set by the server | Last metadata or audio change |

Categories are paths on voices, not separate objects: a category exists
while at least one voice uses it, and there is no create or delete category
operation (my call: it keeps one source of truth and nothing to orphan).

The bundled `narrator.json` gets `category: "narrator/neutral"`, tags,
`description` (its current description text) and `design` (from its
`generated_by` and the designing instructions), and stays read-only.

New error code: 400 `invalid_voice_metadata` (message names the field and
the rule).

#### Routes (all under `/v1/audio/speech/voices`, `model` as query or form field as today)

| Method and path | Change | Notes |
|---|---|---|
| `GET /voices` | extended | Adds `data` (one entry per voice: name, category, tags, description, notes, transcript, duration_seconds, read_only, created_at, updated_at, design) and `categories` (`[{path, count}]` for every path and its ancestors). Keeps `voices` (the name list) for existing clients. Filters: `category=` (matches the path and everything under it), `tag=` (repeatable, all must match), `q=` (case-insensitive substring over name, description, notes, tags). |
| `GET /voices/{name}` | new | One entry, same shape as a `data` item. 404 `voice_not_found`. |
| `GET /voices/{name}/audio` | new | The reference clip as `audio/wav`, so an agent or person can audition a voice without synthesizing. |
| `POST /voices` | extended | Optional form fields `category`, `tags` (comma-separated), `description`, `notes`, `design_model`, `design_instructions`. Overwrite replaces audio and metadata together. |
| `PATCH /voices/{name}` | new | JSON body with any of `category`, `tags`, `description`, `notes`, `design`; a field set to null clears it. Audio and transcript are not patchable (re-save with `overwrite` instead, since the transcript must match the clip). Bundled voices: 409 `voice_read_only`. Rename is not offered (my call: save under the new name, then delete). |
| `DELETE /voices/{name}` | unchanged | Gains an MCP tool. |

PATCH rewrites only `voice.json`, atomically (temp file in the voice
directory, then `os.replace`), and must never pair a transcript with a
different clip.

**Which process does what (my call).** Only `POST /voices` needs the model's
venv (it decodes the uploaded audio with TorchCodec), so only POST goes to
the worker. The three GETs, PATCH and DELETE are light routes (below) and
never load the model: an agent reorganizing dozens of voices should not
cold-load a 3 GB model to edit a JSON file. Because the gateway and the
worker now touch the same voice directory from two processes, every read
and write takes a per-model advisory file lock (`securefs`, a lock file in
the voice root; shared for reads, exclusive for writes), replacing the
in-process `RLock` as the thing that keeps a reader from seeing a
half-swapped directory or a mismatched clip and transcript. The gateway's
`extract_model_from_request` learns to read `model` from the query on
`PATCH` (today a PATCH gets 400 `model_required`).

The discovery validator (`_valid_discovery_route`) and generated MCP tools
gain `PATCH`. New MCP tools: `muse_get_voice`, `muse_get_voice_audio`
(writes the WAV to a file like other media outputs), `muse_update_voice`,
`muse_delete_voice` (destructive annotation).

#### Serving read-only routes without a model: light routes

The three GET routes must answer without loading the speech model. This
adds one generic seam, not voice-specific gateway code:

- A modality package may export `build_light_router(catalog) -> APIRouter`:
  routes that answer from disk or manifests and never need a model in
  memory. Mostly reads; a light route may also write small files under
  `paths.home` (voice metadata, voice deletion) under the owning store's
  file lock. `catalog` gives the route the enabled model ids, their
  manifests and capabilities for this modality (an object the gateway and
  the worker both construct; the builder picks its shape).
- Discovery finds it the way it finds `build_router` (AST check that the
  name exists; imported only by the gateway and worker processes).
- The gateway mounts light routers ahead of its catch-all proxy, so a
  matching GET is answered in the gateway process. The worker also mounts
  the same light router, so a standalone worker and the existing route
  tests keep working.
- A light route resolves `model` the way the proxy would (explicit id, or the
  default-model seam when the route is model-optional), returns the standard
  404 envelope for an unknown model, and for a model that does not support
  saved voices falls back to forwarding (preset-voice models, until phase 3
  gives them a manifest list).
- Light routes are declared in `DISCOVERY` with `"light": true`.
- Light-router modules must not import torch, transformers, diffusers,
  numpy or the modality's runtimes at module level; a meta-test enforces it
  the way `tests/observability/test_public_api.py` keeps fastapi out of
  `muse --help`.

For voices this means splitting `voices.py`: a stdlib-only metadata reader
(list, get, read the WAV bytes, find bundled voices) that the light router
uses, and the existing numpy-backed store for saves. Bundled voices are
found without importing the runtime, by a hidden manifest capability on
`qwen3-tts-0.6b-base` naming the bundled voice files (my call; the builder
can pick another light mechanism).

Federation (my call): voices are per node and stay that way. The merged
capabilities document keeps the first node's `options.voices`, and a request
may land on another node, so with two nodes serving the same voice model an
agent can save on one and get `voice_not_found` on the other. This is
documented as a known limitation in docs/AGENT_DISCOVERY.md; the owner runs
one node today. Routing all voice traffic for a model to one node is a
possible later change.

#### Voices in discovery

- The `qwen3-tts-0.6b-base` model entry in `/v1/capabilities` gains
  `options.voices`: `{count, categories: [{path, count}], sample: first 10
  entries (name, category, description), list_route}`. Filtered views
  (`?model=` and `?modality=audio/speech`) include every entry; the
  unfiltered document keeps only count, categories, sample and the link.
- Computed off the event loop like the rest of the document, cached on the
  voice directory's identity (mtime and entries), and never loads a model.
- The `audio/speech` DISCOVERY text gains the voice-creation recipe in
  `use_for` / route descriptions: design a take with
  `qwen3-tts-1.7b-voicedesign` and `instructions`, listen or check it, save
  it with `POST /voices` including `category`, `description` and
  `design_instructions`, then speak with `voice=<name>` on the base model.

#### Backfill (operational, after phase 1 ships)

Set metadata on the voices already on the owner's server with PATCH:
`narrator_{sage,bard,chronicler,queen,trailer}_{1,2}` get `narrator/epic`
plus their design descriptions and instructions (kept in the session
scratchpad's `epic-narrator/meta.json`); `brineknife_captain` gets
`npc/pirate/captain`. Not part of the code change.

### Phase 2: every route described

Every route in the gateway's `/openapi.json`: the 44 modality operations and
the gateway's own (admin, telemetry, `/v1/models`, `/health`,
`/v1/capabilities`, `/llms.txt`, `/.well-known/muse.json`), since the owner
asked for every endpoint. The audit's 20 undescribed operations are almost
all gateway routes; every modality operation already borrows its modality
summary. Gateway routes have no `DISCOVERY` literal, so (my call) their
`description`, `response_example` and `errors` live in one literal
`SERVER_DISCOVERY` in a new `muse.core.server_discovery` module, validated by
the same rules and published the same way; admin routes are marked as
needing the bearer token.

Strict-parser routes (`/v1/systemone`, `/v1/systemone/batch`,
`/v1/translate`, `/translate`) parse the raw body, so today they have no
request schema in OpenAPI. They gain one (my call): the parser's accepted
shape is declared as a pydantic model attached with `openapi_extra`, used
for documentation only, and a test checks the parser accepts the example and
rejects a body missing a required field, so the schema cannot drift from the
parser.

1. **Field descriptions.** Every field of every pydantic request model and
   every `Form` / `Query` / `File` / path parameter carries a description:
   meaning, unit, range and default when not obvious from the schema, and
   which capability flag gates it if any (the existing `instructions` and
   `pronunciations` descriptions are the model). These reach `/openapi.json`
   and the generated MCP tools with no other change.
2. **Route descriptions.** Each DISCOVERY route gains `description` (one to
   three sentences: what it returns and when to use it rather than a sibling
   route). `merge_openapi` uses it for the operation description in place of
   the modality summary it uses today.
3. **Real examples.** Each route's `example` includes every required field
   with a realistic value, and at least one optional field for routes that
   have them. File fields are named in `file_fields` with a note of what
   file to send. The 9 placeholder routes are fixed.
4. **Response examples.** Each route gains `response_example`: a trimmed
   real response (base64 shortened with `"..."`), so an agent knows what
   comes back without calling it. My call over writing pydantic response
   models for ~66 operations: cheaper, and the test below keeps it honest.
5. **Error codes.** Each route gains `errors`: `[{status, code, when}]`.
   Codes every route can return (404 `model_not_found`, queue and capacity
   503s, 413 upload caps, 422 validation) are declared once in a common list
   that `/v1/capabilities` publishes under `server.errors`.

#### Conformance tests (CI fails on any of these)

- A request field or parameter in `/openapi.json` with no description, or
  one under 10 characters.
- A route with no `description`, an `example` holding only `model`, or no
  `response_example`.
- `example` not validating against the request model (exists today);
  `response_example` keys not a subset of what the route's test client
  returns for the example request against the FakeModel used in that
  modality's route tests (the builder decides how to wire this; a
  per-modality fixture is fine).
- **Error-code drift, both directions** (scope is my call). The codes a
  modality can return are:
  1. literal codes in its own package passed to `error_response(...)` or to
     a code-carrying exception (`VoiceError("voice_exists", ...)`, a class
     whose `code` is a class-level literal);
  2. codes of the shared helpers it imports: each helper module in
     `muse.modalities._common` and `muse.core` that produces error envelopes
     exports a literal `ERROR_CODES` tuple, and its own test checks that
     tuple against that module's literals;
  3. pass-throughs (`error_response(exc.status, exc.code, ...)`) count as
     covered by the literal codes of the exception classes, found by (1) or
     (2), so existing handlers need no rewrite.

  Each reachable code must be declared on one of the modality's routes or in
  the common list, and each declared code must be reachable. A code that is
  neither a literal nor an exception class's literal fails the test until
  it is made one. The common list holds only codes every route can return;
  shared-helper codes such as image decode errors are declared on the
  routes that use them, never in `server.errors`.

### Phase 3: named options everywhere, and the short index

A **named option** is a request field whose valid values are a set of names
an agent cannot guess. Each one is declared in DISCOVERY under a new
top-level `options` list:

```python
"options": [
    {
        "name": "palettes",
        "request_field": "palette",
        "routes": ["/v1/images/pixelize"],
        "description": "Named retro palettes; any of these or a palette spec.",
        "source": {"kind": "static",
                   "check": "muse.modalities.image_pixelart.palettes:NAMED_PALETTES"},
        "items": [{"value": "pico-8", "description": "PICO-8, 16 colours"}, ...],
    },
]
```

Source kinds:

| Kind | Values come from | Examples |
|---|---|---|
| `static` | `items` inline in the literal; `check` names the code constant a test compares against (imported only in the test) | pixel-art palettes, MIDI modes, audio/process ops, autotile formats |
| `model_capability` | a public manifest capability holding a list (per model) | preset TTS voices (Kokoro, Supertonic, Bark gain a `voices` capability), `supported_modes`, `supported_scales`, translation languages |
| `catalog` | models in the catalog matching a capability (computed by `muse.core`) | LoRA adapters with their `base_model` |
| `voice_store` | the phase 1 reader | saved voices |

Every item may carry `description`, `category` and `tags`; categories work
the same everywhere. Model-scoped kinds appear on model entries, others on
the modality entry, both capped in the unfiltered document (count, first 10,
`list_route` or `?modality=` link) and complete in filtered views.

Short index: `GET /v1/capabilities?view=index` returns, per modality, its
summary, route list and model ids, and per model its id, modality,
quality_tier, summary and loaded state. This is the same index `muse_describe`
already builds when the document is over 20,000 chars; move that function
into `muse.core.capabilities` and use it from both. `/llms.txt` gains one
line per declared option ("palettes: 12 named, see ...").

Conformance tests: each option's `request_field` exists on every listed
route's request model; each `static` option's values equal its `check`
constant; each `model_capability` option names a key in
`capability_keys.PUBLIC_KEYS`; a light route listing an option never loads a
model (asserted with a registry that raises on load).

## Constraints

- Discovery and listing never load a model or spawn a worker; computed off
  the event loop.
- `DISCOVERY` stays a pure literal (AST-read); new keys are validated in
  `muse.core.discovery` and malformed declarations are logged and skipped,
  as today.
- Existing response fields, MCP tool names and request shapes keep working.
- Hidden capability keys never appear in public output
  (`muse.core.capability_keys`); new public keys are classified there.
- `muse --help` and `muse pull` still work without ML deps; light routers
  import no heavy libraries.
- Bounded output: unfiltered documents cap every list.
- Tests use isolated catalogs and voice dirs, never `~/.muse`.
- Conventional Commits with a scope; one phase can ship as one release.

## Decided by the owner

- Approach 1, phases in the order voices, then route descriptions, then
  named options.
- Agents may create voices themselves; voices get categories, not a flat
  list.
- HTTP plus a shell is the primary agent interface; MCP follows from the
  same metadata.
- Combining modalities is the agent's job.
- Voice save, PATCH and delete stay open to anyone who can reach the
  server (no admin token); bundled voices stay read-only.

## Settled after the builder review (my calls; the owner can overrule)

- PATCH and DELETE on voices are light and model-free; a cross-process file
  lock guards the voice directory (Phase 1).
- Phase 2 covers the gateway's own routes through `SERVER_DISCOVERY`, and
  strict-parser routes gain a documentation-only schema.
- The error-code scan follows shared helpers through their `ERROR_CODES`
  tuples and treats exception pass-throughs as covered.
- Federation keeps voices per node, documented as a known limitation.

## Left to the builder

- The shape of the `catalog` object passed to `build_light_router`, and how
  the gateway and worker construct it.
- How bundled voices are located without importing the runtime, if not the
  hidden manifest capability suggested above.
- Wiring for the `response_example` test fixture per modality.
- Exact wording of all descriptions, within the rules above.
- Task breakdown, and whether phase 2's per-modality descriptions are split
  into several tasks.

## Open items (owner, not blocking)

- Whether the three phases ship as one release or one each.
