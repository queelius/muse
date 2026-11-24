# Described Routes (AI-friendly discovery, phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every route in the gateway's `/openapi.json` gets the following, and CI fails on any gap or drift:
- a description of every field and parameter;
- a route description;
- a realistic request example and a checked response example;
- its error codes.

**Architecture:**
- **Where metadata lives:** route-level metadata (`description`, `response_example`, `errors`) goes in each modality's `DISCOVERY` literal. The gateway's own routes get a new literal, `SERVER_DISCOVERY`, in `muse.core.server_discovery`, which also holds `COMMON_ERRORS`. Field descriptions live on the pydantic models and FastAPI parameters themselves.
- **Where it shows up:** `merge_openapi` puts the route descriptions and error codes into `/openapi.json`. `/v1/capabilities` publishes `server.errors` and `server.routes`. The generated MCP tools reuse all of it.
- **How it stays honest:** four conformance suites, each with a shrinking `PENDING` set, let content land one modality group per task. A final task deletes `PENDING` and makes the keys required.

**Tech Stack:** Python 3.10+, FastAPI, pydantic v2, the stdlib `ast` (for the error-code scan) and pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-ai-friendly-discovery-design.md`, "Phase 2: every route described", including its builder-review settlements.

## Global Constraints

- **Scope:** every route in the gateway's `/openapi.json`: the 47 modality operations (light routes included) and the gateway's own 22: `/health`, `/v1/models`, `/v1/capabilities`, `/dashboard`, 12 `/v1/admin/*` and 6 `/v1/telemetry/*`.
- **Field descriptions:** every request field and parameter (path, query, form, file, JSON body) has a `description` of at least 10 characters. It states the meaning, plus the unit, range and default when the schema does not make them obvious, and which capability flag gates the field if any. The existing `instructions` and `pronunciations` descriptions in `audio_speech/routes.py` are the model to follow.
- **Route `description`:** 1-3 sentences, 20-600 characters, saying what the route returns and when to use it instead of a sibling route. `merge_openapi` uses it as the operation description in place of the modality summary.
- **`example`:** every required field has a realistic value; at least one optional field appears when the route has any; it has at least one key besides `model` when the route has other inputs; at most 1500 characters as JSON.
- **`response_example`:**
  - For a JSON response: a trimmed real response, with base64 cut to `"..."`.
  - For any other content type: a one-line string describing the body, e.g. `"WAV audio bytes (mono, the model's sample rate)"`.
  - At most 1500 characters as JSON.
- **`errors`:** a list of `{"status": int, "code": str, "when": str}`. `status` is 400-599, `code` matches `^[a-z][a-z0-9_]*$`, and `when` is 10-200 characters. Codes every routed request can return live once in `COMMON_ERRORS`, never per route. Shared-helper codes (image decode, upload caps) are declared on the routes that use them, never in `COMMON_ERRORS`.
- **Accuracy:** descriptions are written from the code (route, runtime, `docs/MODALITIES.md`), never guessed. When the code and the docs disagree, the code wins, and the docs are fixed in the same task.
- **Compatibility:** existing request shapes, response shapes, MCP tool names and DISCOVERY keys keep working. The new keys are optional until Task 11.
- **Commits and tests:** Conventional Commits with a scope; each ends with the session's `Co-Authored-By` line. Tests never touch `~/.muse`. Work in a worktree and run tests with `PYTHONPATH=$PWD/src`.

## Review Focus

1. **A description that contradicts the schema**, e.g. it says "default 30" while the field defaults to 50. Presence tests can't catch that. Task 2 adds `test_description_defaults_match_schema`: a description containing `default <value>` must name the schema's default.
2. **An error code declared with the wrong status.** Task 3 adds `test_literal_statuses_match_declared`: where the code is `error_response(<int literal>, "<code>", ...)`, that status must be among the statuses declared for that code.
3. **Capabilities or MCP output ballooning** from examples and errors. The per-item caps above are enforced in Task 2. Task 1 adds `test_mcp_tool_description_is_bounded`: generated tool descriptions stay at most 4000 characters.
4. **An example that validates but fails against a capable model**, for instance because it needs a capability flag the fake lacks. The Task 4 harness runs every example against a fake with the route's capabilities and requires a 2xx.
5. **A gateway route that needs the admin token being documented as open.** Task 5 adds `test_admin_routes_marked_auth`: every `/v1/admin/*` entry in `SERVER_DISCOVERY` has `"auth": "admin"`, and the OpenAPI operation documents 401/403/503 `admin_disabled`.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/muse/core/server_discovery.py` (new) | `SERVER_DISCOVERY` (the gateway's own routes, as route dicts) and `COMMON_ERRORS` |
| `src/muse/core/discovery.py` | the validator accepts `description`, `response_example`, `errors` (and `auth` for server routes) |
| `src/muse/core/capabilities.py` | `server_block` adds `errors` and `routes`; `render_llms_txt` adds one line |
| `src/muse/cli_impl/agent_routes.py` | `merge_openapi`: route descriptions, error responses, server-route descriptions and tags, injected strict-parser body schemas |
| `src/muse/core/openapi_docs.py` (new) | `body_schema(model) -> (schema, defs)` for documentation-only request bodies |
| `src/muse/mcp/generated.py` | the tool description prefers the route description and lists error codes |
| `src/muse/modalities/_common/{fields,inference,uploads,images}.py` | each exports a literal `ERROR_CODES` tuple |
| `tests/discovery/` (new package) | `pending.py` (the one shared `PENDING` set), `test_route_docs.py` (descriptions, examples, caps), `test_error_codes.py` (drift scan), `test_response_examples.py` + `fakes.py` (live check) |
| modality `__init__.py` / `routes.py` / `light.py` files | content (Tasks 6-10) |

---

### Task 1: Route keys, server discovery module, publishing

**Files:**
- Create: `src/muse/core/server_discovery.py`
- Modify:
  - `src/muse/core/discovery.py`: `_valid_discovery_route` / `_OPTIONAL_ROUTE_KEYS`
  - `src/muse/core/capabilities.py`: `server_block`, `render_llms_txt`
  - `src/muse/cli_impl/agent_routes.py`: `merge_openapi`
  - `src/muse/mcp/generated.py`: `_description`
- Test: `tests/core/test_discovery.py`, `tests/core/test_capabilities.py`, `tests/cli_impl/test_agent_routes.py`, `tests/mcp/test_generated.py`

**Interfaces:**
- Produces:
  - `muse.core.server_discovery.SERVER_DISCOVERY: dict`, shaped `{"routes": [route dicts]}`. Each route dict has the modality route keys, minus `file_fields`, `media_fields` and `mcp_tool`, plus `auth: "none" | "admin" | "telemetry"`.
  - `muse.core.server_discovery.COMMON_ERRORS: list[dict]`, entries `{status, code, when}`.
  - Both are pure literals. This task seeds them with the three discovery routes (`/v1/capabilities`, `/v1/models`, `/health`) and an empty `COMMON_ERRORS`; Tasks 3 and 10 fill them.
  - `muse.core.discovery.valid_error_entry(e) -> bool` and `valid_route_metadata(route) -> bool`, which check the three new keys against the Global Constraints rules. `_valid_discovery_route` calls `valid_route_metadata` when the keys are present.
  - `server_block(...)` gains `"errors": copy of COMMON_ERRORS` and `"routes": copy of SERVER_DISCOVERY["routes"]`.
  - `merge_openapi`:
    - The operation description is `route["description"]` if present, else the modality summary (as today).
    - For each declared error, `op["responses"][str(status)]` gets `"description"` set to `"; ".join(f"{code}: {when}")` for that status, merged with what FastAPI generated.
    - Gateway operations named in `SERVER_DISCOVERY` get its description and `tags: ["server"]`.

- [ ] **Step 1: Write the failing tests**
  - `test_valid_route_metadata_rules`: valid and invalid `errors` entries (bad status 302, code `Bad-Code`, short `when`), a description under 20 or over 600 characters, and an oversize `example` / `response_example`.
  - `test_server_block_publishes_errors_and_routes`: `server_block(...)["errors"] == COMMON_ERRORS` and `["routes"][0]["path"] == "/v1/capabilities"`.
  - `test_llms_txt_mentions_server_errors_and_routes`.
  - `test_merge_uses_route_description_and_error_responses`: build `merge_openapi` from a fake discovery carrying one route with `description` and `errors`. Assert the op description and `responses["400"]["description"]` contains `code: when`.
  - `test_mcp_tool_description_uses_route_description_and_lists_errors`.
  - `test_mcp_tool_description_is_bounded`: every generated tool description is at most 4000 characters.

- [ ] **Step 2: Run the tests and see them fail.**
  Run: `PYTHONPATH=$PWD/src pytest tests/core/test_discovery.py tests/core/test_capabilities.py tests/cli_impl/test_agent_routes.py tests/mcp/test_generated.py -q`

- [ ] **Step 3: Implement it**
  - MCP `_description`: start with `route["description"]` when present, keeping the modality summary as the second paragraph. End with `"Errors: " + ", ".join(codes)` when the route declares any, truncating that list to stay within 4000 characters.
  - `render_llms_txt`: add one Discovery bullet: ``- Errors and server routes: `server.errors` (codes any route can return) and `server.routes` (health, models, admin with a bearer token, telemetry) in [Capabilities](/v1/capabilities)``.

- [ ] **Step 4: Run the tests and see them pass**, plus `tests/core tests/mcp tests/cli_impl/test_agent_routes.py`.

- [ ] **Step 5: Commit** with `feat(discovery): route descriptions, response examples and error codes`.

---

### Task 2: Route documentation conformance suite

**Files:**
- Create: `tests/discovery/__init__.py`, `tests/discovery/conftest.py`, `tests/discovery/pending.py`, `tests/discovery/test_route_docs.py`

**Interfaces:**
- Consumes: Task 1's keys and `SERVER_DISCOVERY`.
- Produces:
  - Fixture `merged_openapi`: `build_gateway(state=SupervisorState(workers=[], device="cpu")).openapi()`, with telemetry enabled so the telemetry routes appear.
  - Helper `owner(op) -> str`: the operation's first tag, which is the modality tag or `"server"`.
  - `tests/discovery/pending.py: PENDING: set[str]`, the owners not yet done: every modality tag plus `"server"`. This is the one set every discovery suite (Tasks 2-5) imports and skips by. Tasks 6-10 remove owners from it and Task 11 deletes it.

- [ ] **Step 1: Write the tests.** Each is parametrized over (method, path) in the merged OpenAPI, or over declared routes:
  - `test_every_operation_has_a_description`: at least 20 characters.
  - `test_every_parameter_and_body_field_is_described`: at least 10 characters. Resolve `$ref` and `allOf`/`anyOf` wrappers. A property is described if it or its resolved schema carries `description`.
  - `test_every_declared_route_has_route_metadata`: `description`, `response_example` and `errors` are present (in `DISCOVERY` or `SERVER_DISCOVERY`).
  - `test_examples_cover_required_fields_and_one_optional`: required fields come from path, query, form and JSON body plus `file_fields`. At least one optional field appears when any exists, and at least one non-`model` key when the route has other inputs.
  - `test_examples_and_response_examples_are_bounded`: each at most 1500 characters as JSON.
  - `test_description_defaults_match_schema`: a regex `\bdefault(?:s to| is|:)?\s+` followed by a value; that value must equal `str(schema["default"])`, case-insensitively, with quotes and backticks stripped.

- [ ] **Step 2: Run them.**
  Run: `PYTHONPATH=$PWD/src pytest tests/discovery/test_route_docs.py -q`
  Expected: pass, because every owner is in `PENDING` and so every case is skipped. Then temporarily remove `"audio/speech"` from `PENDING`, confirm the suite reports that owner's gaps, and restore it.

- [ ] **Step 3: Commit** with `test(discovery): route documentation conformance suite`.

---

### Task 3: Error-code drift scan, shared-helper codes, COMMON_ERRORS

**Files:**
- Create: `tests/discovery/test_error_codes.py`
- Modify:
  - `src/muse/modalities/_common/fields.py`, `inference.py`, `uploads.py`, `images.py`: add `ERROR_CODES: tuple[str, ...]`.
  - `src/muse/core/server_discovery.py`: `COMMON_ERRORS` content.
- Test: the new file

**Interfaces:**
- Consumes: Task 1 (`COMMON_ERRORS`, `SERVER_DISCOVERY`).
- Produces: `scan_codes(paths: Iterable[Path]) -> dict[str, set[int | None]]`, which maps each code to the literal statuses seen (`None` when not literal). It lives in the test module.

**The scan** (AST, stdlib only). A file contributes a code for each of the following:
1. `error_response(<status>, "<code>", ...)`, recording the status when it is an int literal.
2. A call to a callee whose name ends in `Error` with a first positional snake_case literal (`VoiceError("voice_exists", ...)`, `OperationError("model_disabled", ...)`, `DecisionBackendError(...)`, `OptionError(...)`).
3. `super().__init__("<code>", ...)` inside a class whose name ends in `Error`.
4. A class-level `code = "<literal>"` in a class whose name ends in `Error`.
5. A dict literal with a `"code"` key and a literal string value.

`NOT_CODES` is a small allowlist in the test of strings the heuristic catches that are not codes; each entry carries a reason comment.

**What counts as reachable:**
- Modality: the files of `src/muse/modalities/<pkg>/**`, plus the `ERROR_CODES` of every `_common` module the package imports (directly or through other `_common` modules), plus `model_not_found` (from `muse.core.errors.ModelNotFoundError`).
- Server: `cli_impl/gateway.py`, `cli_impl/load_director.py`, `cli_impl/light_catalog.py`, `core/server.py`, `core/errors.py`, `core/light.py`, `admin/**` and `observability/dashboard.py`.

- [ ] **Step 1: Write the failing tests**
  - `test_shared_helper_error_codes_match_their_module(helper)`: each `_common` module's `ERROR_CODES` equals `scan_codes([that file])`, minus `NOT_CODES`.
  - `test_modality_codes_are_declared(tag)`: reachable is a subset of (codes declared on the tag's routes, union `COMMON_ERRORS` codes). Skipped for tags in `PENDING`.
  - `test_declared_modality_codes_are_reachable(tag)`: the reverse direction.
  - `test_server_codes_are_declared`: reachable server codes are a subset of `COMMON_ERRORS` union the `SERVER_DISCOVERY` route errors. Skipped while `"server"` is in `PENDING`.
  - `test_common_errors_are_reachable_somewhere`: every `COMMON_ERRORS` code is found by some scan.
  - `test_literal_statuses_match_declared`: for every code with literal statuses, each of them appears among that code's declared statuses (on the routes of the scope that raised it, or in `COMMON_ERRORS`).
  - `test_error_response_codes_are_literal_or_passthrough`: every `error_response(...)` code argument in the scanned files must be one of: a string literal; an attribute named `code` (`exc.code`, a pass-through covered by rule 2-4); or a local name whose every assignment in that function is a string literal. Anything else fails, naming file:line, until it is made one of those. Today's 16 non-literal calls (e.g. `gateway.py`'s `code` variable) must pass as they are, or be rewritten in this task.

- [ ] **Step 2: Run them and see them fail.**
  The helper test fails because `ERROR_CODES` is missing, and the common-errors tests fail because `COMMON_ERRORS` is empty.

- [ ] **Step 3: Implement it**
  - Add each module's `ERROR_CODES`.
  - Fill `COMMON_ERRORS` with every code a routed request can get from the gateway or worker shell, read from `cli_impl/gateway.py`, the `OperationError` codes in `cli_impl/load_director.py`, and `core/server.py`. Leave out codes owned by a single server route (e.g. `not_found` from `/v1/capabilities` filters), which go on that `SERVER_DISCOVERY` route in Task 10.
  - Each entry's `when` states the condition: e.g. 400 `model_required`, "the request names no model and the route is not model-optional"; 503 `queue_full`, "the model's request queue is at server.max_queue_depth".

- [ ] **Step 4: Run them**; everything not skipped by `PENDING` passes.

- [ ] **Step 5: Commit** with `test(discovery): error-code drift scan; ERROR_CODES on shared helpers; COMMON_ERRORS`.

---

### Task 4: Response-example harness

**Files:**
- Create: `tests/discovery/fakes.py`, `tests/discovery/test_response_examples.py`

**Interfaces:**
- Produces:
  - `FakeSpec = NamedTuple(factory: Callable[[], Any], model_id: str, capabilities: dict)`.
  - `FAKES: dict[str, FakeSpec]` (modality tag to spec). This task adds `audio/speech` and `text/translation`; Tasks 6-9 add the rest.
  - `SAMPLES`: sample upload bytes keyed by kind (`"wav"` for 2 s of 24 kHz silence, `"png"` for 8x8 RGB, `"glb"` only if some route takes one). `sample_for(tag, field) -> tuple[filename, bytes, mime]` picks by field name, falling back to the modality's input kind.
  - `SKIP_LIVE: dict[tuple[str, str], str]`: routes that cannot run against a fake, each with a reason (e.g. one needing a real llama.cpp grammar).

- [ ] **Step 1: Write the test**
  - `test_response_example_matches_a_live_response(tag, method, path)`, one case per declared route of each tag in `FAKES` and not in `PENDING`.
  - It builds a worker-style app (`create_app` with `build_router` and `build_light_router` over a registry holding the fake, under the spec's capabilities and a tmp `MUSE_HOME`).
  - It sends the route's example: `"<model>"` becomes the fake's id; path parameters go in the path; GET and DELETE take query parameters; JSON routes take a JSON body; multipart routes take form fields plus `sample_for` files.
  - It asserts a 2xx. For a JSON response, the `response_example` keys must be a subset of the response keys at the top level, and, for one level of nested dicts or lists of dicts, of the first element's keys. For any other response, `content-type` starts with `response_content_type`.
  - `test_every_declared_tag_has_a_fake_or_is_pending`.

- [ ] **Step 2: Run it.** Speech and translation fail where their `response_example` is still missing. Mark those two `PENDING` until Tasks 6 and 7, so the suite passes.

- [ ] **Step 3: Commit** with `test(discovery): live response-example harness`.

---

### Task 5: Documentation-only request schemas for strict-parser routes; server-route harness

**Files:**
- Create:
  - `src/muse/core/openapi_docs.py`
  - `src/muse/modalities/decision_evaluation/schema.py`: doc-only `SystemOneRequest` and `SystemOneBatchRequest`, mirroring `validation.parse_request` / `parse_batch_request`.
- Modify:
  - `decision_evaluation/routes.py`, `text_translation/routes.py`: `openapi_extra={"x-muse-body-model": "<module>:<Class>"}` on `/v1/systemone`, `/v1/systemone/batch`, `/v1/translate` and `/translate`.
  - `cli_impl/agent_routes.py`: resolve `x-muse-body-model`.
  - `tests/discovery/test_response_examples.py`: server-route cases.
- Test: `tests/core/test_openapi_docs.py`, `tests/modalities/decision_evaluation/test_schema.py`, `tests/modalities/text_translation/test_schema.py`

**Interfaces:**
- Produces:
  - `body_schema(model: type[BaseModel]) -> tuple[dict, dict]`: `(schema, defs)`, using `TypeAdapter(model).json_schema(ref_template="#/components/schemas/{model}")`, with `$defs` popped into `defs`.
  - In `_modality_schema` / `merge_openapi`, an operation whose `x-muse-body-model` resolves (by import) gets `requestBody: {"required": true, "content": {"application/json": {"schema": schema}}}`, and `defs` are merged into `components.schemas`, under the same rename rules as today's clashes. Drop the `x-muse-body-model` key from the published operation.
  - The doc models carry `Field(description=...)` on every field; the content tasks (7) fill those in.

- [ ] **Step 1: Write the failing tests**
  - `test_body_schema_hoists_defs`.
  - `test_systemone_schema_matches_parser`: the DISCOVERY example parses with `parse_request` **and** validates against `SystemOneRequest`. For each required field, removing it makes both reject.
  - The same test for the batch route and for translation (`_TranslateRequest` is already the model; reuse it).
  - `test_openapi_has_request_bodies_for_strict_routes`: the merged OpenAPI operation has a `requestBody` schema with `properties`.
  - Server-route harness `test_server_response_example_matches(method, path)`:
    - Use `build_gateway` with `MUSE_ADMIN_TOKEN` set in the env, sent as a bearer header for `auth: "admin"`, telemetry enabled over a tmp store, and a `SupervisorState` with a director spy.
    - Run each `SERVER_DISCOVERY` route not in `SKIP_LIVE_SERVER`. Seed that set with the job-starting and destructive routes, with reasons: enable, probe, pull, DELETE model, worker restart and disable.
    - Skipped while `"server"` is in `PENDING`.

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement it.**

- [ ] **Step 4: Run them and see them pass**, plus `tests/modalities/decision_evaluation tests/modalities/text_translation tests/cli_impl/test_agent_routes.py tests/mcp`. MCP now generates typed schemas for `muse_decide` and `muse_translate`; update any test that asserted the untyped fallback for them, and keep `test_file_fields_become_path_and_url`'s untyped-`state` assertion true by leaving `state` untyped in the doc model.

- [ ] **Step 5: Commit** with `feat(discovery): documented request bodies for strict-parser routes`.

---

### Tasks 6-10: Content (one modality group per task)

Every content task follows the same steps for each modality in its group:

1. **Write field descriptions** on every request model field, `Form` / `Query` / `File` / path parameter and doc-only model field the routes use, following the Global Constraints. Use `Field(..., description=)`, `Form(..., description=)` and so on, and never change types or defaults.
2. **Fill the DISCOVERY route keys** for every route:
   - `description`;
   - a realistic `example` (fix the placeholder ones named in the spec);
   - `response_example`, written from what the route actually returns (run the harness to see it);
   - `errors`: the route's codes that are not in `COMMON_ERRORS`. Run `test_error_codes.py` for the tag to list them; give each its true status and condition.
3. **Add the tag's fake** to `tests/discovery/fakes.py`. It is a minimal object satisfying the modality's protocol and returning a small valid result. Reuse the modality's existing test fake where one exists; else write one, under 40 lines. Add a `SKIP_LIVE` entry only with a reason.
4. **Remove the tags from `PENDING`** in `tests/discovery/pending.py`.
5. **Fix docs:** correct any statement in `docs/MODALITIES.md` the descriptions showed to be wrong.

- [ ] **Step 1:** steps 1-3 for each tag; remove the tags from `PENDING`.
- [ ] **Step 2:** run `PYTHONPATH=$PWD/src pytest tests/discovery tests/modalities/<each tag dir> tests/core/test_discovery_meta.py tests/mcp -q`. Expected: pass.
- [ ] **Step 3:** commit with `docs(<group>): describe routes, examples and errors`.

| Task | Group | Tags |
|---|---|---|
| 6 | audio | audio/alignment, audio/classification, audio/embedding, audio/generation, audio/process, audio/quality, audio/speech, audio/transcription |
| 7 | text, chat, decision | chat/completion, decision/evaluation (fills the Task 5 doc models' descriptions), embedding/text, text/classification, text/rerank, text/summarization, text/translation |
| 8 | image I | image/animation, image/cv, image/embedding, image/generation, image/matting, image/ocr |
| 9 | image II and others | image/pixelart, image/segmentation, image/upscale, image/vectorization, 3d/generation, music/midi, video/generation |
| 10 | server | the 22 gateway routes in `SERVER_DISCOVERY` (`auth` set; admin ones document 401/403/503 `admin_disabled`), plus parameter and body descriptions in `admin/routes/*.py`, `observability/dashboard.py` and gateway's `/v1/models`, `/health` and `/v1/capabilities`; remove `"server"` from `PENDING`. Add `test_admin_routes_marked_auth` here. |

`chat/completion` note: the request model passes extra OpenAI fields through. Describe the declared fields only, and say in the route description that other OpenAI fields pass through to llama.cpp.

---

### Task 11: Make it required; docs

**Files:**
- Modify:
  - `src/muse/core/discovery.py`: `description`, `response_example` and `errors` become required route keys for modality DISCOVERY (validator) and for `SERVER_DISCOVERY`.
  - `tests/discovery/pending.py`: delete it and its imports.
  - `docs/AGENT_DISCOVERY.md`: the new keys, `server.errors`, `server.routes`, the conformance suites and how a new modality satisfies them.
  - `CLAUDE.md`: the "Agent discovery" section, in one or two lines.
  - The "Adding a new modality" steps in CLAUDE.md: add "describe every field; fill description/example/response_example/errors; add a fake in tests/discovery/fakes.py".
- Test: `tests/core/test_discovery.py`, with `test_route_without_metadata_is_rejected`.

- [ ] **Step 1: Write the failing test**: a route dict missing `errors` is now invalid.
- [ ] **Step 2: Run it and see it fail.**
- [ ] **Step 3: Implement it**; delete `PENDING`.
- [ ] **Step 4: Run** `python scripts/preflight.py`, then `pytest tests -q -m slow`. Expected: both green.
- [ ] **Step 5: Commit** with `feat(discovery): route metadata is required for every route`.

## After the plan

- With your OK, push and restart the server.
- Then spot-check `/openapi.json`, `/llms.txt` and `muse_describe` on the live server for size and readability. The unfiltered `/v1/capabilities` was 79 KB before this work; report the new size.
