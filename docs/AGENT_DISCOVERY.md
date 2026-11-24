# Agent discovery and model provenance

A muse server describes itself, from only its URL, so an AI agent (or a
pipeline an agent is writing) never has to learn muse from prose. Every
inference response also says which weights produced it.

Design: `docs/superpowers/specs/2026-10-21-agent-discovery-design.md`.

## Endpoints

All are public (no token), read-only, and never trigger a model load.

| Endpoint | Returns |
|---|---|
| `GET /` | Pointer document: links to `/v1/capabilities`, `/openapi.json`, `/llms.txt`, `/v1/models`. Not proxied to a worker. |
| `GET /v1/capabilities` | The main document: `server` (version, node, links, MCP install line), `modalities[]` (summary, `use_for`, `not_for`, routes with request/response content types, an example, the MCP tool name, file fields, and `model_optional`), and `models[]` (id, modality, license, `revision`, `components`, one-line `summary`, `strengths`, `limits`, `quality_tier` of best/good/draft, filtered capability flags, memory estimate, and live state: loaded, last cold-load seconds, queue depth, `unservable_reason`). Only enabled models are listed. |
| `GET /v1/capabilities?modality=audio/speech` | The same document narrowed to one modality. `?model=<id>` narrows to one model. An unknown value is a 404. |
| `GET /openapi.json` | Complete OpenAPI for every inference route (alignments, vectorize, systemone and the rest), generated from the same declarations. |
| `GET /llms.txt` | A short markdown page in the llmstxt.org shape, rendered from the capabilities document: an H1, a one-line summary, the workflow (pick a modality, read `?modality=` / `?model=`, call with `model`, expect a cold load when a model is not `loaded`), relative links (correct behind any proxy; only the `MUSE_SERVER=` install line uses the request's base URL), the MCP install line, each modality that has an enabled model (linked to its filtered document, with routes, `model optional` marks, and up to 8 models, loaded first then by tier, each on one line with its tier, live state and a summary cut to 120 characters, then "and K more"), and an `Optional` section naming the modalities with no model enabled here. |
| `GET /.well-known/muse.json` | Same pointer document as `/`, at a well-known path. |
| `GET /v1/models` | Unchanged OpenAI-shape list, plus a `revision` field per model. |

A route with `model_optional: true` (its package lists it in
`MODEL_OPTIONAL_PATHS`) accepts a request without `model`: the gateway uses
the first enabled model of that modality (pixelize, autotile, translate,
`/languages`, alignment, audio quality, vectorize). Other routes need `model`.

`quality_tier` is absolute within a modality across bundled and curated
models, so an agent can pick the best model for a task without a table of ids.
Model notes live next to each model (bundled script MANIFEST or curated
entry); a model without notes still appears, with only the facts.

## First three HTTP calls

```bash
# 1. What is this server?
curl -s http://host:8000/

# 2. What speech models and voices exist, and how do I call them?
curl -s 'http://host:8000/v1/capabilities?modality=audio/speech' | jq .

# 3. Call one; read the revision from the response headers.
curl -sD- -o out.wav http://host:8000/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"kokoro-82m","input":"hello","response_format":"wav"}' | grep -i x-muse
```

Write pipeline code against capability flags (`supports_*`, `supported_scales`
and so on) from step 2 rather than hardcoding model ids; the flags are the
stable contract and the ids are the choice.

## Provenance headers

Every inference response carries:

| Header | Meaning |
|---|---|
| `X-Muse-Model` | Catalog model id that served the request. |
| `X-Muse-Model-Revision` | 40-hex Hugging Face commit of the weights. Omitted when unknown. |
| `X-Muse-Model-Artifacts` | Short digest over all components (weights plus, for example, a LoRA base, or the pinned `code_revision` of a remote-code repo, labelled with `capabilities.code_repo` when the code lives in another repo) when a model has more than one. Omitted for other single-component models. A weightless model (no weights, such as `pixel-tools`) always sends it: the first 16 hex of `sha256("<script sha256>:<running muse version>")`, so it changes when either the script or muse changes. |

Chat completions also set `system_fingerprint` to `<model_id>@<revision[:12]>`
(omitted when the revision is unknown). The revision is recovered from the
HF snapshot path or the persisted pin, and is cached against the catalog
file's identity (device, inode, size, mtime, ctime). `GET /v1/capabilities?model=<id>` returns the same value as
`models[0].revision`, so a pipeline can check it before a batch.

Recording provenance in a pipeline: store `{model, revision, artifacts}` from
the three headers next to every generated artifact (the audio file, the
alignment, the image). Frozen records then say exactly which weights made
them, and a later run can compare the current revision to detect drift before
reusing a cached result. With MCP, the same triple is returned in each
tool result (below).

## First three MCP calls

1. Register the local MCP program (only the server URL is needed):

   ```bash
   claude mcp add muse -e MUSE_SERVER=http://host:8000 -- uvx --from "museq[mcp]>=0.70" muse mcp
   ```

   The capabilities document's `server.mcp.install` gives this line with the
   server's own minor version as the floor (unpinned for a dev build), so
   uvx does not reuse an older cached museq. At initialize the server sends
   `instructions` (`muse.mcp.server.INSTRUCTIONS`): the describe, pick a
   model, call, read files workflow; the path-input rules and the `_paths` /
   `_urls` list forms; that admin tools need the token except
   `muse_list_models` and `muse_search_models`; and that long jobs return a
   job_id to poll. An admin-only or inference-only filter sends just its
   half.

2. Call `muse_describe`: returns the capabilities document for the live
   server. Narrowed by `modality` or `model`, it comes back inline. Unfiltered
   and over 20,000 characters (a full server), the full document is written
   to a file and the result is an inline index: per modality its summary,
   tool names and model ids; per model its id, modality, license,
   quality_tier, summary and loaded state.
3. Call `muse_speak` with `model`, `input` and `out_path`. The result is a
   short text block, a `resource_link` per written file, and
   `structuredContent` `{model, revision, artifacts, files: [{path, sha256,
   mime, bytes, ...}], response}`, where `response` is the JSON body with
   media replaced by file paths. Pass `preview: true` on image tools to also
   get one inline 512 px PNG.

`muse mcp` is a thin local HTTP client. It reads `/openapi.json` and
`/v1/capabilities` (cached 30 s) and generates one tool per inference route,
so tools track the server: a new route or model appears without a client
release. It lists `muse_describe`, the generated tools, and the 11 admin
tools. If the server is unreachable, only `muse_describe` and the admin tools
are listed (never stale generated tools). The server advertises
`tools.listChanged` and notifies a session once when the tool set changes
(including the server coming up after the client connected), so clients
re-list. A `model` argument is a string whose description lists the serving
models, not an enum, so a model enabled mid-session is accepted; on a
`model_optional` route it is not required. Every tool has a `title` and
annotations. Generated inference tools are not read-only (any can write a
local file) and not destructive (a cold load may evict idle models, which is
not counted as destructive), and are open-world only when they take `*_url` /
`*_urls` inputs. `muse_describe` and the admin read tools are read-only;
`muse_remove_model`, `muse_disable_model` and `muse_pull_model` (a re-pull
replaces weights; a failed probe disables the model) are destructive;
`muse_search_models` and `muse_pull_model` are open-world (Hugging Face Hub).
The `model` argument has `minLength: 1`.
`muse mcp --http` serves Streamable HTTP at `/mcp` (stateless). Behaviour:

- Media outputs are written to files and returned as path, sha256, model and
  revision, never inline bytes. `out_path` names the file; otherwise a
  content-named file goes in `mcp.output_dir` (`MUSE_MCP_OUTPUT_DIR`,
  default `./muse-output`, relative to the MCP process cwd; when the cwd is
  `/`, as Claude Desktop launches it, `~/muse-output`). SVG from
  `muse_vectorize_image` is written as an `.svg` file too.
- File inputs take a local `<name>_path`. With
  `fetch.mcp_allowed_path_prefixes` (`MUSE_MCP_ALLOWED_PATH_PREFIXES`) empty,
  paths under the current directory or `mcp.output_dir` are allowed; a cwd of
  `/` is never allowed implicitly. A set value replaces the current-directory
  default with its colon-separated directories; `mcp.output_dir` stays
  allowed for reads and writes either way, so one tool's output file can be
  another tool's input (speak, then align). Multipart `*_url` inputs are fetched by the MCP process through muse's
  SSRF guard; JSON `*_url` inputs are passed to the server, whose SSRF guard
  fetches them. Base64 inputs are not supported.
- Speech with `timestamps: true` (Kokoro only) returns one timing per written
  word in `response.words`; `response.audio.b64_json` becomes the audio file
  path. Models without native timings refuse the request. `pronunciations`
  carries respellings or supported IPA separately from the written input.
- Speech headers surface as `response.voice`, `response.speed`,
  `response.audio_sha256`, and, when supplied, `response.pronunciations_digest`.
- `X-Alignment` from `audio/speech align=true` contains Soprano's paragraph
  timings and is surfaced as `response.alignment`.
- Tool descriptions come from the capabilities document, so they carry the
  model notes.
- A route's `DISCOVERY` entry may declare `mcp_defaults`, a mapping of
  default arguments for its generated tool. A user-supplied argument always
  wins. `muse_autotile` defaults `response_format` to `zip`: MCP saves each
  output under a content-hash name, which would break the Godot `.tres` /
  Tiled `.tsx` references to `<name>_blob47.png`, whereas the zip keeps the
  five bundle names intact.
- Pixel-art tools: `muse_pixelize_image` and `muse_autotile` (modality
  `image/pixelart`, see `docs/PIXELART.md`).

## Implementation notes

`modality_discovery()` adds `model_optional` to every route (true when the
package lists the path in `MODEL_OPTIONAL_PATHS`), returning deep copies so
the memoized declarations are never mutated. `server.mcp.install` pins `museq[mcp]>=<major>.<minor>`
for a release version and leaves dev or local builds unpinned, so uvx does
not reuse an older cached client.
