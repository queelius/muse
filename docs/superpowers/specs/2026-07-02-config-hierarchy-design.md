# Config hierarchy + persistent config file (v0.52.0)

**Goal:** Replace ~28 ad-hoc `os.environ.get("MUSE_*", default)` reads with one
declarative settings registry, a persistent `~/.muse/config.yaml`, a standard
precedence chain, and a `muse config` CLI group.

## Problem

muse has no server config file. Every setting is a `MUSE_*` env var read
inline at its call site with a hardcoded default, scattered across ~15 files.
There is no way to (a) persist server config, (b) see the effective config, or
(c) discover what knobs exist. Operators must know the env-var names by reading
source.

## Design

### Single source of truth: the settings registry

`muse.core.config` declares every setting ONCE:

```python
@dataclass(frozen=True)
class Setting:
    key: str          # dotted config path, e.g. "server.idle_timeout_seconds"
    env: str          # env var, e.g. "MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS"
    type: str         # "int" | "float" | "str" | "bool" | "opt_int" | "opt_float" | "opt_str"
    default: Any      # built-in default (may be None for optionals)
    group: str        # "server" | "paths" | "limits" | "fetch" | ...
    help: str         # one-line description (for `config generate` comments)

SETTINGS: list[Setting] = [ ... all ~28 ... ]
```

Env reads, file reads, CLI overrides, `config show`, and `config generate` all
DERIVE from this list. No parallel lookup table anywhere (this is the core
principle: derive behavior from structure).

### Resolution precedence (standard hierarchy)

`Config.get(key)` walks, first-wins:

1. **CLI override** (a dict the CLI passes in for `--flag`s on a command).
2. **Env var** (`os.environ[setting.env]`).
3. **Config file** (`~/.muse/config.yaml`, nested by group -> key-leaf).
4. **Built-in default** (`setting.default`).

Each source's raw value is coerced via `setting.type`. `opt_*` types treat
empty string / missing as None. Numeric `<= 0` handling stays where the
consumer already defines it (e.g. idle-timeout <= 0 means off) -- the registry
only parses; semantics stay at the call site.

**Coercion policy: lenient at runtime, strict at write.** `config.get(key)`
NEVER raises on an un-coercible value: it logs one warning naming the env var /
key, the bad value, and the fallback, then returns the built-in default. This
exactly preserves what the existing per-request route caps do today (they parse
`raw`, log on failure, fall back to the hardcoded default) and is the safe
server behavior -- a typo in one env var must not 500 every request. It is
also strictly more robust than the import-time constant sites
(`_MAX = int(os.environ.get(...))`), which currently CRASH the worker at import
on a bad value; after migration they warn + use the default. Valid values
behave identically at every site; only the invalid-input path is unified (all
warn + default). By contrast `muse config set <key> <value>` and `config get
<key>` on an explicit bad value ARE strict: they validate against the registry
and refuse / exit nonzero with a clear message -- that is the interactive path
where a hard error is the right feedback.

### Config file

- Location: `<catalog_dir>/config.yaml` where catalog_dir honors the existing
  `MUSE_CATALOG_DIR` (default `~/.muse`). Overridable via `MUSE_CONFIG` (full
  path) for an explicit file.
- Format: yaml, nested by group:
  ```yaml
  server:
    idle_timeout_seconds: 600
    idle_sweep_interval_seconds: 30
    gpu_headroom_gb: 1.0
    admin_token: null
  limits:
    image_input_max_bytes: 10485760
    rerank_max_documents: 100
  fetch:
    allow_private: false
  ```
- Missing file = all-defaults (no error). Unknown keys in the file = warn + ignore.

### Defaults changed by this work

- `server.idle_timeout_seconds` default flips **None -> 600** (10 min). This is
  the "sensible default" -- omitting the env/file now gives a 10-minute global
  idle timeout (bundled models included). Off is still reachable: set 0.

### CLI: `muse config` group (mirrors `muse models`)

- `muse config generate [--force]` -- write `config.yaml` from the registry:
  every setting with its default value, a `# <help>` comment, and its env-var
  name. Refuses to overwrite without `--force`.
- `muse config show [--json]` -- table of every setting: effective value AND
  source (default / file / env / cli). `--json` for machine output.
- `muse config path` -- print the resolved config-file path.
- `muse config get <key>` -- print one effective value.
- `muse config set <key> <value>` -- write one value into the file (create if
  missing; validate against the registry; coerce/validate the value).

### Migration (scope A: all settings)

Every `os.environ.get("MUSE_*", ...)` site routes through `config.get("...")`.
A meta-test AST-scans `src/muse/` for `os.environ.get("MUSE_` (and `getenv`)
and fails if any remain outside `muse.core.config` (mirrors the existing
`runtime_helpers` meta-test).

**Read-timing is preserved per site.** Some caps are import-time module
constants (`_MAX_BATCH = int(os.environ.get(...))`); others are per-request
functions (`_max_batch()`), which several modalities document as "read
per-request so changes take effect without a restart." The migration swaps the
SOURCE (`os.environ.get` -> `config.get`) but keeps each site's timing: an
import-time constant stays a module-level `config.get(...)`; a per-request
function stays a function calling `config.get(...)`. `config.get` reads env
LIVE on every call (env is fixed per-process, but tests monkeypatch it and the
per-request caps depend on live reads); the config FILE is parsed once and
cached (no live file reload, per Out of Scope).

**Bootstrap ordering.** `config.py` resolves its own config-file path from
`MUSE_CONFIG` (env) and `catalog_dir` (from `MUSE_CATALOG_DIR` env or
`~/.muse`) using env+default ONLY, via a small standalone `_catalog_dir()`
that does NOT import `catalog.py` (avoids an import cycle: `catalog.py` may
import `config`, never the reverse). `paths.catalog_dir` is still a registry
row for `config show`, but the file can never override the path that locates
the file.

**Client vs server settings.** `MUSE_SERVER` / `MUSE_ADMIN_TOKEN` are read in
client + CLI code that may run on a different host than the server. They route
through the registry too (`client.server_url`, `admin.token`), so a user who
drops a `config.yaml` on the client box gets the same resolution; explicit
constructor args still win (each client does `arg or config.get(...)`).

### Doc drift fixed by this work

`MUSE_GPU_BUDGET_GB`, `MUSE_CPU_BUDGET_GB`, `MUSE_GPU_HEADROOM_GB`,
`MUSE_CPU_HEADROOM_GB` are documented in CLAUDE.md as active env knobs but are
**never read from the environment** -- `LoadDirector.__init__` takes them as
params (`gpu_budget_gb=None`, `cpu_budget_gb=None`, `gpu_headroom_gb=1.0`,
`cpu_headroom_gb=2.0`) and `supervisor.py`'s sole construction site passes
none of them. This work makes the four real: the supervisor reads them via
`config.get` and passes them into `LoadDirector`. Defaults match today's
hardcoded values (budgets None, headroom 1.0/2.0), so a deployment that sets
nothing sees identical behavior -- the knobs simply start working.

### Authoritative settings inventory

Group `server` (read at supervisor/worker start):

| key | env | type | default | site |
|---|---|---|---|---|
| server.idle_sweep_interval_seconds | MUSE_IDLE_SWEEP_INTERVAL_SECONDS | float | 30.0 | supervisor.py:1015 |
| server.idle_timeout_seconds | MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS | opt_float | **600.0** | supervisor.py:1023 |
| server.shutdown_grace_seconds | MUSE_SHUTDOWN_GRACE_SECONDS | opt_float | None | serve_util.py:43 |
| server.gpu_budget_gb | MUSE_GPU_BUDGET_GB | opt_float | None | (wire into LoadDirector) |
| server.cpu_budget_gb | MUSE_CPU_BUDGET_GB | opt_float | None | (wire into LoadDirector) |
| server.gpu_headroom_gb | MUSE_GPU_HEADROOM_GB | float | 1.0 | (wire into LoadDirector) |
| server.cpu_headroom_gb | MUSE_CPU_HEADROOM_GB | float | 2.0 | (wire into LoadDirector) |
| server.device | MUSE_DEVICE | str | auto | server | `muse serve --device` default (new env) |

`server.device` is new: the `muse serve --device` default was a CLI-only
flag (no env). Its typer default flips from `"auto"` to `None` ("unset ->
consult config"), and the resolved device becomes
`config.get("server.device", override=<--device flag or None>)`. The registry
default `"auto"` preserves current behavior when nothing is set; an explicit
`--device cuda` still wins (it is the override arg).

Group `admin`:

| key | env | type | default | site |
|---|---|---|---|---|
| admin.token | MUSE_ADMIN_TOKEN | opt_str | None | cli.py, admin/client.py, mcp/*, cli_impl/mcp_server.py |

Group `client`:

| key | env | type | default | site |
|---|---|---|---|---|
| client.server_url | MUSE_SERVER | str | http://localhost:8000 | ~13 client.py + cli.py:304 + runtime_state.py:34 |

Group `paths` (bootstrap; env+default for catalog_dir/config path):

| key | env | type | default | site |
|---|---|---|---|---|
| paths.catalog_dir | MUSE_CATALOG_DIR | str | ~/.muse | catalog.py:398 |
| paths.home | MUSE_HOME | str | ~/.muse | audio_speech/backends/base.py:14 |
| paths.models_dir | MUSE_MODELS_DIR | opt_str | None | catalog.py:129 |
| paths.modalities_dir | MUSE_MODALITIES_DIR | opt_str | None | worker.py:37, discovery.py:129/391 |
| paths.config_file | MUSE_CONFIG | opt_str | None (-> catalog_dir/config.yaml) | (new; config.py self) |

Group `fetch`:

| key | env | type | default | site |
|---|---|---|---|---|
| fetch.allow_private | MUSE_ALLOW_PRIVATE_FETCH | bool | False | net_fetch.py:73 |
| fetch.mcp_allowed_path_prefixes | MUSE_MCP_ALLOWED_PATH_PREFIXES | str | "" | mcp/binary_io.py:117 |

Group `limits` (per-modality request caps; keep each site's read timing):

| key | env | type | default | site |
|---|---|---|---|---|
| limits.image_input_max_bytes | MUSE_IMAGE_INPUT_MAX_BYTES | opt_int | 10485760 | image_generation/image_input.py:49 |
| limits.audio_cls_max_bytes | MUSE_AUDIO_CLS_MAX_BYTES | opt_int | 52428800 | audio_classification/routes.py:41 |
| limits.audio_quality_max_bytes | MUSE_AUDIO_QUALITY_MAX_BYTES | opt_int | 52428800 | audio_quality/routes.py:24 |
| limits.audio_quality_max_duration_seconds | MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS | opt_float | 600.0 | audio_quality/routes.py:32 |
| limits.audio_embeddings_max_bytes | MUSE_AUDIO_EMBEDDINGS_MAX_BYTES | opt_int | 52428800 | audio_embedding/routes.py:49 |
| limits.asr_max_mb | MUSE_ASR_MAX_MB | int | 100 | audio_transcription/routes.py:48 |
| limits.embeddings_max_batch | MUSE_EMBEDDINGS_MAX_BATCH | int | 2048 | embedding_text/routes.py:46 |
| limits.embeddings_max_chars_per_item | MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM | int | 100000 | embedding_text/routes.py:48 |
| limits.image_embeddings_max_batch | MUSE_IMAGE_EMBEDDINGS_MAX_BATCH | int | 64 | image_embedding/routes.py:49 |
| limits.segmentation_max_input_side | MUSE_SEGMENTATION_MAX_INPUT_SIDE | int | 2048 | image_segmentation/routes.py:65 |
| limits.upscale_max_input_side | MUSE_UPSCALE_MAX_INPUT_SIDE | int | 1024 | image_upscale/routes.py:40 |
| limits.model_3d_input_max_bytes | MUSE_3D_INPUT_MAX_BYTES | opt_int | 20971520 | model_3d_generation/routes.py:58 |
| limits.moderations_max_batch | MUSE_MODERATIONS_MAX_BATCH | int | 1024 | text_classification/routes.py:68 |
| limits.moderations_max_chars_per_item | MUSE_MODERATIONS_MAX_CHARS_PER_ITEM | int | 100000 | text_classification/routes.py:76 |
| limits.classifications_max_labels | MUSE_CLASSIFICATIONS_MAX_LABELS | int | 200 | text_classification/routes.py:84 |
| limits.rerank_max_documents | MUSE_RERANK_MAX_DOCUMENTS | int | 1000 | text_rerank/routes.py:40 |
| limits.rerank_max_query_chars | MUSE_RERANK_MAX_QUERY_CHARS | int | 4000 | text_rerank/routes.py:41 |
| limits.rerank_max_doc_chars | MUSE_RERANK_MAX_DOC_CHARS | int | 100000 | text_rerank/routes.py:42 |
| limits.summarize_max_text_chars | MUSE_SUMMARIZE_MAX_TEXT_CHARS | int | 100000 | text_summarization/routes.py:42 |
| limits.video_max_frames_b64 | MUSE_VIDEO_MAX_FRAMES_B64 | int | 240 | video_generation/routes.py:43 |

35 settings total. Confirm each default against the cited line during
migration (the table is the intent; the code is the ground truth for any
default that has drifted).

## Acceptance

- One registry; a meta-test proves no stray `os.environ.get("MUSE_` remains
  outside `muse.core.config`.
- `Config.get` precedence: override > env > file > default, unit-tested per
  source and for each type. Un-coercible value -> warn + default (lenient),
  never raises; `config set`/`config get` on an explicit bad value is strict.
- `muse config generate/show/get/set/path` work; `show` reports the right
  source per setting.
- `server.idle_timeout_seconds` defaults to 600; omitting all sources yields a
  10-min idle timeout end-to-end.
- The four budget/headroom knobs become live: `supervisor` reads them via
  `config.get` and passes them into `LoadDirector`; defaults match today's
  hardcoded values so a deployment setting nothing is unchanged.
- Full fast lane green; for VALID values, behavior is unchanged for every
  existing env var (same name, same default, same parse) except the
  idle-timeout default flip. INVALID values are uniformly warn + default (some
  import-time sites previously crashed; this is a deliberate robustness gain).

## Out of scope

- Live config reload (config is read at process start / per-request as today).
- A web UI for config.
- Secrets management beyond the existing admin-token handling.
