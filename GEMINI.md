# GEMINI.md

Concise engineering guide for Gemini / Antigravity agents working in this repository. For deeper architecture notes and per-modality contracts, see `CLAUDE.md`, `docs/MODALITIES.md`, and `docs/INTERNALS.md`.

## 1. Project Overview & Packaging

- **Identity & Packaging**: Python 3.10+ (`src/` layout). Distributed on PyPI as **`museq`** (`pyproject.toml` `[project].name = "museq"`), while the Python import path, CLI binary (`muse`), and repo stay **`muse`**.
- **Purpose**: Self-hostable, model-agnostic multi-modality AI server with OpenAI-compatible (plus Cohere, LibreTranslate, TypeSafe System One, and muse-native) HTTP endpoints.
- **Scale (live codebase)**: **25 modalities** under `src/muse/modalities/`, **30 bundled model scripts** under `src/muse/models/`, **12 admin endpoints**, and a dynamic **MCP server** (11 fixed admin tools + `muse_describe` + dynamic per-route inference tools generated from `/v1/capabilities` + `/openapi.json`).
- **Package Data**: `src/muse/curated.yaml`, `src/muse/chat_formats.yaml`, and bundled speech reference voices (`src/muse/modalities/audio_speech/data/*.{wav,json}`) are shipped as wheel package data and loaded via `importlib.resources`.
- **CLI Philosophy**: Deliberately admin-only (`serve`, `pull`, `search`, `models`, `config`, `doctor`, `storage`, `telemetry`, `federate`, `mcp`). Never add per-modality CLI verbs (`muse speak`, etc.); inference belongs on HTTP and MCP.

## 2. Repository Layout & Key Modules

```text
src/muse/
├── cli.py                   # Typer CLI wiring ONLY (no heavy logic or ML imports)
├── cli_impl/                # CLI implementations, supervisor, gateway, worker, director, telemetry
│   ├── supervisor.py        # `muse serve` process supervisor, worker lifecycle, auto-restart monitor
│   ├── gateway.py           # In-process FastAPI proxy, RequestBodyLimitMiddleware, aggregation
│   ├── agent_routes.py      # Agent discovery (`/llms.txt`, `/v1/capabilities`, merged `/openapi.json`)
│   ├── worker.py            # Hidden `muse _worker` single-venv FastAPI worker
│   ├── load_director.py     # Lazy-load orchestrator, live VRAM/RAM admission, LRU eviction
│   ├── idle_sweeper.py      # Background idle-timeout eviction thread
│   ├── queueing.py          # Event-loop ConcurrencyGate + CapacityNotifier
│   ├── telemetry.py         # Local `muse telemetry` CLI implementation
│   ├── federation.py        # `muse federate` coordinator ASGI app
│   ├── mcp_server.py        # `muse mcp` stdio / HTTP+SSE entrypoint
│   ├── resource_doctor.py   # `muse doctor resources [--repair]`
│   ├── storage.py           # `muse doctor storage`, `muse storage prune`, `muse storage dedupe`
│   └── runtime_state.py     # Public `/v1/models` loaded-state helper for CLI
├── core/                    # Modality-agnostic infrastructure
│   ├── config.py            # Single declarative SETTINGS registry (env > file > default)
│   ├── discovery.py         # Runtime scanning for modalities, model scripts, MODEL_OPTIONAL_PATHS
│   ├── catalog.py           # Facade over catalog_{store,locks,models,pull,remove} (`~/.muse/catalog.json`)
│   ├── registry.py          # ModalityRegistry keyed by (modality, model_id); no shared base class
│   ├── server.py            # FastAPI factory, `/health`, `/v1/models`, build_model_entry
│   ├── resolvers.py         # URI scheme dispatch (`hf://...`)
│   ├── resolvers_hf.py      # HuggingFace resolver delegating to per-modality `hf.py` plugins
│   ├── runtime_helpers.py   # Shared select_device, dtype_for_name, set_inference_mode, LoadTimer, resolve_model_source
│   ├── memory_probe.py      # Live pynvml (GPU) + psutil (CPU) memory probes
│   ├── securefs.py          # Owned-dir checks, no-follow opens, advisory locks, atomic fsync writes
│   ├── owned_process.py     # Isolated process groups, leader pinning, bounded TERM->KILL teardown
│   ├── venv.py              # Per-model venv creation (`~/.muse/venvs/<id>/`) and rollback transactions
│   ├── errors.py            # ModelNotFoundError + OpenAI-shaped error helpers
│   ├── storage.py           # Owned-storage accounting, safe pruning, venv hardlink deduplication
│   ├── resource_registry.py # Owned process/resource tracking for clean teardown & doctor
│   └── image_preprocessing.py # Shared 4-tier image processor selection helper
├── modalities/              # 25 self-contained modality packages + shared `_common/` helpers
├── models/<model_id>.py     # 30 bundled drop-in model scripts (MANIFEST + Model)
├── admin/                   # `/v1/admin/*` routes, shared bearer auth, JobStore, operations, AdminClient
├── observability/           # TelemetryStore (SQLite WAL), events, recorder, traces, sampler, logs, dashboard, `_lifecycle`
├── federation/              # Static node membership, NodeRegistry polling, model-locality router
└── mcp/                     # Dynamic MCPServer (`muse.mcp.generated` + `tools/admin.py`), MuseClient, binary_io
scripts/                     # preflight.py, smoke_fresh_venv.py, validate_installed_distribution.py, bench/
tests/                       # Mirrors src/muse/ + integration/ + test_cli.py + audit/meta tests
docs/                        # MODALITIES.md, INTERNALS.md, CONFIG.md, TELEMETRY.md, AGENT_DISCOVERY.md, DECISIONS.md, PIXELART.md, STORAGE.md, MODEL_SCRIPTS.md, HF_PLUGINS.md, RESOLVERS.md, CHAT_COMPLETION.md
examples/                    # Runnable HTTP/SDK clients
```

### The 25 Modalities (`src/muse/modalities/`)

Python directory names use underscores; runtime `MODALITY` tags are MIME-style (`3d/generation` lives in `model_3d_generation/`):
`audio/alignment`, `audio/classification`, `audio/embedding`, `audio/generation` (`/v1/audio/music` + `/v1/audio/sfx`), `audio/quality`, `audio/speech` (`/v1/audio/speech` + saved voices `/v1/audio/speech/voices`), `audio/transcription`, `chat/completion` (text LLMs + VLMs), `decision/evaluation` (`/v1/systemone`), `embedding/text`, `image/animation`, `image/cv` (`/v1/images/{depth,keypoints,detect}`), `image/embedding`, `image/generation` (`/v1/images/{generations,edits,variations}`), `image/ocr`, `image/pixelart` (`/v1/images/{pixelize,autotile}`), `image/segmentation`, `image/upscale`, `image/vectorization`, `3d/generation` (`/v1/3d/{from-image,generations}`), `text/classification` (`/v1/moderations` + `/v1/text/classifications`), `text/rerank`, `text/summarization`, `text/translation` (`/v1/translate`, `/translate`, `GET /languages`), `video/generation`.

## 3. Architecture Invariants

- **Three runtime-discovered plugin surfaces** (no hardcoded allowlist or registration list):
  1. `src/muse/modalities/<name>/` (or `$MUSE_MODALITIES_DIR`) exporting `MODALITY` and `build_router`.
  2. `src/muse/models/<id>.py` (or `~/.muse/models/`, `$MUSE_MODELS_DIR`) exporting `MANIFEST` and `Model` (including weightless scripts with `MANIFEST["weights"] = "none"`, e.g. `pixel-tools`).
  3. `src/muse/core/resolvers*.py` + per-modality `hf.py` (`HF_PLUGIN`) for URI pulls (`hf://...`).
  - Collision precedence: **bundled scripts > resolver-pulled persisted manifests**.
- **No shared supertype across modalities**: `ModalityRegistry` stores backends as `Any`; each modality defines its own `Protocol` and result dataclasses structurally. Never introduce a cross-modality `GenerationModel` ABC. Shared per-modality route/codec/upload helpers live in `src/muse/modalities/_common/`.
- **Strict import hygiene (lazy ML & server imports)**:
  - `src/muse/__init__.py`, `src/muse/cli.py`, and `src/muse/core/config.py` must stay free of heavy ML libs (`torch`, `transformers`, `diffusers`, `llama_cpp`, `sentence_transformers`) so `muse --help` and `muse pull` run with core deps only.
  - `src/muse/observability/__init__.py` eagerly exports only stdlib-backed symbols and lazily loads `dashboard` / `dashboard_auth` via PEP 562 `__getattr__` so importing `recorder` never pulls in `fastapi` (enforced by `tests/observability/test_public_api.py`).
  - Runtimes and bundled model scripts defer heavy imports via module-level `None` sentinels + `_ensure_deps()`, and MUST reuse `muse.core.runtime_helpers` (`select_device`, `dtype_for_name`, `set_inference_mode`, `LoadTimer`, `resolve_model_source`) — enforced via AST scan in `tests/core/test_runtime_helpers_meta.py`.
- **`MODEL_OPTIONAL_PATHS`**: Modalities whose wire clients may omit `model` export a `MODEL_OPTIONAL_PATHS` tuple in `__init__.py` (currently `audio_alignment`, `audio_quality`, `image_pixelart`, `image_vectorization`, and `text_translation`). `discover_modalities` / `model_optional_paths()` aggregates them; the gateway resolves the first enabled catalog model of that modality (or 503 `no_default_model`) instead of 400 `model_required`.
- **Capability & device placement precedence**:
  - **Device** (resolved via `effective_device` / `load_backend`): `catalog.device_override` (`muse models set-device`) > `manifest.capabilities.device` (if `!= "auto"`) > `muse serve --device` / `MUSE_DEVICE` > `"auto"` (`cuda` > `mps` > `cpu`). Default for CUDA-safe models is `"auto"` (only ONNX `supertonic-3` pins `"cpu"`).
  - **Consistent pool resolution**: `auto` resolves to GPU VRAM iff live GPU memory is present, consistently across `LoadDirector`, `supervisor._servability_reason`, and `/v1/admin/memory`. (`mps` pools against host RAM but records probe measurements under `mps`.)
  - **GGUF `n_gpu_layers`**: `catalog.gpu_layers_override` (`muse models set-gpu-layers <id> N|--all|--clear`) > `capabilities.n_gpu_layers` > `-1`.
  - **GGUF `chat_format`**: `capabilities.chat_format` > `src/muse/chat_formats.yaml` (first match wins) > llama-cpp GGUF metadata.
  - At `load_backend`, `manifest.capabilities` is merged into runtime constructor kwargs (caller kwargs win), and `model_id` is injected for generic runtimes.
- **Process model (Supervisor + Gateway + Per-Model Workers)**:
  - `muse serve` runs the supervisor + in-process FastAPI gateway (port 8000) and spawns isolated subprocess workers (`muse _worker` on internal ports) inside each model's venv (`~/.muse/venvs/<id>/`) using `muse.core.owned_process`.
  - **Catalog `enabled` vs runtime `loaded` are decoupled**: `enabled: true` means eligible to lazy-load; `loaded` means resident on a worker. Lazy-load/eviction call `load_model_into_worker` / `unload_model_from_worker` (no catalog flip); operator `enable`/`disable` flip the catalog.
  - **Auto-restart monitor** polls loaded workers every 5s; after 3 consecutive health failures or exit, restarts with exponential backoff (1s..30s) and marks dead after 10 attempts (never reviving workers retired by disable, eviction, or failed load).
- **Lazy load, LRU/idle eviction, and request queueing**:
  - Boot is instant with zero workers; `/v1/models` and `/health` union loaded workers with enabled-but-unloaded catalog entries.
  - `LoadDirector.acquire()` runs **off the event loop** (`asyncio.to_thread` + `asyncio.shield`) with cancellation-safe refcount release, same-model `cold_load_gates` coalescing (no thread-pool exhaustion), and same-pool in-flight memory reservations.
  - **3-tier memory sizing ladder**: (1) `capabilities.memory_gb` or `measurements.<device>.peak_bytes`, (2) on-disk weights fallback (single `gguf_file` for GGUF; dtype/format deduped weights for diffusers/transformers), (3) none -> `unservable_reason` (re-validated live on request). Observed cold-load peaks self-heal upward when a load runs solo.
  - **Headroom & budgets**: `server.gpu_headroom_gb` (default `1.0`), `server.cpu_headroom_gb` (default `2.0`), optional `server.gpu_budget_gb` / `server.cpu_budget_gb`.
  - **Idle eviction**: `IdleSweeper` (interval `server.idle_sweep_interval_seconds`, default `30s`) evicts refcount-0 models past `capabilities.idle_timeout_seconds` else `server.idle_timeout_seconds` (default `600s`; `<= 0` disables).
  - **Request queueing** (`src/muse/cli_impl/queueing.py`): event-loop `ConcurrencyGate` (`capabilities.max_concurrency` else `server.default_max_concurrency`, default `0` = unlimited) + `CapacityNotifier` retry for transient capacity 503s. Shared deadline `server.queue_timeout_seconds` (default `300s`, `0` disables); depth cap `server.max_queue_depth` (default `256`, `0` = unbounded -> 503 `queue_full`).
- **Central configuration (`src/muse/core/config.py`)**:
  - Single `SETTINGS` registry; precedence **CLI override > `MUSE_*` env (read live) > `~/.muse/config.yaml` (cached) > default**.
  - Bootstrap keys `paths.catalog_dir` (`MUSE_CATALOG_DIR`) and `paths.config_file` (`MUSE_CONFIG`) resolve from env+default only.
  - `config.get()` is lenient (warns + falls back to default); `muse config set` is strict. Server-global settings live in `config.yaml`; per-model state (`enabled`, `device_override`, `gpu_layers_override`, `measurements`) lives in `catalog.json`.

## 4. Admin API, Observability, Federation, Agent Discovery & MCP

- **Admin REST API (`src/muse/admin/`)**: 12 endpoints under `/v1/admin/*` — `POST /models/{id}/{enable,disable,warmup,probe,pull}`, `DELETE /models/{id}?purge=bool`, `GET /models/{id}/status`, `GET /workers`, `POST /workers/{port}/restart`, `GET /memory`, `GET /jobs`, `GET /jobs/{job_id}`. Closed-by-default (`503 admin_disabled`) until `MUSE_ADMIN_TOKEN` (`admin.token`) is set; uses shared `http_bearer_check` (`Authorization: Bearer <token>`, case-insensitive scheme, constant-time compare). Async ops (`enable`, `probe`, `pull`) return `202 + job_id` via `JobStore` (10-minute retention). `install_admin_error_handler` unwraps OpenAI-shaped auth errors. **Never echo the admin token** in errors, logs, or jobs.
- **Observability (`src/muse/observability/`, `docs/TELEMETRY.md`)**:
  - **Store**: `<catalog_dir>/telemetry.db` (SQLite WAL mode, safe to query while `muse serve` runs, auto-migrates new nullable columns with SQLite type affinities, default 7-day retention). Single sparse `events` table (`EVENT_COLUMNS` in `events.py`, strict validation in `event_to_row`) covering `request`, `model_load`, `model_evict`, and `sample`. Exposes `last_cold_load_seconds()`, `request_report()`, `recent_requests()`, `series()`, `samples()`, `status()`, and `export_events()`.
  - **Fire-and-forget recorder**: `record(type, **fields)` enqueues onto bounded `TelemetryRecorder` (queue 10,000, 0.5s batch flush, increments `dropped` instead of blocking; `_NoopRecorder` when `telemetry.enabled=false`). Call sites wrap in `try/except`.
  - **Measured request traces**: `traces.py` carries `RequestTrace` via a `ContextVar` (propagated across `asyncio.to_thread`) so `LoadDirector` annotates exact `cold`, `load_ms`, and `evicted_models` while the gateway records `request_id`, `latency_ms`, `queued_ms`, `forward_ms`, `stream`, and `peak_vram_gb`.
  - **Resource sampler**: `Sampler` + `VramTracker` (`sampler.py`) record device-wide VRAM (`gpu_used_gb`, `free_vram_gb` — `None`, never `0`, on CPU-only/no-NVML hosts) and RAM, speeding up from `telemetry.sample_interval_seconds` (`10.0s`) to `telemetry.trace_sample_interval_seconds` (`0.25s`) on active requests with bounded wake throttling.
  - **Dashboard & HTTP API (`dashboard.py`)**: `GET /dashboard` (un-gated HTML shell with automatic SSE log-ticket reconnect), gated `GET /v1/telemetry/{summary,series,report,traces}`, `POST /v1/telemetry/logs-ticket`, and `GET /v1/telemetry/logs/{model_id}` (SSE via `LogHub.subscribe_with_snapshot`, authenticated by short-lived `?ticket=` or `Authorization: Bearer`, never `?access_token=`). Reuses `admin.token`; gated by `telemetry.require_auth` (`MUSE_TELEMETRY_REQUIRE_AUTH`, default `true` — only set `false` on trusted loopback dev hosts).
  - **CLI (`src/muse/cli_impl/telemetry.py`)**: `muse telemetry {status,summary,traces,vram,export,prune}`. Summary rows distinguish `measured`, `estimated`, and `mixed` bases; `export` validates upfront, writes CSV headers even when empty, and refuses overwrite without `--force`.
- **Agent Discovery (`src/muse/cli_impl/agent_routes.py`, `docs/AGENT_DISCOVERY.md`)**: Exposes `GET /llms.txt`, `GET /v1/capabilities` (including `last_cold_load_seconds` and provenance/artifact digests), and merged `/openapi.json` across discovered modalities.
- **Federation (`src/muse/federation/`, `src/muse/cli_impl/federation.py`)**: `muse federate --port 8100 --node <url> [--config <yaml>]` fronts unmodified `muse serve` nodes. `NodeRegistry` polls `/v1/models`, `/health`, and (if a per-node token is configured) `/v1/telemetry/summary` every `federation.refresh_interval_seconds` (`3.0s`). `select_node` routes by model-locality (prefer already `loaded`, tie-break lowest `in_flight` requests, else round-robin) with one connect/timeout failover retry. Exposes aggregated `GET /v1/models`, `GET /health`, and `GET /v1/federation/nodes`.
- **MCP Server (`src/muse/mcp/`, `src/muse/cli_impl/mcp_server.py`)**: `muse mcp` (stdio default; `--http --port 8088` for Starlette `/mcp`; `--filter all|admin|inference`; lightweight install via `museq[mcp]`). Dynamically generates inference tools from the live server's `/v1/capabilities` + `/openapi.json` alongside `muse_describe` and 11 fixed admin tools (requiring `MUSE_ADMIN_TOKEN`), plus `muse://capabilities` and `muse://models/{id}` resources. Binary inputs accept mutually exclusive `<name>_b64`, `<name>_url`, or `<name>_path` (guarded by `fetch.mcp_allowed_path_prefixes`).

## 5. Build, Preflight & Test Commands

```bash
# Editable install matching CI's CPU stack
pip install -e ".[dev,server,audio,images,embeddings]" \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Preflight: verifies all fast-lane deps + version specifiers, then runs fast lane
python scripts/preflight.py
python scripts/preflight.py --check-only
python scripts/preflight.py -- -k telemetry          # forward args to pytest

# Fast lane (CI default; excludes real-subprocess @pytest.mark.slow tests)
pytest -m "not slow" -q

# Full lane (includes slow e2e supervisor tests; run before supervisor/runtime changes)
pytest tests/ -q

# Targeted tests
pytest tests/modalities/audio_speech/ -q
pytest tests/observability/ -q
pytest tests/models/test_kokoro_82m.py -q

# Fresh-venv smoke check (mirrors .github/workflows/fresh-venv-smoke.yml)
python scripts/smoke_fresh_venv.py --model_id kokoro-82m

# Opt-in live integration tests (auto-skip if unreachable or model absent)
MUSE_REMOTE_SERVER=http://127.0.0.1:8000 pytest tests/integration/ -q

# Common local server / diagnostics commands
muse serve --host 127.0.0.1 --port 8000
muse doctor resources [--repair]
muse doctor storage [--json]
muse storage prune [--dry-run] [--include-unreferenced]
muse storage dedupe [--dry-run]
muse telemetry status && muse telemetry summary --since 24h
```

## 6. Coding, Testing & Commit Conventions

- **Python style**: 4-space indent, type annotations on public APIs, concise docstrings for non-obvious contracts, `snake_case` modules/functions, `PascalCase` classes, `UPPER_CASE` constants. No mandatory formatter; match neighboring code.
- **CLI architecture**: Put Typer signatures in `src/muse/cli.py` and implementation in `src/muse/cli_impl/<cmd>.py`. Signal CLI errors with `raise typer.Exit(code)` (propagated by `main()` via `standalone_mode=False`). Respect TTY vs piped output (`cli_impl/console.py`) and support `--json` for machine-readable commands.
- **HTTP & streaming contracts**:
  - Raise `ModelNotFoundError` (`muse.core.errors`) rather than raw `HTTPException(detail=...)` so 404/422/admin errors keep the OpenAI envelope `{"error": {"code", "message", "type"}}`.
  - Stream with a producer thread + `asyncio.Queue` (`muse.modalities._common.streaming`) — never materialize `list(generator)` before yielding.
  - Audio protocol boundary: `float32` in `[-1, 1]` (codecs convert to int16 PCM via `* 32768` + `np.clip`). Images: `Any` (codecs normalize PIL/numpy/torch via `muse.modalities._common.images`).
  - Chat responses must return the Muse catalog `model_id`, never an internal GGUF file path.
- **Testing**:
  - Mirror source layout (`tests/core/`, `tests/modalities/<name>/`, `tests/models/`, `tests/cli_impl/`, `tests/observability/`, `tests/admin/`, `tests/federation/`, `tests/mcp/`).
  - Use `FakeModel` classes and fresh `ModalityRegistry()` instances in unit tests (no real weights; mock heavy libs at their module-level import sentinels).
  - Mark subprocess end-to-end tests with `@pytest.mark.slow`. In `tests/integration/`, prefix hard invariants `test_protocol_*` and model-behavior watchdogs `test_observe_*`.
  - Every bug fix must include a regression test.
- **Commits & PRs**: Conventional Commits (`feat(observability): ...`, `fix(core): ...`), imperative scoped subject, cohesive changes. Never commit model weights, secrets, local venvs, or caches.

## 7. Quick Checklists

### Adding a New Model
1. **Curated alias (preferred if HF resolver supports the family)**: Add an entry to `src/muse/curated.yaml` pointing to `hf://...` (or a bundled id). Verify with `muse models list` / `muse models info <id>`.
2. **Resolver URI only**: No code changes — run `muse pull hf://<org>/<repo>[@variant]`. For GGUFs needing a specific llama-cpp template, add a pattern to `src/muse/chat_formats.yaml` (most-specific first).
3. **Custom model script** (`src/muse/models/<id>.py` for bundled, or `~/.muse/models/`):
   - Define `MANIFEST` (`model_id`, `modality`, `hf_repo` or `weights: "none"`, `description`, `license`, `pip_extras`, `system_packages`, `capabilities`) and class `Model`.
   - Constructor accepts `hf_repo=`, `local_dir=`, `device=`, `**_`, prefers `local_dir`, defers ML imports via `_ensure_deps()`, and uses `muse.core.runtime_helpers`.
   - Ensure `pip_extras` covers all direct and load-time imports (`tests/models/test_pip_extras_audit.py`; lightweight CPU models can join `.github/workflows/fresh-venv-smoke.yml`).
   - Add mocked unit tests in `tests/models/test_<id>.py`. No catalog/registry/worker edits needed.

### Adding a New Modality (Rare)
1. Create `src/muse/modalities/<mime_name>/` (underscores; MIME tag has `/`).
2. Implement `protocol.py`, `routes.py` (`build_router(registry) -> APIRouter`), `codec.py`, `client.py`, and optional `runtimes/`, reusing `src/muse/modalities/_common/`.
3. Export `MODALITY`, `build_router`, `PROBE_DEFAULTS`, and (if clients may omit `model`) `MODEL_OPTIONAL_PATHS` from `__init__.py`.
4. Optionally add `hf.py` exporting `HF_PLUGIN` (loaded via single-file import — **no relative imports**; see `docs/HF_PLUGINS.md`).
5. Use `muse.core.runtime_helpers` inside runtimes (checked by `tests/core/test_runtime_helpers_meta.py`).
6. Add route/codec/plugin/client tests in `tests/modalities/<mime_name>/` (and bundled model test if adding one). No edits to `worker.py`, `gateway.py`, `catalog.py`, `registry.py`, `server.py`, or `resolvers_hf.py` are needed.
