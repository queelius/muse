# Muse comprehensive code review, synthesis backlog

_Date: 2026-07-03. Source: parallel per-subsystem reviewers + adversarial verifiers. Refuted findings dropped; the 50 confirmed/unclear findings below are deduplicated and ranked._

## Summary

- **Total findings:** 50 (18 important, 32 minor; no critical).
- **Quick wins:** 28 (8 important, 20 minor), S-effort, clear correctness / consistency / dead-code fixes.
- **Later:** 22 (10 important, 12 minor), design work, larger refactors, concurrency-subtle, test-gaps.
- **Federation-relevant:** 22 findings flagged. The federation seams (request routing, sizing/placement, budget fallback, MCP transport auth) carry a cluster of important issues that should be closed before building peer placement on top.

### Counts by subsystem

| Subsystem | Findings | Notable |
|---|---|---|
| core/catalog | 4 | curated-overlay split-brain (get_manifest), discover_models on hot path |
| core/net_fetch | 3 | streamed-response close, to_thread pool exhaustion, SSRF is_global gap |
| core/memory_probe | 1 | single-device only (multi-GPU placement gap) |
| core (config/resolvers/image_preprocessing/errors) | 4 | assorted S consistency/bug nits |
| cli_impl/load_director | 2 | warmup sizes at 0GB; GPU budget can't rescue None probe |
| cli_impl/gateway | 6 | multipart double-parse, coalescing re-acquire, forward leak, federation seams |
| cli_impl/supervisor | 5 | orphan-worker respawn, budget servability, restart_count, idle clamp, dead code |
| cli_impl (cli/refresh/console/search/probe_worker) | 6 | silent usage errors, MODALITY_EXTRAS divergence |
| admin/operations + routes | 6 | sibling-claim race, coalesce lost-update, remove TOCTOU |
| mcp | 5 | HTTP auth not armed, blocking-on-loop, audio enum drift |
| modalities (video/image_cv) | 3 | env-read-once, decode ordering, empty-frames 500 |
| models (sd_turbo/mert) | 3 | fp16-on-CPU crash, lazy-pipe race, dup helper |
| tests | 3 | pip_extras alias gap, concurrency + MCP-auth test gaps |

### Related-finding notes

- **GPU budget fallback** is one root gap with two independent fix sites: `supervisor.py:720` (boot/request servability stamp, #33) and `load_director.py:1086` (request-path fit check, #4). Both must be fixed; the director 503s even if the supervisor stamp is cleared.
- **MCP HTTP auth**: `mcp_server.py:77` (code, #1) is paired with `test_cli_integration.py:116` (the test-gap that hid it, #52).
- **Shared-venv admin races**: `operations.py:208` (#36), `:219` (#37), and `test_operations.py:1` (#51) describe the same multi-model-worker concurrency surface.
- **fp16-on-CPU** (`sd_turbo.py:169`, #39) is systemic; the fix should also route the generic diffusers runtime and `transformers_vlm` through a shared `dtype_for_device` helper.

---

## Quick wins

S-effort, clear correctness / consistency / dead-code fixes. Ordered: important (federation first), then minor (federation first).

### Important

#### 1. `src/muse/cli_impl/mcp_server.py:77`: MCP HTTP endpoint left unauthenticated when token passed via `--admin-token` [federation]
- **Severity:** important · **Category:** bug · **Effort:** S
- CLI `--admin-token` is passed to `MuseClient` for outbound calls but `run_http()` is called without it; `build_http_app` then reads only `MUSE_ADMIN_TOKEN` env, so `--admin-token SECRET` without also exporting the env starts the remotely-reachable HTTP+SSE transport fully open (warn-only).
- **Fix:** `asyncio.run(server.run_http(host="127.0.0.1", port=port, admin_token=admin_token))`, `run_http` already forwards `admin_token` to `build_http_app`.

#### 2. `src/muse/mcp/server.py:119`: blocking httpx handler runs on the asyncio event loop [federation]
- **Severity:** important · **Category:** concurrency · **Effort:** S
- `_call_tool` is async but runs `blocks = handler(client, args)` synchronously; each handler does a blocking httpx request up to 600s. In HTTP+SSE mode (single loop) one in-flight `muse_generate_video` freezes all tool calls, handshakes, and SSE frames, the same class the gateway fixed via off-loop acquire.
- **Fix:** `blocks = await asyncio.to_thread(handler, client, args)`; keep `call_handler` synchronous for tests. `MuseClient` is stateless per call.

#### 3. `src/muse/cli_impl/load_director.py:596`: admin warmup sizes models at 0 GB (over-admit / OOM) [federation]
- **Severity:** important · **Category:** bug · **Effort:** S
- `_decide`/`_load_and_commit` read `capabilities.memory_gb` straight (fallback 0.0). The gateway path runs `backfill_manifest_memory` first, but `operations.warmup_model` passes the bare `get_manifest` with no backfill, so a warmed never-probed (or probe-in-`measurements`) model sizes at 0.0, always "fits," reserves 0, never counts against concurrent loads. Backfill also folds `device_override`, so warmup additionally sizes against the wrong pool for pinned models.
- **Fix:** move the sizing ladder into the director (`backfill`-equivalent before `_decide`), or at minimum call `backfill_manifest_memory` in `operations.warmup_model` before `director.warmup`.

#### 4. `src/muse/cli_impl/load_director.py:1086`: declared GPU budget can never rescue a missing live probe [federation]
- **Severity:** important · **Category:** bug · **Effort:** S
- `_free_for_device` returns 0.0 when `gpu_free_gb()` is None; `_available_for_device` then does `min(0.0, budget) = 0.0`, so any cuda/auto model 503s `model_too_large_for_device` regardless of `MUSE_GPU_BUDGET_GB`. Budget is only ever a cap on a live reading, never a fallback. Independent from the supervisor site (#33), the director 503s even if the stamp is cleared.
- **Fix:** detect an unknown live reading (None) distinctly from 0.0; when a budget is declared, size against `budget - headroom`.

#### 5. `src/muse/cli_impl/supervisor.py:441`: auto-restart monitor respawns an orphan of a just-unloaded sole-tenant worker [federation]
- **Severity:** important · **Category:** concurrency · **Effort:** S
- Sole-tenant removal paths (`operations.py` unload ~L513, disable ~L595) `state.workers.remove(spec)` under lock but do NOT set `spec.job_id`, then shut down outside the lock. A monitor tick that snapshotted the spec earlier still iterates it (status running, job_id None → not skipped); `poll()` non-None forces failure threshold → `_attempt_restart` spawns a new subprocess on the same port that is never torn down. Restart-in-place siblings correctly guard with job_id; only these paths omit it.
- **Fix:** set `spec.job_id` (e.g. `director-unload-<id>` / `admin-disable-<id>`) under the lock before releasing on the sole-tenant paths; add a regression test.

#### 6. `src/muse/core/net_fetch.py:188`: streamed httpx responses never closed on redirect/error paths [federation]
- **Severity:** important (verifier: minor, fresh per-call client bounds impact) · **Category:** bug · **Effort:** S
- `client.send(request, stream=True)` is only consumed/auto-closed on the success path; the redirect branch `continue`s and the error paths raise without `response.close()`, holding a live connection until the `with httpx.Client` block exits. Bounded per-call (fresh client, no cross-request pool), but every redirect hop / error abandons a connection for the call duration.
- **Fix:** wrap send/consume in try/finally and `response.close()` on the redirect branch and error paths, or use `with client.stream(...)` per hop.

#### 7. `src/muse/cli.py:984`: `main()` swallows all usage-error messages on the shipped `muse` binary
- **Severity:** important · **Category:** bug · **Effort:** S
- Entry point runs `app(..., standalone_mode=False)`, where click re-raises `ClickException` WITHOUT `.show()`; the handler returns `e.exit_code` but never calls `e.show()`, so every UsageError (unknown command/option, missing arg, bad value) exits rc≠0 with empty stdout+stderr. The `python -m` tests use the standalone `__main__` block and never catch it (same gap that hid the prior exit-code bug).
- **Fix:** call `e.show()` before `return e.exit_code`; add a regression test asserting non-empty stderr on a bad flag.

#### 8. `src/muse/mcp/tools/inference_audio.py:210`: audio `response_format` enums advertise values the server rejects
- **Severity:** important · **Category:** consistency · **Effort:** S
- Route validates `^(wav|mp3|opus|flac)$` but `muse_generate_music`/`muse_generate_sfx` advertise `[wav, ogg, mp3]` (ogg → 422; opus/flac hidden), and `_AUDIO_MIME` has no `ogg` key. `muse_speak` advertises `[wav, opus, mp3]` but the speech route accepts `^(wav|opus)$` (mp3 → 422). Hand-copied enums drifted.
- **Fix:** align enums (music/sfx → `[wav, mp3, opus, flac]`; speak → `[wav, opus]`), ideally derived from the modality Field pattern so they can't drift.

### Minor

#### 9. `src/muse/cli_impl/gateway.py:233`: `/v1/models` and `/health` aggregation hardcode a 5.0s timeout [federation]
- **Severity:** minor · **Category:** consistency · **Effort:** S
- `list_models` (L233) and `health` (L273) build `httpx.AsyncClient(timeout=5.0)`, ignoring the gateway's configurable `timeout`; a >5s worker is dropped/counted-down. Fine locally, wrong for federation fan-out to remote peers.
- **Fix:** thread an aggregation-timeout (env/param, default 5s) through `build_gateway` and reuse it in both endpoints.

#### 10. `src/muse/core/net_fetch.py:90`: SSRF guard misses ranges `not ip.is_global` would reject (e.g. 100.64.0.0/10 CGNAT)
- **Severity:** minor · **Category:** bug · **Effort:** S
- The explicit `is_private/is_loopback/...` disjunction lets `100.64.0.1` (RFC 6598) through, yet `is_global` is False. Live on Python 3.10/3.11 (CI/runtime here).
- **Fix:** use `if not ip.is_global: raise ...` (keep the `MUSE_ALLOW_PRIVATE_FETCH` message), or add `is_global` to the disjunction.

#### 11. `src/muse/core/catalog.py:458`: `_read_catalog` returns `{}` on JSONDecodeError, silently vanishing every pulled model
- **Severity:** minor · **Category:** bug · **Effort:** S
- A truncated/corrupt read logs one warning, returns `{}`, doesn't populate the cache (so the parse repeats every hot-path call), and every consumer behaves as no-models (404s, empty `/v1/models`).
- **Fix:** raise a distinct `CatalogCorruptError` the supervisor/gateway can surface as 500, or return the last-known-good cached parse instead of `{}`.

#### 12. `src/muse/core/catalog.py:1098`: `load_backend` uses unbounded `split(":")` while `get_manifest` uses `split(":", 1)`
- **Severity:** minor · **Category:** consistency · **Effort:** S
- A `backend_path` with a second colon raises "too many values to unpack" in `load_backend` but is tolerated by `get_manifest`.
- **Fix:** use `split(":", 1)` in `load_backend`, or centralize parsing in one validated helper.

#### 13. `src/muse/core/config.py:340`: `set_value`/`unset_value` write config.yaml but never reset the module `_CONFIG` singleton
- **Severity:** minor · **Category:** consistency · **Effort:** S
- In-process write is invisible to subsequent `config.get()` (stale singleton). Latent today (CLI writes in a separate process) but a footgun for a future in-process/admin config mutation; inconsistent with the catalog module.
- **Fix:** call `reset_config()` at the end of both (guard on `path == config_path()`), or document them as write-only requiring a caller reset.

#### 14. `src/muse/core/resolvers.py:218`: empty resolver registry raises a misleading "multiple resolvers registered []"
- **Severity:** minor · **Category:** bug · **Effort:** S
- With `len(_RESOLVERS)==0` and `backend=None`, control falls into the else branch that claims multiple resolvers.
- **Fix:** branch on `len==0` explicitly with "no resolvers registered"; reserve disambiguation for `len>1`.

#### 15. `src/muse/core/image_preprocessing.py:228`: `DerivedImageProcessor` silently defaults mean/std to 0.5 when config omits them
- **Severity:** minor · **Category:** bug · **Effort:** S
- Tier 3 builds from `read_encoder_hints`; when `image_mean/image_std` are absent it substitutes 0.5 with only a debug log, reintroducing the silently-wrong normalization the Tier-3 removal targeted. (Partly a documented ViT default, and now a narrower trigger.)
- **Fix:** only build the derived processor when mean/std are actually present or overridden; otherwise raise `ImageProcessorError` pointing at the override hatch.

#### 16. `src/muse/core/errors.py:39`: `ModelNotFoundError` hardcodes `type='invalid_request_error'` instead of `error_type_for_status`
- **Severity:** minor · **Category:** consistency · **Effort:** S
- Correct value today, but the one envelope builder in the file that bypasses the single-source-of-truth helper; drift risk on a future status change.
- **Fix:** build the envelope via `error_type_for_status(404)`.

#### 17. `src/muse/cli_impl/gateway.py:640`: unmapped loader failure gives loader a 500 but same-model waiters a 503
- **Severity:** minor · **Category:** consistency · **Effort:** S
- A non-`OperationError` from `director.acquire` propagates out of `_route_via_director` (only OperationError caught) → 500 for the loader, while `_settle` makes waiters raise `OperationError(status=503)`, different status for one root cause.
- **Fix:** wrap the loader's unexpected acquire failure to a consistent 503, matching waiters.

#### 18. `src/muse/cli_impl/supervisor.py:399`: `restart_count` is cumulative for the worker lifetime and never resets on successful restarts
- **Severity:** minor · **Category:** bug · **Effort:** S
- `_attempt_restart` bumps `restart_count` unconditionally and resets only `failure_count` on success; a worker that cleanly recovers ~10 times over its life hits `_MAX_RESTARTS` and is marked dead, contradicting the "10 unsuccessful restart attempts" doc.
- **Fix:** reset `restart_count` after N consecutive healthy ticks, or only increment on a FAILED restart.

#### 19. `src/muse/cli_impl/supervisor.py:259`: `plan_workers`, `_wait_for_first_ready`, `_promote_workers` are vestigial (test-only callers)
- **Severity:** minor · **Category:** dead-code · **Effort:** S
- Lazy-load `run_supervisor` never calls them; port-allocation logic is re-implemented in `operations._pick_free_port`.
- **Fix:** remove the three functions (and orphaned tests), or mark legacy and consolidate port allocation with `_pick_free_port`.

#### 20. `src/muse/cli_impl/supervisor.py:1048`: idle-sweep interval not guarded against 0/negative → busy-loop
- **Severity:** minor · **Category:** bug · **Effort:** S
- `MUSE_IDLE_SWEEP_INTERVAL_SECONDS=0` (or negative) makes `_stop_event.wait(interval)` return immediately, spinning `tick()` and hammering the director lock. The adjacent `default_idle_timeout` IS guarded.
- **Fix:** clamp/validate in `run_supervisor` (fall back to 30.0 when not a positive finite number).

#### 21. `src/muse/cli_impl/console.py:88`: `truncate()` is tested-but-dead; `get_console` docstring stale
- **Severity:** minor · **Category:** dead-code · **Effort:** S
- No production caller (tables use rich `overflow='ellipsis'`/`no_wrap`); docstring describes a `force_terminal=False` override the code does not pass (`no_color=True, highlight=False`).
- **Fix:** remove `truncate()`+tests (or wire it into the one plain-text renderer that needs it); fix the docstring.

#### 22. `src/muse/cli_impl/search.py:114`: size/downloads formatting treats 0 as unknown and duplicates format expressions 3×
- **Severity:** minor · **Category:** bug · **Effort:** S
- `if r.size_gb`/`if r.downloads` render genuine 0.0/0 as `?`, losing the None-vs-0 distinction (inconsistent with the `is None` filter at L79); expressions copy-pasted across rich/plain renderers + width calc.
- **Fix:** guard on `is not None`; factor two tiny shared helpers.

#### 23. `src/muse/cli.py:897`: `_try_admin_action` enable falls back to catalog-only when the admin response lacks `job_id`
- **Severity:** minor · **Category:** bug · **Effort:** S
- Every `return True` is nested inside `if job_id:`; a falsy job_id reaches the trailing `return False`, so the caller runs the catalog-only fallback + "catalog only" message even though the admin API accepted the enable. Latent (async contract sends job_id) but bites on a sync/contract change.
- **Fix:** treat a missing-job_id admin success as handled (`return True` with an explicit message), or assert the async contract and error clearly.

#### 24. `src/muse/admin/routes/jobs.py:23`: jobs routes call `get_default_store()` directly while models routes use `_resolve_store()`
- **Severity:** minor · **Category:** consistency · **Effort:** S
- Inconsistent dependency-resolution seam across sibling admin route modules; a monkeypatch targeting `models._resolve_*` won't be seen by jobs/workers/memory.
- **Fix:** pick one convention, drop the indirection in `models.py`, or add matching `_resolve_*` helpers to jobs/workers/memory.

#### 25. `src/muse/modalities/video_generation/routes.py:43`: `MUSE_VIDEO_MAX_FRAMES_B64` read once at import (needs restart)
- **Severity:** minor · **Category:** consistency · **Effort:** S
- Sibling caps (`image_input.py`, `text_classification/routes.py`) read per-request so env changes take effect live; video freezes the cap at first import.
- **Fix:** replace the module constant with a per-request `_max_frames_b64()` helper calling `config.get`.

#### 26. `src/muse/modalities/image_cv/routes.py:100`: image_cv/image_ocr decode the upload before resolving model / checking capability
- **Severity:** minor · **Category:** consistency · **Effort:** S
- CV (decode L101 → resolve L105 → gate L106) and OCR (decode L59 → get L64) pay a full size-capped decode on 404 / wrong-primitive requests; image_generation/upscale/segmentation gate first.
- **Fix:** reorder to resolve backend + capability gate before `decode_image_file`.

#### 27. `src/muse/modalities/video_generation/routes.py:119`: empty-frames result surfaces as a bare 500
- **Severity:** minor · **Category:** bug · **Effort:** S
- `_encode` raises plain `ValueError("...frames list is empty")` for zero-frame results; the route catches only `UnsupportedFormatError`, so it escapes to FastAPI's default 500 (non-OpenAI envelope).
- **Fix:** catch `ValueError` alongside `UnsupportedFormatError` and return an OpenAI-shape `error_response`.

#### 28. `src/muse/models/sd_turbo.py:249`: lazy img2img/inpaint pipeline caches built without a lock (double-build race)
- **Severity:** minor · **Category:** concurrency · **Effort:** S
- `self._i2i_pipe`/`self._inp_pipe` read-then-assign; with route handlers dispatched via `asyncio.to_thread`, two concurrent first requests each call `from_pipe`, briefly doubling VRAM and leaking one instance.
- **Fix:** guard the lazy `from_pipe` init with a per-instance double-checked `threading.Lock`.

---

## Later

Design work, larger refactors, concurrency-subtle, and test-gaps. Ordered: important (federation first), then minor.

### Important

#### 29. `src/muse/core/catalog.py:1188`: `get_manifest` omits the curated-capabilities overlay that `known_models` applies (split-brain) [federation]
- **Severity:** important · **Category:** consistency · **Effort:** M
- `known_models()` re-applies the curated overlay onto resolver-pulled persisted manifests (docstring: edits to curated.yaml take effect on restart without a re-pull); `get_manifest()`'s persisted branch applies only `base_override`. So post-pull curated.yaml edits (supports_tools, context_length, memory_gb, device) reach the runtime constructor (via `entry.extra` from `known_models`) but NOT route-gating, `/v1/models`, or the director's sizing/placement (which read `get_manifest`).
- **Fix:** factor the overlay re-application into a shared `_apply_overlays(manifest, model_id, entry_data)` and call it from both paths.

#### 30. `src/muse/core/catalog.py:1209`: `get_manifest` calls `discover_models()` live on the gateway hot path for alias-runtime bundled models [federation]
- **Severity:** important · **Category:** design · **Effort:** M
- For bundled scripts whose `Model` is aliased from a shared runtime (smolvlm, ace-step), the module-level MANIFEST mismatches, so the fallback calls `discover_models(_model_dirs())` directly, bypassing the process-lifetime cache, on every routing request (`gateway.py:516`) and `/v1/models` aggregation (`gateway.py:439`). Bodies are sys.modules-cached but the glob/spec/getattr churn runs per request; federation multiplies routing traffic.
- **Fix:** cache the discovered manifest (store alongside `CatalogEntry` in `_discovered_entries_cache`, or add a cached `discover_manifest(model_id)`), read from it in `get_manifest`.

#### 31. `src/muse/core/net_fetch.py:240`: `afetch_url_bytes` wraps the blocking sync fetch in `asyncio.to_thread` (shared-pool exhaustion) [federation]
- **Severity:** important · **Category:** concurrency · **Effort:** M
- Each async fetch pins one default-executor worker for up to the 30s timeout; a burst of image_url inputs (or a federation gateway proxying many multimodal requests) exhausts the shared pool and stalls every other `to_thread` user, the exact class documented for director coalescing, at the request-routing seam.
- **Fix:** give `afetch_url_bytes` a native `httpx.AsyncClient` path (share the redirect/pin/size-cap loop), or at least route through a bounded dedicated executor.

#### 32. `src/muse/core/memory_probe.py:75`: `gpu_free_gb` queries a single device (device 0); no enumerate/aggregate API [federation]
- **Severity:** important · **Category:** design · **Effort:** M
- The whole control plane sizes/admits/evicts against GPU 0's free memory. Federation hardware-aware placement wants per-device VRAM to route to the node/GPU with capacity; there is no way to reason about GPUs 1..N.
- **Fix:** add `gpu_devices()`/`gpu_free_map()` (`nvmlDeviceGetCount` + per-index free) → `{device_id: free_gb}`; thread device selection through the director's admission/eviction.

#### 33. `src/muse/cli_impl/supervisor.py:720`: `MUSE_GPU_BUDGET_GB` never consulted for servability (documented fallback nonexistent) [federation]
- **Severity:** important · **Category:** design · **Effort:** M
- `_available_pools`/`_servability_reason` compute GPU availability purely from live `gpu_free_gb()`; when pynvml is absent they hard-stamp "exceeds device capacity (... or set memory budget)", but neither reads `server.gpu_budget_gb`, so the advertised budget escape hatch does not exist at this seam and the gateway 503s before the director runs. (Pairs with #4.)
- **Fix:** thread `gpu_budget_gb` in; when live GPU info is None but a budget is declared, use `gpu_available_gb = max(0, budget - headroom)`; hard-fail only when both are absent.

#### 34. `src/muse/cli_impl/gateway.py:104`: gateway fully parses every multipart upload just to read the `model` field (double-buffer) [federation]
- **Severity:** important · **Category:** design · **Effort:** M
- `extract_model_from_request` does `await request.body()` (caches raw bytes) then `await request.form()` (re-parses, materializes UploadFiles / spools >1MB to disk), discarded without `close()`. Every image/audio/CV multipart request holds the upload ~2× at the front door and pays a full parse for a small string; plausible temp-file leak. The routing seam federation will reverse-proxy.
- **Fix:** extract `model` without full-body materialization (bounded streaming read of leading fields), or cap `form()` and explicitly `close()` returned UploadFiles.

#### 35. `src/muse/cli_impl/refresh.py:53`: `MODALITY_EXTRAS` is a second, divergent, stale source of truth for venv provisioning
- **Severity:** important · **Category:** consistency · **Effort:** M
- `pull` provisions `muse[server]` + the model's `pip_extras`; `refresh_one` installs `muse[server,<MODALITY_EXTRAS[modality]>]` + pip_extras, folding in extras pull never installed (and `--upgrade` can bump torch/diffusers past a model's pins). The map is hand-maintained and already missing audio/classification, image/cv, image/ocr, 3d/generation. The hardcoded-mapping pattern CLAUDE.md warns against.
- **Fix:** make refresh mirror pull (`muse[server]` via the shared `_muse_server_install_args` + pip_extras) and delete `MODALITY_EXTRAS`; if muse-side modality extras are truly needed, add them in ONE shared helper used by both.

#### 36. `src/muse/admin/operations.py:208`: `enable_model` sibling-claim ignores an already-set `job_id`, stealing an in-flight spec (double restart) [federation]
- **Severity:** important · **Category:** concurrency · **Effort:** M
- Claim test `status=="running" or job_id is None` runs before the coalesce branch; unload/disable set `job_id` but leave `status=="running"`, so a concurrent same-venv enable claims the spec, overwrites job_id, and runs a second `_restart_worker_inplace` (double SIGTERM+respawn on one port → orphan VRAM, port race, corrupted `models`). `load_model_into_worker` (L405) has the identical hole. Reachable only for shared-venv multi-model workers, the co-location case federated placement leans on.
- **Fix:** route a sibling with `job_id is not None` to coalesce regardless of status; claim only when `job_id is None and status=="running"`. Apply the same guard to `load_model_into_worker`.

#### 37. `src/muse/admin/operations.py:219`: coalescing appends the model to `spec.models` after `spawn_worker` may have read it (lost update, false success) [federation]
- **Severity:** important · **Category:** concurrency · **Effort:** M
- The coalesce-onto-sibling branch appends `model_id` under the lock and returns a done/coalesced job, but the in-flight restart runs outside the lock and `spawn_worker` may already be past reading `spec.models`. The model is then absent from the spawned worker yet the routing source lists it and the job reports done → gateway routes requests to a worker that never loaded it.
- **Fix:** don't treat append+coalesce as a load guarantee; wait on the owning job then re-validate membership (poll `/v1/models`) and re-drive a load if absent.

#### 38. `src/muse/models/sd_turbo.py:169`: default `dtype=float16` kept on CPU → documented CPU-runnable model crashes [federation]
- **Severity:** important · **Category:** bug · **Effort:** M
- sd-turbo declares `device:auto` and defaults `dtype=float16`; on a CPU-only host `select_device→cpu` but torch_dtype stays fp16 and fp16 variant weights load, so diffusers CPU inference raises (e.g. "LayerNormKernelImpl not implemented for Half"). Systemic: `image_generation/runtimes/diffusers.py` (~L182) and `chat_completion/runtimes/transformers_vlm.py` (L122-131) share it (smolvlm documented CPU-runnable). Only `bark_small.py` guards. Tests mock torch so CI never exercises fp16-on-CPU.
- **Fix:** add `runtime_helpers.dtype_for_device(name, device, torch)` forcing float32 on cpu; route sd_turbo, the generic diffusers runtime, and transformers_vlm through it; skip `variant="fp16"` on CPU.

#### 39. `tests/models/test_pip_extras_audit.py:180`: audit only AST-walks the script file, so alias-runtime scripts' real deps are unverified
- **Severity:** important · **Category:** test-gap · **Effort:** M
- For the ~9 scripts that `from ...runtimes import X as Model`, `_walk_imports` returns only `{'muse'}` (skipped), so pip_extras coverage of the runtime module's imports (torch/transformers/acestep/librosa/timm/torchvision) is unasserted; the fresh-venv smoke matrix covers none of them, so a runtime adding an import ships a broken cold-load silently.
- **Fix:** when the only relevant import is the aliased runtime, resolve the `Model.__module__` file and AST-walk it too (union import sets), or add the light alias scripts to the smoke matrix.

### Minor

#### 40. `src/muse/cli_impl/gateway.py:517`: request-path failure seams (404 unknown-model, retained 503) have no federation delegation point [federation]
- **Severity:** minor · **Category:** design · **Effort:** L
- `_route_via_director` resolves locally: `get_manifest` KeyError → 404, retained unservable stamp / OperationError → 503. These are exactly where federation wants to reverse-proxy to a peer, but there is no seam.
- **Fix:** introduce an injectable placement-resolver seam; on local KeyError / unfit-503, consult a peer registry and forward if a peer serves/fits, else error. Local-only default is a no-op resolver.

#### 41. `src/muse/cli_impl/gateway.py:895`: `_forward` duplicates `_forward_with_release` and leaks the httpx client if `stream_ctx.__aexit__` raises
- **Severity:** minor · **Category:** simplification · **Effort:** M
- The legacy static-routes `_forward` runs `await stream_ctx.__aexit__(...)` before `await client.aclose()` unguarded (L895-897, L909-911); if `__aexit__` raises, `aclose()` is skipped and the client leaks, the failure `_forward_with_release` was hardened against. Low blast radius (production uses `_forward_with_release`).
- **Fix:** unify into one `_forward(request, url, timeout, *, on_release=None)` with the guarded per-step cleanup and release as an optional callback.

#### 42. `src/muse/cli_impl/gateway.py:659`: coalescing waiters' post-'ok' re-acquire is uncoalesced (re-introduces #319 thread-park under churn)
- **Severity:** minor · **Category:** concurrency · **Effort:** M
- If the model is evicted between the loader's commit and the waiters' bare `_acquire_off_loop` re-acquire, the model is cold again and N-1 waiters park in the director's `event.wait` inside ThreadPoolExecutor threads, the exhaustion coalescing was built to prevent. Rare but reachable.
- **Fix:** route the waiter's re-acquire back through `_acquire_coalesced` (recurse/loop) so a re-cold model re-collapses to a single loader.

#### 43. `src/muse/cli_impl/probe_worker.py:148`: `_resolve_device` re-implements `load_backend`'s device-precedence ladder [federation]
- **Severity:** minor · **Category:** simplification · **Effort:** M
- The override > cap-pin > requested > auto ladder is duplicated so the probe knows VRAM-vs-RSS; the docstring itself warns it must stay in lockstep or the probe records a bogus peak against the wrong pool (which then mis-sizes admission/eviction).
- **Fix:** extract a shared `resolve_load_device(entry, requested, override)` used by both `load_backend` and `probe_worker` (cover the auto→concrete step too). Useful for the federation placement layer.

#### 44. `src/muse/admin/operations.py:753`: `remove_model` checks `live_host` under the lock but purges the venv outside it (TOCTOU)
- **Severity:** minor · **Category:** concurrency · **Effort:** M
- A concurrent enable/load can spawn a worker in the window between the check and `catalog_remove(purge=...)`, which then deletes the venv under a live process holding open FDs, exactly what the guard's docstring warns about.
- **Fix:** hold `state.lock` across the check and a state mutation blocking concurrent spawns (mark pending-removal), or re-check `live_host` immediately before purge and abort 409.

#### 45. `src/muse/admin/routes/memory.py:142`: `_per_model_breakdown` re-derives the auto→pool verdict per model per request [federation]
- **Severity:** minor · **Category:** design · **Effort:** S
- Calls `_resolve_auto_side()` per auto-device model (up to 2N times/request across gpu+cpu summaries), redundantly re-querying GPU free and re-computing the pool decision the director resolves once. (Verifier: the per-model pynvml init/shutdown cost claim is overstated, init is sticky/idempotent; real cost is a cheap query.)
- **Fix:** resolve the auto side once per request and pass into both breakdowns; ideally share the director's `_resolve_pool_device` result.

#### 46. `src/muse/admin/operations.py:188`: `enable_model`/`disable_model` do catalog disk I/O while holding `state.lock`
- **Severity:** minor · **Category:** concurrency · **Effort:** S
- `set_enabled` (atomic write-then-rename) runs inside `with state.lock` (L189/L577); the gateway's per-request unservable short-circuit read also takes `state.lock` (`gateway.py:495`), so a catalog fsync/rename briefly blocks request routing. (Detail nit: `_read_catalog` at L173 is outside the lock.)
- **Fix:** move the `set_enabled` write outside the critical section (plan under lock, flush the flag before/after), matching the spawn/shutdown plan-then-execute pattern.

#### 47. `src/muse/mcp/tools/inference_text.py:30`: inference handlers don't catch `httpx.HTTPStatusError`, dropping the structured error body
- **Severity:** minor · **Category:** consistency · **Effort:** M
- Admin handlers wrap errors into a structured envelope; inference handlers let `HTTPStatusError` hit the blanket `except`, yielding `{error: str(e)}` that drops muse's OpenAI-shape `{error:{code,message,type}}` (e.g. `vision_not_supported`), so the LLM loses the reason to self-correct.
- **Fix:** catch `HTTPStatusError` in a shared helper (or in `MuseClient`) and re-raise/return the parsed OpenAI error body, mirroring `_admin_error_block`.

#### 48. `src/muse/mcp/binary_io.py:84`: input size cap enforced only on http(s) URLs; b64 and data: URLs bypass it
- **Severity:** minor · **Category:** design · **Effort:** S
- The b64 branch (L85) and data: branch (L93) `b64decode` with no ceiling, fully materializing arbitrarily large blobs, a cheap memory-pressure vector in open HTTP+SSE mode. (Docstring does scope the cap to URL inputs, so it's documented, but the C3 guarantee is inconsistent.)
- **Fix:** apply `_default_max_bytes()` to the decoded length of the b64 and data: paths too, raising the same `ValueError`.

#### 49. `src/muse/models/mert_v1_95m.py:246`: `_move_to_device` copy-pasted verbatim across 5 files
- **Severity:** minor · **Category:** simplification · **Effort:** S
- Identical helper in mert, dinov2_small, image_embedding/audio_embedding runtimes, and sam2_runtime, the cross-cutting device util `runtime_helpers` exists to consolidate; the meta-test flags the other four helpers but not this fifth.
- **Fix:** move `_move_to_device` into `muse.core.runtime_helpers`, import in the five sites, add it to the meta-test's flagged set.

#### 50. `tests/admin/test_operations.py:1`: no test exercises the job_id ownership protocol under concurrency [federation]
- **Severity:** minor · **Category:** test-gap · **Effort:** S
- Happy-path plans are covered but no test asserts a running sibling with a non-None job_id (owned by an in-flight unload/disable) is NOT re-claimed by enable/load, exactly the race in #36, silently uncovered.
- **Fix:** add a test pre-populating a running multi-model spec with `job_id` set, asserting `enable_model` for a same-venv sibling coalesces/waits rather than overwriting job_id.

#### 51. `tests/mcp/test_cli_integration.py:116`: no test asserts a CLI-supplied admin token actually arms the HTTP auth gate [federation]
- **Severity:** minor · **Category:** test-gap · **Effort:** S
- `test_runs_http_when_flag_set` only asserts `run_http` was the coroutine invoked (closed unexecuted); the CLI→run_mcp_server→run_http→build_http_app wiring is untested, which is why #1 is invisible.
- **Fix:** drive `run_mcp_server(http=True, admin_token='t')` with `MUSE_ADMIN_TOKEN` unset and assert the resulting Starlette app rejects an unauthenticated `/mcp` with 401.
