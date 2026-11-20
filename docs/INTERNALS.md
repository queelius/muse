# Runtime internals

Long-form notes on device placement extras, lazy loading and eviction, request
queueing, observability and federation, moved out of CLAUDE.md so it stays short.
CLAUDE.md keeps a summary of each with the invariants that bite; the detail is here.

## Device placement extras

**GPU-layers pin (v0.56.0, GGUF only).** `muse models set-gpu-layers <id> <N>`
(N >= 0), `--all` (all layers on GPU, i.e. -1; a bare `-1` positional is not
shell-passable through click), or `--clear` writes a top-level catalog
`gpu_layers_override` (the `device_override` pattern): llama.cpp `n_gpu_layers`,
-1 = all layers on GPU, 0 = pure CPU, N > 0 = a static GPU/CPU layer split for
GGUFs bigger than VRAM. Precedence: pin > `capabilities.n_gpu_layers` > runtime
default (-1). Refuses non-GGUF models (other runtimes silently ignore the kwarg).
Takes effect on next cold load; run `muse models probe <id>` after pinning
so admission sizing measures the split's real (smaller) VRAM peak. The
probe honors the pin automatically (it constructs via `load_backend`).
**CUDA llama-cpp-python wheels (v0.62.1).** pip's source build is CPU-only
without `nvcc`. `install.llama_cpp_wheel: cu130` (env `MUSE_LLAMA_CPP_WHEEL`)
makes `muse pull` / `muse models refresh` install llama-cpp-python
binary-only from the abetlen cu130 index, capped at the verified version in
`muse.core.llama_cuda.WHEELS`, plus `nvidia-cuda-runtime`/`nvidia-cublas`
(`pip_install.pip_install_commands`; a same-version CPU build is force-reinstalled
`--no-deps`). The cap is folded into the requirement (`>=0.2.90,<=0.3.33`): a separate
`<=` requirement makes pip download the newest ~800 MB wheel first. The wheel step
uses `--no-cache-dir` because the index serves signed one-off URLs pip never reuses. The wheel does not ship libcudart/libcublas and its RPATH is
`$ORIGIN`, so `llama_cuda.prepare_llama_cpp()` dlopens them RTLD_GLOBAL from
`site-packages/nvidia/*/lib` before `import llama_cpp` (both GGUF runtimes'
`_ensure_deps`, `placement.native_gpu_offload/backend`, and the supervisor's
`_QUERY` capability script, which also adds them to its cache identity).
With `cpu_only=True` (a cpu placement or 0 GPU layers) it first sets
`CUDA_VISIBLE_DEVICES=""`: a CUDA build otherwise allocates VRAM and offloads
host matmuls even at `n_gpu_layers=0`. No-op for CPU builds. See
docs/CHAT_COMPLETION.md "GPU builds of llama-cpp-python".
**Venv dedupe (v0.63.0).** `muse.core.venv_dedupe` hardlinks content-identical
(SHA-256 + size + mode + uid) regular files >= 1 MiB under each model venv's
`site-packages` (never `bin/`, `__pycache__`, dot-prefixed transaction
workspaces, or files changed in the last 60s), via link-to-temp + rename after
re-checking both files' stat and comparing their bytes (the digest cache in
`venv-digests.json` only groups candidates and is pruned to live inodes); the
linked name must still be the keeper's inode. A venv with a live
`.<name>.transaction-*` workspace is skipped: a refresh or re-pull runs pip in
the canonical path. Safe because pip unlinks before writing installed files,
which is also why `venv_transaction` hardlinks those same files
(`venv_dedupe.is_package_file`) instead of copying when it clones a venv, so
re-pulls keep dedupe's links and need far less free space.
`muse storage dedupe [--dry-run]` does every venv; `dedupe_after_install` runs an
`only=<venv>` pass after each successful pull/refresh (`storage.dedupe_venvs`,
default true; never fails the install). Saves disk and shared-library page
cache, not VRAM. First pass on the dev host: 118.5 GiB of 154 GiB.
`muse.core.hf_seed.seed_repo` hardlinks SHA-verified shared-HF-cache blobs
(`HF_HUB_CACHE`) into `<catalog_dir>/weights` before cache-mode downloads (bundled
`snapshot_download` and resolver `resolved.download`), so huggingface_hub reuses
them by etag name instead of re-downloading; `drop_unused_seeds` removes the ones
the snapshot did not reference. It links first and verifies the linked inode,
links only blobs this user owns (clearing group/world write, since the inode is
shared), honors trailing-`/` patterns like huggingface_hub, and reads only the
pinned revision's snapshot when present. Resolver plugins pass their download
allow-list as `ResolvedModel.allow_patterns` (the GGUF plugin's names only the
chosen quant), so seeding hashes only what the pull will fetch. `muse doctor storage` names catalog models still
loading from the shared cache (`StorageReport.shared_hf_models`): pre-0.61 pulls
with no provenance receipt, which are unservable until re-pulled.
The chat HF resolver caps the active `context_length` at 8192 and records the
trained maximum as `max_context_length` (it used to size the KV cache for the
full trained context: Qwen3-0.6B 6.65 GB -> 1.66 GB VRAM).
Known limitation: a split model occupies both VRAM and host RAM, but the
director accounts it against its resolved device's pool only -- the
Tier-1 static-offload simplification (automatic offload was evaluated and
rejected; see docs/superpowers/specs/2026-07-08-gpu-layers-pin-design.md).

## Lazy load + LRU eviction (v0.40.0+)

`muse serve` no longer eager-loads enabled models at boot. The gateway
comes up instantly with zero workers; per-model workers spawn on first
request and are evicted under memory pressure via on-demand LRU. Behind
this is a runtime decoupling: catalog `enabled: true` declares "in
service, may serve requests" but does NOT imply "in memory right now."
A 12GB GPU can have 30 models enabled and serve them all, just not
simultaneously; the live working set is sized to fit current free VRAM
minus a headroom margin.

The orchestrator is `muse.cli_impl.load_director.LoadDirector`,
attached to `SupervisorState`. The gateway wraps every forwarded
request in `director.acquire(model_id)` / `director.release(model_id)`.
`acquire` is a three-phase critical section: under-lock decision (read
loaded set, query live free memory, pick eviction victims if needed) ->
outside-lock load (worker spawn + `/health` poll, possibly with
preceding evictions) -> under-lock commit (insert `LoadEntry`,
increment refcount, update `last_touched_at`). Concurrent acquires for
the *same* cold model collapse to one load via an `asyncio.Future` /
`threading.Event` registered in `in_flight_loads`; concurrent acquires
for *different* cold models proceed in parallel because the load phase
runs outside the lock.

**Acquire runs OFF the event loop (v0.50.3+).** The gateway dispatches
`director.acquire` via `await asyncio.to_thread(...)` under
`asyncio.shield` (`gateway.py:_route_via_director`), NOT as a blocking
synchronous call. A cold load takes tens of seconds (worker spawn +
health poll); calling it synchronously froze the single gateway event
loop and stalled EVERY concurrent request, including ones for
already-hot models (measured: a 37s cold load stalled 6 hot requests
~35s each -- see `examples/concurrent/RESULTS.md`). Moving it off-loop
makes request-path acquires genuinely concurrent, which required three
companion safety fixes the serialized path had masked:
1. **In-flight memory reservation.** `in_flight_loads` now maps to an
   `InFlightLoad(event, memory_gb, pool)` record, and `_decide` debits
   the sum of same-pool in-flight loads (`_reserved_for_pool`) from
   available memory before the fit check (the eviction no-candidates
   re-check nets the same reservations). Without this, two concurrent
   cold loads for different models would both pass the fit check against
   the same live free-VRAM reading and over-commit -> OOM. The
   reservation is deliberately conservative (it may over-reserve while a
   worker is mid-allocation, trading a possible spurious evict/503 for
   guaranteed no-OOM; precise per-load allocation accounting is a
   deferred follow-up).
2. **Cancellation-safe release.** If a request is cancelled (client
   disconnect / timeout) while its off-loop acquire is still running,
   `shield` keeps the acquire alive and a done-callback releases the
   refcount if the acquire later succeeds -- else a bumped refcount leaks
   and pins the model non-evictable forever. `_forward`'s single idempotent
   cleanup was likewise widened to `except BaseException` so a cancel on
   the forward leg also releases.
3. **Overlap-gated observed-peak writeback.** `_load_and_commit` only
   writes the self-heal peak when the load was "solo" (the sole in-flight
   load at both endpoints of its free_before..free_after window AND no
   concurrent load/eviction bumped `_inflight_epoch`); an overlapping
   load or eviction pollutes the global free-memory delta the writeback
   infers a peak from.

**Same-model cold-load coalescing (v0.51.0+).** Off-loop acquire fixed
loop-blocking but left a second hazard: a burst of concurrent requests
for the SAME cold model each dispatched their own `to_thread(acquire)`,
and the N-1 that lost the director's singleton-collapse race parked in
`event.wait` INSIDE ThreadPoolExecutor threads, exhausting the default
`min(32, cpu+4)` pool and stalling unrelated hot traffic (measured 11.5s
on the GPU box; loop stayed responsive, `/health` fine). `gateway.py`
`_acquire_coalesced` elects ONE loader per cold model via a per-model
`asyncio.Future` gate in `SupervisorState.cold_load_gates` (a plain dict:
the loader election is await-free, hence atomic on the single loop thread
-- no lock, no two-loader TOCTOU). The loader runs one off-loop acquire;
same-model waiters `await asyncio.shield(gate)` ON THE LOOP (no thread
each). The gate is settled from a done-callback on the acquire future
(NOT the loader body), so a loader cancelled mid-load still resolves its
waiters; waiters `shield` so one cancellation can't poison the group;
waiters PROPAGATE the loader's failure (no retry herd) and on success take
their own refcount via a full re-acquire that re-decides hot-or-cold. The
director is unchanged (its own threading collapse still serves admin
warmup / other callers). Coalescing is same-model only; cross-model
over-admission is still handled by the in-flight reservation above.

Memory accounting is live, not declared. `muse.core.memory_probe`
wraps `pynvml.nvmlDeviceGetMemoryInfo` (GPU) and
`psutil.virtual_memory().available` (CPU). `nvidia-ml-py` is a soft
dep under `[server]`: when missing (CPU-only host, AMD GPU, driver
mismatch) `gpu_free_gb()` returns None and GPU loads either fall back
to a declared `MUSE_GPU_BUDGET_GB` cap or 503 with `unservable_reason`.
Every cold load also captures `free_before - free_after`; if the
observed delta exceeds `measurements.<device>.peak_bytes` it gets
written back via atomic write-then-rename. Estimates self-heal upward
toward reality on every load.

A model's load size comes from a four-step sizing ladder (in
`supervisor.py:_has_memory_data`; its docstring uses the same numbering),
most-honest first:

1. `capabilities.memory_gb` (declared annotation), or
   `measurements.<device>.peak_bytes` from a prior probe -- the
   authoritative numbers.
2. *Base chase:* when neither exists and the manifest declares
   `capabilities.base_model` (a LoRA adapter, whose own dir holds only the
   adapter), the ladder sizes it from that base's catalog entry
   (`supervisor._base_model_memory_gb`, one level, from the catalog the
   caller already read). The base is a muse catalog id, or an HF repo id
   (a raw `hf://` LoRA pull takes it from the Hub tag, e.g.
   `black-forest-labs/FLUX.2-klein-4B`) matched against the catalog
   entries' `hf_repo` case-insensitively, ignoring any `@revision`; the
   largest estimate wins if several entries carry that repo. Only the
   base's probe measurement for the pool the LoRA loads into, else its
   declared `memory_gb`, else a measurement from another device counts.
   It never falls back to the base's weights on disk: a klein snapshot
   sums to ~24 GB against a ~9.5 GB offloaded peak and would make the LoRA
   unservable. So when the base has no `memory_gb` and no probe
   measurement (or is not in the catalog), the LoRA falls through to
   step 3 and is sized from its own adapter weights (tens of MB) until
   the pull probe measures the real base+adapter peak (or the first load
   self-heals it upward). Boot validation, the request-path re-check and
   `backfill_manifest_memory` all use it.
3. *Weights-on-disk fallback* (v0.47.3): when neither exists, the
   supervisor sizes the model from the bytes already in its `local_dir`
   (sum over the HF snapshot tree, symlinks followed). A pulled model
   that was never probed is therefore sized from its weights and loads
   on demand rather than 503'ing with "no memory estimate." This is the
   common case -- every `muse pull` leaves weights on disk -- so the old
   "run `muse models probe` first" wall is gone. *GGUF exception
   (v0.50.2):* a GGUF snapshot dir routinely holds several quant variants
   of one model (q3/q4/q5/q8/f16), but only the declared
   `capabilities.gguf_file` actually loads. Summing the whole tree would
   OVERestimate wildly (a 4B q4 whose repo ships six quants sums to ~15 GB
   vs its ~2.6 GB weight) and 503 a servable model as "exceeds device
   capacity", so when a specific `gguf_file` is declared the sizer counts
   that one file (every shard of it, for a split GGUF), falling back to
   the tree walk only if it is absent on disk. This bug stayed hidden until v0.50.1 routed undeclared-device
   models to the cuda pool, where the inflated estimate first exceeded the
   card. *Diffusers/transformers dedup (v0.54.3):* the same over-count bites
   any snapshot that ships redundant encodings of one component -- fp16 AND
   fp32, `.safetensors` AND `.ckpt`/`.bin`, `model-*.safetensors` AND
   `pytorch_model-*.bin` -- plus large non-weight blobs (dataset-attribution
   CSVs, READMEs). Stable Audio Open summed 14.6 GB vs its ~2.6 GB real load
   and 503'd on a 12 GB card. The tree walk now counts only WEIGHT-extension
   files and, per component (same dir + canonical stem, with dtype tags and
   the `model`/`pytorch_model`/`tf_model`/`flax_model` prefix normalized),
   keeps ONE variant -- fp16 over fp32, safetensors over pickle. Counting the
   loaded variant (not the largest) leans the estimate DOWN, the safe
   direction: an under-estimate self-heals via the observed-peak writeback
   and a rare OOM is recovered by the auto-restart monitor, whereas an
   over-estimate 503s a servable model outright.
4. Nothing sizable (no annotation, no probe, no weights on disk): the
   model is flagged `unservable_reason` and 503s before any worker
   spawn.

`backfill_manifest_memory` applies the same ladder to the manifest the
gateway hands `director.acquire`, so the director sizes a never-probed
model from its weights too (not from a default 0.0). A model that DOES
fit (possibly after evicting idle LRU models) loads on demand; the
director runs the live fit/evict decision. A model that cannot fit even
an empty device is caught earlier by the request-path servability
re-check (see below) and 503s at the gateway WITHOUT reaching the
eviction loop -- deferring an impossible model to the director would
only evict the whole idle working set before 503'ing.

Memory-accounting settings (all optional; read via the config registry,
so each can be set as a `MUSE_*` env var OR in `~/.muse/config.yaml` OR
via `muse config set` -- see "Configuration" below and docs/CONFIG.md).
As of v0.52.0 the four budget/headroom knobs are actually wired into the
LoadDirector (before v0.52.0 they were documented but inert):

- `MUSE_GPU_BUDGET_GB` (`server.gpu_budget_gb`): declared cap on GPU
  memory; muse uses `min(declared, live)` when both are available.
- `MUSE_CPU_BUDGET_GB` (`server.cpu_budget_gb`): declared cap on host RAM.
- `MUSE_GPU_HEADROOM_GB` (`server.gpu_headroom_gb`, default `1.0`):
  subtracted from live free VRAM before deciding fit; protects against
  driver allocations and fragmentation.
- `MUSE_CPU_HEADROOM_GB` (`server.cpu_headroom_gb`, default `2.0`):
  subtracted from live free RAM before deciding fit.

`muse models list` shows a five-state status enum:

- `enabled_loaded` (filled circle): catalog-enabled and currently
  resident on a worker.
- `enabled_unloaded` (half circle): catalog-enabled but not resident;
  next request triggers a cold load.
- `disabled` (open circle): catalog-disabled; requests 503 without
  attempting a load.
- `recommended` (star): curated entry not yet pulled.
- `available` (mid-dot): pullable from a resolver but not curated.

The breaking behavior change is cold-start latency on first request
to each model. A 9B GGUF can take 10-30s to load; a 3B diffusion
model 5-15s. `muse models warmup <id>` is the manual pre-load: the
admin endpoint `POST /v1/admin/models/{id}/warmup` runs the load
without bumping refcount, so subsequent requests are hot. Operators
who really want the v0.39 eager-boot semantics put a warmup loop in
their startup script.

`muse pull` runs probe at the end (`--no-probe` opts out for
cross-device pulls, e.g., pulling on a CPU host for later GPU
deployment). This gives freshly-pulled models a measured memory
estimate up front. As of v0.47.3 the estimate is no longer *required*
to serve: even a `--no-probe` pull (or a probe that was skipped) loads
on demand, because the supervisor falls back to sizing the model from
its on-disk weights (step 3 of the sizing ladder above). Probe still
matters -- it yields a more honest peak than raw weights size, and it
self-heals on every subsequent load -- but for sizing it is an
optimization, not a gate.

The probe IS a gate on the pull itself (v0.61.0): it proves the model
loads and runs a representative inference. A successful pull leaves the
model `enabled: true`. If the probe fails (load error, inference error,
or timeout), `muse pull` exits 1 and
the entry is kept but set `enabled: false`, so weights and venv survive
for a retry; the message points at `muse models probe <id>` then
`muse models enable <id>`. `catalog.pull()` returns the id it installed,
so the probe targets the right entry even on a URI re-pull. The admin
pull job runs `muse pull` as a subprocess and therefore ends `failed`.
Admin probe jobs wait the model's own probe timeout plus 120 s; admin pull jobs
wait 1800 s plus the largest declared probe timeout plus 120 s (the modality is
unknown before resolution).
The in-venv probe worker exits 0 when the model loads but its inference
raises (the record carries `inference_error`); `run_probe` treats that
record as a failure (returns 6), so it fails the pull too.
A `--no-probe` pull exits 0 and stays enabled.

Decoupling principle: `enabled` is catalog state; `loaded` is runtime
state. Two new admin operations, `load_model_into_worker` and
`unload_model_from_worker` (in `muse.admin.operations`), perform the
runtime mutation *without* flipping the catalog `enabled` bit. The
LoadDirector calls these on lazy-load and eviction paths so LRU
eviction of model X doesn't accidentally disable X for the next
request. The legacy `enable_model` / `disable_model` ops keep
catalog-flip semantics for explicit operator intent and back the
`muse models enable/disable` CLI verbs.

Catalog format stays backward-compatible. `measurements.<device>.peak_bytes`
gains self-healing semantics (passively updated on every cold load),
and older muse readers ignore unknown fields. `/v1/models` gains
`loaded: bool`, `last_loaded_at: iso8601 | null`, and `unservable_reason:
str | null` per entry, plus the OpenAI-compat `created: int` (a stable
`0`; muse has no per-model creation time) and `owned_by: str` (the HF org
slug from `hf_repo`, else `"muse"`) added in v0.47.4 so strict OpenAI SDK
clients that validate the model-object shape accept it. The OpenAI-shape
envelope is otherwise unchanged. All these fields are written *after* the
`capabilities` splat in `build_model_entry`, so a manifest can never
clobber the authoritative `id`/`object`/`created`/`owned_by`/`loaded`
keys.

The gateway's `/v1/models` lists *every enabled catalog model*, not just
the ones with a live worker (v0.47.3). Loaded workers are aggregated
from their own `/v1/models`; enabled-but-unloaded catalog entries are
appended with `loaded: false` (skipping disabled models and any without
a `python_path`). Both paths render through the shared
`muse.core.server.build_model_entry` helper so the entry shape is
identical whether the model is resident or cold. Before v0.47.3 the
gateway listed only loaded workers, so a freshly-booted lazy supervisor
reported an empty `/v1/models` until the first request warmed a worker
-- a doc-vs-code drift that hid the entire enabled-but-unloaded set from
clients.

Worker-reported `last_loaded_at` is filled in by the gateway, not the
worker (v0.47.4). Per-model workers run outside the supervisor and own no
`LoadDirector`, so their own `/v1/models` reports `last_loaded_at: null`
even for a resident model. The gateway's `_enrich_loaded_at` joins the
real load timestamp from `state.director.loaded[id].loaded_at` (snapshot
under the director lock, rendered via the shared `_format_loaded_at`)
onto each aggregated resident entry before appending the unloaded rows;
it never overwrites an already-non-null value. `/health` is broadened the
same way: it unions enabled-but-unloaded catalog models + modalities into
its `models`/`modalities` sets (via `_unloaded_catalog_entries`) so a
zero-worker lazy supervisor reports the full serviceable surface, not an
empty one.

Servability is re-checked live at request time, not frozen at boot.
Boot and the request-path re-check share ONE verdict function,
`_servability_reason(entry, ...)` (sizing ladder, then a device-capacity
check against live free memory), so the two can never drift.
`validate_catalog_at_boot` stamps `unservable_reason` once on startup;
`revalidate_servability` re-derives the full verdict on the first
request to a stamped model against the LIVE catalog + live free memory.
It clears the stamp only when the model is sizable AND fits: a probe or
weights landing after boot clears a "no memory estimate" stamp, and
freed memory clears a stale "exceeds device capacity" stamp -- both
without a supervisor restart. A GENUINE "exceeds device capacity" stamp
(the model cannot fit even an empty device) is RETAINED, so the gateway
503s `model_unservable` directly instead of routing an impossible model
into the director's eviction loop. So `muse models probe <id>` (which
writes a measurement to the catalog) takes effect on the *next request*.
The mtime-cached `_read_catalog` is the read path; a probe's catalog
write bumps the mtime so the re-check sees fresh measurements.

Lock discipline:

- `state.director.lock` (RLock) guards in-memory mutations of the
  loaded set, in_flight_loads, and the recent_decisions deque. Held
  during decision and commit phases of `acquire`; released around the
  long-running worker spawn / wait.
- Admin operations (`load_model_into_worker`, `disable_model`,
  `unload_model_from_worker`) plan their mutations under `state.lock`,
  release the lock before slow worker spawn or shutdown, then reacquire
  to commit. Same plan-then-execute pattern as the existing admin
  endpoints; lets `acquire` and admin endpoints share the runtime
  without serializing on slow I/O.
- `catalog._CATALOG_WRITE_LOCK` serializes the read-modify-write on
  `measurements` across observed-peak writebacks for *different* models
  (and every other catalog RMW), so catalog.json round-trips stay atomic.

### Idle eviction (v0.40.1+)

In addition to memory-pressure LRU eviction, models can declare a
per-model `capabilities.idle_timeout_seconds` in their manifest. A
background sweeper thread runs every `MUSE_IDLE_SWEEP_INTERVAL_SECONDS`
(default 30) and evicts loaded models whose `last_touched_at` exceeds
the timeout AND whose refcount is 0. This frees memory without
waiting for traffic-driven LRU.

A GLOBAL default idle timeout (`server.idle_timeout_seconds`, env
`MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS`, v0.51.0+) applies to any model that
declares no per-model `idle_timeout_seconds`. Precedence: per-model
`capabilities.idle_timeout_seconds` wins; otherwise the global default;
otherwise never idle-evict. As of v0.52.0 the global default is **600
seconds (10 minutes)**, not off -- an untouched model with refcount 0 is
reclaimed after 10 minutes by default. The value is resolved through the
config registry (lenient: a non-numeric env/file value logs a warning and
falls back to the 600s default, so a bad value cannot crash boot). This
is the knob for "reclaim any idle model after N minutes" on a shared
single-GPU box, where bundled models (which declare no per-model timeout)
would otherwise sit resident until memory pressure.

To DISABLE idle eviction entirely, set `server.idle_timeout_seconds` (or
`MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS`) to `0` (or any value `<= 0`); the
IdleSweeper's `<= 0` guard then never idle-evicts, and only memory
pressure releases a model. Per-model `capabilities.idle_timeout_seconds`
still overrides the global default in both directions.

The sweeper reuses the on-demand eviction's disable_fn primitive,
so the orphan-worker-on-disable-failure remediation (re-insert
LoadEntry) applies uniformly. Idle eviction is logged in
`recent_decisions` with reason="idle_timeout:Ns".

Configuration:
- Per-model: `capabilities.idle_timeout_seconds: <number>` in manifest. Null/absent = fall back to the global default.
- Global default: `server.idle_timeout_seconds` (env `MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS`), default **600** (10 min); set `<= 0` to disable. Applied to models without a per-model timeout.
- Sweep interval: `server.idle_sweep_interval_seconds` (env `MUSE_IDLE_SWEEP_INTERVAL_SECONDS`), default 30.

### Request queueing (v0.55.0+)

Two gateway-side mechanisms (spec docs/superpowers/specs/2026-07-08-request-queueing-design.md), both waiting ON the event loop (never in pool threads):

- **Per-model concurrency cap:** `capabilities.max_concurrency` (else `server.default_max_concurrency`, default 0 = unlimited = no gating). Excess requests park FIFO on a per-model asyncio.Semaphore in `muse.cli_impl.queueing.ConcurrencyGate`. The CPU box's 32B runs with `max_concurrency: 1`.
- **Capacity-wait:** the LoadDirector tags its capacity 503 `retryable=True` when the failure is TRANSIENT -- any of the three in-flight states holds: an in-use loaded model (refcount > 0) blocks eviction, another model's cold load is in flight (its reservation made the fit fail; v0.57.1), or an eviction is mid-flight (`director.evicting`: victim popped from `loaded` but disable_fn still shutting the worker down, so the director looks idle while VRAM is still occupied; v0.57.2). The gateway parks on `CapacityNotifier` (generation asyncio.Event, fired threadsafe by the director on release-to-zero, post-eviction, and load commit/abort) and retries under the shared deadline. Only a truly idle-and-empty device that still cannot fit the model 503s immediately.

A modality may export a literal `CONCURRENCY_CEILING = {"max_concurrency": n, "max_queue_depth": n}` (AST-read by `muse.core.discovery.concurrency_ceilings`, like `MODEL_OPTIONAL_PATHS`). The gateway clamps every model of that modality to it, so no manifest, catalog edit or `server.default_max_concurrency` can raise it; `acquire_slot(..., max_queue_depth=)` takes the smaller positive of the ceiling and `server.max_queue_depth`, so a global 0 does not remove it. The gate applies a LOWERED cap live (surplus permits are retired: free ones at once, held ones as their holders release, never over-issuing); a raised cap still applies on restart. decision/evaluation declares 1 / 32.

One budget covers both: `server.queue_timeout_seconds` (default 300; 0 = today's immediate-503). `server.max_queue_depth` (default 256; 0 = unbounded) fast-fails 503 `queue_full`. Timeout -> 503 `queue_timeout` (message reports actual elapsed wait). Telemetry `request` events carry `queued_ms`; `/v1/admin/memory` and `/v1/telemetry/summary` expose per-model `queue_depth` (the seam a future queue-aware federation router consumes). `/v1/admin/memory` omits queue_depth when no gate is bound (unknown is never rendered as 0), while `/v1/telemetry/summary` always includes it (0 when unbound). Cold-starting models with parked waiters (not yet loaded) are visible too (v0.57.4): the summary lists them in a separate always-present `queued` array, and the admin per-model breakdown includes them as rows marked `loaded: false`. `acquire_slot` returns whether a slot was actually taken and the gateway releases only on True, so a live `server.default_max_concurrency` change can no longer desync the gate's accounting.

## Observability

`muse.observability/` (v0.51.x) is a self-contained telemetry package: a sparse single-table
event model, a fire-and-forget recorder, a per-worker log ring buffer,
a periodic resource sampler, and a `/dashboard` HTML page backed by a
small set of gated JSON/SSE endpoints. It has no dependency on any
other modality; the rest of muse depends on it only through the thin
`record(...)` call.

Import-hygiene contract: `muse.observability/__init__.py` re-exports
`EVENT_COLUMNS`, `event_to_row`, `TelemetryStore`, `TelemetryRecorder`,
`record`, `get_recorder`, `init_recorder`, `reset_recorder`, `Sampler`,
and `LogHub` EAGERLY (all stdlib-only), but re-exports
`build_dashboard_router`, `DASHBOARD_HTML`, `require_dashboard_auth`,
and `check_dashboard_token` LAZILY via a PEP 562 module `__getattr__`,
because `dashboard.py` and `dashboard_auth.py` import fastapi and
sse_starlette at module top. The director, gateway, and supervisor all
import `muse.observability.recorder` from the hot request path, and
importing any submodule runs `__init__.py` first -- an eager dashboard
import there would drag fastapi into `muse --help` / `muse pull`,
violating the project's deferred-imports convention. Regression guard:
`tests/observability/test_public_api.py` re-imports `recorder` in a
clean subprocess and asserts `"fastapi" not in sys.modules`.

**Event model.** One sparse SQLite table, `events`
(`muse.observability.store.TelemetryStore`), with columns
`EVENT_COLUMNS` (`muse.observability.events`): `ts`, `type`, plus every
field any event type might carry (`model_id`, `pool`, `gb`,
`latency_ms`, `queued_ms`, `status`, `reason`, `cold_load_seconds`,
`stream`, `free_vram_gb`, `free_ram_gb`, `loaded_count`,
`in_flight_count`, `modality`) -- unset fields are `NULL`, not one
table per event type. `event_to_row(type, ts, **fields)` builds the
full row and raises `ValueError` on an unknown field name (a typo'd
kwarg fails loud instead of silently writing `NULL` to the wrong
column). Four event types are recorded today: `model_load` and
`model_evict` (from `LoadDirector.acquire` / eviction, `pool` +
`gb`/`cold_load_seconds`), `request` (from the gateway's forwarding
path, `latency_ms`/`status`/`stream`/`modality`), and `sample` (from
`Sampler`, a periodic snapshot of `free_vram_gb`/`free_ram_gb`/
`loaded_count`/`in_flight_count`, where the supervisor feeds
`in_flight_count` the number of in-flight COLD LOADS). The retired
`gpu_used_gb` column was never written; old databases keep it as a
nullable column and inserts name their columns, so they still work. `TelemetryStore.series(metric,
since_ts, bucket_seconds)` buckets rows into a fixed set of named
metrics (`request_rate`, `latency`, `vram`, `ram`, `load_evict`); v1
latency is avg+max per bucket, not exact percentiles.

**Fire-and-forget recording.** `muse.observability.recorder.record(type,
**fields)` is meant to be called from every hot path (director,
gateway) and must never block or raise. `TelemetryRecorder` enqueues
onto a bounded `queue.Queue` (`max_queue=10000`) and a daemon flush
thread batches inserts into the store every `flush_interval` (0.5s
default); when the queue is full the event is dropped and
`recorder.dropped` increments rather than blocking the caller --
`/v1/telemetry/summary` surfaces the running drop count so an operator
can tell if the recorder is falling behind. Every call site
(`load_director.py`, `gateway.py`) wraps `record(...)` in its own
try/except so a telemetry regression can never break a real model load,
eviction, or request. `init_recorder(store, enabled=...)` /
`get_recorder()` / `reset_recorder()` manage a module-level singleton;
`enabled=False` (or `telemetry.enabled: false`) swaps in a
`_NoopRecorder` so a disabled deployment pays no queue/thread cost at
all.

**Log capture.** Each worker's stdout is piped into a per-model
`LogHub` (`muse.observability.logs`) ring buffer, byte-bounded by
`telemetry.log_buffer_kb` (KB, not lines), when telemetry is enabled;
`spawn_worker(..., log_hub=state.log_hub)` in the supervisor starts a
daemon reader thread (`_pump_worker_logs`) that tees each line to both
the aggregate supervisor stdout and `hub.append(model_id, line)`. When
telemetry is disabled `log_hub` stays `None` and workers spawn exactly
as before (no reader thread, no behavior change). `LogHub` also fans
out live lines to subscriber queues for the SSE tail;
`subscribe_with_snapshot(id)` returns the buffered history and a live
queue atomically for a late-connecting client (`snapshot(id)` is a plain
copy kept for tests and diagnostics).

**`/dashboard` + telemetry endpoints.** `build_dashboard_router(state)`
(lazily imported, see above) mounts:
- `GET /dashboard` -- a single self-contained HTML/CSS/JS page
  (`DASHBOARD_HTML`), UN-GATED so it always loads (even with no token
  configured yet) and prompts the browser for a token before hitting
  any gated endpoint, storing it in `sessionStorage`.
- `GET /v1/telemetry/summary` -- gated JSON: currently loaded models
  (`model_id`/`pool`/`gb`/`last_used`/`queue_depth`), cold-start
  `queued` models, `in_flight` (in-flight REQUESTS: director refcounts
  plus gate-parked waiters; federation balances on it. It excludes
  requests waiting on a cold load or a capacity wait, which hold neither
  a refcount nor a gate queue slot),
  `in_flight_loads` (cold loads in progress),
  `dropped_events`, and a federation-forward `node` id (`state.node_url`
  or `state.node_id` if set, else `socket.gethostname()`) so a future
  multi-node aggregator can tell which box a summary came from.
- `GET /v1/telemetry/series?metric=...&window=...` -- gated JSON time
  series from `TelemetryStore.series`; unknown `metric` returns 400
  `invalid_metric`.
- `POST /v1/telemetry/logs-ticket` -- gated (header): mints a
  short-lived ticket (`muse.observability.log_tickets.LogTicketStore`)
  the dashboard exchanges for SSE access without putting the admin
  token in a URL.
- `GET /v1/telemetry/logs/{model_id}` -- SSE tail, authenticated INSIDE
  the handler (not via the `require_dashboard_auth` dependency, since
  `EventSource` sends no custom headers): a valid `?ticket=` OR an
  `Authorization: Bearer <token>` header (for curl-style clients)
  is accepted; `?access_token=<token>` is NOT. Once authenticated, it
  takes the buffered history and a subscription atomically
  (`hub.subscribe_with_snapshot`, so no line falls between them), then
  polls every 250ms until the client disconnects, unsubscribing in a
  `finally` so a disconnect/cancel/exception never leaks a subscriber
  queue.

**Auth.** `muse.observability.dashboard_auth.check_dashboard_token`
mirrors the admin API's closed-by-default policy: with no `admin.token`
configured, every gated endpoint 503s `dashboard_closed` (same token as
`MUSE_ADMIN_TOKEN`/admin API -- there is no separate dashboard secret).
The admin token is HEADER-ONLY everywhere: a request needs it as
`Authorization: Bearer <token>`; there is no `?access_token=` query-param
fallback. The one place a header alone cannot work is the SSE log
stream (`EventSource` cannot set custom headers), which instead
authenticates with a short-lived, random ticket
(`telemetry.log_ticket_ttl_seconds`, default 60s) minted from the
header-gated `POST /v1/telemetry/logs-ticket`. The ticket is reusable
within its TTL (not single-use, so `EventSource` auto-reconnect inside
the TTL keeps working; a reconnect after it 401s, the browser closes the
stream, and the dashboard JS mints a new ticket and reopens it with
backoff) and unscoped (it authorizes the logs SSE surface generally,
matching the admin token's own all-or-nothing scope). So the admin
token itself never rides a URL, and a leaked ticket in an access log
is useless once its TTL elapses. The check is `muse.admin.auth.check_bearer`
(shared with the admin API: case-insensitive `Bearer` scheme, exact
credential, `secrets.compare_digest`); the token is never echoed in an
error message or log line.

**Config.** Five `telemetry.*` settings (see "Configuration" above and
`docs/CONFIG.md`): `telemetry.enabled` (`MUSE_TELEMETRY_ENABLED`,
default `true`), `telemetry.retention_days` (`MUSE_TELEMETRY_RETENTION_DAYS`,
default `7`), `telemetry.log_buffer_kb` (`MUSE_TELEMETRY_LOG_BUFFER_KB`,
default `64`), `telemetry.sample_interval_seconds`
(`MUSE_TELEMETRY_SAMPLE_INTERVAL_SECONDS`, default `10.0`),
`telemetry.log_ticket_ttl_seconds`
(`MUSE_TELEMETRY_LOG_TICKET_TTL_SECONDS`, default `60.0`). The
supervisor wires the store, recorder, sampler, and log hub together
once at boot (`_init_telemetry`, gated on `telemetry.enabled`) and tears
them down symmetrically on shutdown (`sampler.stop()` joins the sampler
thread; `store.close()` closes the sqlite connection).

### Eviction telemetry

- **`model_evict` covers every director eviction** (memory-pressure LRU,
  idle sweep, admin disable) and carries `pool` and `gb`; it is emitted from
  `finish_eviction` once the teardown succeeds, so a failed disable records
  no event (it logs a `disable_fn_raised` decision instead).

## Federation

`muse federate` (see
`docs/superpowers/specs/2026-07-04-federation-design.md` for the full
design) runs a thin coordinator process that fronts a static list of
unmodified muse `serve` nodes so a client sees "one server with all the
GPUs" instead of N separate boxes. This is the second sub-project of
the federation arc; it consumes the per-node observability surface
(`/v1/models`, `/v1/telemetry/summary`) shipped in v0.53.0.

```
                 clients (OpenAI SDK, curl, ...)
                          |
                 [ muse federate ]  :8100        <- new coordinator
                  /       |       \
             node A     node B     node C         <- unmodified muse servers
```

Nodes need NO code changes: they are plain `muse serve` instances. The
coordinator lives in `muse.federation` (node-membership model, async
poll-and-cache registry, pure model-locality router) plus
`muse.cli_impl.federation` (the FastAPI app + `run_coordinator`,
structurally a sibling of the existing single-host gateway in
`muse.cli_impl.gateway`, reusing its `_forward` + SSE-relay machinery
and `extract_model_from_request`).

**Node membership (static, v1).** Nodes come from two merged sources,
CLI entries winning on url collision. A node's default name is
`host:port` (the scheme's port when the url has none), so two nodes on
one host differ; duplicate names are rejected and `muse federate` exits 2
on an invalid node list:
- `--node http://host:8000` or `--node name=http://host:8000` (repeatable).
- A yaml file, `nodes: [{url, name?, token?}, ...]`. Default location is
  `<catalog_dir>/federation.yaml` when it exists and neither `--config`
  nor `federation.config_file` names an explicit path. The optional
  per-node `token` is used ONLY by the coordinator's own outbound poll
  to read that node's gated `/v1/telemetry/summary`; it is never
  exposed to clients. Dynamic membership (registration, gossip,
  health-based ejection) is deferred to v2.

**Node-state refresh.** `muse.federation.registry.NodeRegistry` polls
every node concurrently on a background interval
(`federation.refresh_interval_seconds`, default 3.0s) and caches a
`NodeState` snapshot per node: `GET <node>/v1/models` (reachability +
per-model `loaded`) and (only if that node has a
configured token) `GET <node>/v1/telemetry/summary` for `in_flight`
(the node's in-flight requests).
Route handlers read the cache via `.snapshot()`, so routing is a fast
local lookup, never a per-request fan-out. A node whose poll fails is
marked unreachable and skipped by the router until it recovers
(per-node failure isolation: one bad node never aborts the refresh for
the others). A connect failure while forwarding also marks the node
unreachable in the cache (`mark_unreachable`) until the next poll, which
hides ALL of that node's models from routing, `/v1/models` and `/health`
until that poll succeeds. The
registry remembers each node's model list from its last reachable poll
(`was_listed`).

**Routing policy (model-locality, v1).** `muse.federation.router.select_node`
is a pure function over the cached snapshot:
1. Candidates = reachable nodes whose `/v1/models` lists the requested
   model.
2. Prefer loaded: among candidates, those with the model currently
   `loaded` win over those that would have to cold-load it.
3. Tie-break: lowest `in_flight` (in-flight requests from the node's
   telemetry, if a token is configured for that node), else round-robin
   over the tied set. An unknown `in_flight` (no token) is "no
   information", not "infinitely busy": it ties with the lowest known
   value, so a token-less node is never starved.
4. No candidate -> 404 `model_not_available` (OpenAI-shape), except a
   retryable 503 `no_node_available` before the first poll lands, when no
   node is reachable, or when a now-unreachable node listed the model in
   its last reachable poll.

Deferred to v2: hardware-aware routing (route a big model to the node
whose GPU actually fits it), true queue-depth/latency-aware routing,
sticky sessions, an aggregated cluster dashboard (v1 links out to each
node's own `/dashboard` via `/v1/federation/nodes`), and
coordinator-side inference auth (v1's inference routes are open,
mirroring an unmodified node's own gateway; per-node tokens are used
only for the coordinator's outbound telemetry poll, never exposed to
clients).

**Request forwarding + failover.** The catch-all proxy extracts `model`
from the request body/query (the same logic the single-host gateway
uses), asks `select_node`, and forwards via the shared `_forward`
helper (buffered + SSE relay, unchanged). A forward that fails at
connect time (`ConnectError`/`ConnectTimeout`; never mid-stream -- once
a `StreamingResponse` starts, the relay is on its own) retries ONCE
against a different candidate for the same model; if no other candidate
exists, or the retry also fails, the coordinator returns 502
`no_node_available`. A read timeout (the node accepted the request) is a
504 `node_timeout` without failover.

**Aggregated read endpoints**, mounted before the catch-all so they are
never shadowed:
- `GET /v1/models` -- OpenAI-shape union of model ids across reachable
  nodes, each annotated with `loaded` (true if loaded on ANY node) and
  `nodes` (which node names carry it).
- `GET /v1/capabilities` -- the nodes' discovery documents merged, each
  document matched to its node by position (not name).
- `GET /health` -- `"ok"` if at least one node is reachable, else
  `"degraded"`; per-node reachability/model-count/in_flight breakdown
  in the body.
- `GET /v1/federation/nodes` -- operator view: each node's url/name,
  reachable state, loaded model list, in_flight (if known), and
  last-poll age in seconds.

**Config** (four settings under the `federation.*` group in
`muse.core.config.SETTINGS`; see "Configuration" above and
`docs/CONFIG.md`):
- `federation.refresh_interval_seconds` (`MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS`,
  default `3.0`) -- seconds between background polls of each node.
- `federation.forward_timeout_seconds` (`MUSE_FEDERATION_FORWARD_TIMEOUT_SECONDS`,
  default `300.0`) -- per-request timeout when forwarding to a node
  (long, since generation requests can run for minutes).
- `federation.poll_timeout_seconds` (`MUSE_FEDERATION_POLL_TIMEOUT_SECONDS`,
  default `10.0`) -- per-node timeout for each
  background poll (above a node's own aggregation timeout, so a briefly
  slow node is not dropped from routing).
- `federation.config_file` (`MUSE_FEDERATION_CONFIG`, default unset) --
  explicit path to the node-list yaml, overriding the
  `<catalog_dir>/federation.yaml` default-if-exists lookup.

**CLI:** `muse federate --port 8100 --node http://192.168.0.204:8000
--node http://192.168.0.50:8000` (or `--config path/to/federation.yaml`).
`run_coordinator` (`muse.cli_impl.federation`) resolves the node list
(`--config` > `federation.config_file` > `<catalog_dir>/federation.yaml`
if it exists), and refuses to start a server -- exiting 2 with a clear
stderr message -- if the merged node list is empty; a coordinator with
zero nodes would otherwise boot healthy and 404 every request, a worse
failure mode than refusing outright. Once nodes resolve, it builds a
`NodeRegistry` + the coordinator app and blocks on
`muse.cli_impl.serve_util.run_uvicorn` (the same bounded-graceful-
shutdown uvicorn wrapper `muse serve` uses). Registry lifecycle is
wired through the coordinator app's ASGI lifespan (an
`asynccontextmanager` passed to `FastAPI(..., lifespan=...)` inside
`build_coordinator`): `registry.start()` runs on startup, launching the
background refresh loop on the SERVER'S OWN running event loop (which
`NodeRegistry.start`'s `asyncio.ensure_future` requires), and
`await registry.aclose()` runs on shutdown. This only fires when the
ASGI lifespan protocol actually executes (a real uvicorn run, or a
`with TestClient(app) as c:` block); a bare `TestClient(app).get(...)`
(the style the existing route tests in `tests/cli_impl/test_federation_app.py`
use) never triggers lifespan, so wiring it into `build_coordinator`
did not disturb those tests' plain `.snapshot()`-only fakes.

Client usage is unchanged OpenAI-compat, just pointed at the
coordinator: `OpenAI(base_url="http://coordinator:8100/v1",
api_key="not-used")`.
