# Lazy-load servability fixes (v0.47.3)

Date: 2026-06-30
Status: approved (user: "go with your recommendation", ultracode)

## Context

Diagnosing a running muse server (.204) surfaced confusion about
`enabled_unloaded` models, then three real defects, all stemming from one
root cause: **the lazy-load runtime reads `capabilities.memory_gb`, but
`muse models probe` writes `measurements.<device>.peak_bytes`.** The two
are never reconciled on the request path.

All three reproduced live on a CPU server (muse 0.47.1).

## The three defects

### Bug #2 (primary): `muse models probe` ignored until `muse serve` restart

`validate_catalog_at_boot` (supervisor.py:583) stamps
`state.unservable_reasons[model_id] = "no memory estimate..."` once at
boot for every enabled model lacking a memory estimate. The gateway's
`_route_via_director` (gateway.py:349-360) reads that boot snapshot and
returns 503 `model_unservable` BEFORE calling the director. Running
`muse models probe` writes `measurements.<device>` to catalog.json on
disk, but the in-memory snapshot is never refreshed, so the model stays
503 until the supervisor restarts.

### Bug #1: `/v1/models` omits enabled-but-unloaded models

The gateway's `list_models` (gateway.py:214-232) only unions the
`/v1/models` of currently-loaded workers. With lazy-load + zero resident
workers, `/v1/models` is `{"data": []}` even with many enabled models.
This contradicts CLAUDE.md ("enabled-but-unloaded models appear with
`loaded: false`") and is why the catalog is invisible over HTTP until a
model is summoned.

### Gap #2b (discovered): director sizes probed-only models at 0.0 GB

`get_manifest` does not backfill `memory_gb` from `measurements`, and the
director's `_decide` (load_director.py:559) sizes a load from
`manifest.capabilities.memory_gb` only. A model whose size exists only as
a probe measurement is sized at 0.0 -> "fits anywhere", never triggers
eviction. Verified live: `all-minilm` loaded with `memory_gb=0.0` while
its catalog measurement was 0.753 GB. Latent over-commit / OOM risk on
memory-tight hosts; defeats the v0.40.0 "live accounting" intent.

## Fix A (added per user clarification): on-disk weights size fallback

User intent: "if a model is available but unloaded, it should be usable
through the REST API (just needs to load, which may mean it needs to
unload other models to make room)." The `model_unservable` "no memory
estimate" 503 violates this -- a never-probed model refuses to load
rather than loading + evicting.

Resolution: a single SIZING LADDER, used by boot validation, the
request-path re-check, and director sizing alike:

    capabilities.memory_gb (declared)
      > measurements.<device>.peak_bytes (probe / self-healed)
      > on-disk weights size summed from the catalog entry's local_dir

`_has_memory_data` is extended to consult the weights-size fallback, so
every PULLED model (weights on disk) is sizable -> servable -> loads on
demand with eviction. The "no memory estimate" 503 effectively disappears
for pulled models; the only remaining hard 503 is the director's
`model_too_large_for_device` (genuinely too big even after evicting
everything). The observed-peak writeback self-heals the estimate upward
after the first real load, so an initial weights-size under-estimate
(weights < live runtime) is corrected on the next load; a crashed worker
from an under-estimate is recovered by the existing auto-restart monitor.

New helper `_weights_size_gb(entry) -> float`: sums regular-file sizes
under `entry["local_dir"]` (follows HF snapshot symlinks via getsize),
returns 0.0 on missing/unreadable. Pure read; no caching (catalog small).

## Design

A single per-model servability/sizing seam, reused by boot and the
request path. New functions in `cli_impl/supervisor.py` (beside
`_has_memory_data` + `validate_catalog_at_boot`):

- `revalidate_servability(state, model_id) -> str | None` (Bug #2):
  re-reads the live (mtime-cached) catalog for one model, recomputes the
  "no memory estimate" condition via the existing `_has_memory_data`, and
  updates `state.unservable_reasons` under `state.lock`. Returns the
  current reason (None = now servable). Scoped to one model; does NOT
  re-walk the catalog or re-probe memory. A model that gained an estimate
  is cleared and allowed through to the director, which performs the live
  fit/eviction decision (proven safe: a no-estimate model is gated
  upstream because `memory_gb=0.0` would bypass the director's fit-check;
  a GPU model with `gpu_free=None` resolves to `available=0` -> safe 503,
  no worker spawn).

- `backfill_manifest_memory(manifest, model_id) -> dict` (Gap #2b): when
  `capabilities.memory_gb` is absent, set it from the catalog measurement
  (`_has_memory_data` returns declared-or-measured GB). Declared value
  always wins. Returns a copy; never mutates the input.

Gateway `_route_via_director`:
1. Read `state.unservable_reasons[model_id]`. If set, call
   `revalidate_servability` (live re-check). If still set, 503.
2. `get_manifest` (unchanged; patchable). KeyError -> 404.
3. `backfill_manifest_memory(manifest, model_id)` before `acquire`.

Gateway `list_models` (Bug #1): after aggregating loaded workers,
enumerate enabled+`python_path` catalog rows not already present and
append entries via a shared `build_model_entry(...)` helper extracted
from `core/server.py` (loaded=False, last_loaded_at=None,
unservable_reason from `state.unservable_reasons`). Skipped when
`routes_state is None` (legacy static-routes test mode).

`build_model_entry` is extracted into `core/server.py` and reused by both
`core/server.py:list_models` and the gateway (no parallel entry shapes).

## Test plan (TDD, red first)

- `test_supervisor_lazy.py`: `revalidate_servability` clears stamp when a
  measurement/annotation appears; keeps it when still absent; no-op when
  model absent from catalog. `backfill_manifest_memory` sets memory_gb
  from measurement; declared wins; copy semantics.
- `test_gateway_lazy.py`: request to a stale-unservable model whose
  catalog now has an estimate proceeds to `acquire` (no 503); request to a
  still-no-estimate model 503s without acquire (existing contract, made
  deterministic via patched `_read_catalog`); `acquire` receives a
  manifest sized from the measurement; `/v1/models` lists enabled-unloaded
  entries with loaded=false.
- `tests/core/test_server.py`: `build_model_entry` shape; existing
  /v1/models entry tests still green after extraction.
- `test_e2e_lazy_supervisor.py` (slow): probe-after-boot makes a model
  servable WITHOUT restart (seed unservable catalog -> boot stamps it ->
  first request 503 -> rewrite catalog.json with a measurement -> second
  request proceeds to acquire). Mirrors the live reproduction exactly.

## Post-review correction (adversarial review, ultracode)

An adversarial review of the diff found one HIGH regression, corroborated by
all four review dimensions: the first cut of `revalidate_servability` cleared
ANY stamp once a model was sizable -- INCLUDING the "exceeds device capacity"
stamp. That routed a request for a model larger than total device memory into
the director, whose eviction loop tears down the entire idle working set
before discovering the model can never fit and 503'ing -- and repeats the wipe
on every retry (the cleared stamp never short-circuits again). At lazy boot the
working set is empty, so a capacity stamp means "does not fit even empty" =
genuinely too big; deferring it to the director can only ever wipe + 503.

Fix: a SINGLE servability seam used by boot AND the request-path re-check, so
the two verdicts cannot drift:

- `_available_pools(probe, *, gpu_headroom_gb, cpu_headroom_gb)` -> live
  `(cpu_available_gb, gpu_available_gb|None)`.
- `_servability_reason(entry, *, cpu_available_gb, gpu_available_gb)` -> the
  full verdict string or None. Sizing ladder, then the device-capacity check.
- `validate_catalog_at_boot` and `revalidate_servability` both call it.

`revalidate_servability` now re-derives the FULL verdict (estimate AND
capacity) against live free memory. It clears the stamp only when the model is
sizable AND fits; a genuine capacity stamp is RETAINED, so the gateway 503s
`model_unservable` directly and the request never reaches the director's
eviction loop. Because the re-check reads LIVE free memory (not the stale boot
snapshot), it also correctly clears a capacity stamp once memory frees up --
the legitimate "load later" path -- without ever deferring an impossible model.

This preserves the user directive (models that CAN fit, possibly after
eviction, load on demand) while a model that cannot fit even an empty device
gets a clean 503 with no collateral eviction.

Known pre-existing follow-up (NOT introduced here, left out of scope): the
director's `_evict_lru_until_fits` does not filter eviction candidates by
device, so a GPU-pressure load could evict idle CPU models (which frees no GPU
memory) before reaching a GPU victim. The capacity guard above keeps this from
being reachable for impossible models; the device-filtering refinement is its
own task.

## Out of scope

- `last_loaded_at` rendering for loaded entries (separate minor).
- Restructuring `unservable_reasons` into a typed enum (bigger refactor).
- Device-aware eviction candidate filtering in the director (pre-existing).

## Release

v0.47.3, per the established muse ritual (preflight -> build -> wheel
smoke -> bump/tag/push/GitHub release/PyPI), gated on explicit user go.
