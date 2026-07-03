# Video CPU offload for large-pipeline T2V models (v0.52.1)

**Goal:** Make `wan2-1-t2v-1-3b` (and other diffusers video pipelines) load and
generate on ~8-12 GB GPUs by supporting diffusers CPU offload, and correct the
wan memory annotation / docs that ignored the ~11 GB UMT5-XXL text encoder.

## Problem

`WanRuntime` moves the ENTIRE pipeline to CUDA
(`self._pipe = self._pipe.to(self._device)`). Wan2.1-T2V-1.3B bundles a
UMT5-XXL text encoder (~11 GB fp16) plus the 1.3B transformer plus a VAE, so the
full-resident load is ~11.5 GB and OOMs a 12 GB card (confirmed on frodo: the
pipeline loads all 5 components, then dies allocating the last 80 MiB with
23 MiB free). The bundled manifest declares `memory_gb: 6.0` and the docstring
says "~3GB / fits 8GB" -- both count only the 1.3B transformer and are wrong.

## Design

CPU offload is a general diffusers mechanism (`DiffusionPipeline.enable_model_cpu_offload`
/ `enable_sequential_cpu_offload`, backed by `accelerate`), so it belongs in the
runtime, exposed as a per-model capability (whether to use it is per-model:
pointless for small pipelines, essential for wan).

### Capability: `cpu_offload` (mode) + `vae_tiling` (bool)

`capabilities.cpu_offload` accepts a MODE, not a bool:
- `false` / absent -> `self._pipe.to(device)` (today's behavior; unchanged for
  every model that does not set it).
- `"model"` -> `self._pipe.enable_model_cpu_offload(device=...)`, and do NOT
  call `.to(device)` (the two are mutually exclusive; offload manages placement).
  Whole-component granularity; peak VRAM ~ largest single component.
- `"sequential"` -> `self._pipe.enable_sequential_cpu_offload(device=...)`,
  no `.to()`. Sub-module granularity; fits <=12 GB, slower.

`capabilities.vae_tiling: true` -> call `self._pipe.enable_vae_tiling()` (and
`enable_vae_slicing()` when present) after placement, to cap the VAE-decode
spike at higher resolutions. Best-effort: guarded by `hasattr`.

Offload only applies when the resolved device is CUDA; on CPU the pipeline
stays on CPU (the existing `device != "cpu"` guard). `accelerate` is already in
wan's `pip_extras`.

Same dispatch is added to `CogVideoXRuntime` (identical `.to(device)` shape).

### Global override: `server.video_cpu_offload`

New registry setting `server.video_cpu_offload` (opt_str, default None). The
runtime resolves the effective mode as:

    config.get("server.video_cpu_offload")  # operator override, if set
    or capabilities.get("cpu_offload")       # per-model default
    or False

So an operator can force a mode across all video models without editing a
manifest: `muse config set server.video_cpu_offload sequential` (or `model`, or
`off`). The literal string `"off"`/`"false"`/`"none"` resolves to no offload.
This dogfoods the v0.52.0 config registry.

### Bundled wan defaults

`wan2-1-t2v-1-3b` manifest:
- `capabilities.cpu_offload: "sequential"` -- guarantees load+generate on
  8-12 GB out of the box (the bundle's stated identity). A bigger-card user who
  wants speed sets `server.video_cpu_offload model` (or `off`).
- `capabilities.vae_tiling: true`.
- `capabilities.memory_gb: 3.0` -- honest sequential-peak estimate; the probe
  self-heals it to the measured value on first load.
- Docstring + `description`: drop "~3GB / fits 8GB"; state the UMT5-XXL encoder
  makes the full-resident footprint ~11-12 GB, hence sequential offload by
  default.

## Acceptance

- `WanRuntime` / `CogVideoXRuntime`: with `cpu_offload="model"` the pipeline
  calls `enable_model_cpu_offload` and NOT `.to(cuda)`; with `"sequential"` it
  calls `enable_sequential_cpu_offload` and NOT `.to(cuda)`; with `false`/absent
  it calls `.to(cuda)` as before. `vae_tiling=true` calls the tiling helpers
  when present. All unit-tested with a mock pipe (assert which method was
  called, and that `.to(cuda)` and offload are never both called).
- The global override `server.video_cpu_offload` beats the manifest capability;
  `off`/`false`/`none` disables; unset falls through to the capability.
- Bundled wan manifest declares `cpu_offload: "sequential"`, `vae_tiling: true`,
  a corrected `memory_gb`, and a corrected docstring/description.
- Full fast lane green; the existing video_generation tests still pass (models
  without `cpu_offload` are unchanged).
- CLAUDE.md video-generation section corrected (the "~3GB / fits 8GB" wan claim
  and the offload knobs documented).

## Out of scope

- A `muse models set-capability` CLI (per-model capability editing) -- the
  global `server.video_cpu_offload` override covers the immediate need; the
  per-model CLI is a separate future item.
- 8-bit / fp8 text-encoder quantization (a further VRAM reduction; deferred).
- LTX / Mochi / Hunyuan dedicated runtimes (already deferred).
