# GPU-Layers Operator Pin Design (`muse models set-gpu-layers`)

- **Date:** 2026-07-08
- **Status:** approved design, pre-implementation
- **Ships as:** v0.56.0

## Problem

A GGUF model larger than free VRAM cannot run GPU-accelerated on muse today
even though llama.cpp natively supports splitting layers between GPU and
CPU (`n_gpu_layers`). `LlamaCppModel` already accepts and forwards
`n_gpu_layers` (default `-1` = offload everything the GPU fits), and
manifest `capabilities.n_gpu_layers` already flows into the constructor
via the load_backend kwargs merge -- but there is no operator control:
pinning a split requires hand-editing catalog.json.

Concrete target: the ~20 GB Qwen2.5-32B q4 GGUF on frodo's 12 GB card.
`set-gpu-layers 30` puts ~half the layers on the GPU for a predictable
speedup over pure-CPU, statically, per explicit operator pin.

This is the Tier-1 scope agreed after evaluating automatic VRAM-overflow
offload: static, declared, zero runtime decisions. Explicitly REJECTED
(Tier 2+): the director choosing offload automatically at admission time
(silent performance cliffs, dual-pool accounting, contention policy).

## Design (clone the `set-device` pattern)

### 1. Catalog field + setter

New TOP-LEVEL catalog entry field `gpu_layers_override: int` (NOT in the
manifest -- operator state, mirroring `device_override`). New
`muse.core.catalog.set_gpu_layers_override(model_id: str, n: int | None)`
mirroring `set_device_override`: `None` pops the field, else writes it;
atomic write-then-rename; mtime bump makes the next cold load see it.

### 2. Load precedence

In the same place `load_backend` resolves `device_override`, resolve
`n_gpu_layers`, most authoritative first:

1. catalog `gpu_layers_override` (operator pin, read live)
2. manifest `capabilities.n_gpu_layers` (model author; already merged
   into kwargs today -- the pin must WIN over this merge)
3. runtime default (`-1` in `LlamaCppModel`)

### 3. CLI verb

`muse models set-gpu-layers <id> <N>` and
`muse models set-gpu-layers <id> --clear`, sibling of `set-device` in
`cli.py` + a thin `cli_impl` handler:

- `N` is an int `>= -1`: `-1` = all layers on GPU, `0` = pure CPU,
  `N > 0` = first N layers on GPU, rest on CPU. Validate; reject other
  values with exit code 2.
- REFUSES a model whose catalog manifest lacks `capabilities.gguf_file`
  (non-GGUF models ignore the kwarg; an honest error beats a silent
  no-op pin). Exit 2 with a message naming the constraint.
- Takes effect on the model's NEXT COLD LOAD (evict or restart to
  apply) -- same documented semantics as `set-device`.

### 4. Probe honors the pin

`probe_worker` reads `gpu_layers_override` exactly where it reads
`device_override` and passes it into the backend construction, so
`muse models probe <id>` after pinning measures the SPLIT's real VRAM
peak. This matters: a split model uses less VRAM than its whole-weights
estimate, and the probe measurement is what corrects admission sizing
downward (the observed-peak writeback only self-heals UPWARD).

### 5. Surfacing

`muse models info <id>` shows the pin beside the device pin (e.g.
`gpu layers: 30 (operator pin)`), omitted when unset.

## Known limitation (stated, accepted)

A split model occupies BOTH pools (part VRAM, part host RAM). The
director still accounts it against its resolved device's single pool
(VRAM when cuda). The probe makes the VRAM side honest; the host-RAM
share stays unaccounted -- the same simplification as every GPU model's
host-side overhead today. Dual-pool accounting is out of scope (Tier 2+,
deliberately skipped).

## Testing

- catalog: set/clear round-trip, atomic write, unknown model error --
  mirror the `set_device_override` tests.
- load precedence: pin beats capability beats default (unit, fake
  backend capturing kwargs).
- CLI: valid N, -1, 0, junk value rejected, --clear, non-GGUF refusal,
  exit codes -- mirror `test_set_device_cli.py`.
- probe_worker: pin threaded into construction kwargs (unit).
- models info: pin rendered / omitted.

## Rollout

v0.56.0. Backward compatible: no field set = today's behavior exactly.
Post-deploy validation on frodo: pin the 32B (pull it there first) to
~30 layers, probe, then one chat request -- expect tok/s well above the
CPU box's ~1.3.
