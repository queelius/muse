# LoRA Adapter Support for image/generation: Design

**Date:** 2026-07-02
**Status:** Approved (user-reviewed section by section)
**Target:** v0.50.0

## Goal

`muse pull hf://nerijs/pixel-art-xl` turns a LoRA adapter repo into a servable
image/generation model: muse pairs the adapter with a base diffusers pipeline,
loads it via `pipe.load_lora_weights(...)`, and serves it through the existing
`/v1/images/generations` route with an optional per-request `lora_scale`.

## Decisions (user-approved)

1. **Base pairing: declared base, overridable.** Default to the base repo the
   adapter's HF tags declare; `muse pull <lora> --base <muse-id-or-hf-repo>`
   and curated `capabilities.base_model` override it. The curated
   `pixel-art-xl` entry ships pre-paired with `sdxl-turbo` (SDXL LoRAs are
   architecture-compatible with SDXL-Turbo; 1-4 steps vs 25).
2. **`lora_scale` is per-request** with a per-model default (1.0). Requires
   the adapter to stay unfused.
3. **v1 scope: one LoRA per catalog entry.** Multi-LoRA composition,
   per-request adapter switching, and per-request scale on the edit/variation
   routes are out of scope.
4. **Approach A: extend the existing plugin + runtime.** No new modality, no
   new runtime class. A LoRA'd pipeline IS a text-to-image pipeline; it
   inherits routes, codec, img2img/inpaint/variations, and capability gating.

## Ground truth (verified 2026-07-02 via HF API)

`nerijs/pixel-art-xl`: `pipeline_tag=text-to-image`; tags include `lora`,
`base_model:stabilityai/stable-diffusion-xl-base-1.0`, AND
`base_model:adapter:stabilityai/stable-diffusion-xl-base-1.0`; siblings are
just `pixel-art-xl.safetensors` + README. No `model_index.json`.

## 1. Resolver plugin (`src/muse/modalities/image_generation/hf.py`)

**Sniff** accepts a second shape (in addition to the current diffusers-t2i
shape):

- NO `model_index.json` sibling, AND
- at least one `*.safetensors` sibling, AND
- text-to-image signal: `pipeline_tag == "text-to-image"` or `"text-to-image"`
  in tags, AND
- LoRA signal: `"lora"` in tags OR any tag starting with `base_model:adapter:`.

**Resolve** for the adapter shape:

- Base repo extraction: first tag matching `base_model:adapter:<repo>`;
  fall back to a plain `base_model:<repo>` tag that is not itself an
  `:adapter:`/`:finetune:` qualified form. A `--base` override (threaded
  through `pull()` / `expand_curated_pull` the same way `modality_override`
  is today) wins over both.
- No base derivable and no override: **pull fails** with an actionable
  message ("repo declares no base model; re-run with
  `muse pull <id> --base <muse-id-or-hf-repo>`"). No guessing.
- Adapter weight file: exactly one top-level `*.safetensors` means automatic
  selection. More than one: pull fails with an actionable message listing
  the files (a `--weight-name` flag can lift this later; most adapter repos
  ship one).
- Manifest: `model_id` from repo name (`pixel-art-xl`), modality
  `image/generation`, `hf_repo` = adapter repo, runtime path unchanged
  (`DiffusersText2ImageModel`). Capabilities:
  - `lora_adapter: true`
  - `base_model: <resolved base, a muse id or HF repo>`
  - `lora_scale: 1.0` (default strength)
  - `default_size` / `default_steps` / `default_guidance` derived by running
    the existing `_infer_defaults()` against the BASE id, so a turbo pairing
    automatically gets steps=1 / guidance=0.
  - `supports_negative_prompt` / `supports_seeded_generation` /
    `supports_img2img` / `supports_inpainting` / `supports_variations`: true
    (the base pipeline provides them; the adapter rides along via shared
    components).
- Download: snapshot of the ADAPTER repo only, allow_patterns
  `["*.safetensors", "*.json", "*.txt"]` (roughly 20-50 MB).
- `pip_extras`: existing list + `peft`. Modern diffusers requires the PEFT
  backend for `load_lora_weights`; the exploration report's "peft not needed"
  holds only for old diffusers. Verified against the .204 stack during
  implementation (Step B1).
- Memory estimate (see section 4): when the base is an HF repo, one extra
  HF API call sums the base repo's weight sizes into `capabilities.memory_gb`.
  When the base is a muse id, omit it; sizing derives from that entry.
- `search` is unchanged in v1.

## 2. Runtime (`runtimes/diffusers.py`) + base resolution

`DiffusersText2ImageModel.__init__` gains a branch driven by the capability
kwargs it already receives via the manifest splat:

```python
if lora_adapter:
    base_src = resolve_model_source(base_model)   # muse id -> local_dir, else verbatim
    self._pipe = AutoPipelineForText2Image.from_pretrained(base_src, ...)
    self._pipe.load_lora_weights(self._src)       # adapter local_dir (or hf_repo)
else:
    self._pipe = AutoPipelineForText2Image.from_pretrained(self._src, ...)
```

- **Never `fuse_lora()`.** Fusing bakes the adapter into the base weights
  and kills per-request scale.
- `resolve_model_source(ref)` is a new helper in
  `muse.core.runtime_helpers`: if `ref` is a pulled catalog id with a
  `local_dir`, return that path; else return `ref` verbatim (an HF repo id,
  which `from_pretrained` downloads into the HF cache at first load, the
  AnimateDiff precedent). This keeps the runtime catalog-dumb and
  `load_backend` LoRA-dumb.
- Load-time failure when the base is a muse id that is not pulled: raise with
  the actionable fix (`muse pull sdxl-turbo`) so the worker error and the
  gateway 5xx carry the remedy. A pull-time check in `expand_curated_pull` /
  `pull()` catches the curated case earlier with the same message.
- Per-request scale: `generate(...)` accepts optional `lora_scale`; when the
  model is a LoRA and the caller supplied one, pass it via diffusers'
  per-call scale mechanism (`cross_attention_kwargs={"scale": s}` on the
  PEFT backend); otherwise use the configured default. The exact kwarg
  plumbing is an implementation detail verified on .204.
- img2img / inpaint / variations: the existing lazy `from_pipe` paths share
  the UNet/text-encoder components of `self._pipe`, so the loaded adapter
  carries over automatically. These routes use the configured default scale
  in v1.

## 3. Wire surface (`routes.py` + protocol)

- `GenerationsRequest` gains optional `lora_scale: float | None`, validated
  to [0.0, 2.0].
- Model without `capabilities.lora_adapter` receiving `lora_scale` returns
  400 `lora_not_supported` (consistent with the img2img capability-mismatch
  precedent).
- OpenAI SDK usage: `client.images.generate(..., extra_body={"lora_scale": 0.8})`.

## 4. Memory sizing (no hardcoded family table)

Most-honest-first, all structural:

1. Probe measurement: `muse pull` ends with a probe that loads base+adapter
   and runs inference; its `measurements.<device>.peak_bytes` is
   authoritative and self-heals on every cold load (existing machinery).
2. Base-is-muse-id: the LoRA entry's sizing derives from the BASE entry's
   measurement / weights-on-disk via the sizing ladder
   (`backfill_manifest_memory` learns to chase `capabilities.base_model`
   when `lora_adapter` is set and the entry itself has no measurement).
3. Base-is-HF-repo: resolve-time `capabilities.memory_gb` estimate from the
   HF API weight sizes (one call; on API failure it is omitted, and the
   probe covers it).

This closes the hole where a 20 MB adapter dir would fool the
weights-on-disk fallback while the base needs about 7 GB.

## 5. Curated + CLI

- `curated.yaml` gains:

```yaml
- id: pixel-art-xl
  uri: hf://nerijs/pixel-art-xl
  modality: image/generation
  size_gb: 0.05
  description: "Pixel-art LoRA on SDXL (trigger: 'pixel'); pre-paired with sdxl-turbo for 1-4 step generation"
  capabilities:
    lora_adapter: true
    base_model: sdxl-turbo
    lora_scale: 1.0
```

- `muse pull` gains `--base <muse-id-or-hf-repo>` (typer option in `cli.py`,
  threaded to the resolver like the existing overrides).
- `muse models info` shows the pairing via existing capabilities rendering.

## 6. Error handling summary

| Condition | Where | Behavior |
|---|---|---|
| No base tag, no `--base` | pull | fail, actionable `--base` hint |
| Multiple adapter safetensors | pull | fail, list files |
| Curated/`--base` muse id not pulled | pull + load | fail, print `muse pull <base>` |
| `lora_scale` to non-LoRA model | route | 400 `lora_not_supported` |
| `lora_scale` outside [0, 2] | route | 422 validation |
| peft missing in venv | load | prevented: `peft` in pip_extras |

## 7. Testing

- **Plugin** (`tests/modalities/image_generation/test_hf_plugin.py`): adapter
  sniff true (pixel-art-xl real shape) / false (safetensors without lora
  signal; lora tag with model_index.json goes down the t2i path); base-tag
  parsing incl. `:adapter:` preference and plain-`base_model:` fallback;
  `--base` override wins; defaults derived from base id; multi-safetensors
  pull error; missing-base pull error.
- **Runtime** (`tests/modalities/image_generation/runtimes/`): mocked
  diffusers checks that the base source is loaded (not the adapter), that
  `load_lora_weights` is called with the adapter dir, that `fuse_lora` is
  never called, per-call scale pass-through, default scale fallback, and
  that the non-LoRA path is unchanged.
- **runtime_helpers**: `resolve_model_source` (muse id resolves to
  local_dir; unknown ref passes through verbatim; entry without local_dir
  passes through verbatim).
- **Routes**: `lora_scale` 400 on non-LoRA; forwarded on LoRA; bounds 422.
- **Sizing**: `backfill_manifest_memory` chases `capabilities.base_model` for
  a lora_adapter entry with no own measurement (muse-id base with a
  measurement; muse-id base with weights only; HF-repo base falls back to
  the resolve-time estimate).
- **CLI**: `--base` persists into the manifest.
- **Real-API verification on .204 (Step B1)**: pull `pixel-art-xl` paired
  with `sdxl-turbo`, enable, generate with trigger word `pixel` at steps=1,
  verify style + turbo-class latency + `lora_scale` visibly changes output.
- Fresh-venv CI smoke: skipped (needs a 7 GB base; over free-tier budget).

## Out of scope (v1)

Multi-LoRA composition; per-request adapter switching; per-request scale on
edits/variations; LoRA for non-diffusers modalities; `--weight-name` flag;
`muse search` LoRA filtering.
