# Image Generation HF Plugin + Generic Diffusers Runtime

**Date:** 2026-04-26
**Driver:** unlock `muse pull hf://...` for any text-to-image diffusers repo (SDXL-Turbo, FLUX.1-schnell, SD 3.5, Playground, etc.). Today `image/generation` is bundled-script-only (just `sd-turbo`); this task brings it in line with how `chat/completion`, `embedding/text`, `audio/transcription`, and `text/classification` already work.

## Goal

1. New generic runtime `DiffusersText2ImageModel` that wraps `diffusers.AutoPipelineForText2Image` with manifest-driven defaults (size, steps, guidance).
2. New HF resolver plugin at `src/muse/modalities/image_generation/hf.py`. Sniff: any HF repo with `model_index.json` sibling AND `text-to-image` tag.
3. Curated entries for `sdxl-turbo` and `flux-schnell` in `curated.yaml`.

## Non-goals

- img2img / `image` + `strength` route extension. That's the next task.
- Deprecating the bundled `sd_turbo.py` script. It stays. (Curated `sd-turbo` continues to alias the bundled script via first-found-wins.)
- Refiner pipelines (SDXL-Refiner), inpainting, controlnet. Different routes/modalities.

## Architecture

```
muse pull hf://stabilityai/sdxl-turbo
    |
    v
HFResolver.resolve  (Task #129's plugin dispatch)
    |
    | iterates plugins; image_generation/hf.py sniff matches
    v
plugin.resolve  -> ResolvedModel(manifest, backend_path, download)
    |
    | manifest.backend_path = ".../runtimes/diffusers:DiffusersText2ImageModel"
    | manifest.capabilities = {default_size, default_steps, default_guidance, ...}
    v
catalog._pull_via_resolver  ->  install pip_extras into per-model venv
                                persist manifest in catalog.json
                                snapshot_download weights
    |
    v
muse serve  ->  worker imports DiffusersText2ImageModel, instantiates with
                manifest.capabilities (default_size etc. injected as kwargs)
```

## Plugin contract

`src/muse/modalities/image_generation/hf.py`:

```python
HF_PLUGIN = {
    "modality": "image/generation",
    "runtime_path": "muse.modalities.image_generation.runtimes.diffusers:DiffusersText2ImageModel",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "priority": 100,  # file-pattern + tag, very specific
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

Sniff: True iff (`model_index.json` is a sibling) AND (`"text-to-image"` is in tags). The `model_index.json` test alone would also match img2img-only and inpaint-only repos; the tag adds the "this is a text-to-image entry point" filter. Lower bound on false positives.

Resolve: synthesizes a manifest with capabilities sourced from a per-model lookup OR from a defaults YAML. For this task, defaults come hardcoded based on tags / repo-name heuristics (Turbo variants get `steps=1, guidance=0`; FLUX gets `steps=4, guidance=0`; everything else gets `steps=25, guidance=7.5`). Curated entries can override via the `capabilities` overlay.

Search: HF `list_models(filter="text-to-image")`, yield rows.

## Generic runtime

`src/muse/modalities/image_generation/runtimes/diffusers.py:DiffusersText2ImageModel`:

Parameterized from manifest capabilities at construction time:

```python
class DiffusersText2ImageModel:
    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        default_size: tuple[int, int] = (512, 512),
        default_steps: int = 1,
        default_guidance: float = 0.0,
        **_,
    ) -> None: ...

    def generate(
        self, prompt: str, *,
        negative_prompt=None, width=None, height=None,
        steps=None, guidance=None, seed=None, **_,
    ) -> ImageResult: ...
```

Implementation mirrors `src/muse/models/sd_turbo.py` (which already uses `AutoPipelineForText2Image`); the only differences are:
- `model_id` injected from catalog (one runtime serves many models)
- `default_size`, `default_steps`, `default_guidance` injected from manifest.capabilities
- No `default_size: tuple` typing nuance (sd_turbo had `(512, 512)` hardcoded)

Lazy imports for torch + diffusers; same `_ensure_deps()` pattern as `sd_turbo.py`.

## Capability defaults (for the plugin's `_resolve`)

Inferred from repo name / tags at resolve time (no live HF call beyond `repo_info`):

| Pattern in repo_id (lowercased) | default_steps | default_guidance | default_size |
|---|---|---|---|
| `*turbo*` (sd-turbo, sdxl-turbo, etc.) | 1 | 0.0 | (512, 512) |
| `*flux*schnell*` | 4 | 0.0 | (1024, 1024) |
| `*flux*dev*` | 28 | 3.5 | (1024, 1024) |
| `*sdxl*` (no `turbo`) | 25 | 7.5 | (1024, 1024) |
| `*stable-diffusion-3*` | 28 | 4.5 | (1024, 1024) |
| (default fallback) | 25 | 7.5 | (512, 512) |

These are reasonable starting points. Users can override per-call (request fields) or per-model (curated `capabilities` overlay).

## Curated entries

In `src/muse/curated.yaml`:

```yaml
- id: sdxl-turbo
  uri: hf://stabilityai/sdxl-turbo
  modality: image/generation
  size_gb: 7.0
  description: "SDXL Turbo: 1-step distilled SDXL, 512x512, fast"

- id: flux-schnell
  uri: hf://black-forest-labs/FLUX.1-schnell
  modality: image/generation
  size_gb: 24.0
  description: "FLUX.1 Schnell: 4-step distilled, 1024x1024, Apache 2.0"
```

## Manifest shape after `muse pull hf://stabilityai/sdxl-turbo`

```json
{
  "model_id": "sdxl-turbo",
  "modality": "image/generation",
  "hf_repo": "stabilityai/sdxl-turbo",
  "description": "Diffusers text-to-image: stabilityai/sdxl-turbo",
  "license": "...",
  "pip_extras": ["torch>=2.1.0", "diffusers>=0.27.0", ...],
  "system_packages": [],
  "capabilities": {
    "default_size": [512, 512],
    "default_steps": 1,
    "default_guidance": 0.0,
    "supports_negative_prompt": true,
    "supports_seeded_generation": true
  },
  "backend_path": "muse.modalities.image_generation.runtimes.diffusers:DiffusersText2ImageModel"
}
```

## Route impact: zero

`/v1/images/generations` already accepts `prompt`, `model`, `n`, `size`, `negative_prompt`, `steps`, `guidance`, `seed`. The route calls `model.generate(prompt, width=, height=, steps=, guidance=, ...)` which the new runtime satisfies. No route changes.

## Constraints

- `hf.py` import rules (per Task #129): stdlib + `huggingface_hub` + `muse.core.*` only. No relative imports, no sibling-modality imports, no heavy deps.
- The runtime DOES import torch + diffusers (heavy), but that's fine because it's loaded inside the per-model venv at worker time, not at discovery time.

## Test coverage

- `tests/modalities/image_generation/runtimes/test_diffusers.py`: mocked diffusers; runtime instantiation, generate() returns ImageResult, manifest defaults applied.
- `tests/modalities/image_generation/test_hf_plugin.py`: 7 tests (mirror prior plugin tests). Plugin keys, metadata, sniff true/false, resolve, search.
- `tests/core/test_curated.py`: 2 new entries asserted (sdxl-turbo, flux-schnell).

## Migration / risk

- Bundled `sd_turbo.py` script is untouched; existing `muse pull sd-turbo` keeps working unchanged.
- The new plugin's sniff requires `model_index.json` sibling, which the SD-Turbo HF repo also has, so a future user pulling `hf://stabilityai/sd-turbo` directly would route through the plugin (not the bundled script). That's fine: same model, new path.
- No breaking changes to any wire contract.

## Out of scope (filed for later)

- img2img extension to `/v1/images/generations` (next task; img2img via `image` + `strength` request fields, OpenAI SDK compatible via `extra_body`).
- AnimateDiff / `image/animation` modality (separate task #107-adjacent).
- Diffusers-based modalities other than text-to-image (image-to-image, image-edits, image-variations).
