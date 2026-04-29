# `image/upscale` modality design

**Date:** 2026-04-28
**Driver:** add the 13th muse modality, `image/upscale`, exposing `POST /v1/images/upscale` for super-resolution. Diffusion-based upscalers (SD x4 upscaler) are the v0.25.0 target. GAN-based upscalers (AuraSR, Real-ESRGAN) need separate runtimes and are deferred.

## Goal

1. Mount `POST /v1/images/upscale` (multipart/form-data) on a new `image/upscale` modality. Inputs: source image file + optional prompt + scale + steps + guidance + negative_prompt + seed + n + response_format. Behavior: super-resolution upscaling via `diffusers.StableDiffusionUpscalePipeline`.
2. Bundle `stabilityai/stable-diffusion-x4-upscaler` (~3GB, Apache 2.0) as the default model, alias-mapped through `curated.yaml`.
3. New `DiffusersUpscaleRuntime` generic runtime in `image_upscale/runtimes/diffusers_upscaler.py`, callable for any HF repo whose `model_index.json` shape matches the SD x4 upscaler family.
4. New HF plugin sniffing diffusers-shape upscalers (`model_index.json` + `image-to-image` tag + repo name pattern hints). Priority 105 (between image/animation 110 and image/generation 100).
5. New `ImageUpscaleClient` HTTP client (multipart upload).
6. `supported_scales` capability flag gates the `scale` request parameter (returns 400 for unsupported scales).
7. Optional input-side cap (env-tunable `MUSE_UPSCALE_MAX_INPUT_SIDE`, default 1024) prevents runaway VRAM use.

## Non-goals

- AuraSR, Real-ESRGAN, GFPGAN, and other non-diffusers upscalers. They need their own runtime classes; the diffusers HF plugin must not claim them. Filed as v1.next.
- Tile-based upscaling for very large inputs. Future enhancement.
- Face restoration (CodeFormer, GFPGAN). Future modality (`image/restoration` or kept as a sub-feature here).
- Video upscaling. Out of scope.
- Multi-pass / cascaded upscalers (e.g. 2x then 2x). Future feature.
- GUI / preview. Wire-only modality.

## Architecture

```
client.images.upscale(image=src.read(), model="stable-diffusion-x4-upscaler", scale=4, prompt="")
   |
   v  HTTP POST /v1/images/upscale, multipart/form-data
   |  fields: image (PNG file), model, scale, prompt?, negative_prompt?,
   |          steps?, guidance?, seed?, n?, response_format?
   v
routes.upscale handler
   - decode_image_file(image) into PIL.Image (mirrors image_generation helper)
   - input-side cap MUSE_UPSCALE_MAX_INPUT_SIDE check
   - capability gate: supported_scales must include `scale`
   - call backend.upscale(img, scale=..., prompt=..., ...)
   - codec.encode each UpscaleResult into the OpenAI envelope
   v
DiffusersUpscaleRuntime.upscale(image, scale=4, prompt="", ...)
   - lazy-load StableDiffusionUpscalePipeline once, cache as self._pipe
   - call self._pipe(image=image, prompt=prompt, num_inference_steps=...,
                     guidance_scale=..., generator=..., negative_prompt=...)
   v
UpscaleResult -> codec -> JSON envelope
```

The MIME tag is `image/upscale`; the URL is `/v1/images/upscale`. The directory is `src/muse/modalities/image_upscale/` (underscore in dir name; slash in MIME).

## Wire contract

`POST /v1/images/upscale`, multipart/form-data:

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `image` | file (PNG/JPEG/WebP) | yes | - | size cap 10MB; max side cap MUSE_UPSCALE_MAX_INPUT_SIDE (default 1024) |
| `model` | str | no | None | bundled or pulled model id |
| `scale` | int | no | 4 | must be in model's `supported_scales` |
| `prompt` | str | no | "" | 0 to 4000 chars (SD x4 supports prompt-guided upscale; empty is fine) |
| `negative_prompt` | str | no | None | up to 4000 chars |
| `steps` | int | no | None | 1 to 100 (model default if omitted) |
| `guidance` | float | no | None | 0 to 20 (model default if omitted) |
| `seed` | int | no | None | reproducibility |
| `n` | int | no | 1 | 1 to 4 (smaller cap than t2i; upscale is heavy) |
| `response_format` | str | no | "b64_json" | "b64_json" or "url" |

Response (mirrors `/v1/images/generations` envelope):
```json
{
  "created": 1730000000,
  "data": [
    {"b64_json": "...", "revised_prompt": null}
  ]
}
```

`revised_prompt` carries the input prompt verbatim (or null when not provided), matching the OpenAI convention from `/v1/images/generations` and `/v1/images/edits`. When `response_format="url"`, each entry has `url` (data URL) instead of `b64_json`.

## Module structure

```
src/muse/modalities/image_upscale/
|-- __init__.py          # MODALITY="image/upscale", build_router, exports, PROBE_DEFAULTS
|-- protocol.py          # UpscaleResult dataclass, ImageUpscaleModel Protocol
|-- codec.py             # to_bytes / to_data_url (reuses image_generation codec)
|-- routes.py            # POST /v1/images/upscale (multipart)
|-- client.py            # ImageUpscaleClient (HTTP, multipart)
|-- hf.py                # HF_PLUGIN sniffing diffusers-shape upscalers
`-- runtimes/
    |-- __init__.py
    `-- diffusers_upscaler.py  # DiffusersUpscaleRuntime
```

`src/muse/models/stable_diffusion_x4_upscaler.py` is the bundled model script. It mirrors `sd_turbo.py`'s lazy-import pattern (module-level sentinels, `_ensure_deps()`, importlib-friendly tests).

## `UpscaleResult` dataclass

```python
@dataclass
class UpscaleResult:
    image: Any                          # PIL.Image (codec normalizes)
    original_width: int
    original_height: int
    upscaled_width: int
    upscaled_height: int
    scale: int                          # actual scale (may differ from request)
    seed: int                           # -1 if no seed supplied
    metadata: dict = field(default_factory=dict)
```

## Protocol

```python
@runtime_checkable
class ImageUpscaleModel(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def supported_scales(self) -> list[int]: ...

    def upscale(
        self,
        image: Any,
        *,
        scale: int | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> UpscaleResult: ...
```

Duck-typed; backends only need to satisfy the structural shape.

## Runtime: `DiffusersUpscaleRuntime`

```python
class DiffusersUpscaleRuntime:
    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        default_scale: int = 4,
        supported_scales: list[int] | None = None,
        default_steps: int = 20,
        default_guidance: float = 9.0,
        **_: Any,
    ) -> None:
        # lazy-import torch + StableDiffusionUpscalePipeline
        # StableDiffusionUpscalePipeline.from_pretrained(local_dir or hf_repo,
        #                                                torch_dtype=...)
        # honor device + dtype

    @property
    def supported_scales(self) -> list[int]: ...

    def upscale(self, image, *, scale=None, prompt=None, negative_prompt=None,
                steps=None, guidance=None, seed=None, **_) -> UpscaleResult:
        # SD x4 always upscales by exactly 4x. The `scale` arg is informational
        # for fixed-scale models; the route layer enforces supported_scales.
        # Call self._pipe(image=image, prompt=prompt or "", ...)
        # Return UpscaleResult(image, original_size, upscaled_size, scale, seed, metadata)
```

Mirrors `DiffusersText2ImageModel`'s structure: lazy-import sentinels at module scope (`torch`, `StableDiffusionUpscalePipeline`), `_ensure_deps()` with short-circuit, `_select_device()` helper, the rest of the class. The runtime doesn't subclass anything; it satisfies the Protocol structurally.

The constructor reads `default_scale`, `supported_scales`, `default_steps`, `default_guidance` from manifest capabilities (registry merges them in at `load_backend` time, same as `DiffusersText2ImageModel`).

## Bundled `stable_diffusion_x4_upscaler.py`

```python
MANIFEST = {
    "model_id": "stable-diffusion-x4-upscaler",
    "modality": "image/upscale",
    "hf_repo": "stabilityai/stable-diffusion-x4-upscaler",
    "description": "SD x4 upscaler: 4x super-resolution via latent diffusion, Apache 2.0",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cuda",          # CPU is too slow at 20 steps for 256->1024
        "default_scale": 4,
        "supported_scales": [4],   # SD x4 is fixed-ratio
        "default_steps": 20,
        "default_guidance": 9.0,
        "memory_gb": 6.0,          # peak fp16 inference at 256->1024
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "feature_extractor/*.json",
        "scheduler/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
        "unet/*.fp16.safetensors", "unet/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
    ],
}
```

The `Model` class lazy-imports `torch` and `diffusers.StableDiffusionUpscalePipeline`, mirrors the runtime's behavior, and is itself testable with importlib-shaped patches.

## Capability flags

| Flag | Default in resolver-pulled diffusers upscalers | Default in stable-diffusion-x4-upscaler |
|---|---|---|
| `default_scale` | 4 | 4 |
| `supported_scales` | inferred per pattern (see HF plugin) | [4] |
| `default_steps` | 20 | 20 |
| `default_guidance` | 9.0 | 9.0 |

## HF plugin

Priority 105. `_sniff` runs after image/animation (110) and before image/generation (100), so an upscaler repo gets correctly assigned even though it shares the `image-to-image` tag and `model_index.json` pattern with regular diffusers t2i / i2i checkpoints.

```python
def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(Path(f).name == "model_index.json" for f in siblings)
    has_i2i_tag = "image-to-image" in tags
    repo_id = (getattr(info, "id", "") or "").lower()
    is_upscaler_name = any(
        s in repo_id
        for s in [
            "upscaler", "super-resolution", "esrgan", "upscale",
            "x4-upscaler", "ldm-super",
        ]
    )
    return has_pipeline_config and has_i2i_tag and is_upscaler_name
```

The `_resolve` infers per-pattern defaults:

| Repo pattern | default_scale | supported_scales | default_steps | default_guidance |
|---|---|---|---|---|
| `*x4-upscaler*` (or `*x4*`) | 4 | [4] | 20 | 9.0 |
| `*aura*` (deferred runtime; not claimed in v0.25.0) | n/a | n/a | n/a | n/a |
| `*esrgan*` (deferred; not claimed) | n/a | n/a | n/a | n/a |
| fallback | 4 | [4] | 20 | 9.0 |

For v0.25.0, the plugin's `_sniff` returns False on `aura*` / `esrgan*` patterns until a non-diffusers runtime exists. The repo-name allowlist is conservative: only checkpoints that look like SD-style upscalers and ship the `model_index.json` shape get claimed.

`_search` filters by `image-to-image` HF tag and post-filters the repo names by the `is_upscaler_name` rule, mirroring the per-modality search pattern (image/embedding does the same).

## Codec

Reuses `to_bytes` and `to_data_url` from `muse.modalities.image_generation.codec` to keep the surface small. The route assembles the OpenAI envelope inline:

```python
{
  "created": int(time.time()),
  "data": [
    {"b64_json": ..., "revised_prompt": prompt_or_null} | {"url": ..., "revised_prompt": ...},
    ...
  ],
}
```

The codec module re-exports the two functions:
```python
from muse.modalities.image_generation.codec import to_bytes, to_data_url

__all__ = ["to_bytes", "to_data_url"]
```

This keeps the modality self-describing without code duplication.

## Multipart input decoding

Reuses `decode_image_file` from `muse.modalities.image_generation.image_input`. No need for a third copy. The route also runs the `MUSE_UPSCALE_MAX_INPUT_SIDE` cap after decode (default 1024 per side; environment-tunable). Oversize inputs return 400 with code `invalid_parameter`.

The cap rationale: a 4x upscale of a 2048x2048 input becomes 8192x8192, which is ~200MB of decoded image and likely OOMs the upscaler on consumer GPUs.

## PROBE_DEFAULTS

```python
def _make_probe_image():
    """Generate a small synthetic test image. Lazy import so the modality
    package loads without PIL on the host python."""
    from PIL import Image
    return Image.new("RGB", (128, 128), (128, 128, 128))


PROBE_DEFAULTS = {
    "shape": "128x128 -> 512x512 (4x), 20 steps",
    "call": lambda m: m.upscale(_make_probe_image(), scale=4),
}
```

`muse models probe stable-diffusion-x4-upscaler` runs this lambda; the probe image is 128x128 (small enough that a 4x upscale lands at 512x512, within VRAM budget on small GPUs).

## Curated entries

```yaml
- id: stable-diffusion-x4-upscaler
  bundled: true
```

AuraSR-v2 deferred to v1.next: it's a 600MB GAN-based super-resolution model. The `fal/AuraSR-v2` repo doesn't ship a `model_index.json`; it has a custom `aura_sr.py` PyTorch module and a single safetensors file. Adding a separate `AuraSRRuntime` (or a generic `TorchHubRuntime`) is doable but expands scope past v0.25.0. Filed in spec sentinels for follow-up.

## Behavioral resilience

1. **SD x4 is fixed at 4x.** `supported_scales=[4]` gates the `scale` request parameter. A request with `scale=2` or `scale=8` returns 400 with code `invalid_parameter`, message `"model 'stable-diffusion-x4-upscaler' only supports scales: [4]"`. The capability is honored end-to-end so users see the constraint before paying for a forward pass.

2. **VRAM is heavy.** SD x4 at 256x256 -> 1024x1024 uses ~6GB peak at fp16. At 512x512 -> 2048x2048, much more. The default `MUSE_UPSCALE_MAX_INPUT_SIDE` of 1024 keeps the bundled model in fp16-on-12GB-GPU territory. Users with more VRAM can raise the cap explicitly.

3. **Long runtime.** Diffusion upscaling at 20 steps for 512 -> 2048 is ~30-60s on a 12GB GPU. Default request timeout in `ImageUpscaleClient` is 600s (10 min), generous for n=1 to n=4.

4. **PIL output normalization.** The runtime returns whatever the diffusers pipeline's `images[0]` is (PIL.Image in practice). The codec calls `to_pil(image)` first which normalizes PIL / numpy / torch to PIL; tests can return a real `PIL.Image.new("RGB", (1024, 1024))` rather than mocking deeper.

5. **Input size validation.** The route runs the cap after decode but before invoking the pipeline. Cap is read at request time so the env var change takes effect on the next request, not just at supervisor restart.

6. **Empty prompt is valid.** SD x4 upscaler accepts empty prompt for plain upscaling. The routes default `prompt` to `""` when omitted; the runtime forwards the empty string to the pipeline.

## Modality `__init__.py` exports

```python
from muse.modalities.image_upscale.client import ImageUpscaleClient
from muse.modalities.image_upscale.protocol import ImageUpscaleModel, UpscaleResult
from muse.modalities.image_upscale.routes import build_router

MODALITY = "image/upscale"

PROBE_DEFAULTS = {
    "shape": "128x128 -> 512x512 (4x), 20 steps",
    "call": lambda m: m.upscale(_make_probe_image(), scale=4),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageUpscaleClient",
    "ImageUpscaleModel",
    "UpscaleResult",
]
```

## Migration / risk

- New modality directory; no impact on existing routes.
- HF plugin priority 105 sits between image/animation (110) and image/generation (100). Conservative `is_upscaler_name` allowlist keeps the plugin from accidentally claiming non-upscaler diffusers checkpoints.
- New per-model venv when users `muse pull stable-diffusion-x4-upscaler`. Reuses the diffusers + transformers + accelerate stack, identical to sd-turbo's pip_extras.
- `MUSE_UPSCALE_MAX_INPUT_SIDE` is opt-in env-tunable; default 1024 protects out-of-the-box users.

## Test coverage

```
tests/modalities/image_upscale/
|-- test_protocol.py     # dataclass shape, structural protocol check
|-- test_codec.py        # passthrough re-exports
|-- test_routes.py       # POST /v1/images/upscale (happy path + 4xx)
|-- test_client.py       # ImageUpscaleClient (multipart upload, env var)
|-- test_hf_plugin.py    # _sniff and _resolve, including SD x4 + fallback
`-- runtimes/
    `-- test_diffusers_upscaler.py  # patched StableDiffusionUpscalePipeline

tests/models/test_stable_diffusion_x4_upscaler.py  # bundled-script tests

tests/cli_impl/test_e2e_image_upscale.py           # @pytest.mark.slow
tests/integration/test_remote_image_upscale.py     # opt-in MUSE_REMOTE_SERVER
```

Specifically:

- `test_protocol.py`: UpscaleResult fields, ImageUpscaleModel duck-typing.
- `test_codec.py`: re-exports work.
- `test_hf_plugin.py`:
  - `_sniff(x4-upscaler)` returns True (i2i tag + repo name match).
  - `_sniff(stable-diffusion-2-1)` returns False (no upscaler-name).
  - `_sniff(repo_without_model_index)` returns False.
  - `_resolve("stabilityai/stable-diffusion-x4-upscaler", ...)` produces the right capabilities.
  - `_resolve` for a generic upscaler-name repo gets the fallback defaults.
- `test_routes.py`:
  - happy path POST returns 200 + envelope + b64_json.
  - missing image returns 422 (FastAPI Form validation).
  - empty image returns 400 (decode error).
  - malformed image returns 400 (decode error).
  - `scale=2` on a model with `supported_scales=[4]` returns 400.
  - over-cap input returns 400 (`MUSE_UPSCALE_MAX_INPUT_SIDE`).
  - unknown model returns 404 (OpenAI envelope).
  - n=2 returns 2 entries.
  - response_format=url returns data URL.
  - revised_prompt echoes the input prompt.
- `test_client.py`:
  - default base url + MUSE_SERVER override.
  - multipart shape (image, scale, prompt, n, response_format).
  - decode b64_json -> bytes.
  - non-200 raises RuntimeError.
- `runtimes/test_diffusers_upscaler.py`:
  - lazy-import sentinels.
  - constructor calls `StableDiffusionUpscalePipeline.from_pretrained`.
  - upscale calls the pipe with prompt="" by default.
  - upscale uses seeded generator when seed supplied.
  - upscale honors steps and guidance overrides.
  - upscale returns UpscaleResult with correct dimensions.
- `models/test_stable_diffusion_x4_upscaler.py`:
  - manifest required fields.
  - manifest pip_extras has torch + transformers + diffusers.
  - manifest capabilities advertise supported_scales=[4].
  - Model class loads via patched `StableDiffusionUpscalePipeline`.
  - upscale returns UpscaleResult with original/upscaled dims.
- `cli_impl/test_e2e_image_upscale.py`: full TestClient flow with a fake backend that satisfies the protocol structurally.
- `integration/test_remote_image_upscale.py`: protocol assertion on a real server, optional via MUSE_REMOTE_SERVER.

## Open questions

None. The diffusers `StableDiffusionUpscalePipeline` is documented and stable. The OpenAI SDK has no `images.upscale(...)` method but raw multipart POST is straightforward.

## Out of scope (filed for later)

- AuraSR / Real-ESRGAN: needs `AuraSRRuntime` or `TorchHubRuntime` (not diffusers-shaped). Track in v1.next.
- Tile-based upscaling for inputs >1024 per side: future feature.
- Face restoration via CodeFormer / GFPGAN: future modality / sub-feature.
- Cascaded upscalers (2x then 2x then 2x): future feature.
- AnimateUpscale (frame-by-frame upscaling on image/animation outputs): cross-modal pipeline; future.
- The `muse.modalities._common.uploads` shared module: with three multipart modalities (audio_transcription, image_generation, image_upscale), the case for extraction is yet stronger but still optional.
