# `/v1/images/edits` and `/v1/images/variations` route extensions

**Date:** 2026-04-28
**Driver:** unlock OpenAI's two remaining canonical image routes on the existing `image/generation` modality. `/v1/images/edits` is mask-conditioned inpainting (regenerate the masked region per prompt). `/v1/images/variations` is alternates of one image with no prompt (image-to-image with empty prompt + high strength). Both are additive route mounts on the same modality, mirroring the v0.17.0 img2img extension pattern.

## Goal

1. Mount `POST /v1/images/edits` (multipart) on the existing `image_generation` build_router. Inputs: source image file + mask file + prompt + n + size + response_format. Behavior: regenerate masked region per prompt via `AutoPipelineForInpainting.from_pipe(self._pipe)`.
2. Mount `POST /v1/images/variations` (multipart) on the same router. Inputs: source image + n + size + response_format (no prompt). Behavior: image-to-image with empty prompt and high strength (default 0.85).
3. Extend `DiffusersText2ImageModel` (the generic runtime) and `sd_turbo.py` (the bundled script) with `inpaint()` and `vary()` methods. Lazy-load and cache the inpaint pipeline alongside the existing img2img pipeline.
4. Capability flags `supports_inpainting` and `supports_variations` gate the routes. Both default True for diffusers-shape models (synthesized by the HF resolver) and for sd_turbo.
5. New `decode_image_file(UploadFile) -> PIL.Image.Image` helper next to `decode_image_input` for the multipart upload paths.
6. New `ImageEditsClient` and `ImageVariationsClient` HTTP clients alongside `GenerationsClient`. OpenAI SDK's `images.edit(...)` and `images.create_variation(...)` continue to work because muse mirrors the multipart wire shape.

## Non-goals

- Image-conditioned ControlNet edits. Future modality.
- Mask-free editing via prompt diffs (instructions). Future feature.
- Multi-image composition / IP-Adapter style conditioning. Same.
- Variations conditioned on multiple images (averaging or interpolation). Same.
- Per-frame edits on an animation. The image/animation modality is independent.
- New modality directory. Both routes share the existing `image/generation` MIME tag and runtime.

## Architecture

```
client.images.edit(image=src.read(), mask=msk.read(), prompt="...", model="sdxl-turbo", n=1, size="512x512")
   |
   v  HTTP POST /v1/images/edits, multipart/form-data
   |  fields: image (PNG file), mask (PNG file), prompt, model, n, size, response_format
   v
routes.edits handler
   - capability gate: registry.manifest(...).capabilities.supports_inpainting must be True
   - decode_image_file(image) and decode_image_file(mask) into PIL.Image
   - mask normalized to PIL.Image.L (grayscale) at runtime layer
   - call model.inpaint(prompt, init_image=img, mask_image=msk, ...)
   - codec.encode each ImageResult into the OpenAI envelope
   v
DiffusersText2ImageModel.inpaint(prompt, init_image, mask_image, strength=0.99, ...)
   - lazy-load AutoPipelineForInpainting via from_pipe(self._pipe), cache as self._inp_pipe
   - call with prompt, image=init_image, mask_image=mask_image, strength
   v
ImageResult -> codec -> JSON envelope
```

```
client.images.create_variation(image=src.read(), model="sdxl-turbo", n=1, size="512x512")
   |
   v  HTTP POST /v1/images/variations, multipart/form-data
   |  fields: image (PNG file), model, n, size, response_format
   v
routes.variations handler
   - capability gate: registry.manifest(...).capabilities.supports_variations must be True
   - decode_image_file(image) into PIL.Image
   - call model.vary(init_image=img, n=n, ...)
   - codec.encode each ImageResult into the OpenAI envelope
   v
DiffusersText2ImageModel.vary(init_image, ...)
   - reuse self._generate_img2img with prompt="" and strength=0.85
   v
ImageResult -> codec -> JSON envelope (no revised_prompt; envelope carries only b64_json/url)
```

## Wire contract: `/v1/images/edits`

Multipart/form-data fields:

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `image` | file (PNG/JPEG/WebP) | yes | - | size cap 10MB |
| `mask` | file (PNG/JPEG/WebP) | yes | - | size cap 10MB; white = regenerate, black = keep |
| `prompt` | str | yes | - | 1 to 4000 chars |
| `model` | str | no | None | bundled or pulled model id |
| `n` | int | no | 1 | 1 to 10 |
| `size` | str | no | "512x512" | "WIDTHxHEIGHT", 64 to 2048 per side |
| `response_format` | str | no | "b64_json" | "b64_json" or "url" |

Response (mirrors `/v1/images/generations`):
```json
{
  "created": 1730000000,
  "data": [
    {"b64_json": "...", "revised_prompt": "make it night"},
    ...
  ]
}
```

When `response_format="url"`, each entry has `url` (data URL) instead of `b64_json`. `revised_prompt` echoes the request prompt (same convention as `/v1/images/generations`).

## Wire contract: `/v1/images/variations`

Multipart/form-data fields:

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `image` | file (PNG/JPEG/WebP) | yes | - | size cap 10MB |
| `model` | str | no | None | bundled or pulled model id |
| `n` | int | no | 1 | 1 to 10 |
| `size` | str | no | "512x512" | "WIDTHxHEIGHT", 64 to 2048 per side |
| `response_format` | str | no | "b64_json" | "b64_json" or "url" |

Response: same envelope as `/v1/images/edits` but each entry has only `b64_json` (or `url`); no `revised_prompt` since there is no prompt.

OpenAI SDK usage:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

# Inpainting
with open("scene.png", "rb") as src, open("mask.png", "rb") as msk:
    r = client.images.edit(
        image=src,
        mask=msk,
        prompt="add a moon to the sky",
        model="sd-turbo",
        n=1,
        size="512x512",
    )

# Variations
with open("scene.png", "rb") as src:
    r = client.images.create_variation(
        image=src,
        model="sd-turbo",
        n=2,
        size="512x512",
    )
```

## Runtime contract

`DiffusersText2ImageModel` and `sd_turbo.Model` grow two new methods:

```python
def inpaint(
    self, prompt: str, *,
    init_image: Any, mask_image: Any,
    negative_prompt: str | None = None,
    width: int | None = None, height: int | None = None,
    steps: int | None = None, guidance: float | None = None,
    seed: int | None = None, strength: float | None = None,
    **_: Any,
) -> ImageResult: ...


def vary(
    self, *,
    init_image: Any,
    width: int | None = None, height: int | None = None,
    steps: int | None = None, guidance: float | None = None,
    seed: int | None = None, strength: float | None = None,
    **_: Any,
) -> ImageResult: ...
```

`inpaint()`:
- Lazy-loads `AutoPipelineForInpainting` via `_inp.from_pipe(self._pipe)` (shares weights, no second VRAM allocation). Cache on `self._inp_pipe`.
- Normalizes the mask to grayscale via `mask_image.convert("L")` if it's RGB/RGBA.
- Calls the inpaint pipe with `prompt`, `image=init_image`, `mask_image=normalized_mask`, `strength=strength or 0.99`.
- Bumps `num_inference_steps` to satisfy the `steps * strength >= 1` contract (same as img2img).
- Returns ImageResult with `metadata.mode = "inpaint"`.

`vary()`:
- Delegates to `self._generate_img2img(prompt="", init_image=..., strength=strength or 0.85)`.
- No new pipeline; reuses the v0.17.0 img2img path.
- Returns ImageResult with `metadata.mode = "variations"` (override the underlying img2img's `"img2img"` mode label).

## Capability flags

| Flag | Default for resolver-pulled diffusers | Default in sd_turbo |
|---|---|---|
| `supports_inpainting` | True | True |
| `supports_variations` | True | True |

The HF resolver plugin's `_resolve` adds both flags to `capabilities`. sd_turbo's MANIFEST adds both. Capability gates in routes:

- `POST /v1/images/edits` with `supports_inpainting=False` -> 400 with `code=invalid_parameter`, `message="model X does not support inpainting"`.
- `POST /v1/images/variations` with `supports_variations=False` -> 400 with `code=invalid_parameter`, `message="model X does not support variations"`.

## Multipart file decoding

New helper next to `decode_image_input` in `image_input.py`:

```python
async def decode_image_file(file: UploadFile, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> Any:
    """Decode an UploadFile into PIL.Image. ValueError on failure."""
```

Reads the upload bytes, applies the size cap, and reuses `_bytes_to_pil`. Returns PIL.Image. Failures (empty file, oversized, undecodable) raise ValueError. Routes wrap into 400 responses with the OpenAI error envelope.

The helper is async because UploadFile.read is async.

## Codec

`/v1/images/edits` and `/v1/images/variations` reuse `codec.to_bytes` and `codec.to_data_url` (no codec changes). Each route builds its own envelope:

- edits: `{"created": ..., "data": [{"b64_json"|"url", "revised_prompt"}, ...]}`
- variations: `{"created": ..., "data": [{"b64_json"|"url"}, ...]}`

## Modality `__init__.py` exports

The modality package adds two new exports to `__all__`:

```python
from muse.modalities.image_generation.client import (
    GenerationsClient,
    ImageEditsClient,
    ImageVariationsClient,
)

__all__ = [
    "MODALITY", "PROBE_DEFAULTS", "build_router",
    "GenerationsClient",
    "ImageEditsClient",
    "ImageVariationsClient",
    "ImageResult", "ImageModel",
]
```

## Migration / risk

- Existing `/v1/images/generations` requests: zero impact. New routes mount alongside.
- Already-pulled diffusers models: synthesized capabilities are persisted at pull time; old pulls miss `supports_inpainting`/`supports_variations` until repulled (see #138). The capability gate returns 400 in that case until the user re-pulls or runs `muse models refresh`.
- New venvs: no new pip_extras (PIL + diffusers are already there).

## Test coverage

New tests:

`tests/modalities/image_generation/test_image_input.py`:
- `decode_image_file` reads PNG bytes via UploadFile-shape and yields PIL.Image
- `decode_image_file` raises ValueError on empty file
- `decode_image_file` raises ValueError on oversized file
- `decode_image_file` raises ValueError on undecodable bytes

`tests/modalities/image_generation/runtimes/test_diffusers.py`:
- `inpaint(init_image, mask_image)` calls `AutoPipelineForInpainting.from_pipe(self._pipe)` and the resulting pipe
- inpaint reuses the cached `_inp_pipe` on a second call
- inpaint uses `from_pipe`, not `from_pretrained`, so VRAM stays shared
- inpaint converts an RGB/RGBA mask to L before passing to the pipe
- inpaint bumps steps to satisfy `steps * strength >= 1`
- `vary(init_image)` delegates to img2img with empty prompt and default strength 0.85
- vary returns ImageResult with `metadata.mode == "variations"`

`tests/models/test_sd_turbo.py`:
- MANIFEST advertises `supports_inpainting` and `supports_variations`
- `Model.inpaint(...)` calls AutoPipelineForInpainting.from_pipe
- `Model.vary(...)` reuses img2img with prompt=""

`tests/modalities/image_generation/test_routes.py`:
- POST `/v1/images/edits` multipart with image+mask+prompt returns 200, JSON envelope with `b64_json`
- POST `/v1/images/edits` with `supports_inpainting=False` model returns 400
- POST `/v1/images/edits` with empty image file returns 400
- POST `/v1/images/edits` with malformed image returns 400
- POST `/v1/images/edits` with `n=3` returns 3 entries
- POST `/v1/images/edits` echoes prompt as `revised_prompt`
- POST `/v1/images/variations` multipart with image returns 200
- POST `/v1/images/variations` with `supports_variations=False` model returns 400
- POST `/v1/images/variations` returns entries WITHOUT `revised_prompt`
- POST `/v1/images/variations` with `n=2` returns 2 entries
- POST both routes with unknown model returns 404 (OpenAI envelope)
- POST `/v1/images/edits` with `response_format=url` returns data URL

`tests/modalities/image_generation/test_hf_plugin.py`:
- `_resolve` includes `supports_inpainting: True` and `supports_variations: True` in synthesized capabilities

`tests/modalities/image_generation/test_client.py`:
- `ImageEditsClient.edit()` POSTs multipart with image, mask, prompt fields
- `ImageEditsClient.edit()` returns list[bytes] decoded from b64_json
- `ImageEditsClient.edit()` raises RuntimeError on non-200 response
- `ImageVariationsClient.vary()` POSTs multipart with image only
- `ImageVariationsClient.vary()` returns list[bytes]
- both clients honor `MUSE_SERVER` env var and trim trailing slashes

`tests/cli_impl/test_e2e_images_edits_variations.py` (slow, in-process):
- Full multipart round trip through FastAPI + codec for both routes

`tests/integration/test_remote_images_edits_variations.py` (opt-in via MUSE_REMOTE_SERVER):
- protocol: edits returns valid PNG+envelope
- protocol: variations returns valid PNG+envelope
- observe: edits actually changes pixels in the masked region
- observe: variations produces a visually-similar-but-different image

## Open questions

None. The OpenAI SDK already exposes `images.edit(...)` and `images.create_variation(...)`; the multipart wire shape is documented and stable.

## Out of scope (filed for later)

- Probe defaults for inpaint/variations: today's PROBE_DEFAULTS exercises only the t2i path. Adding inpaint/variations probes needs a synthetic image+mask. Future enhancement.
- A `muse.modalities._common.uploads` shared module: with two multipart modalities now (audio_transcription, image_generation), the case for extraction is stronger. Can land as a follow-up sweep.
