# Img2img extension on `/v1/images/generations`

**Date:** 2026-04-27
**Driver:** unlock image-to-image (init_image + denoise strength) on the existing `/v1/images/generations` route. OpenAI's API has no img2img endpoint; the cleanest path is to extend the generation route with optional `image` + `strength` fields that the OpenAI SDK can pass via `extra_body`. Foundation for "coherent image sequences" (chain frames N -> N+1) without committing to a separate animation modality.

## Goal

1. Extend the `GenerationsRequest` Pydantic model with two optional fields: `image` (data URL or HTTP URL) and `strength` (float, denoise level 0.0 to 1.0).
2. When `image` is present, the route uses `AutoPipelineForImage2Image` instead of `AutoPipelineForText2Image`. Strength controls how much the source image is preserved (0 = identity, 1 = ignore source).
3. The `DiffusersText2ImageModel` runtime grows a `generate(prompt, init_image=, strength=, ...)` method that branches on `init_image is not None` and lazily loads a separate img2img pipeline.
4. Route layer parses data URLs and HTTP URLs into PIL images before handing off to the runtime.
5. OpenAI SDK round-trip: clients pass `image="data:image/png;base64,..."` and `strength=0.5` via `extra_body=`.

## Non-goals

- `/v1/images/edits` (inpainting with mask). Different OpenAI route, different feature.
- `/v1/images/variations` (no prompt). Same.
- ControlNet conditioning (depth, edge, pose). Adds dependencies; future modality.
- IP-Adapter / face conditioning. Same.
- Multi-image inputs. Single image only for now.
- New endpoint `/v1/images/img2img`. Decision: extend the existing route. OpenAI SDK supports `extra_body` cleanly; preserves wire compat for `OpenAI(...).images.generate(...)`.

## Architecture

```
client.images.generate(
    prompt="moonlit version of this scene",
    model="sdxl-turbo",
    extra_body={"image": "data:image/png;base64,...", "strength": 0.6},
)
    |
    v  HTTP POST /v1/images/generations
    |  body has prompt, model, image, strength
    v
GenerationsRequest (pydantic) parses body
    |
    | route inspects req.image
    | if present: decode data URL / fetch HTTP URL into PIL.Image
    v
ImageModel.generate(prompt, init_image=PIL_or_None, strength=float, ...)
    |
    | if init_image is None: text-to-image (current path)
    | else: lazy-load AutoPipelineForImage2Image, call with image + strength
    v
ImageResult -> codec -> JSON response
```

## Wire contract

`GenerationsRequest` adds two optional fields:

| Field | Type | Default | Validation |
|---|---|---|---|
| `image` | `str | None` | `None` | data URL (`data:image/...;base64,...`) OR `https?://...` URL |
| `strength` | `float | None` | `None` | `0.0 <= strength <= 1.0` when set |

Validation rules:
- `image` and `strength` are independent (you can set strength without image, but it's silently ignored). Documented but not an error.
- `image` URL fetching has a 30-second timeout and a 10MB size cap (configurable via `MUSE_IMG2IMG_MAX_BYTES`).
- Decoded image must be a PIL-compatible format (PNG, JPEG, WebP). Decode failures return 400 with `code=invalid_parameter`, `type=image_decode_error`.

OpenAI SDK usage:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

# Text-to-image (unchanged)
r = client.images.generate(prompt="a cat", model="sdxl-turbo")

# Image-to-image (new, via extra_body)
r = client.images.generate(
    prompt="oil painting of a cat",
    model="sdxl-turbo",
    extra_body={
        "image": "data:image/png;base64,...",
        "strength": 0.7,
    },
)
```

Response shape unchanged. The same `data: [{b64_json: ...}]` envelope.

## Runtime contract

`DiffusersText2ImageModel.generate` grows two new keyword args:

```python
def generate(
    self, prompt: str, *,
    negative_prompt=None,
    width=None, height=None,
    steps=None, guidance=None, seed=None,
    init_image=None,         # NEW: PIL.Image | None
    strength=None,           # NEW: float | None
    **_,
) -> ImageResult: ...
```

Branch logic:
- `init_image is None`: existing text-to-image path (no behavior change).
- `init_image is not None`: lazily-load `AutoPipelineForImage2Image` using the same `local_dir` / `hf_repo`. Cache the loaded pipeline on the instance so repeated img2img calls don't re-load. Pass `image=init_image` and `strength=strength or 0.5` (default mid-strength) to the pipe call.

The two pipelines share weights internally (diffusers reuses the underlying model objects when both pipelines load from the same checkpoint), so the second pipeline is a small marginal cost (~100MB extra in some configurations, 0 in others).

When `width`/`height` are not specified for img2img, default to the input image's dimensions (don't resize). For text-to-image the existing default-from-manifest behavior applies.

## Capability flag

Per-model `capabilities.supports_img2img: bool` declares whether the runtime can handle img2img for this model. Defaults: `DiffusersText2ImageModel` advertises True for any sniff-matched diffusers repo. The bundled `sd_turbo.py` script gets a manual addition (`supports_img2img: True`) since SD-Turbo supports img2img natively.

When a request includes `image` for a model whose capability is `False`, return 400 with `code=invalid_parameter`, `message="model X does not support img2img; use one of: ..."`.

## Image decoding

New helper module `src/muse/modalities/image_generation/image_input.py`:

```python
def decode_image_input(value: str, *, max_bytes: int = 10 * 1024 * 1024) -> PIL.Image.Image:
    """Parse a data URL or HTTP URL into a PIL.Image.

    - data:image/{png,jpeg,webp};base64,XYZ -> b64 decode -> PIL
    - http(s)://... -> fetch via httpx with timeout, validate Content-Type, decode
    - exceeds max_bytes -> ValueError
    - decode failure -> ValueError
    """
```

The route layer wraps decode failures into 400 responses with the OpenAI error envelope.

`PIL` is already a pip_extra of the diffusers runtime. `httpx` is already a server-extras dep. No new dependencies.

## Migration / risk

- **Existing text-to-image requests**: zero impact. New fields are optional and default to None.
- **Bundled `sd_turbo.py`**: untouched in this task except for one new line in MANIFEST capabilities (`"supports_img2img": True`). The Model class needs a small extension to honor `init_image` + `strength`.
- **Resolver-pulled diffusers models**: every existing pulled model gets the new capability advertised because the plugin's resolve sets `supports_img2img: True` (next pull). Already-pulled models miss it until repulled (#138).

## Test coverage

New tests:

`tests/modalities/image_generation/test_image_input.py`:
- decode_image_input parses a data URL correctly
- decode_image_input fetches an HTTP URL (mocked httpx)
- decode_image_input rejects oversized images
- decode_image_input rejects non-image content types
- decode_image_input rejects unsupported MIME types

`tests/modalities/image_generation/runtimes/test_diffusers.py`:
- generate(init_image=PIL, strength=0.5) calls AutoPipelineForImage2Image (not Text2Image)
- generate(init_image=None) keeps using AutoPipelineForText2Image (no regression)
- generate(init_image=PIL) without strength defaults to 0.5
- second img2img call reuses cached pipeline (only one from_pretrained)

`tests/modalities/image_generation/test_routes.py`:
- POST with `image` data URL routes through img2img
- POST with `image` and `strength=0.7` propagates strength
- POST with `strength` but no `image` ignores strength (silent passthrough; no error)
- POST with `image` for a model whose `supports_img2img: False` returns 400
- POST with malformed data URL returns 400 with image_decode_error
- POST with oversized image returns 400

## Open questions

None. The OpenAI SDK + `extra_body` path is well-trodden (vllm, ollama, llama-cpp use it for tools, vision, etc.).

## Out of scope (filed for later)

- Multi-image conditioning (ControlNet, IP-Adapter): different feature; needs its own design.
- `/v1/images/edits` (inpainting with mask): OpenAI's named route; mask is a separate optional input. Could land later.
- `/v1/images/variations` (no prompt): OpenAI's named route; trivially built on top of img2img with empty prompt and high strength.
- Coherent-frame chaining as a server-side primitive: build the loop on the client side first, observe whether server-side state (a session id that holds the previous frame) actually helps before adding statefulness.
