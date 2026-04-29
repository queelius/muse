# `/v1/images/edits` + `/v1/images/variations` Implementation Plan (#100)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mount `POST /v1/images/edits` (inpainting) and `POST /v1/images/variations` (alternates) on the existing `image/generation` modality, mirroring the v0.17.0 img2img pattern. Both routes are multipart/form-data; both reuse the diffusers pipeline via `from_pipe` to share VRAM.

**Architecture:** Two new routes mount inside the existing `image_generation/routes.py:build_router`. Runtime gains `inpaint()` (lazy `AutoPipelineForInpainting.from_pipe`) and `vary()` (delegates to existing img2img with empty prompt + strength 0.85). Capability flags `supports_inpainting` and `supports_variations` gate each route. The bundled `sd_turbo.py` mirrors the runtime additions.

**Spec:** `docs/superpowers/specs/2026-04-28-images-edits-variations-design.md`

**Target version:** v0.21.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/image_generation/image_input.py` | modify | add `decode_image_file(UploadFile)` async helper |
| `src/muse/modalities/image_generation/protocol.py` | modify | optional: extend Protocol with inpaint/vary (kept duck-typed) |
| `src/muse/modalities/image_generation/runtimes/diffusers.py` | modify | add `inpaint()`, `vary()`, lazy AutoPipelineForInpainting sentinel |
| `src/muse/modalities/image_generation/routes.py` | modify | mount `/v1/images/edits` and `/v1/images/variations` |
| `src/muse/modalities/image_generation/client.py` | modify | add `ImageEditsClient`, `ImageVariationsClient` |
| `src/muse/modalities/image_generation/__init__.py` | modify | export the two new clients |
| `src/muse/modalities/image_generation/hf.py` | modify | resolve adds supports_inpainting + supports_variations |
| `src/muse/models/sd_turbo.py` | modify | manifest caps + Model.inpaint() + Model.vary() |
| `tests/modalities/image_generation/test_image_input.py` | modify | + tests for decode_image_file |
| `tests/modalities/image_generation/runtimes/test_diffusers.py` | modify | + tests for inpaint() + vary() |
| `tests/models/test_sd_turbo.py` | modify | + tests for inpaint() + vary() + caps |
| `tests/modalities/image_generation/test_routes.py` | modify | + tests for /v1/images/edits and /v1/images/variations |
| `tests/modalities/image_generation/test_client.py` | modify | + tests for ImageEditsClient + ImageVariationsClient |
| `tests/modalities/image_generation/test_hf_plugin.py` | modify | + test for new capability flags |
| `tests/cli_impl/test_e2e_images_edits_variations.py` | create | slow e2e in-process |
| `tests/integration/test_remote_images_edits_variations.py` | create | opt-in remote integration |
| `pyproject.toml` | modify | bump 0.20.0 -> 0.21.0 |
| `src/muse/__init__.py` | modify | docstring v0.21.0 |
| `CLAUDE.md` | modify | document /v1/images/edits + /v1/images/variations |
| `README.md` | modify | document the two new routes |

---

## Task A: `decode_image_file` helper + tests

Pure helper, easiest to test in isolation. Used by routes for multipart uploads.

**Files:**
- Modify: `src/muse/modalities/image_generation/image_input.py`
- Modify: `tests/modalities/image_generation/test_image_input.py`

- [ ] **Step 1: Append failing tests**

In `tests/modalities/image_generation/test_image_input.py`, add tests that build a fake `UploadFile`-shape via `starlette.datastructures.UploadFile`. Tests should cover: PNG decode, empty-file rejection, oversized rejection, undecodable bytes rejection.

- [ ] **Step 2: Implement decode_image_file**

In `src/muse/modalities/image_generation/image_input.py`, add an async helper:

```python
async def decode_image_file(file, *, max_bytes: int = _DEFAULT_MAX_BYTES):
    raw = await file.read()
    if not raw:
        raise ValueError("empty image file")
    if len(raw) > max_bytes:
        raise ValueError(f"image bytes exceeds max ({len(raw)} > {max_bytes})")
    return _bytes_to_pil(raw)
```

- [ ] **Step 3: Run targeted + fast lane**

```bash
pytest tests/modalities/image_generation/test_image_input.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 4: Commit (Task A)**

```
feat(image-gen): decode_image_file async helper for multipart uploads (#100)
```

---

## Task B: Runtime `inpaint()` and `vary()`

Extend the generic `DiffusersText2ImageModel` with the two new methods. Lazy-load `AutoPipelineForInpainting` via `from_pipe` to share VRAM with the loaded t2i pipeline.

**Files:**
- Modify: `src/muse/modalities/image_generation/runtimes/diffusers.py`
- Modify: `src/muse/modalities/image_generation/protocol.py` (optional Protocol additions)
- Modify: `tests/modalities/image_generation/runtimes/test_diffusers.py`

- [ ] **Step 1: Append failing tests**

In `tests/modalities/image_generation/runtimes/test_diffusers.py`, add:

- `test_inpaint_uses_inpainting_pipeline` (patches `AutoPipelineForInpainting`, asserts `from_pipe` called)
- `test_inpaint_caches_pipeline` (second call doesn't call from_pipe again)
- `test_inpaint_uses_from_pipe_not_from_pretrained_to_share_vram`
- `test_inpaint_normalizes_rgba_mask_to_grayscale`
- `test_inpaint_bumps_steps_to_satisfy_strength_contract`
- `test_vary_delegates_to_img2img_with_empty_prompt_and_default_strength_0_85`
- `test_vary_returns_imageresult_with_mode_variations`

- [ ] **Step 2: Add module-level sentinel + `_ensure_deps` extension**

In `src/muse/modalities/image_generation/runtimes/diffusers.py`:

```python
AutoPipelineForInpainting: Any = None
```

Extend `_ensure_deps` to lazy-import it.

- [ ] **Step 3: Add `inpaint()` and `vary()` methods**

```python
def inpaint(self, prompt, *, init_image, mask_image,
            negative_prompt=None, width=None, height=None,
            steps=None, guidance=None, seed=None, strength=None,
            **_):
    # Lazy-load + cache self._inp_pipe via from_pipe(self._pipe)
    # Normalize mask to L mode
    # Bump steps for strength contract
    # Call self._inp_pipe(prompt=..., image=..., mask_image=..., strength=..., ...)
    # Return ImageResult with metadata.mode = "inpaint"


def vary(self, *, init_image, width=None, height=None,
         steps=None, guidance=None, seed=None, strength=None,
         **_):
    result = self._generate_img2img(
        prompt="", init_image=init_image,
        strength=strength if strength is not None else 0.85,
        steps=steps, guidance=guidance, seed=seed,
        negative_prompt=None,
    )
    result.metadata["mode"] = "variations"
    return result
```

- [ ] **Step 4: Optional Protocol additions**

In `src/muse/modalities/image_generation/protocol.py`, the Protocol stays duck-typed (the runtime can satisfy it structurally). Adding Protocol methods is optional and only useful if external callers want type checking. Skip unless wanted.

- [ ] **Step 5: Run targeted + fast lane**

```bash
pytest tests/modalities/image_generation/runtimes/test_diffusers.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 6: Commit (Task B)**

```
feat(image-gen): inpaint() and vary() methods on DiffusersText2ImageModel (#100)
```

---

## Task C: Bundled `sd_turbo.py` updates

Mirror the runtime additions on the bundled SD-Turbo script. Add capability flags. Add `inpaint()` (similar to runtime). Add `vary()` (delegates to internal `_generate_img2img` with prompt="" and strength 0.85).

**Files:**
- Modify: `src/muse/models/sd_turbo.py`
- Modify: `tests/models/test_sd_turbo.py`

- [ ] **Step 1: Append failing tests**

In `tests/models/test_sd_turbo.py`, add:

- `test_manifest_advertises_supports_inpainting`
- `test_manifest_advertises_supports_variations`
- `test_sd_turbo_inpaint_uses_from_pipe`
- `test_sd_turbo_inpaint_caches_pipeline`
- `test_sd_turbo_inpaint_normalizes_rgba_mask_to_grayscale`
- `test_sd_turbo_vary_delegates_to_img2img_with_empty_prompt`

- [ ] **Step 2: Add module-level sentinel for AutoPipelineForInpainting**

```python
AutoPipelineForInpainting: Any = None
```

Extend `_ensure_deps` to import it.

- [ ] **Step 3: Update MANIFEST capabilities**

```python
"capabilities": {
    ...
    "supports_inpainting": True,
    "supports_variations": True,
    ...
}
```

- [ ] **Step 4: Add `inpaint()` and `vary()` to Model**

Mirror the runtime versions. `inpaint()` lazy-loads via `from_pipe(self._pipe)`. `vary()` delegates to `self._generate_img2img(prompt="", ..., strength=0.85)` and overrides `metadata["mode"]`.

- [ ] **Step 5: Run targeted + fast lane**

```bash
pytest tests/models/test_sd_turbo.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 6: Commit (Task C)**

```
feat(image-gen): sd_turbo bundled script gains inpaint() and vary() (#100)
```

---

## Task D: Routes `/v1/images/edits` and `/v1/images/variations`

Mount the two new routes on the existing `build_router`. Multipart parsing via FastAPI `UploadFile + Form`. Capability gates per route.

**Files:**
- Modify: `src/muse/modalities/image_generation/routes.py`
- Modify: `tests/modalities/image_generation/test_routes.py`

- [ ] **Step 1: Append failing route tests**

Add to `tests/modalities/image_generation/test_routes.py`:

- `test_post_edits_multipart_returns_envelope` (200, b64_json, revised_prompt echoes prompt)
- `test_post_edits_with_uncapable_model_returns_400`
- `test_post_edits_empty_image_returns_400`
- `test_post_edits_malformed_image_returns_400`
- `test_post_edits_n_creates_multiple_images`
- `test_post_edits_response_format_url_returns_data_url`
- `test_post_edits_unknown_model_returns_404`
- `test_post_variations_multipart_returns_envelope` (200, b64_json, NO revised_prompt)
- `test_post_variations_with_uncapable_model_returns_400`
- `test_post_variations_n_creates_multiple_images`
- `test_post_variations_unknown_model_returns_404`

The fixtures need to instantiate fake models with `inpaint` and `vary` methods that record calls, and register manifests with the right capability flags.

- [ ] **Step 2: Define request validators + handlers**

In `src/muse/modalities/image_generation/routes.py`:

Add the two new routes inside `build_router`. Use FastAPI's `UploadFile = File(...)` + `Form(...)` parameter shape (mirroring `audio_transcription/routes.py`). The size and n validators are simple regex/range checks; map to 400 with `error_response`.

```python
@router.post("/edits")
async def edits(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    model: str | None = Form(None),
    n: int = Form(1),
    size: str = Form("512x512"),
    response_format: str = Form("b64_json"),
):
    # Validate prompt, n, size, response_format
    # Look up model -> 404 ModelNotFoundError if missing
    # Capability gate: supports_inpainting must be True
    # Decode image + mask via decode_image_file
    # Call backend.inpaint(prompt, init_image=img, mask_image=mask, ...)
    # Encode envelope with revised_prompt


@router.post("/variations")
async def variations(
    image: UploadFile = File(...),
    model: str | None = Form(None),
    n: int = Form(1),
    size: str = Form("512x512"),
    response_format: str = Form("b64_json"),
):
    # Look up model -> 404 ModelNotFoundError if missing
    # Capability gate: supports_variations must be True
    # Decode image via decode_image_file
    # Call backend.vary(init_image=img, ...)
    # Encode envelope (no revised_prompt)
```

Both routes call inference via `asyncio.to_thread` to keep the event loop free, mirroring `/v1/images/generations`.

- [ ] **Step 3: Run targeted + fast lane**

```bash
pytest tests/modalities/image_generation/test_routes.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 4: Commit (Task D)**

```
feat(image-gen): mount /v1/images/edits and /v1/images/variations (#100)
```

---

## Task E: HTTP clients

`ImageEditsClient` and `ImageVariationsClient` mirror `GenerationsClient` but POST multipart. Both honor `MUSE_SERVER`.

**Files:**
- Modify: `src/muse/modalities/image_generation/client.py`
- Modify: `src/muse/modalities/image_generation/__init__.py`
- Modify: `tests/modalities/image_generation/test_client.py`

- [ ] **Step 1: Append failing client tests**

Add tests for both clients: default base url, MUSE_SERVER env var, trailing-slash trimming, multipart POST shape (image+mask for edits; image only for variations), b64_json decoding, error on non-200.

- [ ] **Step 2: Add ImageEditsClient and ImageVariationsClient**

In `src/muse/modalities/image_generation/client.py`:

```python
class ImageEditsClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0): ...

    def edit(
        self, prompt: str, *,
        image: bytes, mask: bytes,
        model: str | None = None, n: int = 1, size: str = "512x512",
        response_format: str = "b64_json",
    ) -> list[bytes]:
        files = {"image": ("image.png", image), "mask": ("mask.png", mask)}
        data = [("prompt", prompt), ("n", str(n)), ("size", size),
                ("response_format", response_format)]
        if model is not None:
            data.append(("model", model))
        # POST + decode b64_json -> list[bytes]


class ImageVariationsClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0): ...

    def vary(
        self, *,
        image: bytes,
        model: str | None = None, n: int = 1, size: str = "512x512",
        response_format: str = "b64_json",
    ) -> list[bytes]:
        files = {"image": ("image.png", image)}
        data = [("n", str(n)), ("size", size),
                ("response_format", response_format)]
        if model is not None:
            data.append(("model", model))
        # POST + decode b64_json -> list[bytes]
```

- [ ] **Step 3: Update `__init__.py`**

```python
from muse.modalities.image_generation.client import (
    GenerationsClient,
    ImageEditsClient,
    ImageVariationsClient,
)

__all__ = [..., "ImageEditsClient", "ImageVariationsClient", ...]
```

- [ ] **Step 4: Run targeted + fast lane**

```bash
pytest tests/modalities/image_generation/test_client.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 5: Commit (Task E)**

```
feat(image-gen): ImageEditsClient and ImageVariationsClient HTTP clients (#100)
```

---

## Task F: HF plugin capability defaults

Synthesize `supports_inpainting` and `supports_variations` in resolver-pulled diffusers manifests.

**Files:**
- Modify: `src/muse/modalities/image_generation/hf.py`
- Modify: `tests/modalities/image_generation/test_hf_plugin.py`

- [ ] **Step 1: Add the failing test**

Add to `test_hf_plugin.py`:

```python
def test_resolve_advertises_supports_inpainting_and_variations():
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("org/anything", None, info)
    caps = result.manifest["capabilities"]
    assert caps["supports_inpainting"] is True
    assert caps["supports_variations"] is True
```

- [ ] **Step 2: Update `_resolve` capabilities**

```python
capabilities = {
    **defaults,
    "supports_negative_prompt": True,
    "supports_seeded_generation": True,
    "supports_img2img": True,
    "supports_inpainting": True,
    "supports_variations": True,
}
```

- [ ] **Step 3: Run targeted + fast lane**

```bash
pytest tests/modalities/image_generation/test_hf_plugin.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 4: Commit (Task F)**

```
feat(image-gen): HF plugin advertises supports_inpainting + supports_variations (#100)
```

---

## Task G: Slow e2e + integration tests

Slow e2e exercises the full multipart -> FastAPI -> codec chain in-process with a fake backend. Integration tests hit a real muse server (opt-in via MUSE_REMOTE_SERVER).

**Files:**
- Create: `tests/cli_impl/test_e2e_images_edits_variations.py`
- Create: `tests/integration/test_remote_images_edits_variations.py`

- [ ] **Step 1: Slow e2e test**

`tests/cli_impl/test_e2e_images_edits_variations.py`:

```python
@pytest.mark.slow
def test_multipart_edits_flow_end_to_end():
    # Build a fake ImageModel with inpaint + vary
    # Register with supports_inpainting + supports_variations
    # POST a real PNG + mask via TestClient
    # Assert envelope + b64_json decodes to PNG magic
```

Two methods: one for /edits, one for /variations.

- [ ] **Step 2: Integration test**

`tests/integration/test_remote_images_edits_variations.py`:

```python
pytestmark = pytest.mark.slow

@pytest.fixture(scope="session")
def image_model(remote_health):
    model_id = os.environ.get("MUSE_IMAGE_MODEL_ID", "sd-turbo")
    if model_id not in (remote_health.get("models") or []):
        pytest.skip(...)
    return model_id

def test_protocol_edits_returns_png_envelope(remote_url, image_model):
    # Build PNG + mask in memory
    # POST multipart
    # Assert 200, b64_json decodes to PNG magic, revised_prompt echoes


def test_protocol_variations_returns_png_envelope(remote_url, image_model):
    # POST multipart with image only
    # Assert 200, b64_json decodes to PNG magic, no revised_prompt
```

- [ ] **Step 3: Run fast + slow lanes (slow only when local), integration is opt-in**

```bash
pytest tests/ -q -m "not slow"
pytest tests/ -q  # full lane including the slow e2e
```

- [ ] **Step 4: Commit (Task G)**

```
test(image-gen): slow e2e + opt-in integration for /edits and /variations (#100)
```

---

## Task H: Documentation + v0.21.0 release

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `pyproject.toml` (0.20.0 -> 0.21.0)
- Modify: `src/muse/__init__.py` (docstring v0.21.0)

- [ ] **Step 1: CLAUDE.md notes**

Add bullet under the `image/generation` description: "Also exposes `/v1/images/edits` (inpainting via `AutoPipelineForInpainting.from_pipe`) and `/v1/images/variations` (alternates via img2img with empty prompt) since v0.21.0. Multipart/form-data; gated by `supports_inpainting` and `supports_variations` capability flags."

Update the multipart-modalities note: now two modalities use multipart (audio_transcription, image_generation).

- [ ] **Step 2: README updates**

Add the two new routes to the table; add usage examples for the OpenAI SDK + the new clients.

- [ ] **Step 3: Bump version**

`pyproject.toml`: `version = "0.21.0"`. `src/muse/__init__.py` docstring: list `image/generation: /v1/images/generations, /v1/images/edits, /v1/images/variations`.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -q --timeout=300
```

- [ ] **Step 5: Em-dash check**

```bash
python -c "
import sys, pathlib
files = [
    'docs/superpowers/specs/2026-04-28-images-edits-variations-design.md',
    'docs/superpowers/plans/2026-04-28-images-edits-variations.md',
    'CLAUDE.md', 'README.md',
]
em_codepoint = chr(0x2014)
hits = []
for f in files:
    p = pathlib.Path(f)
    if not p.exists():
        continue
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if em_codepoint in line:
            hits.append((f, i, line))
sys.exit(1 if hits else 0)
"
```

Should exit zero (no em-dash codepoint U+2014 anywhere).

- [ ] **Step 6: Commit + tag + push + GitHub release**

```bash
git add CLAUDE.md README.md pyproject.toml src/muse/__init__.py
git commit -m "chore(release): v0.21.0

Mounts /v1/images/edits (inpainting) and /v1/images/variations
(alternates) on the image/generation modality. Both are multipart
POST routes. Inpainting uses AutoPipelineForInpainting.from_pipe to
share VRAM with the loaded t2i pipeline. Variations reuses the
existing img2img path with an empty prompt and high strength.

Capability flags supports_inpainting and supports_variations gate
each route. The HF resolver synthesizes both flags as True. The
bundled sd_turbo script also advertises both.

Closes #100."

git tag -a v0.21.0 -m "v0.21.0: /v1/images/edits + /v1/images/variations"
git push origin main
git push origin v0.21.0
gh release create v0.21.0 --title "v0.21.0: image edits + variations" --notes "..."
```

---

## Self-review checklist

1. **Spec coverage:** Tasks A-H implement all sections of the spec. Wire contract, runtime contract, capability flag, multipart decoding, migration, tests.
2. **Placeholder scan:** zero TBD/TODO/XXX/FIXME outside the self-review meta-text.
3. **Type consistency:** `image: UploadFile`, `mask: UploadFile`, `prompt: str = Form(...)`, `init_image: PIL.Image`, `mask_image: PIL.Image`. Pydantic Form validation -> route logic -> runtime kwargs -> diffusers pipe call.
4. **Migration safety:** every change is additive. Existing `/v1/images/generations` requests are untouched.
5. **Behavior preservation:** existing tests stay green without modification.
6. **Capability gating:** the `supports_inpainting=False` and `supports_variations=False` 400 paths are the only way clients learn a model can't do these. New diffusers models pulled via the plugin advertise True; old already-pulled diffusers models miss the capability until repulled (#138 territory).
7. **Multipart consistency:** both routes use FastAPI `UploadFile + Form` parameters, mirror `audio_transcription/routes.py`. python-multipart already in `muse[server]`.
8. **VRAM efficiency:** inpaint uses `AutoPipelineForInpainting.from_pipe(self._pipe)` (not `from_pretrained`). Variations reuses img2img (no third pipeline). Tests assert `from_pretrained` is NOT called on the inpaint class.
