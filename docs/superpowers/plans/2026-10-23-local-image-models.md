# Local Image Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve FLUX.2 klein 4B on a 12 GB GPU with text-to-image and multi-reference edits behind the OpenAI gpt-image request shape, plus a muse-vs-gpt-image-2 comparison script.

**Architecture:**
- **Request options.** A pure options module (size snapping, quality presets, output format, background, ignored-field reporting) is shared by `/v1/images/generations` and `/v1/images/edits`.
- **Edits route.** `/v1/images/edits` gains mask-less multi-reference dispatch to a new optional `edit()` backend method.
- **Klein runtime.** A new diffusers runtime loads any pipeline class that takes a list of reference images. It is placed by an offload helper lifted out of video_generation.
- **Plugin, curated entry, MCP.** The HF plugin routes klein repos to it, a curated entry describes it, and the MCP client uploads list file fields as repeated multipart parts.

**Tech Stack:**
- Python 3.11, FastAPI, Pydantic v2, Pillow;
- diffusers >= 0.37 (`Flux2KleinPipeline`), torch, httpx;
- pytest, using the FakeModel pattern and mocked diffusers.

**Spec:** `docs/superpowers/specs/2026-10-22-local-image-models-design.md`

## Global Constraints

- **Hardware target:**
  - one RTX 3060, 12 GB VRAM, with 31 GB host RAM;
  - nothing is designed for or verified on A100s.
- **Licensing:** defaults and recommended picks are permissively licensed only. A non-commercial model is never a default.
- **Back-compat:**
  - sd-turbo, sdxl-turbo and LoRA generation behave exactly as today;
  - masked inpainting with one `image` plus `mask` behaves exactly as today;
  - so does an omitted `size` for models that declare neither `size_multiple` nor `max_pixels`.
- **References:** a request carrying reference images to a model without `supports_reference_images` gets a 400, never a silent drop.
- **Dependencies:**
  - klein's `pip_extras` pin `diffusers>=0.37.0` and `transformers>=4.51.0`;
  - muse's shared `images` extra in pyproject.toml is not changed.
- **Error envelope:** use `muse.core.errors.error_response(status, code, message)`.
- **Error codes:** use these exact strings:
  - `invalid_parameter`;
  - `too_many_images`;
  - `reference_images_not_supported`;
  - `background_not_supported`.
- **Deferred imports:**
  - heavy deps load through module-level sentinels and `_ensure_deps()`;
  - device and dtype helpers come from `muse.core.runtime_helpers`.
- **Capability keys:** every new capability key is classified in `muse.core.capability_keys`, which is enforced by the existing meta-test.
- **Tests:**
  - tests use isolated catalogs and never touch `~/.muse`;
  - worktree test runs use `PYTHONPATH=src`, because the editable install points at the main checkout.
- **Repos and process:**
  - `~/github/youtube-channel` is read-only;
  - never print the admin token;
  - ask before pushing or tagging.

## Review Focus

1. **A request with dozens of `image[]` parts.** It is rejected with `too_many_images` before any image is decoded, so no memory is spent on it. *Owner: Task 3.*
2. **An odd size on a snapping model,** such as `size=1000x1001` or `2048x2048` at `quality=low`. It is scaled down keeping the aspect ratio, and each side is a multiple of 16 and at least 64. The response reports the size actually produced. *Owners: Tasks 1 and 2.*
3. **An OpenAI SDK `images.edit(image=[f1, f2], ...)` call.** It sends parts named `image[]`; the route must accept them, and keep accepting the single-`image` form. *Owner: Task 3.*
4. **`output_format=webp` combined with `response_format=url`.** The data URL's MIME must be `image/webp`, not png. *Owner: Task 2.*
5. **MCP `muse_edit_image` with `image_paths` of 3 files.** The server receives 3 file parts, not form strings. *Owner: Task 4.*

---

### Task 1: Image request options and codec formats

**Files:**
- Create: `src/muse/modalities/image_generation/options.py`
- Modify: `src/muse/modalities/image_generation/codec.py` (`to_bytes`, `to_data_url`)
- Modify: `src/muse/core/capability_keys.py` (`PUBLIC_KEYS`, `HIDDEN_KEYS`)
- Test: `tests/modalities/image_generation/test_options.py`, `tests/modalities/image_generation/test_codec.py`

**Interfaces:**
- Produces (in `options.py`):
  - `class OptionError(ValueError)` with attributes `code: str` and `message: str`.
  - `parse_size(size: str | None, caps: Mapping) -> tuple[int, int]`:
    - `None` or `"auto"` gives `tuple(caps["default_size"])` when the model snaps, else `(512, 512)`;
    - `"WxH"` is checked to 64-2048 per side;
    - anything else raises `OptionError("invalid_parameter", ...)`.
  - `snaps(caps: Mapping) -> bool`: true iff `size_multiple` or `max_pixels` is in caps.
  - `resolve_quality(quality: str | None, caps: Mapping) -> tuple[str, dict]`:
    - returns `(effective_name, preset_dict)`;
    - `None` and `"auto"` mean `"medium"`;
    - a model without `quality_presets` returns `(quality or "auto", {})`;
    - a value outside `low`/`medium`/`high`/`auto` raises `OptionError("invalid_parameter")`.
  - `snap_size(width: int, height: int, caps: Mapping, preset: Mapping) -> tuple[int, int]`.
  - `check_output(output_format: str | None, output_compression: int | None) -> str`:
    - returns the format, defaulting to `"png"`;
    - a format other than `png`/`jpeg`/`webp`, or compression outside 0-100, raises `invalid_parameter`.
  - `check_background(background: str | None) -> str`:
    - `None` and `"auto"` give `"auto"`; `"opaque"` gives `"opaque"`;
    - `"transparent"` raises `OptionError("background_not_supported")`;
    - anything else raises `invalid_parameter`.
  - `ignored_fields(sent: Mapping[str, Any], caps: Mapping) -> list[str]`:
    - returns sorted names whose value is not None and which had no effect;
    - `moderation` is always listed;
    - `quality` is listed when caps lack `quality_presets`;
    - `negative_prompt` is listed when `caps.get("supports_negative_prompt") is False`;
    - `guidance` is listed when `caps.get("supports_guidance") is False`;
    - `output_compression` is listed when the effective format is png. Pass it in `sent` as the `output_format` key.
- Produces (codec):
  - `to_bytes(image, *, fmt="png", compression: int | None = None) -> bytes`, accepting `png`, `jpeg`/`jpg` and `webp`. `compression` maps to Pillow `quality=` for jpeg and webp, and is unused for png.
  - `to_data_url(image, *, fmt="png", compression=None) -> str`, with MIME `image/png`, `image/jpeg` or `image/webp`.

- [ ] **Step 1: Write failing tests** in `test_options.py`. Use `caps = {"size_multiple": 16, "max_pixels": 4194304, "default_size": [1024, 1024], "quality_presets": {"low": {"steps": 4, "max_pixels": 1048576}, "medium": {"steps": 4}, "high": {"steps": 6}}}`:

```python
def test_parse_size_default_for_snapping_model(): assert parse_size(None, caps) == (1024, 1024)
def test_parse_size_auto_matches_omitted(): assert parse_size("auto", caps) == (1024, 1024)
def test_parse_size_default_for_legacy_model(): assert parse_size(None, {"default_size": [1024, 1024]}) == (512, 512)
def test_parse_size_rejects_out_of_range():
    with pytest.raises(OptionError) as e: parse_size("4096x64", caps)
    assert e.value.code == "invalid_parameter"
def test_snap_keeps_fitting_size(): assert snap_size(2048, 1152, caps, {}) == (2048, 1152)
def test_snap_rounds_down_to_multiple(): assert snap_size(2000, 1125, caps, {}) == (2000, 1120)
def test_snap_low_preset_caps_pixels(): assert snap_size(1536, 1024, caps, caps["quality_presets"]["low"]) == (1248, 832)
def test_snap_never_enlarges(): w, h = snap_size(100, 70, caps, {}); assert (w, h) == (96, 64)
def test_snap_odd_size_stays_valid():
    w, h = snap_size(1000, 1001, caps, caps["quality_presets"]["low"])
    assert w % 16 == 0 and h % 16 == 0 and w >= 64 and h >= 64 and w * h <= 1048576
def test_snap_noop_for_legacy_model(): assert snap_size(1000, 1001, {}, {}) == (1000, 1001)
def test_quality_auto_is_medium(): assert resolve_quality("auto", caps) == ("medium", {"steps": 4})
def test_quality_without_presets_passthrough(): assert resolve_quality("high", {}) == ("high", {})
def test_quality_invalid(): ...  # "ultra" -> OptionError code invalid_parameter
def test_background_transparent(): ...  # code == "background_not_supported"
def test_output_check(): assert check_output(None, None) == "png"; ...  # "gif" and compression 101 -> invalid_parameter
def test_ignored_fields():
    sent = {"moderation": "low", "negative_prompt": "x", "guidance": 3.0, "quality": "low", "output_format": "png", "output_compression": 80}
    assert ignored_fields(sent, {"supports_negative_prompt": False, "supports_guidance": False}) == \
        ["guidance", "moderation", "negative_prompt", "output_compression", "quality"]
    assert ignored_fields({"moderation": None}, caps) == []
```

  In `test_codec.py`, add `test_webp_roundtrip` (`Image.open(BytesIO(to_bytes(img, fmt="webp"))).format == "WEBP"`), `test_jpeg_compression_changes_size`, and `test_data_url_webp_mime` (the URL starts with `data:image/webp;base64,`).

- [ ] **Step 2: Run them and see them fail.** `PYTHONPATH=src pytest tests/modalities/image_generation/test_options.py tests/modalities/image_generation/test_codec.py -q`. Expected: ImportError, plus webp failures.

- [ ] **Step 3: Implement `options.py` and the codec changes.** Snapping algorithm:
  1. `m = caps.get("size_multiple") or 1`.
  2. `cap` is the smaller of `caps.get("max_pixels")` and `preset.get("max_pixels")`, among those set.
  3. If `w*h > cap`, scale both sides by `sqrt(cap / (w*h))`, flooring each.
  4. Round each side down to a multiple of `m`.
  5. Clamp each side to at least `max(m, 64)` rounded up to a multiple of `m`.

  Never enlarge beyond the input, except for that clamp. Add `"size_multiple"` and `"quality_presets"` to `PUBLIC_KEYS`, and `"pipeline_class"` to `HIDDEN_KEYS`.

- [ ] **Step 4: Run the tests again; all pass.** Also run `PYTHONPATH=src pytest tests/core/test_capability_keys*.py -q`.

- [ ] **Step 5: Commit** with message `feat(images): request options (size snapping, quality presets, formats)`.

### Task 2: Generations route uses the options

**Files:**
- Modify: `src/muse/modalities/image_generation/routes.py` (`GenerationRequest`, `generations`, and a new shared `_build_response`)
- Test: `tests/modalities/image_generation/test_routes.py`

**Interfaces:**
- Consumes: Task 1's `options` functions and codec signatures.
- Produces:
  - `_build_response(results: list[ImageResult], *, prompt: str | None, response_format: str, output_format: str, compression: int | None, quality: str, background: str, ignored: list[str]) -> dict`, used again by Task 3.
  - It returns `{"created", "data": [...], "size": f"{w}x{h}" of results[0], "quality", "output_format", "background", "metadata": {"ignored": ignored}}`.
  - A `None` prompt omits `revised_prompt`.
- `GenerationRequest` field changes:
  - `size: str | None = None`; the pattern now accepts `^(\d+x\d+|auto)$`, and range checking moves to `parse_size`.
  - New optional fields: `quality`, `output_format`, `output_compression`, `background`, `moderation`.
  - `OptionError` maps to `error_response(400, e.code, e.message)`.
- Steps precedence: request `steps`, else the preset's `steps`, else the backend default (pass `None`).

- [ ] **Step 1: Write failing tests.** Add a fake model registered with the Task 1 `caps` plus `supports_negative_prompt: False` and `supports_guidance: False`. It records its `generate` kwargs and returns an `ImageResult` of the requested size.

```python
def test_omitted_size_uses_model_default_for_snapping_model(...): assert (rec["width"], rec["height"]) == (1024, 1024)
def test_omitted_size_legacy_model_still_512(client): ...  # existing FakeImageModel gets width=height=512
def test_low_quality_snaps_and_reports(...):
    r = c.post(url, json={"prompt": "p", "model": "k", "size": "1536x1024", "quality": "low"}).json()
    assert r["size"] == "1248x832" and r["quality"] == "low" and rec["steps"] == 4
def test_high_quality_steps_and_explicit_steps_override(...): ...  # high -> steps 6; steps=3 -> 3
def test_webp_b64_and_url(...): ...  # b64 decodes as WEBP; response_format=url starts "data:image/webp;base64,"
def test_transparent_background_400(...): ...  # status 400, error.code == "background_not_supported"
def test_ignored_listed(...): ...  # moderation + negative_prompt -> metadata.ignored == ["moderation", "negative_prompt"]
def test_auto_quality_echo_medium(...): ...
def test_legacy_response_still_has_data_and_created(client): ...
```

- [ ] **Step 2: Run them and see them fail.** `PYTHONPATH=src pytest tests/modalities/image_generation/test_routes.py -q`.

- [ ] **Step 3: Implement.** Fold `_build_response` into the variations route only where it is behaviour-neutral. Variations keeps its current response keys, apart from the new echo fields. Existing tests must stay green; update a test only where it asserted exact equality on the old dict, and say so in the commit message.

- [ ] **Step 4: Run the image_generation tests and `tests/core/test_discovery_meta.py`; they pass.**

- [ ] **Step 5: Commit** with message `feat(images): gpt-image request fields on generations`.

### Task 3: Multi-reference `/v1/images/edits`

**Files:**
- Modify: `src/muse/modalities/image_generation/protocol.py` (add `ReferenceImageModel`)
- Modify: `src/muse/modalities/image_generation/routes.py` (`edits`)
- Modify: `src/muse/modalities/image_generation/client.py` (`ImageEditsClient`: accept a list of images and an optional mask)
- Test: `tests/modalities/image_generation/test_routes.py`, `tests/modalities/image_generation/test_client.py`

**Interfaces:**
- Consumes: Task 1's `options` functions and Task 2's `_build_response`.
- Produces:
  - A `ReferenceImageModel` Protocol with `edit(self, prompt: str, *, images: list[Any], width: int | None = None, height: int | None = None, steps: int | None = None, guidance: float | None = None, seed: int | None = None, negative_prompt: str | None = None, **kwargs) -> ImageResult`.
  - Route parameters:
    - `image: list[UploadFile] | None = File(None)`;
    - `image_list: list[UploadFile] | None = File(None, alias="image[]")`;
    - `mask: UploadFile | None = File(None)`.
  - Form fields:
    - `prompt` (required) and `model`;
    - `n`, `size` (default `None`) and `response_format`;
    - `quality`, `output_format`, `output_compression`, `background`, `moderation`;
    - `negative_prompt`, `steps`, `guidance`, `seed`.
- Module constant `MAX_EDIT_IMAGES = 16`, a hard cap for any model.

- [ ] **Step 1: Write failing tests,** one per row of the spec's dispatch table, plus:

```python
def test_inpaint_unchanged_single_image_and_mask(client_with_edits_model): ...  # inpaint called, 512x512 as today
def test_mask_with_two_images_400(...): ...  # code invalid_parameter
def test_reference_edit_dispatch_order(...):  # parts: image=a, image[]=b, image[]=c -> backend.edit images sizes in order a,b,c
def test_reference_edit_seed_offsets_with_n(...): ...  # seed=7, n=2 -> seeds 7, 8
def test_too_many_for_model_400(...): ...  # 5 images, max_reference_images 4 -> too_many_images, message contains "4"
def test_hard_cap_rejected_before_decode(monkeypatch, ...):  # 17 parts; patch decode_image_file to fail the test if called
def test_no_reference_support_400(client_with_edits_model): ...  # no mask -> reference_images_not_supported; message contains "/v1/capabilities?modality=image/generation"
def test_no_images_400(...): ...  # invalid_parameter
def test_edit_size_omitted_uses_default_and_snaps(...): ...  # 1024x1024; "1536x1024"+quality low -> 1248x832
def test_decoded_images_closed_on_error(...): ...  # backend.edit raises -> every decoded image's close() called
```

  Client test: `ImageEditsClient().edit(prompt, images=[b1, b2], model=...)` sends two `image[]` parts and no mask.

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement.**
  1. Merge the parts as `image` parts followed by `image[]` parts.
  2. Count before decoding: 0 parts or more than `MAX_EDIT_IMAGES` is a 400.
  3. With a mask:
     - more than 1 image is a 400;
     - otherwise run the existing inpaint path unchanged, still gated by `supports_inpainting`;
     - its size handling is `parse_size` with no snapping change, since inpaint models declare no snapping keys.
  4. Without a mask:
     - gate on `supports_reference_images`;
     - check the count against `max_reference_images`;
     - then `decode_image_file` each part, then `validate_total_image_pixels`.
  5. Resolve quality, size, background and output the same way generations does, then loop `n` times through `run_native_offload` with `backend._inference_lock`, as the current routes do.
  6. Close decoded inputs and results on every path.

  *Ruling vs the spec:* the "no reference support" message points at `GET /v1/capabilities?modality=image/generation` instead of naming models, because a worker's registry holds only its own model.

- [ ] **Step 4: Run the image_generation tests and `tests/core/test_discovery_meta.py`; they pass.**

- [ ] **Step 5: Commit** with message `feat(images): multi-reference edits on /v1/images/edits`.

### Task 4: MCP list file uploads and DISCOVERY text

**Files:**
- Modify: `src/muse/mcp/generated.py` (`_take_files` and its caller near line 375)
- Modify: `src/muse/modalities/image_generation/__init__.py` (`DISCOVERY`)
- Test: `tests/mcp/test_generated*.py` (the existing file for request building), `tests/core/test_discovery_meta.py` (it runs unchanged)

**Interfaces:**
- Consumes: Task 3's route parameter alias `image[]`.
- Produces:
  - On multipart routes, a `[]` file field named `X[]` sends one file part per path or URL, each named `X[]`.
  - `_take_files` returns a `list[tuple[str, tuple[str, bytes, str]]]`, which is what httpx `files=` accepts.
  - JSON routes keep sending a list of data URLs in `args[X]`, unchanged.

- [ ] **Step 1: Write failing tests:**
  - `test_multipart_list_field_sends_repeated_parts`: a fake route with `file_fields: ["image[]", "mask"]` and three temp PNGs under an allowed prefix. Capture the request with `httpx.MockTransport`; it has three parts named `image[]` and no form field `image`.
  - `test_json_list_field_still_data_urls`: an `input[]` field on a JSON route is still sent as data URLs.
  - `test_multipart_list_url_fetched`: `image_urls` are fetched through the existing `_fetch_input` and uploaded.

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement the generated.py change.** Then update DISCOVERY:
  - summary: `"Still image generation. Creates images from a prompt, edits with up to several reference images for consistent characters, inpaints with a mask, or makes variations, returned as base64 PNG, JPEG or WebP."`;
  - add to `use_for`: `"new scenes of the same character from reference images"`;
  - edits route: `file_fields: ["image[]", "mask"]`, with example `{"model": "<model>", "prompt": "the same character, now reading in a library", "size": "1536x1024"}`.

- [ ] **Step 4: Run `PYTHONPATH=src pytest tests/mcp tests/core/test_discovery_meta.py tests/core/test_capabilities.py -q`; it passes.**

- [ ] **Step 5: Commit** with message `feat(mcp): upload list file fields as repeated multipart parts`.

### Task 5: Shared offload helper and the reference runtime

**Files:**
- Create: `src/muse/modalities/_common/offload.py`, moved from `src/muse/modalities/video_generation/runtimes/_offload.py`. Delete the old file.
- Modify: `src/muse/modalities/video_generation/runtimes/wan_runtime.py`, `cogvideox_runtime.py`, and `src/muse/models/wan2_1_t2v_1_3b.py` if it imports the helper.
- Create: `src/muse/modalities/image_generation/runtimes/reference.py`
- Test:
  - move `tests/modalities/video_generation/test_offload.py` to `tests/modalities/_common/test_offload.py`;
  - update patch targets in the video runtime tests;
  - create `tests/modalities/image_generation/runtimes/test_reference.py`.

**Interfaces:**
- Produces (offload):
  - `resolve_offload_mode(capability_mode, *, override_key: str | None = "server.video_cpu_offload") -> str | None`;
  - `place_pipeline(pipe, device, *, cpu_offload=None, vae_tiling=False, override_key: str | None = "server.video_cpu_offload")`;
  - `override_key=None` means the capability alone decides.
- Produces (runtime):
  - `class DiffusersReferenceModel` at `muse.modalities.image_generation.runtimes.reference:DiffusersReferenceModel`;
  - constructor: `(*, hf_repo, local_dir=None, device="auto", dtype="bfloat16", model_id, pipeline_class="Flux2KleinPipeline", default_size=(1024, 1024), default_steps=4, default_guidance=1.0, cpu_offload=None, vae_tiling=False, max_reference_images=4, **_)`;
  - attributes: `model_id`, `default_size` (a tuple), `supports_reference_images = True`, `max_reference_images`;
  - `generate(prompt, *, negative_prompt=None, width=None, height=None, steps=None, guidance=None, seed=None, init_image=None, strength=None, **_) -> ImageResult`. A non-None `init_image` raises `ValueError`.
  - `edit(prompt, *, images, width=None, height=None, steps=None, guidance=None, seed=None, negative_prompt=None, **_) -> ImageResult`. It calls the pipeline with `image=list(images)`.
  - `ImageResult.metadata` holds `{"prompt", "steps", "guidance", "reference_images": int}`.
  - `negative_prompt` is passed only if the pipeline's `__call__` signature has that parameter.
  - The generator is `torch.Generator(device="cpu").manual_seed(seed)` when a seed is given. On the CPU generator: it is the reproducible choice under model offload.
  - The pipeline class is looked up with `getattr(diffusers_module, pipeline_class)` through module sentinels `torch` and `diffusers`, populated by `_ensure_deps()`.
  - Placement: `place_pipeline(pipe, device, cpu_offload=..., vae_tiling=..., override_key=None)`.

- [ ] **Step 1: Write failing tests:**
  - offload: the existing tests still pass from the new path, plus `test_override_key_none_ignores_video_override` (with `MUSE_VIDEO_CPU_OFFLOAD=off` and `cpu_offload="model"`, `enable_model_cpu_offload` is still called);
  - runtime, with a mock diffusers module whose `Flux2KleinPipeline.from_pretrained` returns a recording pipe:
    - `test_loads_named_pipeline_class_with_dtype`;
    - `test_model_offload_used_not_to`;
    - `test_generate_passes_size_steps_and_cpu_generator`;
    - `test_edit_passes_image_list_in_order`;
    - `test_generate_rejects_init_image`;
    - `test_negative_prompt_dropped_when_pipeline_lacks_it`;
    - `test_metadata_counts_references`.
  - Also run `tests/core/test_runtime_helpers_meta.py`, which flags re-implemented helpers.

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement.** Do the move, update importers and test patch targets (grep for `_offload`), then write the runtime following `runtimes/diffusers.py`'s conventions: `LoadTimer`, `dtype_for_name`, `select_device`.

- [ ] **Step 4: Run `PYTHONPATH=src pytest tests/modalities/video_generation tests/modalities/_common tests/modalities/image_generation tests/models/test_wan2_1_t2v_1_3b.py tests/core/test_runtime_helpers_meta.py -q`; it passes.**

- [ ] **Step 5: Commit** with message `feat(images): reference-image diffusers runtime; shared offload helper`.

### Task 6: HF plugin dispatch, curated klein entry, edit probe

**Files:**
- Modify: `src/muse/modalities/image_generation/hf.py` (`_resolve`, plus a new `_reference_family(repo_id) -> dict | None`)
- Modify: `src/muse/curated.yaml` (add `flux2-klein-4b` after `flux-schnell`)
- Modify: `src/muse/modalities/image_generation/__init__.py` (`PROBE_DEFAULTS`)
- Test: `tests/modalities/image_generation/test_hf_plugin.py`, `tests/core/test_curated*.py`, the capability-key meta-test, and a probe test beside the existing PROBE_DEFAULTS tests

**Interfaces:**
- Consumes: Task 5's runtime path `muse.modalities.image_generation.runtimes.reference:DiffusersReferenceModel`.
- `_reference_family(repo_id)` matches `flux.2-klein` or `flux2-klein`, case-insensitively, in the repo name. A match returns these capabilities:

```python
{"pipeline_class": "Flux2KleinPipeline", "default_size": [1024, 1024], "default_steps": 4,
 "default_guidance": 1.0, "cpu_offload": "model", "device": "cuda",
 "supports_reference_images": True, "max_reference_images": 4,
 "size_multiple": 16, "max_pixels": 4194304,
 "quality_presets": {"low": {"steps": 4, "max_pixels": 1048576}, "medium": {"steps": 4}, "high": {"steps": 6}},
 "supports_negative_prompt": False, "supports_guidance": False, "supports_seeded_generation": True,
 "supports_img2img": False, "supports_inpainting": False, "supports_variations": False}
```

- Klein's `pip_extras`: `("torch>=2.1.0", "diffusers>=0.37.0", "transformers>=4.51.0", "accelerate", "Pillow", "safetensors")`.
- The download uses the existing subfolder-only allow-list, non-fp16 branch, so the top-level single-file checkpoint is never fetched.
- `PROBE_DEFAULTS = {"shape": "default size, 1 image; plus 1 reference edit when supported", "call": _probe_call}`. `_probe_call(m)` runs `generate("probe scene")`. If `getattr(m, "supports_reference_images", False)` is set, it then runs `m.edit("the same scene at night", images=[result.image])`. It returns the generate result.

- [ ] **Step 1: Write failing tests:**
  - `test_klein_repo_resolves_to_reference_runtime` checks `backend_path`, `capabilities`, `pip_extras` and the license from the info;
  - `test_klein_download_excludes_top_level_single_file` checks the allow-list patterns all start with `model_index.json` or `*/`;
  - `test_other_repos_unchanged`: sdxl-turbo still resolves to `DiffusersText2ImageModel`;
  - `test_probe_runs_edit_when_supported` and `test_probe_skips_edit_otherwise`;
  - the curated tests load the `flux2-klein-4b` entry, which has `quality_tier: good` and a non-empty `limits`.

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement,** and add the curated entry:
  - `id: flux2-klein-4b`, `uri: hf://black-forest-labs/FLUX.2-klein-4B`, `modality: image/generation`, `size_gb: 16.0`;
  - `description: "FLUX.2 klein 4B: text-to-image and up to 4-reference editing, 4 steps, Apache 2.0"`;
  - `summary: "Local stills and consistent-character scenes from reference images, on a 12 GB GPU"`;
  - `strengths`:
    - "Text-to-image and multi-reference edits in one model (up to 4 references)";
    - "Wide and tall sizes up to 4 MP, snapped to multiples of 16";
    - "Apache 2.0; 4-step distilled";
  - `limits`:
    - "Below gpt-image-2 at understanding scenes and context";
    - "Text inside images is unreliable";
    - "Guidance and negative_prompt have no effect (distilled)";
    - "Runs with CPU offload on 12 GB: slow first load, and about 16 GB of host RAM that muse's memory accounting does not count";
  - `quality_tier: good`;
  - `capabilities: {memory_gb: 8.5}`. This is a placeholder; Task 8 replaces it with the probed value.

- [ ] **Step 4: Run the fast lane (`PYTHONPATH=src pytest tests -q -m "not slow"`); it passes.**

- [ ] **Step 5: Commit** with message `feat(images): FLUX.2 klein resolver, curated entry, reference-edit probe`.

### Task 7: Comparison script

**Files:**
- Create: `scripts/bench/image_compare.py`
- Test: `tests/scripts/test_image_compare.py`. Create `tests/scripts/__init__.py` if the directory is new, and import the script by path the way other script tests do; if none exist, use `importlib.util.spec_from_file_location`.

**Interfaces:**
- CLI options:
  - `--cases FILE` or `--from-visuals FILE` (the case source);
  - `--muse URL` (default: `$MUSE_SERVER`, else `http://localhost:8000`);
  - `--model ID` (repeatable);
  - `--openai MODEL` (optional);
  - `--out DIR` (required);
  - `--limit N`.
- Functions:
  - `load_cases(path) -> list[Case]`, accepting YAML or JSON;
  - `cases_from_visuals(path) -> list[Case]`, best effort: read one youtube-channel `content/**/visuals.yaml` read-only to learn the shape, and skip entries it can't map;
  - `run(cases, targets, out_dir, client: httpx.Client) -> list[dict]`;
  - `write_index(results, out_dir) -> Path`.
- `Case` is a dataclass with `id: str`, `prompt: str`, `avoid: str | None`, `refs: list[str]`, `size: str = "1536x1024"` and `quality: str | None`. The prompt sent is `prompt + f" Do not include: {avoid}."` when `avoid` is set.
- Requests:
  - with refs: multipart to `/v1/images/edits` with one `image[]` part per ref;
  - without refs: JSON to `/v1/images/generations`;
  - OpenAI targets go to `https://api.openai.com/v1`, with `Authorization: Bearer $OPENAI_API_KEY`, and never send `response_format` or `seed`.
- Results: each cell gives `{case, target, path, seconds, size, revision, request_id, error}`, written to `<out>/results.json`, with images at `<out>/<target>/<case>.png` and the page at `<out>/index.html`.

- [ ] **Step 1: Write failing tests** using `httpx.MockTransport`:
  - `test_generation_and_edit_requests_shaped` (refs give multipart `image[]` parts);
  - `test_openai_key_only_sent_to_openai_host`: the muse request has no Authorization header;
  - `test_openai_skipped_without_key`;
  - `test_cell_failure_recorded_run_continues`: a 500 on one case gives `error` set, and the other cases still produce files;
  - `test_index_html_lists_every_cell`;
  - `test_revision_header_recorded`.

- [ ] **Step 2: Run them and see them fail.** **Step 3: Implement.** **Step 4: Run them; they pass.**

- [ ] **Step 5: Commit** with message `feat(bench): image comparison contact sheet (muse vs gpt-image-2)`.

### Task 8: Docs, release notes, and real-GPU verification

This task runs after the branch is merged to main locally, from the main checkout, because pulling into `~/.muse` uses the editable install. The controller runs it, not a subagent.

**Files:**
- Modify: `CLAUDE.md` (the image/generation bullet and `/v1/images/edits` paragraph), `README.md` (the image section), `docs/AGENT_DISCOVERY.md` (if it lists the edit tool's inputs)
- Modify: `src/muse/curated.yaml` (`memory_gb` from the probe)

- [ ] **Step 1: Update the docs** to describe:
  - multi-reference edits and the dispatch table, in summary;
  - the size, quality and format fields;
  - `metadata.ignored`;
  - the klein entry and its 12 GB offload note;
  - the comparison script.

  Commit the docs.
- [ ] **Step 2: Pull and probe klein on muse-local.** Run `muse pull flux2-klein-4b`; it must exit 0, with the probe covering both generate and edit. Record the probed peak. Restart muse-local.
- [ ] **Step 3: Generate at 1536x1024, 2048x1152, 1280x720 and 720x1280,** through the gateway with the OpenAI SDK. Record wall time per size, the cold load time, and peak VRAM from `nvidia-smi`.
- [ ] **Step 4: Run a 3-reference edit** with a real character sheet from youtube-channel (read-only), via `client.images.edit(image=[...], model="flux2-klein-4b")`, and via MCP `muse_edit_image` with `image_paths`.
- [ ] **Step 5: Run the comparison script** on about 10 prompts from one reel, with `--openai gpt-image-2` only if `OPENAI_API_KEY` is set. Publish the contact sheet to the owner.
- [ ] **Step 6: Commit** the probed `memory_gb` and any preset step changes justified by the measurements. Record the measurements in the commit message.

### Task 9: Stretch spike, Qwen-Image-Edit-2511 go/no-go (throwaway, time-boxed to one session)

- [ ] **Step 1: Try loading it** in a scratch venv, outside the muse source tree: `QwenImageEditPlusPipeline` with a quantized transformer (diffusers GGUF `Q3_K_S`/`Q4_K_M` from `unsloth/Qwen-Image-Edit-2511-GGUF`, or Nunchaku int4), the Lightning 4-step LoRA, and model CPU offload. The run must stay within 12 GB VRAM and 31 GB RAM.
- [ ] **Step 2: Run one 3-reference edit** with the same sheet as Task 8. Record the peak VRAM, the host RAM and the seconds taken, and save the output beside klein's for the same prompt.
- [ ] **Step 3: Record a go or no-go** in an appendix to the spec, and commit it.
  - **Go** means it loads through diffusers alone and gives a correct edit within limits. It then becomes a new plan task: a curated entry on `DiffusersReferenceModel` with `pipeline_class: QwenImageEditPlusPipeline`, plus whatever loading capability the spike needed.
  - **No-go** ships nothing.
