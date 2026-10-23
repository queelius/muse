# Local image models: design

Date: 2026-10-22. Status: draft for owner review.
Research notes behind the model choices: the comparison table in this doc's
"Model choices" section (sources listed at the end).

## Intent

The YouTube pipeline (`~/github/youtube-channel`, read-only to muse work)
makes almost all of its stills with OpenAI gpt-image-2: 1,056 of 1,067
logged images. Muse's image models (sd-turbo, sdxl-turbo) were too weak to
use. gpt-image-2 is cheap (roughly $15-25 for all of those stills, an
estimate) and it stays the choice for published stills.

The owner still wants muse to have a modern local image model, for two
reasons:

1. **Volume.** Being able to make many more images later, even at lower
   quality than gpt-image-2: more stills per reel, plus thumbnails,
   backgrounds and B-roll frames.
2. **Test bed.** A place to try new open image models as they come out and
   compare them with gpt-image-2 on the channel's real prompts.

Success means:

- the pipeline can send any image call, including reference-image calls,
  to muse just by changing the model and endpoint in its config;
- the result is a big step up from sd-turbo and sdxl-turbo, on the same
  12 GB card;
- there is a repeatable way to see muse and gpt-image-2 side by side on
  the same prompts.

It is not expected to match gpt-image-2 on understanding a scene.

## Goals

1. Serve **FLUX.2 klein 4B** (`black-forest-labs/FLUX.2-klein-4B`,
   Apache-2.0) on the 12 GB RTX 3060. It must handle text-to-image and
   multi-reference editing with up to 4 reference images.
2. Make `/v1/images/edits` accept the request gpt-image clients already
   send: one or more `image[]` parts, no mask, a prompt and a size. When a
   mask is sent, inpainting keeps working as today.
3. Accept the OpenAI gpt-image request fields on `/v1/images/generations`
   and `/v1/images/edits`: `quality`, `output_format`,
   `output_compression`, `background`, `moderation`, `size: "auto"` and
   `n`. Each one is either honoured or safely ignored; none is rejected
   for being unknown.
4. Handle wide and tall sizes properly, for every model:
   - 1536x1024, 2048x1152 and 1280x720;
   - 9:16 portrait (1152x2048, 720x1280);
   - snap each side to a size the model accepts, and report the size
     actually used.
5. Add a **comparison script** that runs a set of prompts, optionally with
   reference images, through one or more muse models and, when an OpenAI
   key is present, through gpt-image-2. It writes a side-by-side contact
   sheet, an HTML page plus the images, with timings and model revisions.
6. Describe the new model honestly to agents: curated notes,
   `quality_tier`, limits, the DISCOVERY text for the edits route, and the
   MCP edit tool taking `image_paths`.

## Non-goals

- Matching gpt-image-2 quality, or calling OpenAI from muse. Muse never
  proxies to a hosted provider. The comparison script calls OpenAI
  directly, and only from the operator's machine.
- Readable text inside images. The pipeline wants none.
- Moderation. `moderation` is accepted and ignored, because muse has no
  safety classifier on this route. The response says so (below).
- Transparent backgrounds. `background: "transparent"` returns 400
  `background_not_supported`; `"auto"` and `"opaque"` are accepted.
- Training or LoRA work on klein. The base-4B checkpoint exists for that,
  but it's out of scope here.
- The A100s. Everything is designed for and verified on the 12 GB 3060.
- Changes to youtube-channel. Those belong to the owner (see "Pipeline
  side").

## Anti-goals (failure even if it technically works)

- Breaking current clients: sd-turbo, sdxl-turbo or LoRA generation, or
  masked inpainting through `/v1/images/edits` with a single `image` and a
  `mask`.
- A request that silently drops its reference images and returns an
  unrelated picture. A model that can't use references must answer 400,
  not ignore them.
- Shipping a non-commercial model as a default or recommended pick
  without saying so. The channel may be monetized, so defaults are
  permissively licensed only.
- A model that only works on a bigger GPU while its notes claim it fits
  12 GB.

## Constraints

- **Hardware:**
  - one NVIDIA RTX 3060 (12 GB VRAM) and 31 GB host RAM;
  - klein needs `enable_model_cpu_offload()`: a bf16 pipeline is about
    16 GB fully resident, with an offload peak of about 8 GB (the Qwen3-4B
    text encoder);
  - with offload, host RAM holds the parked components (about 16 GB).
- **Dependencies:**
  - klein needs `Flux2KleinPipeline`, from diffusers 0.37 or newer; the
    card says to install from git if no release has it;
  - the dev box has diffusers 0.36.0;
  - these pins live only in the model's own `pip_extras` (per-model venv).
    Do not raise muse's shared `images` extra if the older models would
    break under it.
- **Muse conventions** (CLAUDE.md):
  - deferred heavy imports with module-level sentinels;
  - helpers from `muse.core.runtime_helpers`;
  - an OpenAI error envelope via `error_response`;
  - DISCOVERY declarations, checked by the meta-test against mounted
    routes;
  - curated notes with `quality_tier`;
  - FakeModel-pattern route tests;
  - mocked runtime tests;
  - isolated catalogs in tests.
- **Host-RAM accounting:** the LoadDirector charges a model to a single
  pool. Klein is charged to cuda (about 8 GB, from the probe), and the
  roughly 16 GB that offload parks in host RAM isn't counted. Wan already
  works this way, and CLAUDE.md lists it as a known limitation. *My call:*
  accept it here. Klein's curated `limits` say so, and the real-GPU check
  confirms klein loads while the box's usual CPU-pool models are resident.
  Charging both pools is a separate LoadDirector project, out of scope.
- **Sizing:** the klein repo ships both a single-file
  `flux-2-klein-4b.safetensors` (7.75 GB) and the diffusers `transformer/`
  copy. The pull downloads only the diffusers layout (an allow-list), so
  disk and the weights-on-disk sizing don't double count.
- **Process rules:**
  - ask before pushing or tagging;
  - never print the admin token;
  - don't touch the Codex worktree or the VirtualBox VMs.

## What the owner decided

- Do the images project, then narration, then the vision critic.
- Use cases: both more stills per reel and other uses (thumbnails,
  backgrounds, B-roll).
- Keep gpt-image-2 for published stills; muse is the volume and test-bed
  path.
- Design for the 12 GB card, not the A100s.
- Everything else was left to my judgement. The calls below are mine.

## Design

### Model choices (my calls)

| Role | Model | License | Why |
|---|---|---|---|
| Default, `quality_tier: good` | FLUX.2-klein-4B | Apache-2.0 | One model does text-to-image and multi-reference editing; 4 steps; fits 12 GB with offload |
| Draft | the same model at `quality: low` | | No faster permissive model that also edits was found; a second model adds cost for little gain |
| Stretch (optional, gated) | Qwen-Image-Edit-2511, quantized (Nunchaku int4 or GGUF) with the 4-step Lightning LoRA | Apache-2.0 | Strongest permissive multi-reference editor, but 20B: it fits only quantized, and the diffusers loading path is unverified |

Rejected for now:

| Model | Why not |
|---|---|
| FLUX.2-klein-9B | Non-commercial, and doesn't fit at bf16 |
| Qwen-Image-2.1 / -Turbo | Non-commercial Qwen Research License, despite the best capability (10 refs, native 2752x1536) |
| Z-Image-Turbo | No editing; its edit model isn't released; the transformer alone (about 12.3 GB) doesn't fit with plain offload |
| FLUX.1 Kontext-dev | Non-commercial |
| OmniGen2, Step1X-Edit, HiDream-E1 | Older, single-image, or too heavy |

Revisit Z-Image when Z-Image-Edit ships.

The stretch model is a **go/no-go spike on the 3060**. If it can be loaded
through diffusers (GGUF or Nunchaku) inside 12 GB VRAM and 31 GB RAM and it
returns a correct multi-reference edit, it ships as a curated entry with
`quality_tier: best`. If not, the spike's findings go in the doc and it
doesn't ship. Klein must not wait on it.

### Runtime

- Add a klein runtime under `muse/modalities/image_generation/runtimes/`.
  It satisfies the existing `ImageModel` protocol plus a new optional
  method:

  ```python
  def edit(self, prompt: str, *, images: list[Any], width: int | None = None,
           height: int | None = None, steps: int | None = None,
           guidance: float | None = None, seed: int | None = None,
           **kwargs) -> ImageResult: ...
  ```

  `images` is a list of PIL images, in request order. `generate` and
  `edit` share one loaded pipeline; klein takes `image=[...]` on the same
  pipeline call.
- Placement: the manifest declares `capabilities.cpu_offload: "model"`.
  Reuse `place_pipeline`/`resolve_offload_mode`, which today live in
  `video_generation/runtimes/_offload.py`. Moving them to a shared module
  is the builder's call, but the video override setting
  (`server.video_cpu_offload`) must not start applying to image models.
  Image models follow their own capability only.
- The runtime records `steps`, `guidance`, the size used and the number of
  reference images in `ImageResult.metadata`.
- The runtime is generic over diffusers pipelines that take a list of
  reference images, so the stretch model, or a later one, can use it with
  different capabilities. Whether it is one runtime class or one per
  family is the builder's call.

### Capabilities (new manifest keys)

| Key | Meaning |
|---|---|
| `supports_reference_images: bool` | `/v1/images/edits` without a mask is allowed |
| `max_reference_images: int` | klein: 4 |
| `size_multiple: int` | klein: 16 |
| `max_pixels: int` | klein: 4,194,304 (4 MP); the ceiling for snapping |
| `default_size` (existing key, `[w, h]` list or tuple, unchanged format) | klein: `[1024, 1024]` |
| `quality_presets: {low, medium, high: {steps?, max_pixels?}}` | maps OpenAI `quality` to model settings |

Klein's presets:
- `low`: 4 steps, capped at 1 MP;
- `medium`: 4 steps at full size;
- `high`: 6 steps at full size.

These are starting values; tune them in verification. A model without
presets ignores `quality` and records that.

All the new keys must be classified in
`muse.core.capability_keys.public_capabilities`. Its meta-test covers
bundled, curated and HF-plugin keys.

### `/v1/images/edits` behaviour

The request is multipart. Image parts arrive as `image` (one), `image[]`
(repeated, the OpenAI SDK shape for lists) or both. They're merged in
order. `mask` is now optional.

| Images | Mask | Model capability | Result |
|---|---|---|---|
| 1 | yes | `supports_inpainting` | inpaint, as today |
| >1 | yes | any | 400 `invalid_parameter`: a mask applies to one image |
| 1 to `max_reference_images` | no | `supports_reference_images` | `backend.edit(...)` |
| more than the max | no | `supports_reference_images` | 400 `too_many_images`; the message gives the max |
| any | no | no reference support | 400 `reference_images_not_supported`; the message names the enabled models that support it |

In every case:
- inputs go through `decode_image_file` and `validate_total_image_pixels`;
- decoded images are closed on every path (follow the existing
  `close_decoded_images` and `run_native_offload` pattern).

### Size handling (all image models)

- `size` accepts `WxH` or `"auto"`.
- An omitted `size` and `"auto"` behave the same way (*my call*):
  - for a model that declares `size_multiple` or `max_pixels`, they mean
    its `default_size`;
  - every other model keeps today's 512x512, so `pixel-art-xl` and the
    other LoRA models don't change.

  gpt-image SDK calls usually leave out `size`, so for klein they get a
  1024x1024 image rather than 512x512.
- For a model that declares `size_multiple` or `max_pixels`, the requested
  size is scaled down, keeping its aspect ratio, to fit `max_pixels` and
  the quality preset's cap. Each side is then rounded down to the
  multiple. Snapping never enlarges an image.
- The response reports the size actually used.
- Models that declare neither keep today's behaviour exactly, so current
  clients see no change.
- The per-side 64-2048 check stays.

### Request fields and response

- **`quality`:** `low`, `medium`, `high` or `auto`; anything else is 400.
  `auto` means `medium`.
- **`output_format`:** `png` (the default), `jpeg` or `webp`.
  `output_compression` (0-100) applies to jpeg and webp. Add webp to the
  codec.
- **`background`:** `auto` or `opaque` are accepted; `transparent` is 400
  `background_not_supported`.
- **`moderation`:** accepted and ignored.
- **`response_format`:** unchanged. gpt-image clients omit it and get
  `b64_json`.
- **Muse extensions** (`negative_prompt`, `steps`, `guidance`, `seed`)
  keep working on generations, and are added as optional form fields on
  edits. An explicit `steps` overrides the quality preset.
- **Response:** the existing `{created, data:[{b64_json|url,
  revised_prompt}]}` plus the gpt-image echo fields `size`, `quality`,
  `output_format` and `background`. A `metadata` object lists any request
  fields that were accepted but ignored, e.g.
  `{"ignored": ["moderation", "negative_prompt"]}`, so an agent can tell
  what had no effect. Klein is guidance-distilled, so `negative_prompt`
  and `guidance` have no effect there and are listed.
- **Headers:** the provenance headers from v0.67.0 apply unchanged.

### Discovery, notes, MCP

- **DISCOVERY for image/generation:**
  - the summary and `use_for` mention reference-image editing for
    consistent characters;
  - the edits route declares `file_fields: ["image[]", "mask"]`, so the
    MCP tool takes `image_paths` (a list) and `mask_path`;
  - the edits example becomes a reference edit;
  - **`muse.mcp.generated` must upload a `[]` file field as repeated file
    parts on multipart routes.** Today `_take_files` turns it into a list
    of data-URL strings in form fields, which was built for
    image_embedding's JSON route. JSON routes keep the current behaviour.
    *My call*, matching CLAUDE.md: the MCP process fetches and uploads
    files for multipart routes;
  - the meta-test that examples validate must still pass.
- **Curated entry** (`flux2-klein-4b`):
  - summary, strengths and limits, `quality_tier: good`;
  - the limits say plainly: below gpt-image-2 at understanding scenes;
    text in images unreliable; guidance and negative prompt ignored; first
    load is slow because of offload;
  - `memory_gb` comes from a real probe on the 3060.
- **HF plugin:** a repo whose `model_index.json` names
  `Flux2KleinPipeline` resolves to the klein runtime. Other diffusers
  repos keep today's `AutoPipelineForText2Image` path.
- **Probe:** `PROBE_DEFAULTS` keeps `generate("probe scene")`. For a model
  with `supports_reference_images`, the probe also runs one `edit` with
  one small generated reference image. The probe gate then proves both
  paths, so a pull fails if the edit path is broken.

### Comparison script

- Location: `scripts/bench/image_compare.py`, next to the existing
  benchmarks.
- **Input:** a YAML or JSON file of cases. Each case has `id`, `prompt`,
  an optional `avoid` (appended as "Do not include: ..." the way the
  pipeline does), optional `refs` (paths), `size` and `quality`.
  `--from-visuals <visuals.yaml>` reads cases from a youtube-channel
  visuals file read-only. Parsing that format is a best-effort
  convenience, not a contract.
- **Targets:** `--muse URL` with one or more `--model` ids, and
  `--openai gpt-image-2` when `OPENAI_API_KEY` is set. The key is only
  ever sent to `https://api.openai.com`.
- **Output:** a directory of images, a `results.json` (per case and model:
  path, seconds, size used, the `X-Muse-Model-Revision` header or the
  OpenAI request id, any error) and an `index.html` contact sheet.
- Failures in one cell are recorded and never abort the run.
- Later, the vision critic can score these pairs.

### Pipeline side (owner's work in youtube-channel, for reference only)

These changes are not part of this project:
- lift the reference-image gate (`image.py:84`) so it is per model, not
  limited to gpt-image models;
- send `quality` and `output_format` to muse too;
- map `style_avoid` to `negative_prompt` only for models that honour it;
- record the `X-Muse-Model-Revision` header in the provenance JSON.

## Verification

- **Fast lane:** route tests with fake models covering:
  - every row of the edits table;
  - `image` + `image[]` merging;
  - size snapping, including the never-enlarge rule and models that
    declare no snapping keys;
  - quality presets;
  - each output format;
  - the ignored-fields list;
  - `background: transparent`;
  - DISCOVERY and capability-key meta-tests.

  Runtime tests mock diffusers.
- **Real-GPU checks on the 3060,** recorded in the release notes:
  - pull `flux2-klein-4b`; the probe passes, including the edit probe;
  - generate at 1536x1024, 2048x1152, 1280x720 and 720x1280;
  - run a 3-reference edit using a real character sheet from
    youtube-channel;
  - report wall times and the peak VRAM;
  - run the comparison script on about 10 real prompts from one reel, with
    gpt-image-2 if a key is available, and give the owner the contact
    sheet.
- The OpenAI Python SDK's `client.images.edit(image=[f1, f2], prompt=...,
  model="flux2-klein-4b")` and `client.images.generate(..., quality="low",
  size="1536x1024")` work against muse unmodified.

## Left to the builder

- One runtime class or one per family.
- Where the shared offload helper lives.
- How webp gets encoded (Pillow).
- How the sniff detects klein.
- The exact preset step counts, after measuring.
- The contact-sheet layout.
- Whether the stretch spike uses GGUF or Nunchaku.

## Sources

- FLUX.2-klein-4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
  (Apache-2.0; `Flux2KleinPipeline`; Qwen3-4B text encoder),
  https://docs.bfl.ai/flux_2/flux2_overview (up to 4 refs, 4 steps)
- 9B non-commercial: https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
- Diffusers Flux2: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2
- Z-Image-Turbo (edit model unreleased): https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Qwen-Image-Edit-2511 (Apache-2.0): https://huggingface.co/Qwen/Qwen-Image-Edit-2511;
  quantized builds: https://huggingface.co/QuantFunc/Nunchaku-Qwen-Image-EDIT-2511,
  https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF,
  https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
- Qwen-Image-2.1 license (non-commercial): https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
