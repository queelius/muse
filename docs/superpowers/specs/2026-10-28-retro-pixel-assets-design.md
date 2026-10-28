# Retro pixel assets: pixelize, autotiles, klein pixel LoRA

Status: design, 2026-10-28. The owner opted out of questions ("skip questions"),
so every choice not quoted from them below is marked **(my call)**.

## Intent

The owner is making a retro-style fantasy RPG with "autotiles and some of those popular
retro solutions". In their words, AI artwork and sprites "leave something to be desired
unless i download public free to use art packs". They want muse to help design the game's
art, audio/music/SFX, dialog and narration. They want to experiment with an LLM writing
lines that are then spoken, but "static assets are first thing".

This project covers only static visual assets. Audio, dialog and live LLM→TTS are later
projects, and this design must not block them. Research notes, with links:
`/tmp/claude-1000/-home-spinoza-github-muse/715aa660-bcda-4b0b-b0c0-93b3639bedad/scratchpad/retro-rpg-research.md`.
That path is a session scratch file. The links that matter are repeated below.

The research shows why AI output falls short:

- **Fake pixels.** Diffusion models paint at high resolution with soft "pixels" that sit
  off any grid, too many colours, and half-transparent edges.
- **Edges don't match.** No model reliably produces autotile sheets whose edges match
  across every neighbour combination.

Both are fixed best by deterministic code wrapped around the models, not by a bigger
model. Commercial tools (PixelLab, Retro Diffusion) set the quality bar and have no open
equivalent.

## Goals

1. **Pixelize.** Turn any image into real pixel art: AI output, a photo, or a sprite from
   a downloaded pack that needs recolouring. That means:
   - detect the true pixel grid and downscale to it;
   - map colours onto a fixed palette (named or custom) or an automatic palette of N colours;
   - make alpha binary;
   - remove the background;
   - clean up orphan pixels;
   - optionally add a 1 px outline;
   - optionally make the result tile seamlessly.

   It must also work on a whole sprite sheet with one shared grid and palette, so frames
   stay consistent.
2. **Autotile.** Produce engine-ready autotiles:
   - From a fill texture plus an edge style, build an RPG Maker MZ/MV A2 autotile block
     (2×3 tiles).
   - Accept an existing A2 block, for example from a downloaded pack.
   - Expand either one to the 47-tile "blob" set.
   - Export a Godot 4 TileSet (`.tres`, terrain set "match corners and sides") and a Tiled
     tileset (`.tsx`, mixed wang set) that reference the blob sheet.
3. **Klein pixel LoRA.** Serve FLUX.2 klein 4B with `Limbicnation/pixel-art-lora`
   (Apache-2.0, trained on CC0 data, distilled 4-step, about 325 MB) as a curated model.
   The klein runtime needs general LoRA support for this.
4. **No weights as a first-class model type.** A bundled model with no weights is a
   supported muse concept. Pulling it installs its venv only, and it loads, sizes, probes
   and reports provenance without a Hugging Face repo. Pixelize and autotile are served
   by one such model. The sfxr-style SFX synth planned for the audio project will reuse
   this seam.
5. **Agents can drive it.** Everything is reachable through the generated MCP tools, so
   an agent (Claude via `muse mcp`) can chain generate → pixelize → autotile and get
   files on disk.

## Non-goals

- Walk-cycle and sprite-sheet generation. It needs the undistilled `FLUX.2-klein-base-4B`
  plus `svntax-dev/pixel_spritesheet_4walk_small_lora_v1` at 20–50 steps. It is a final,
  optional go/no-go spike (Task "walk-sheet spike" below), not a requirement.
- Qwen-Image-Edit-2511 (20B, quantized on 12 GB) and its 32×48 walk-sheet LoRA.
- Training LoRAs.
- Audio, dialog and TTS.
- A1 (animated water), A3/A4 (walls) and A5 RPG Maker formats. Only A2 ground autotiles
  are in scope **(my call)**: A2 is the format the owner most likely means by "autotiles",
  and the others follow the same quadrant idea.
- Dithering. Output is flat-colour, nearest-palette **(my call)**: RPG Maker-style art
  rarely dithers, and dithering fights palette consistency across frames.
- A UI. The interfaces are muse HTTP routes, MCP tools and the bench script's HTML
  contact sheet.

## Anti-goals (it fails even if it "works")

- **Non-integer or blurry pixels.** Every output pixel must sit on the detected grid, with
  no anti-aliasing at all. Preview upscales use nearest-neighbour only.
- **Semi-transparent alpha.** Any alpha other than 0 or 255 in a pixelized or autotile
  output is a bug.
- **Seams.** A blob-47 tile placed next to any neighbour it can legally touch shows a seam
  or a mismatched edge.
- **Silent palette drift.** A sheet processed with a shared palette must use exactly that
  palette in every cell.
- **Unloadable exports.** Files that Godot 4 or Tiled cannot load.
- **Hidden changes to existing images.** No change to the image models' existing
  behaviour unless a request asks for the new feature.
- **Faked provenance.** A weightless model must still say what it is (code version), not
  pretend to have an HF revision.

## Constraints

- Local RTX 3060 12 GB only (no A100s). Pixelize and autotile run on the CPU.
- No new heavy dependencies: Pillow and numpy only, both already muse deps.
  `unfake`, `perfectPixel`, Blobsmith and similar tools are references, not dependencies
  **(my call)**: unfake pulls a Rust extension, and Blobsmith is closed and paid.
- Deferred-imports rule: `muse --help` and `muse pull` keep working without ML deps.
- Pillow 11 and 13 both supported. Never pass `mode=` to `Image.fromarray`, since
  Pillow 13 removed it.
- Tests are fully offline and use isolated catalogs, never `~/.muse`. Never signal the
  owner's server (`muse-local.service`). Ask before pushing or tagging.
- Everything must work through the gateway, so the owner's other LAN machine can use it.

## The how: decided parts

### Weightless models (core)

- **Manifest.** A bundled script may declare `MANIFEST["weights"] = "none"`, with no
  `hf_repo` (or `hf_repo: None`) and no `revision`.
- **Pull.** `muse pull <id>` creates the venv and installs `pip_extras`. It skips every
  download, creates an empty owned `local_dir` (so existing "files present" checks stay
  meaningful), and writes a provenance receipt of kind `code`, carrying the muse version
  and a SHA-256 of the model script's source.
- **Loading.**
  - `model_load_blocker` and provenance validation accept that receipt.
  - `X-Muse-Model-Revision` is omitted, since it is defined as a 40-hex HF commit.
  - `X-Muse-Model-Artifacts` carries the code digest.
  - `/v1/models` `revision` is `null`.
- **Sizing.** Comes from `capabilities.memory_gb`, which these models must declare.
- **Out of scope here.** Resolver pulls (`hf://...`) are unchanged.
- **Where the code goes.** The builder finds every place that assumes `hf_repo`; the
  research found `catalog_models.py`, `catalog_pull.py` and `catalog.py`. The seam stays
  generic: nothing names pixel art.

### New modality `image/pixelart` (package `image_pixelart`)

A bundled model `pixel-tools` serves both routes. It is CPU-pinned (`device: cpu`),
weightless, and declares `memory_gb: 0.5` **(my call)**. Both routes are multipart, like
`/v1/images/upscale`, and read uploads through the shared bounded-upload helpers. Image
size limits reuse `limits.image_input_max_bytes`. Pixelize also refuses inputs over 4096
px on either side with 413 **(my call)**, configurable via a new
`MUSE_PIXELART_MAX_INPUT_SIDE` setting.

#### `POST /v1/images/pixelize`

**Fields**

- `image` (file)
- `model`
- `grid`: `auto` (default) or an integer source-pixels-per-art-pixel ≥ 1
- `target_size`: `WxH`, an alternative to `grid`
- `palette`:
  - a named palette: `pico-8`, `db16`, `db32`, `endesga-32`, `nes`, `gameboy`;
  - `auto:N` (2 ≤ N ≤ 256, default `auto:32`);
  - `none`;
  - or a comma-separated list of `#rrggbb` values.

  The named palettes ship as data in the package, with their public hex values and
  sources noted.
- `background`: `auto`, `none`, `#rrggbb` (a chroma key), or `transparent-keep`. `auto`
  removes a uniform border colour only when the image's border pixels are at least 90%
  one colour (ΔE < 6) **(my call)**.
- `alpha_threshold`: 0–255, default 128
- `cleanup`: bool, default true. It removes isolated single pixels whose 8 neighbours all
  share one other colour.
- `outline`: `none`, `dark` (each outline pixel takes the darkest palette colour near the
  pixel it borders), or `#rrggbb`. Default `none`.
- `seamless`: bool. It makes the output wrap.
- `sheet`: `CxR`, which splits into C×R cells processed with one shared grid and palette.
  Output keeps the sheet layout.
- `preview_scale`: integer 1–16, nearest-neighbour only, default 1
- `response_format`: `json` or `png`

**Grid detection.** The defaults below are **(my call)**, and the builder may improve
the method within them.

- Search the dominant period of colour-edge positions on each axis: autocorrelation or
  FFT of gradient magnitude, as in perfectPixel and unfake.
- Snap the cell offset.
- Sample each cell by majority colour. This is the "k-centroid" downscale the 4walk LoRA
  author recommends.
- When no period is confident, fall back to `grid=1` and report it.

**JSON response.** `{id, object: "image.pixelize", model, data: [{b64_json}], grid: {cell,
offset_x, offset_y, confidence, detected: bool}, size: [w, h], palette: ["#rrggbb", ...],
background_removed: bool, metadata}`.

**Ordering is fixed** **(my call)**:

1. background removal
2. grid snap
3. palette mapping, nearest in OKLab
4. alpha binarize
5. cleanup
6. outline
7. seamless
8. preview scale

Every output satisfies the anti-goals. Seamless means: repeat the palette-mapped image
in a 3×3 layout, and no edge pixel's 4-neighbour across the wrap breaks the cleanup rule.
How seams get blended is the builder's choice.

#### `POST /v1/images/autotile`

**Fields**

- `model`
- `tile_size`: 16, 24, 32 or 48; default 48, MZ's size **(my call)**. It must be even,
  since a quadrant is half a tile.
- **Source**, exactly one of:
  - `a2`: an existing A2 block file of exactly 2T × 3T, used verbatim;
  - `fill`: a fill texture file, plus optional `base` texture file, plus edge style.
    `base` absent means the outside is transparent, like an RPG Maker A2 "decoration"
    autotile; present means opaque, a "ground" autotile.
- **Edge style:**
  - `edge_width` (px, default T/8)
  - `outline` (`none` / `dark` / `#rrggbb`, default `dark`)
  - `highlight` (bool: one inner pixel line lighter, default true)
  - `roughness` (0–1, default 0.3, seeded)
  - `corner_radius` (px, default T/4)
  - `seed`
  - `palette` (same grammar as pixelize; the output is palette-mapped when set)
- `name`: a safe slug used in file names, default `terrain`
- `response_format`: `json` or `zip`

**Behaviour**

- **Seamless fill.** Fill and base textures are made seamless and scaled or cropped to
  tile-multiple size, so a fill rendered across any run of tiles has no seam.
- **The A2 block.** It is drawn from analytic masks: a rounded inset rectangle and the
  inner-corner notches, broken up with roughness noise. The quadrant layout and the
  48-shape composition table must match RPG Maker MV's `Tilemap.FLOOR_AUTOTILE_TABLE`
  (`rpg_core.js`, MIT, rpgtkool/corescript), since MZ uses the same A2 layout. The
  builder copies the table with attribution.
- **Blob-47.** Compose the 47 distinct shapes from the A2 block. RPG Maker's 48 entries
  include one duplicate; the builder decides which entry to drop and documents it. Lay
  them out in a fixed 8-column sheet **(my call)**, with the layout returned as data:
  `[{index, x, y, mask}]`, where `mask` is the 8-bit neighbour bitmask in a documented
  bit order.
- **Godot `.tres`.** A `TileSet` with one `TileSetAtlasSource` pointing at
  `res://<name>_blob47.png`, one terrain set in `TERRAIN_MODE_MATCH_CORNERS_AND_SIDES`,
  and one terrain, with peering bits per tile set from `mask`.
- **Tiled `.tsx`.** One tileset image `<name>_blob47.png` and one `<wangset
  type="mixed">`, with each tile's `wangid` derived from `mask`.

**Responses**

- JSON: `{id, object: "image.autotile", model, tile_size, files: {a2: {b64_json},
  blob47: {b64_json}, godot_tres: <text>, tiled_tsx: <text>}, layout: [...], metadata}`.
- Zip: `<name>_a2.png`, `<name>_blob47.png`, `<name>.tres`, `<name>.tsx` and
  `<name>_layout.json`, with consistent cross-references, so unzipping into a Godot or
  Tiled project folder just works.

**MCP.** The DISCOVERY entries declare `media_fields` and `text_media_fields` so the
generated MCP tools write all four files. The builder checks how `mcp/generated.py`
names the saved text files; the `.tres` reference to the PNG must still resolve after
saving. If it can't, the MCP tool uses `zip` and says so in its description.

#### Probe

`PROBE_DEFAULTS` runs one small pixelize on a synthetic 4× upscaled 32×32 image. The
probe must detect `grid=4`. It also runs one autotile from a synthetic fill at T=16.

### Klein LoRA support and the curated pixel model

- **LoRA support in `DiffusersReferenceModel`.** It gains `lora_adapter` and
  `base_model`, with the same semantics, errors and catalog resolution as
  `runtimes/diffusers.py` (the base must be pulled).
  - Load with `load_lora_weights` before offload placement.
  - Per-request `lora_scale`, with the manifest default from `capabilities.lora_scale`.
  - Probe, sizing and the supervisor's LoRA-from-base sizing work as they do for
    SDXL LoRAs today.
- **HF plugin.** It must route a LoRA repo whose base tag is a klein repo to the
  reference runtime, not to the SDXL diffusers runtime. The research found
  `_lora_base_from_tags` in `image_generation/hf.py`.
- **New capability `prompt_prefix` (string).** When the prompt doesn't already contain it
  (case-insensitive), the route prepends it, and `metadata` reports that it did
  **(my call)**: the LoRA needs its trigger phrase.
- **Curated entry `klein-pixel-art`:**
  - `uri: hf://Limbicnation/pixel-art-lora`
  - `base_model: flux2-klein-4b`
  - `lora_scale: 1.0` (the card's tested range is 0.85–1.4)
  - `prompt_prefix: "pixel art sprite, "`
  - `default_size: [512, 512]` (the training size), 4 steps
  - `quality_tier: draft`
  - notes saying to send the output through `/v1/images/pixelize`

### Bench / acceptance script

`scripts/bench/pixel_compare.py` runs a YAML case file against a live server. Each case
is a prompt plus model plus pixelize options, or an autotile request. It writes
`results.json` and an `index.html` contact sheet: raw, then pixelized at 1× and at a
nearest-neighbour preview, with grid, palette and timings, plus autotile previews. The
autotile previews are blob-47 tiles randomly painted on a 16×12 map, so seams are
visible. It follows `scripts/bench/image_compare.py`.

### Docs

Update the following:

- `CLAUDE.md`: modality list, plus a paragraph like the other modalities get.
- `docs/AGENT_DISCOVERY.md`, if routes are listed there.
- `README`.
- A short `docs/PIXELART.md` recipe page: generate a sprite with klein-pixel-art, then
  pixelize with `palette=db32`, then autotile a grass fill over a dirt base and import it
  into Godot, Tiled or RPG Maker MZ.

## Acceptance (live, on the 3060, through the gateway)

1. **Unit tests pass, offline.** Exact round-trips:
   - a synthetic 4×/6×/8× nearest-upscaled sprite with jitter and blur is recovered at
     its original size with ≥ 98% pixel agreement;
   - palette mapping uses only palette colours;
   - alpha ∈ {0, 255};
   - every blob-47 edge matches its legal neighbours, checked by composing all legal
     pairs;
   - the `.tres` and `.tsx` parse, and their peering bits / wang ids agree with `mask`.
2. **Generation.**
   - `klein-pixel-art` generates 8 fantasy-RPG sprite prompts: knight, mage, slime,
     goblin, chest, potion, sword icon, tavern sign.
   - `pixelize` with `palette=db32` and `palette=auto:16` produces grid-clean sprites.
   - The contact sheet goes in front of the owner.
3. **Autotile.**
   - A klein-generated "seamless grass texture, top-down" fill over a "dirt path" base
     makes an A2 block, blob-47, `.tres` and `.tsx`.
   - The Godot 4 TileSet loads in a headless Godot 4 binary fetched into the scratch dir.
     If that can't run, the structural checks plus the reason are recorded instead.
4. **Existing art.** An A2 block from a CC0 pack (e.g. an OpenGameArt CC0 RPG Maker-style
   tileset, downloaded into scratch, licence recorded) expands to blob-47 + `.tres` +
   `.tsx`.
5. **MCP.** The generated tools `muse_pixelize_image` and `muse_autotile` (names follow
   the generator) write usable files.
6. **Weightless model.** `pixel-tools` pulls into a fresh isolated catalog with no network
   download, loads, probes, and shows in `/v1/models` and `/v1/capabilities`.
7. **Report.** The results go in `docs/benchmarks/2026-10-28-pixel-assets-acceptance.md`.

## Optional final task: walk-sheet spike (go/no-go)

- Add `flux2-klein-base-4b` as a curated reference-runtime model. This lifts the plugin's
  `base` exclusion for an explicit curated entry only, with 28-step presets **(my call)**.
- Add the 4walk LoRA as `klein-pixel-walk`. Its `prompt_prefix` is the card's spritesheet
  template sentence; the builder takes it from the model card.
- Generate 4 characters, then pixelize them with `sheet=4x4`.
- **Go** if at least 3 of 4 sheets read as consistent 4-direction walk cycles after
  pixelize, judged by me from the contact sheet and then by the owner.
- **No-go**: record the reason, and keep or remove the entries based on the result.

Watch disk and RAM: klein-base is another ~16 GB of weights, about 194 GB disk is free,
and offload parks about 16 GB in host RAM.

## Left to the builder

- The grid-detection algorithm details within the stated contract.
- The A2 mask drawing.
- Exact error codes, which must follow muse's `{"error": {code, message, type}}` style
  and existing naming (`invalid_parameter`, `invalid_image`, ...).
- Where the 47-shape duplicate is dropped.
- The internal module split. Suggested: `pixelize.py`, `palettes.py`, `autotile.py`,
  `exporters.py`.
