# Pixel assets acceptance, 2026-10-28

Isolated server on 127.0.0.1:8011 (catalog `~/mpx`, outside `~/.muse`), run from
the `feature/pixel-assets` worktree, on an RTX 3060 12 GB. Models: `pixel-tools`,
`flux2-klein-4b` and `klein-pixel-art`. The bar is the spec's "Acceptance" section
(docs/superpowers/specs/2026-10-28-retro-pixel-assets-design.md); misses are
reported below and the bar is unchanged. Scratch outputs (images, zips, logs) are
not committed. They are under
`.superpowers/sdd/2026-10-28-retro-pixel-assets/live/` (`live/` below).

**Contact sheet:** `live/pixel/index.html` (run 3, after the fixes; it also has
the CC0 A2 maps). Run 1, with the overdriven LoRA, is kept in
`live/pixel_run1_lora_overdriven/`. Quick-look composites are in `live/view/`.

## Summary

| # | Criterion | Result |
|---|---|---|
| 1 | Unit tests, offline | Pass: fast lane 6326 passed, 77 skipped (including 3 new tests and 3 updated) |
| 2 | klein-pixel-art: 8 sprite prompts | Pass only after fix 1. At the shipped default every image was near-black noise |
| 2 | pixelize db32 / auto:16 gives grid-clean sprites | **Miss with `grid=auto`**: 0/8 grids detected in run 3. Pass with `target_size=64x64` (8/8, by construction). The recipe now uses target_size (fix 3) |
| 2 | Palette and alpha | Pass: 56/56 outputs use only palette colours (db32 subset, or at most 16 for auto:16), and alpha is in {0, 255} |
| 3 | Generated grass over dirt: A2, blob-47, `.tres`, `.tsx` | Pass |
| 3 | Godot 4 loads the TileSet headless | Pass: Godot 4.7.2, 47/47 tiles with matching peering bits, 368/368 terrain-connect cells correct |
| 4 | CC0 A2 block to blob-47, `.tres`, `.tsx` | Pass: 2 blocks from a CC0 OpenGameArt pack; Godot and Tiled checks pass |
| 5 | MCP `muse_pixelize_image` / `muse_autotile` write usable files | Pass: files land in `mcp.output_dir`, and Godot loads the unzipped `.tres` |
| 6 | Weightless model: no download, loads, probes, listed | Pass |

Fixes made during acceptance (each test-first, fast lane green):

1. `4b3282a` fix(images): klein-pixel-art lora_scale 1.0 no longer destroys the image
   (`capabilities.lora_scale_factor`; curated entry 0.0625).
2. `b0083ea` fix(bench): pixel_compare's map painted a waffle of fill corners into
   every unfilled cell; it now uses the base texture.
3. `73649dd` fix(docs): the recipe and bench cases pixelize generated sprites with
   `target_size`, and the bench dirt prompt says "seamless".

## Step 1: Isolated server

- Worktree code everywhere. With `PYTHONPATH=<worktree>/src`,
  `catalog_pull._muse_repo_root()` resolves to the worktree. Each venv therefore
  gets `pip install -e /home/spinoza/github/muse-pixel[server]`, and
  `~/mpx/venvs/pixel-tools/bin/python -c "import muse"` loads
  `/home/spinoza/github/muse-pixel/src/muse`. No override was needed.
- `pixel-tools`: `HF_HUB_OFFLINE=1 muse pull pixel-tools` exited 0 in 27.6 s (pip
  only). Any HF call would have raised; none did, and no `~/mpx/weights` directory
  was created. Probe on CPU: 0.10 GB peak, load 0.0004 s. The catalog has
  `weights: "none"`, an empty `local/pixel-tools` and a code receipt
  (`source_sha256 11b71f6d…`). The receipt says `muse_version: 0.68.0` while
  pyproject says 0.69.0, because it reads the host's installed dist metadata. This
  is a stale editable install of the main checkout, not a branch bug.
- `flux2-klein-4b`: `HF_HUB_CACHE=~/.muse/weights muse pull flux2-klein-4b
  --no-probe` took 7 min 8 s (venv build plus hashing 15 GB). Blobs were
  hardlinked (link count 2), so there was no weight download, only HF
  metadata GET/HEAD calls.
- `klein-pixel-art`: the pull took 8 min 2 s, with the probe on cuda at 8.03 GB
  peak and an 8.3 s load. The probe passed even though every image at the
  shipped default was noise; see the concerns.
  After fix 1, a re-pull took 3 min 52 s and the manifest gained
  `lora_scale_factor: 0.0625`.
- `muse serve --host 127.0.0.1 --port 8011`. Before every klein load, `nvidia-smi`
  showed 11.6-11.8 GB free and the owner's :8000 `/v1/models` had nothing loaded.
  Only one klein worker ran at a time (the server was restarted between the
  LoRA and base runs). The owner's server was never touched.

## Step 2: Contact sheet (`scripts/bench/pixel_compare.py`)

### Run 1: shipped default (lora_scale 1.0). Critical

All 8 sprites and both autotile textures were near-black noise
(`live/view/sprites_overview.png`). Pixelize "detected" cell 16 at offset (1, 12)
on 5 of 8 images, the same offset every time. That is a latent-patch artifact in
the noise, not art.

### LoRA diagnosis (seeded, same prompt, through the gateway)

| adapter weight handed to `set_adapters` | result (knight, seed 42) |
|---|---|
| 0 | clean pixel-art knight |
| 0.03 / 0.0625 / 0.09 | clean, with a visible style shift |
| 0.125 | softer, less pixel-like, checker bleeding |
| 0.25 | brown mush |
| 0.5 / 1.0 | near-black noise (mean RGB about 3 versus 160 at weight 0) |

The adapter does load and apply. There was no "unexpected keys" or "no LoRA keys"
warning, and all 172 keys are `transformer.*` with shapes that fit the 4B
transformer (hidden 3072, context_embedder in 7680). `load_lora_weights` runs
before `enable_model_cpu_offload` (the worker log shows that order), and
`set_adapters` changes the output on every request under model offload.

Cause: the safetensors' `lora_adapter_metadata` declares `use_rslora: true`
(r=64, alpha=128), so diffusers scales by alpha/sqrt(r) = 16. The card's tested
strengths (ComfyUI, 0.85-1.4) assume a scale of 1, and the card's own diffusers
snippet would produce noise.

Fix 1 maps the user's `lora_scale` to weight `lora_scale * 0.0625`.

Evidence after the fix (`live/lora_final/`):

- `lora_scale` 0 is bit-identical to `flux2-klein-4b` with the same prompt and seed
  (max pixel difference 0, seeds 42 and 7).
- `lora_scale` 0 versus 1: mean absolute difference 15-24 per channel.
- `lora_scale` 1 versus 2: mean absolute difference 15-23 per channel.
- Omitting `lora_scale` gives the same image as `lora_scale` 1.

### Run 3: final (`live/pixel/`)

Timings:

- Generation, hot: 12.5-12.9 s per 512x512 image (4 steps). The first request,
  including the cold worker, took 72.9 s; the base model's cold first request took
  76 s.
- Pixelize: 0.10-0.28 s.
- Autotile: 0.10 s.
- The 8 sprites plus 2 textures took 3 min 24 s end to end.

| sprite | grid=auto (cell, confidence) | db32 / auto:16 / 64x64 db32 colours used |
|---|---|---|
| knight | not detected (0.30) | 20 / 16 / 15 |
| mage | not detected (0.29) | 19 / 16 / 15 |
| slime | not detected (0.13) | 18 / 16 / 14 |
| goblin | not detected (0.47) | 23 / 16 / 18 |
| chest | not detected (0.40) | 22 / 16 / 15 |
| potion | not detected (0.00) | 24 / 16 / 11 |
| sword icon | not detected (0.03) | 16 / 16 / 12 |
| tavern sign | not detected (0.07) | 19 / 16 / 14 |

**Grid detection miss.** On clean klein output, `grid=auto` found a grid on 0 of 8
sprites in run 3 and 1 of 8 in run 2. The run-2 hit was the goblin at cell 3,
confidence 0.555, which is likely spurious.

The diagnosis: across periods 3-16 the harmonic comb peaks at the smallest
periods, at 0.1-0.57, and never at a single cell. Diffusion "pixels" drift
between about 6 and 10 px, so there is no period to find. Uniformity is high
(0.93-0.99), so the comb term is what fails.

Without a grid, pixelize keeps 512x512 and palette-maps it. That output is clean
in palette and alpha but is not on a grid (`live/view/knight_zoom.png` shows
sub-cell jaggies and speckle).

With `target_size=64x64` every sprite is grid-clean by construction (8/8, both
palettes; `96x96` was also run, in `live/pixel_target/`). The recipe's example
used auto, so the owner would have hit this. Fix 3 changes the recipe; the
detector itself was not changed.

**Palette and alpha.** All 24 run-3 outputs and the 32 `target_size` outputs
(64 and 96, db32 and auto:16) pass. Every opaque pixel's colour is in db32 (for
db32) or there are at most 16 colours (for auto:16), and alpha is exactly
{0, 255}. Backgrounds were removed on all of them (62-89% transparent). The
script and its output are in `live/palette_alpha_checks.txt`.

## Step 3: Autotile and Godot

Generated `seamless grass texture` (fill) over `seamless dirt texture` (base),
`tile_size=48`. This produced the A2 block (96x144), blob-47 (384x288, 8x6), the
`.tres`, the `.tsx` and the layout. Requesting the zip again gave a byte-identical
blob sheet.

Godot: the official 4.7.2-stable Linux x86_64 build.

- Zip sha256 `cadd3204e728a35d3f13adb7fd0d7902636b79f6b95c40c265eb73b6c35329e4`;
  its SHA-512 matches the release's `SHA512-SUMS.txt`.
- Binary sha256 `8d106cbe6144c2dc7e881d61d2429c1a8a76e6b22ef48bd5e48dcf934953f71e`.
- Method: `--headless --import`, then a `SceneTree` script (`live/godot/*/check.gd`).
- The script loads `res://grass.tres` and finds:
  - a TileSet with tile size 48, 1 terrain set (mode 0, match corners and sides)
    and 1 terrain;
  - an atlas pointing at `res://grass_blob47.png` with 47 tiles;
  - **47/47 tiles whose 8 peering bits equal the layout `mask`**.
- It then paints a random 24x18 blob region with `set_cells_terrain_connect`.
  **All 368 painted cells received the tile whose mask matches their
  neighbours.** Result: PASS.
- Tiled `.tsx`: 47/47 `wangid`s match the layout masks. `tilecount` is 48 because
  the 8x6 sheet has one empty cell.

## Step 4: Existing CC0 art

Source: "Tailfeather Studios PM Tileset" by Tailfeather Studios,
https://opengameart.org/content/tailfeather-studios-pm-tileset, licence **CC0**.

- File: https://opengameart.org/sites/default/files/tfs_pm.zip
  (sha256 `dde8fc3e86f2fc436c0fa0a725a1d7ee8a96fb994f53efa4084ebb7cadc9cd7d`).
- `A2.png` is 512x384, the VX Ace layout with 64x96 blocks of 32 px tiles.
- Two blocks were sent as `a2=` with `tile_size=32`: "Path" (x 64-128) and
  "stone path" (x 192-256). Each autotile took 0.02 s.

Godot 4.7.2: both PASS (47/47 peering bits, 368/368 terrain-connect cells). Tiled:
47/47 wangids each.

Both maps, painted over the pack's own grass tile, appear in the contact sheet
(`live/pixel/cc0-*`) and in `live/view/cc0_maps.png`. They are seamless. The stone
block shows correct inner corners.

The "Path" block's inner-corner quarter is plain fill in the source art, so its
concave corners have no border. That faithfully reproduces the source.

## Step 5: MCP

Real stdio session: the MCP SDK client ran `python -m muse.cli mcp --filter
inference` with `MUSE_SERVER=http://127.0.0.1:8011`. It listed 38 tools,
including `muse_pixelize_image` and `muse_autotile`. The autotile schema's
`response_format` defaults to `zip`.

- `muse_pixelize_image` (`image_path`, `palette=db32`) wrote
  `live/mcp/out/5d4ea25e3c42bf99.png` (512x512; grid not detected, as above).
  Its structured result has `revision: null` and `artifacts: 11b71f6d2bb8fda6`.
- `muse_autotile` (`fill_path`, `base_path`, `tile_size=32`, `name=grass`) wrote
  `live/mcp/out/544c92ad83b69e1d.zip`. Unzipped, `grass.tres` references
  `res://grass_blob47.png` and `grass.tsx` references `grass_blob47.png`. Both
  files are present.
- Godot 4.7.2 loads that `.tres`: PASS (47/47 and 368/368).
- Cosmetic: the text block prints `revision: unknown` for the weightless model,
  where the structured result has `null`.

## Step 6: Weightless model

- `/v1/models`: `pixel-tools` has `revision: null` and `unservable_reason: null`.
- `/v1/capabilities?modality=image/pixelart` describes `/v1/images/pixelize`
  (`mcp_tool: muse_pixelize_image`) and `/v1/images/autotile`
  (`muse_autotile`, `mcp_defaults: {response_format: zip}`, text media fields
  for `.tres` and `.tsx`). The model entry has
  `components: [{role: code, repo: muse/pixel-tools, revision: null}]`.
- Pixelize response headers: `x-muse-model: pixel-tools` and
  `x-muse-model-artifacts: 11b71f6d2bb8fda6`. There is no
  `x-muse-model-revision`.
- The cold load to the first answer took 5.7 s, and an exact 6x round trip
  (16x16 random sprite) was detected as cell 6 at confidence 1.0.

## Quality verdict (retro fantasy RPG)

**Sprites.** With `lora_scale` 1 (after the fix) or even 0, klein draws
convincing, readable 16-bit-style sprites at 512. Pixelize gives a hard palette
and clean alpha with no halos. At `target_size=64x64` they read as real pixel
art: the chest, potion, sword, slime and tavern sign are game-ready.

Full-body characters (knight, goblin, mage) lose their faces at 64 px. Use
96x96 for characters.

Problems:

- Enclosed background is kept: the gap between the mage's staff and body, between
  the goblin's legs, and the triangle between the tavern sign's ropes. The flood
  starts at the image border only.
- db32 can shift hues: teal patches on the slime in run 2, and an orange heart on
  the potion. auto:16 stays truer.
- Without `target_size` the output is not grid art.
- The LoRA itself adds little: `lora_scale` 0 (plain klein plus the trigger
  phrase) is about as good. Its quality tier "draft" is fair.

**Autotiles.** The geometry is right: generated, CC0 and MCP tilesets all paint
correctly in Godot's terrain system, and the blob transitions on the painted maps
are continuous.

On the generated grass, faint horizontal banding is visible every tile row.
The texture repeat shows, and the seamless pass's equal first and last rows read
as lines; the grass texture is also very busy at 48 px. The dirt reads well.

The CC0 pack's A2 blocks expand cleanly with no visible seams. For a polished RPG
map, generated textures need hand touch-up or a calmer fill prompt. The tooling is
sound.

## Concerns

- The pull probe passed for a model whose every output was noise; it checks that
  the model runs, not the image. A cheap guard (for example "not almost all one
  colour") would have caught this.
- `grid=auto` is weak on AI art by design (no period exists). If the owner expects
  auto to work on generated sprites, the detector needs a different approach
  (for example a scale search on reconstruction error). Out of scope here.
- `/v1/models` shows the synthesized description ("Diffusers LoRA adapter: …") for
  curated URI entries rather than the curated text. This predates the branch
  (`flux2-klein-4b` does the same).
- The code receipt's `muse_version` comes from installed dist metadata (0.68.0 on
  this host versus pyproject 0.69.0).

## Walk-sheet spike

**Verdict: GO, 4 of 4.** All four sheets read as consistent 4-direction walk
cycles after pixelize: knight, witch, goblin and elf ranger. Down, left, right
and up rows hold one character with stable colours and outfit, the left and
right rows really face opposite ways, and the three walk frames alternate the
legs. A second seed (11) for each character gave four more coherent sheets, so
the result is not one lucky seed.

Setup:

- `flux2-klein-base-4b` (curated, `reference_runtime: true`).
- `klein-pixel-walk` (svntax-dev/pixel_spritesheet_4walk_small_lora_v1, the
  final 3000-step weight only).
- 512x512, 28 steps, guidance 4.0, `lora_scale` 1.0, seed 7, on the local
  RTX 3060 under model CPU offload.
- The prompt is "A pixel art spritesheet of <character>.", with the card's
  4x4 template sentence prepended by `prompt_prefix`.
- Pixelize: `sheet=4x4 grid=4 palette=auto:24`, giving a 128x128 sheet of
  32x32 frames. That is the card's 4x downscale.

Results:

- **Grid detection.** Unlike klein-pixel-art, this LoRA draws on a real 4 px
  grid. `grid=auto` detected cell 4 on all four sheets.
- **Adapter scale.** It is a plain ai-toolkit LoRA: rank 32, no `.alpha`
  keys, no rsLoRA metadata. So no `lora_scale_factor` is needed.
- **Seeded sweep.** At 0 the output is bit-identical to `flux2-klein-base-4b`
  (max diff 0), and the base alone draws an unstructured 3-row sheet. 0.5 and
  1.0 give clean 4x4 sheets, and 1.5 smears.
- **Timings.**
  - Base pull: 30 min, mostly the download. Probe peak 8.98 GB.
  - LoRA pull: 6.5 min, mostly the venv. Probe peak 8.04 GB.
  - Generation: 43-52 s per sheet once loaded, 87-142 s for the first request
    including the worker load. Pixelize takes 0.5-3 s.

Weak points:

- Back-view frames drift, as the card warns: the knight's plume and the elf's
  hood.
- The auto:24 palette shifts the knight's orange face toward tan.
- Props (bow, dagger) show up only in the jump and back frames.

Outputs are under `.superpowers/sdd/2026-10-28-retro-pixel-assets/live/walk/`:
`sheet/index.html` is the contact sheet, `sheet/<case>/v0.png` the pixelized
sheets, `sweep/` the scale sweep and `seed11/` the second seed. The cases are
in `scripts/bench/pixel_walk_cases.yaml`.
