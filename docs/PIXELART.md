# Pixel-art assets recipe

Goal: from a text prompt to a game-ready sprite and a terrain tileset, all on
your own muse server. The pieces are `klein-pixel-art` (image generation),
`pixel-tools` (the `image/pixelart` modality: `/v1/images/pixelize` and
`/v1/images/autotile`, CPU only, no weights).

```bash
muse pull flux2-klein-4b      # the FLUX.2 klein 4B base (needs a GPU); pull it first
muse pull klein-pixel-art     # the pixel-art LoRA on that base (refused until the base is pulled)
muse pull pixel-tools         # venv only; nothing to download
export MUSE=http://localhost:8000
```

## 1. Generate a sprite

The curated model prepends `pixel art sprite, ` to your prompt unless the
prompt already contains `pixel art sprite` anywhere (any case). Output is 512x512 and soft: AI "pixel art" is rarely on a
true grid.

```bash
curl -s $MUSE/v1/images/generations -H 'content-type: application/json' -d '{
  "model": "klein-pixel-art",
  "prompt": "a knight in plate armor, full body, plain background",
  "size": "512x512", "response_format": "b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 -d > knight_raw.png
```

`lora_scale` (extra field, 0 to 2, default 1.0) turns the style up or down per
request. It matches the LoRA card's ComfyUI strength (tested 0.85-1.4): the
curated entry's `lora_scale_factor` maps it to diffusers' adapter weight,
because diffusers applies the adapter's rsLoRA scaling (16x) and weights of
0.25 or more destroy the image. Even the top of the allowed range is too much:
at `lora_scale` 2 (adapter weight 0.125) sprites already look soft, with the
background bleeding into them, so stay near 1. `lora_scale: 0` is plain klein,
which with the trigger phrase already draws clean sprites.

## 2. Pixelize with a hard palette

Pixelize snaps the image to a grid, maps it to a palette and makes alpha
binary. Generated sprites have no consistent grid (their "pixels" drift between
about 6 and 10 px), so `grid=auto` usually finds none and keeps all 512x512
pixels. Give the size instead: `target_size=64x64` (about 8 px cells at 512) or
`96x96` for more detail. `grid=auto` is for art that really is on a grid, such
as an upscaled sprite.

```bash
curl -s $MUSE/v1/images/pixelize -F model=pixel-tools -F image=@knight_raw.png \
  -F target_size=64x64 -F palette=db32 -F response_format=png -o knight.png
```

`-F model=pixel-tools` is optional on both pixel routes: without it the server
uses its first enabled `image/pixelart` model.

Palettes: `pico-8`, `db16`, `db32`, `endesga-32`, `nes`, `gameboy`, `auto:N`
(N colours picked from the image), `none`, or `#rrggbb,#rrggbb,...`. Useful
extras: `outline=dark` (or `#rrggbb`), `seamless=true` (tileable), `sheet=4x2`
(a 4-column by 2-row grid of cells inside ONE image, sharing one grid and one
palette; when the image does not divide evenly every cell is trimmed to the
smallest, reported as `metadata.cropped`), `preview_scale=8` (nearest-
neighbour enlargement for viewing; use 1 for the game asset). With the default
`response_format=json` the grid, palette and size come back too.

Background keying floods inward from the image border, so background that the
sprite encloses (between a character's legs, inside the crook of an arm,
beside a staff) stays opaque. For characters add `-F remove_enclosed=true`: it
also keys regions of the background colour (same tolerance) that the flood
cannot reach, but only regions of at least 4 art pixels (or 0.05% of the
grid, whichever is larger), so a single background-coloured highlight
survives. Leave it off when the sprite itself has larger areas close to the
background colour (grey armour on a grey backdrop): they would be punched
out. It applies to `background=auto`; an explicit `background=#rrggbb` is
already a global key.

## 3. Autotile a grass fill over dirt

Generate two seamless textures, then hand them to autotile. `base` is what
shows where there is no grass.

```bash
for n in "grass" "dirt"; do
  curl -s $MUSE/v1/images/generations -H 'content-type: application/json' -d "{
    \"model\": \"klein-pixel-art\", \"size\": \"512x512\", \"response_format\": \"b64_json\",
    \"prompt\": \"seamless $n texture, top-down, pixel art\"}" \
    | jq -r '.data[0].b64_json' | base64 -d > "$n.png"
done
curl -s $MUSE/v1/images/autotile -F model=pixel-tools \
  -F fill=@grass.png -F base=@dirt.png -F tile_size=48 -F name=grass \
  -F response_format=zip -o grass.zip && unzip -o grass.zip
```

The zip holds `grass_a2.png` (RPG Maker A2 block, 2x3 tiles), `grass_blob47.png`
(47 tiles, 8 wide), `grass.tres` (Godot 4), `grass.tsx` (Tiled) and
`grass_layout.json` (each tile's `x`/`y` in tile units and its neighbour
`mask`). Both tileset files point at `grass_blob47.png`, so keep them together.
Edge options: `edge_width`, `outline`, `highlight`, `roughness`,
`corner_radius`, `seed`, and `palette` to palette-map the result.

To re-cut a block you already have, send it as `a2` instead of `fill`.

## 3b. A 4-direction walk sheet

`klein-pixel-walk` draws a whole character sheet in one 512x512 image: a 4x4
grid whose rows are walking down, left, right and up (3 frames each), with an
arms-raised, two jumping and a lying frame in the last column. It is a LoRA
on the undistilled `flux2-klein-base-4b` (28 steps, guidance 4.0), so pull
both (about 16 GB of weights). The card's template sentence is prepended for
you; describe only the character.

```bash
muse pull flux2-klein-base-4b && muse pull klein-pixel-walk
curl -s $MUSE/v1/images/generations -H 'content-type: application/json' -d '{
  "model": "klein-pixel-walk", "seed": 7, "response_format": "b64_json",
  "prompt": "A pixel art spritesheet of a witch with long black hair, wearing a pointed black hat and a dark green robe."}' \
  | jq -r '.data[0].b64_json' | base64 -d > witch_sheet_raw.png
curl -s $MUSE/v1/images/pixelize -F model=pixel-tools -F image=@witch_sheet_raw.png \
  -F sheet=4x4 -F grid=4 -F palette=auto:24 -F response_format=png -o witch_sheet.png
```

`grid=4` is the card's 4x downscale: a 128x128 sheet of 32x32 frames, one
shared palette. This output is on a real grid, so `grid=auto` finds the same
4 px cell. Each sheet takes about 45 s on a 12 GB RTX 3060 once loaded. Back
views are the least consistent frames (hair and headwear drift), as the card
warns. `lora_scale` 0.5 to 1.0 is safe; at 1.5 the sprites smear.

## 4. Import

- **Godot 4.** Drop `grass.tres` and `grass_blob47.png` into the project, then
  use the TileSet in a TileMapLayer. It is set up with a terrain set (match
  corners and sides), so paint with the Terrains tab.
- **Tiled.** Map > Add External Tileset, choose `grass.tsx`. It defines a
  wang set over the blob-47 sheet; pick it in the Terrain Sets view.
- **RPG Maker MZ.** `grass_a2.png` is the A2 layout (2x3 tiles). In the
  Database, Tilesets tab, load it into an A2 slot (A2 ground autotile)
  and paint with the Map Editor as usual. Use `tile_size=48` for MZ.

## Through MCP

`muse mcp` exposes `muse_pixelize_image` and `muse_autotile`. `muse_autotile`
defaults to `response_format=zip` because MCP saves outputs under content-hash
names, which would break the tileset files' reference to `<name>_blob47.png`;
unzip the saved file. Inputs come as `image_path` / `fill_path` / `base_path`
(or the matching `*_url`).

## Notes

- A non-curated `hf://` klein LoRA pulled with `--no-probe` starts under-sized
  (sized from the adapter weights until probed). Use the curated id or let the
  pull probe run.
- Input side is capped by `limits.pixelart_max_input_side` (default 4096), and
  so is the pixelize preview's long side (art size x `preview_scale`): both are
  413 `input_too_large`, checked before the preview is allocated.
- `scripts/bench/pixel_compare.py --server $MUSE --cases scripts/bench/pixel_cases.yaml
  --out /tmp/pixel` renders a contact sheet (raw vs pixelized, and a random
  16x12 map painted from the blob tiles) for judging a model or palette.
  `scripts/bench/pixel_walk_cases.yaml` does the same for four walk sheets.
