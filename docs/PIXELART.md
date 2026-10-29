# Pixel-art assets recipe

Goal: from a text prompt to a game-ready sprite and a terrain tileset, all on
your own muse server. The pieces are `klein-pixel-art` (image generation),
`pixel-tools` (the `image/pixelart` modality: `/v1/images/pixelize` and
`/v1/images/autotile`, CPU only, no weights).

```bash
muse pull klein-pixel-art     # FLUX.2 klein 4B + the pixel-art LoRA (needs a GPU)
muse pull pixel-tools         # venv only; nothing to download
export MUSE=http://localhost:8000
```

## 1. Generate a sprite

The curated model prepends `pixel art sprite, ` to your prompt unless you
already wrote it. Output is 512x512 and soft: AI "pixel art" is rarely on a
true grid.

```bash
curl -s $MUSE/v1/images/generations -H 'content-type: application/json' -d '{
  "model": "klein-pixel-art",
  "prompt": "a knight in plate armor, full body, plain background",
  "size": "512x512", "response_format": "b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 -d > knight_raw.png
```

`lora_scale` (extra field) turns the style up or down per request.

## 2. Pixelize with a hard palette

Pixelize finds the true grid (or use `target_size=64x64`), snaps to it, maps to
a palette and makes alpha binary.

```bash
curl -s $MUSE/v1/images/pixelize -F model=pixel-tools -F image=@knight_raw.png \
  -F palette=db32 -F response_format=png -o knight.png
```

Palettes: `pico-8`, `db16`, `db32`, `endesga-32`, `nes`, `gameboy`, `auto:N`
(N colours picked from the image), `none`, or `#rrggbb,#rrggbb,...`. Useful
extras: `outline=black`, `seamless=true` (tileable), `sheet=4x2` (shared
palette across several images in one call), `preview_scale=8` (nearest-
neighbour enlargement for viewing; use 1 for the game asset). With the default
`response_format=json` the grid, palette and size come back too.

## 3. Autotile a grass fill over dirt

Generate two seamless textures, then hand them to autotile. `base` is what
shows where there is no grass.

```bash
for n in "grass" "dirt path"; do
  curl -s $MUSE/v1/images/generations -H 'content-type: application/json' -d "{
    \"model\": \"klein-pixel-art\", \"size\": \"512x512\", \"response_format\": \"b64_json\",
    \"prompt\": \"seamless $n texture, top-down, pixel art\"}" \
    | jq -r '.data[0].b64_json' | base64 -d > "${n%% *}.png"
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
- Input side is capped by `limits.pixelart_max_input_side` (default 4096).
- `scripts/bench/pixel_compare.py --server $MUSE --cases scripts/bench/pixel_cases.yaml
  --out /tmp/pixel` renders a contact sheet (raw vs pixelized, and a random
  16x12 map painted from the blob tiles) for judging a model or palette.
