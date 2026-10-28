# Retro Pixel Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make muse turn AI output and art-pack art into real pixel art and engine-ready
autotiles, and serve FLUX.2 klein with a pixel-art LoRA.

**Architecture:**
- **Core.** A generic "weightless model" seam lets a bundled script with `weights: "none"`
  pull, load, size and report provenance without any Hugging Face repo.
- **New modality.** `image/pixelart` is served by one weightless, CPU-pinned bundled model,
  `pixel-tools`. It exposes `/v1/images/pixelize` and `/v1/images/autotile`, built on
  pure Pillow+numpy libraries.
- **Klein LoRA.** The klein reference runtime gains LoRA support, plus a `prompt_prefix`
  capability, so a curated `klein-pixel-art` entry works.

**Tech Stack:** Python 3.10+, FastAPI, Pillow (11 and 13), numpy, diffusers
(`Flux2KleinPipeline`, `load_lora_weights` / `set_adapters`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-retro-pixel-assets-design.md` (read it; it is
the authority, and its "(my call)" items are binding defaults).

## Global Constraints

- No new dependencies for pixelize/autotile: Pillow and numpy only. `unfake`,
  `perfectPixel` and Blobsmith are references, not imports.
- Never pass `mode=` to `Image.fromarray` (Pillow 13 removed it). Tests must pass on
  Pillow 11.2 and 13.0.
- Deferred imports: `muse --help` and `muse pull` must work without numpy/PIL installed;
  modality `__init__` imports PIL/numpy lazily.
- Output invariants:
  - every pixelize/autotile output pixel has alpha 0 or 255;
  - preview scaling is nearest-neighbour;
  - a shared-palette sheet uses only that palette.
- Errors use `muse.core.errors.error_response(status, code, message)`, giving the
  `{"error": {code, message, type}}` shape; reuse codes `invalid_parameter` and
  `invalid_image`. Oversize input is 413 `input_too_large`.
- Limits:
  - pixelize/autotile inputs over `limits.pixelart_max_input_side` (new setting, env
    `MUSE_PIXELART_MAX_INPUT_SIDE`, default 4096) → 413;
  - byte cap `limits.image_input_max_bytes`.
- Defaults pinned by the spec:
  - `palette` `auto:32`, `alpha_threshold` 128, `cleanup` true, `outline` `none`
    (pixelize) / `dark` (autotile);
  - `tile_size` 48 ∈ {16, 24, 32, 48};
  - `edge_width` T/8, `corner_radius` T/4, `roughness` 0.3, `highlight` true;
  - `preview_scale` 1 ∈ [1, 16];
  - `auto:N` N ∈ [2, 256];
  - background `auto` keys a border colour only when ≥ 90% of border pixels are within
    ΔE < 6 of one colour;
  - pixelize fixed op order: background → grid snap → palette (nearest in OKLab) →
    alpha binarize → cleanup → outline → seamless → preview.
- Named palettes, exactly: `pico-8`, `db16`, `db32`, `endesga-32`, `nes`, `gameboy`,
  stored as data with a source note per palette.
- The A2 quadrant layout and 48-entry composition table follow RPG Maker MV
  `Tilemap.FLOOR_AUTOTILE_TABLE` (`rpg_core.js`, rpgtkool/corescript, MIT), copied with an
  attribution comment.
- `pixel-tools` manifest:
  - `model_id: "pixel-tools"`, `modality: "image/pixelart"`, `weights: "none"`, no
    `hf_repo`, no `revision`;
  - `capabilities: {device: "cpu", memory_gb: 0.5}`;
  - `pip_extras: ("Pillow>=10.0", "numpy")`.
- Curated `klein-pixel-art`:
  - `uri: hf://Limbicnation/pixel-art-lora`, `base_model: flux2-klein-4b`,
    `lora_scale: 1.0`;
  - `prompt_prefix: "pixel art sprite, "`, `default_size: [512, 512]`,
    `default_steps: 4`, `quality_tier: draft`.
- Tests are offline, use isolated catalogs (`MUSE_CATALOG_DIR` tmp), and never touch
  `~/.muse`.
- Never signal `muse-local.service` (the owner's server on :8000). Use only the local
  RTX 3060. Ask before pushing or tagging.
- Tests run with `PYTHONPATH=$PWD/src`, since the editable install points at the main
  checkout.

## Review Focus

1. **Grid-snap edge cases.** An input that is already 1:1 pixel art (no upscale) must come
   back unchanged in size with `grid.detected: false`, not shrunk. Owned by Task 2.
2. **Odd-sized and non-multiple inputs.** A 4×-upscaled sprite cropped by 1–3 px on any
   side (offset grid, partial edge cells) still recovers the original cells. Owned by
   Task 2.
3. **Palette inputs that are already exact.** A sprite already inside the requested
   palette is returned with identical colours, with no drift from the OKLab round-trip.
   Owned by Task 2.
4. **Transparent-PNG inputs from art packs.** RGBA sprites with existing hard alpha keep
   their transparent area. `background: auto` must not key out an opaque sprite colour
   that merely touches the border less than 90% of the time. Owned by Task 2.
5. **Autotile from an A2 block whose size isn't 2T×3T.** It returns 400
   `invalid_parameter` naming the expected size, never a crash or a silently cropped
   sheet. Owned by Task 4.

---

### Task 1: Weightless bundled models (core seam)

**Files:**
- Modify: `src/muse/core/catalog_models.py` (CatalogEntry construction at the `m["hf_repo"]`
  read, about line 114)
- Modify: `src/muse/core/catalog_pull.py` (`_bundled_download_spec` about line 581,
  `_pull_bundled_transaction` about line 706)
- Modify: `src/muse/core/catalog.py` (`_validate_bundled_catalog_provenance` about line
  240, `model_load_blocker` about line 307)
- Modify: `src/muse/core/provenance.py` (`model_revision`)
- Modify: whatever the builder finds that assumes a weights snapshot for a bundled model.
  Grep `local_dir`, `hf_repo` and `artifact_provenance` under `src/muse/core` and
  `src/muse/cli_impl`. Check at least: `muse models remove --purge`, `muse models refresh`,
  storage cleanup / `muse doctor storage`, `models info` display, and supervisor sizing.
- Test: `tests/core/test_weightless_models.py`

**Interfaces:**
- Produces:
  - **Manifest.** `MANIFEST["weights"] == "none"` marks a weightless bundled script.
    `CatalogEntry.hf_repo == ""` for it, and the entry carries `extra["weights"] = "none"`.
  - **Catalog row after pull:** `{"weights": "none", "local_dir": "<catalog>/local/<model_id>",
    "venv_path", "python_path", "enabled": true, "pulled_at",
    "artifact_provenance": [{"kind": "code", "model_id", "muse_version", "source_sha256"}]}`.
    There is no `hf_repo` and no `revision` key. `local_dir` is an empty owned directory.
  - **Provenance.** `muse.core.provenance.model_revision` returns `revision=None` and
    `components=(ArtifactComponent("code", "muse/<model_id>", None),)`, with
    `artifacts_digest = source_sha256[:16]`. `revision_headers` then emits
    `x-muse-model-artifacts`, which also requires emitting the digest for a single code
    component.
  - **Helper:** `muse.core.catalog_models.script_source_sha256(model_id) -> str`, the
    SHA-256 of the model script file bytes.

- [ ] **Step 1: Write failing tests**

They use a fake weightless script in a tmp models dir. Monkeypatch
`catalog_models._bundled_models_dir` to point at it, and the catalog dir via
`MUSE_CATALOG_DIR`. Stub `ensure_venv` / `install_into_venv` as the existing pull tests do;
reuse their fixtures from `tests/core/test_catalog*.py`.

```python
def test_weightless_pull_downloads_nothing(...):
    # snapshot_download / download_hf_artifact_bundle / seed_repo patched to raise
    model_id = pull("toy-tools")
    row = read_catalog()[model_id]
    assert row["weights"] == "none"
    assert "hf_repo" not in row and "revision" not in row
    assert Path(row["local_dir"]).is_dir() and not any(Path(row["local_dir"]).iterdir())
    assert row["artifact_provenance"][0]["kind"] == "code"
    assert len(row["artifact_provenance"][0]["source_sha256"]) == 64

def test_weightless_load_blocker_is_none_after_pull(...):
    assert model_load_blocker("toy-tools", read_catalog()["toy-tools"]) is None

def test_weightless_receipt_mismatch_blocks_load(...):
    # edit the script file after pull -> blocker names the provenance mismatch
    assert "re-run `muse pull toy-tools`" in model_load_blocker(...)

def test_weightless_revision_headers(...):
    rev = model_revision("toy-tools", row, manifest)
    assert rev.revision is None
    headers = dict(revision_headers(rev))
    assert b"x-muse-model-revision" not in headers
    assert headers[b"x-muse-model-artifacts"] == row["artifact_provenance"][0]["source_sha256"][:16].encode()

def test_bundled_script_without_hf_repo_and_without_weights_none_still_errors(...):
    # KeyError / CatalogError exactly as today: the seam is opt-in
```

Also test that `load_backend("toy-tools")` constructs the class with `hf_repo=""` and
`local_dir=<empty dir>`. It must also skip any snapshot existence check. The `--purge`
path and refresh path must tolerate the empty `local_dir`; add one test each.

- [ ] **Step 2: Run, and verify the tests fail**

Run: `PYTHONPATH=$PWD/src pytest tests/core/test_weightless_models.py -v`
Expected: FAIL (KeyError `hf_repo` or CatalogError "must declare an immutable 40-character revision").

- [ ] **Step 3: Implement the seam**

- `_bundled_download_spec` returns a spec with `revision=None` and the code receipt for
  weightless entries.
- `_pull_bundled_transaction` skips the whole download block for weightless entries. It
  creates `<catalog>/local/<model_id>` with `_ensure_owned_directory(..., private=True)`
  and writes the row above.
- `_validate_bundled_catalog_provenance` compares the stored receipt with a freshly
  computed one.

- [ ] **Step 4: Run the new tests and the core/cli suites**

Run: `PYTHONPATH=$PWD/src pytest tests/core tests/cli_impl -q -m "not slow"`
Expected: all pass.

- [ ] **Step 5: Commit** `feat(core): weightless bundled models (weights: none)`

---

### Task 2: Pixelize library

**Files:**
- Create: `src/muse/modalities/image_pixelart/__init__.py`. Minimal for now: module
  docstring and `MODALITY = "image/pixelart"`. Task 4 fills in the rest.
- Create: `src/muse/modalities/image_pixelart/palettes.py`
- Create: `src/muse/modalities/image_pixelart/pixelize.py`
- Test: `tests/modalities/image_pixelart/test_palettes.py`,
  `tests/modalities/image_pixelart/test_pixelize.py`

**Interfaces:**
- Produces (`palettes.py`):
  - `NAMED_PALETTES: dict[str, tuple[str, ...]]`, with keys exactly `pico-8`, `db16`,
    `db32`, `endesga-32`, `nes`, `gameboy`. Values are `#rrggbb` lowercase.
  - `parse_palette(spec: str) -> PaletteSpec`.
    - `PaletteSpec` is a frozen dataclass: `kind: Literal["named", "auto", "none",
      "custom"]`, `colors: tuple[str, ...]`, `count: int | None`.
    - It raises `ValueError` with a user-facing message on a bad spec: unknown name, N out
      of [2, 256], a bad hex.
  - `srgb_to_oklab(rgb: np.ndarray) -> np.ndarray` (float, shape `(..., 3)`), and
    `nearest_palette_index(rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray`.
  - `auto_palette(rgb: np.ndarray, n: int, mask: np.ndarray | None) -> np.ndarray`, using
    Pillow `quantize` (median cut) over opaque pixels only.
- Produces (`pixelize.py`):
  - `@dataclass(frozen=True) class GridInfo: cell: int; offset_x: int; offset_y: int;
    confidence: float; detected: bool`
  - `@dataclass(frozen=True) class PixelizeOptions`, with fields `grid: int | None`
    (None = auto), `target_size: tuple[int, int] | None`, `palette: PaletteSpec`,
    `background: str` (`auto` / `none` / `transparent-keep` / `#rrggbb`),
    `alpha_threshold: int = 128`, `cleanup: bool = True`, `outline: str = "none"`,
    `seamless: bool = False`, `sheet: tuple[int, int] | None = None`,
    `preview_scale: int = 1`.
  - `@dataclass class PixelizeResult: image: PIL.Image.Image` (RGBA), `grid: GridInfo`,
    `size: tuple[int, int]`, `palette: list[str]`, `background_removed: bool`,
    `metadata: dict`.
  - `detect_grid(img: PIL.Image.Image) -> GridInfo`
  - `pixelize(img: PIL.Image.Image, opts: PixelizeOptions) -> PixelizeResult`

**Algorithm notes** (the builder chooses details within these):
- **Grid.** Period of colour-change positions per axis via autocorrelation of the summed
  absolute horizontal/vertical gradient. Cell = the best common period of both axes in
  [2, 64]. Offset = argmax phase. `detected = confidence >= 0.5` (confidence normalised
  to [0, 1]). Undetected → cell 1.
- **Downscale.** Majority (mode) colour per cell over opaque pixels, computed on the
  colour-quantized image, so near-identical colours vote together (the k-centroid
  approach). A cell is transparent when most of its alpha is below `alpha_threshold`.
- **Outline `dark`.** Every transparent pixel 4-adjacent to an opaque one becomes the
  darkest palette colour in OKLab L. The candidates are restricted to colours within
  ΔE 0.25 of the neighbour's hue, falling back to the palette's darkest colour.
- **Cleanup.** An opaque pixel whose 8 neighbours all share one other colour takes that
  colour.
- **Seamless.** After palette mapping, make the left/right and top/bottom edge columns
  and rows consistent. Copy-blend each edge pair to the majority colour of the pair. The
  result must satisfy the wrap test below.
- **Sheet `CxR`.** One shared grid (detected on the whole sheet) and one shared palette:
  `auto:N` is computed over all cells together. The output keeps the layout:
  `(C·cw) × (R·ch)` art pixels.

- [ ] **Step 1: Write failing tests.** The synthetic fixtures are built in the test file:
  a 16×16 or 32×32 sprite of random palette colours with a transparent border.

```python
@pytest.mark.parametrize("k", [4, 6, 8])
def test_recovers_upscaled_sprite(k):
    small = make_sprite(32, 32, colors=NAMED_PALETTES["db32"][:12])
    big = small.resize((32*k, 32*k), Image.NEAREST)
    big = jitter(blur(big, 0.6), noise=6)        # soft AI-style pixels
    r = pixelize(big, opts(palette=parse_palette("db32")))
    assert r.grid.detected and r.grid.cell == k
    assert r.size == (32, 32)
    assert agreement(r.image, small) >= 0.98

def test_offset_crop_still_recovers():          # Review Focus 2
    big = small.resize((128, 128), Image.NEAREST).crop((3, 1, 126, 127))
    r = pixelize(big, opts()); assert r.grid.cell == 4 and agreement_cropped(...) >= 0.98

def test_true_1to1_art_is_unchanged():          # Review Focus 1
    r = pixelize(small, opts(palette=parse_palette("none")))
    assert r.grid.detected is False and r.size == small.size
    assert np.array_equal(np.asarray(r.image), np.asarray(small.convert("RGBA")))

def test_exact_palette_input_keeps_colours():  # Review Focus 3
    r = pixelize(small_in_db32, opts(palette=parse_palette("db32")))
    assert colours(r.image) == colours(small_in_db32)

def test_only_palette_colours_and_binary_alpha():
    a = np.asarray(r.image); assert set(np.unique(a[..., 3])) <= {0, 255}
    assert colours(r.image) <= set(NAMED_PALETTES["pico-8"])

def test_background_auto_keys_uniform_border_only():   # Review Focus 4
    # magenta-bordered sprite -> border transparent, background_removed True
    # opaque sprite whose colour touches 40% of the border -> nothing keyed, False
    # RGBA input with existing hard alpha -> transparent area preserved

def test_outline_dark_adds_one_pixel_ring(): ...
def test_cleanup_removes_orphans(): ...
def test_seamless_wraps(): # tile 3x3, no edge pixel breaks the cleanup rule across the seam
def test_sheet_shares_grid_and_palette():  # 4x4 sheet; every cell's colours ⊆ result.palette; one GridInfo
def test_preview_scale_is_nearest(): # every preview_scale x preview_scale block is uniform
def test_parse_palette_errors(): # "auto:1", "auto:300", "nope", "#12345g" -> ValueError
def test_named_palettes_sizes(): # pico-8 16, db16 16, db32 32, endesga-32 32, gameboy 4, nes >= 52
```

- [ ] **Step 2: Run, and verify the tests fail**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/image_pixelart -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** `palettes.py` and `pixelize.py` to the interfaces above.

- [ ] **Step 4: Run the tests on Pillow 11 and Pillow 13**

Run:
```
PYTHONPATH=$PWD/src pytest tests/modalities/image_pixelart -v
S=$(mktemp -d); pip install -q --no-deps --target $S pillow==13.0.0; PYTHONPATH=$S:$PWD/src pytest tests/modalities/image_pixelart -q
```
Expected: all pass both times.

- [ ] **Step 5: Commit** `feat(pixelart): pixelize library (grid snap, palettes, alpha, outline, seamless, sheets)`

---

### Task 3: Autotile library and exporters

**Files:**
- Create: `src/muse/modalities/image_pixelart/autotile.py`
- Create: `src/muse/modalities/image_pixelart/exporters.py`
- Test: `tests/modalities/image_pixelart/test_autotile.py`,
  `tests/modalities/image_pixelart/test_exporters.py`

**Interfaces:**
- Consumes:
  - `palettes.parse_palette`, `PaletteSpec`;
  - `pixelize.pixelize` / `PixelizeOptions`, for palette mapping an autotile output and
    for making fill/base textures seamless.
- Produces (`autotile.py`):
  - `FLOOR_AUTOTILE_TABLE: tuple[tuple[tuple[int, int], ...], ...]`, the 48 entries ×
    4 quadrant (qx, qy) sources in half-tile units within the 4×6 quadrant grid of an A2
    block, copied from RPG Maker MV `rpg_core.js` with attribution.
  - `@dataclass(frozen=True) class EdgeStyle`, with fields `edge_width: int | None`
    (None → T/8), `outline: str = "dark"`, `highlight: bool = True`,
    `roughness: float = 0.3`, `corner_radius: int | None` (None → T/4), `seed: int = 0`.
  - `build_a2(fill: Image, base: Image | None, tile_size: int, style: EdgeStyle,
    palette: PaletteSpec | None) -> Image`, which returns RGBA of exactly
    (2T, 3T).
  - `validate_a2(img: Image, tile_size: int) -> None`, which raises `ValueError` naming
    the expected `2T×3T` size.
  - `@dataclass(frozen=True) class BlobTile: index: int; x: int; y: int; mask: int`.
    `mask` is the 8-neighbour bitmask, with bit order N=1, NE=2, E=4, SE=8, S=16, SW=32,
    W=64, NW=128. Corner bits count only when both adjacent sides are set (the standard
    blob reduction that yields 47).
  - `expand_blob47(a2: Image, tile_size: int) -> tuple[Image, list[BlobTile]]`. The sheet
    is 8 columns × 6 rows (47 tiles plus one empty cell). It takes each of the 47
    reduced masks to the RPG Maker shape index with the same quadrants. Which of RPG
    Maker's 48 entries is the duplicate gets documented in a comment.
- Produces (`exporters.py`):
  - `godot_tileset_tres(name: str, tile_size: int, tiles: list[BlobTile]) -> str`, a
    Godot 4 `TileSet` resource. It has one `TileSetAtlasSource`
    (`texture = ExtResource` pointing at `res://<name>_blob47.png`), `terrain_set_0/mode
    = 0` (match corners and sides) and one terrain, and per-tile `terrain_set`, `terrain`
    and `terrains_peering_bit/*` from `mask`.
  - `tiled_tileset_tsx(name: str, tile_size: int, tiles: list[BlobTile]) -> str`, a Tiled
    1.10+ `.tsx` with `<image source="<name>_blob47.png">` and one
    `<wangset type="mixed">`, with each tile's `wangid` (8 comma values, clockwise from
    top) derived from `mask`.
  - `bundle_zip(name: str, a2_png: bytes, blob_png: bytes, tres: str, tsx: str,
    layout: list[dict]) -> bytes`, containing `<name>_a2.png`, `<name>_blob47.png`,
    `<name>.tres`, `<name>.tsx` and `<name>_layout.json`.
  - `safe_name(name: str) -> str`, a slug of `[a-z0-9_-]{1,40}`. It raises `ValueError`
    otherwise.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("T", [16, 24, 32, 48])
def test_build_a2_size_and_alpha(T):
    a2 = build_a2(fill_tex(T), None, T, EdgeStyle(), None)
    assert a2.size == (2*T, 3*T); assert set(np.unique(np.asarray(a2)[..., 3])) <= {0, 255}

def test_build_a2_with_base_is_opaque(): ...           # base present -> alpha all 255
def test_build_a2_seeded_deterministic(): ...          # same seed -> identical bytes; different seed -> differs
def test_validate_a2_wrong_size():                      # Review Focus 5
    with pytest.raises(ValueError, match=r"96x144"): validate_a2(Image.new("RGBA", (96, 140)), 48)

def test_blob47_has_47_distinct_masks():
    sheet, tiles = expand_blob47(a2, 16); assert len({t.mask for t in tiles}) == 47
    assert sheet.size == (8*16, 6*16)

def test_blob47_seams_match_for_every_legal_neighbour_pair():
    # for every ordered pair (a, b) of tiles and each direction d where a's mask says "connected toward d"
    # and b's mask says "connected toward opposite(d)", the touching edge rows/cols are pixel-identical
    # in the fill region (compare against a reference composition of a plain fill A2).

def test_known_shapes(): # mask 255 -> centre fill tile; mask 0 -> isolated tile equals A2 top-left tile
def test_table_attribution_present(): assert "rpg_core.js" in inspect.getsource(autotile)

def test_godot_tres_parses_and_peering_matches_mask():
    text = godot_tileset_tres("grass", 16, tiles)
    assert text.startswith("[gd_resource type=\"TileSet\"")
    assert 'path="res://grass_blob47.png"' in text
    # parse per-tile peering bits; each tile's set of connected directions == decode(mask)

def test_tiled_tsx_parses_and_wangids_match_mask():
    root = ET.fromstring(tiled_tileset_tsx("grass", 16, tiles))
    assert root.find("image").get("source") == "grass_blob47.png"
    assert root.find("wangsets/wangset").get("type") == "mixed"

def test_bundle_zip_names(): ...  # five names, layout json round-trips
def test_safe_name(): ...         # "Grass Path!" -> ValueError; "grass_path" ok
```

- [ ] **Step 2: Run, and verify the tests fail**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/image_pixelart/test_autotile.py tests/modalities/image_pixelart/test_exporters.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** `autotile.py` and `exporters.py` to the interfaces above.

- [ ] **Step 4: Run** the same tests, then repeat on Pillow 13 as in Task 2 Step 4.
  Expected: all pass.

- [ ] **Step 5: Commit** `feat(pixelart): RPG Maker A2 autotiles, blob-47 expansion, Godot/Tiled export`

---

### Task 4: `image/pixelart` modality, routes, bundled `pixel-tools`, MCP

**Files:**
- Modify: `src/muse/modalities/image_pixelart/__init__.py` (`MODALITY`, `DISCOVERY`,
  `PROBE_DEFAULTS`, exports, `build_router`)
- Create: `src/muse/modalities/image_pixelart/protocol.py`, `routes.py`, `client.py`,
  `codec.py`
- Create: `src/muse/modalities/image_pixelart/runtime.py` (`PixelTools`)
- Create: `src/muse/models/pixel_tools.py`
- Modify: `src/muse/core/config.py` (the new `limits.pixelart_max_input_side` setting) and
  `docs/CONFIG.md`
- Modify: `.github/workflows/fresh-venv-smoke.yml` (add `pixel-tools` to the matrix: it is
  tiny and CPU-only)
- Test: `tests/modalities/image_pixelart/test_routes.py`,
  `tests/modalities/image_pixelart/test_discovery.py`, `tests/models/test_pixel_tools.py`

**Interfaces:**
- Consumes:
  - from Task 1: weightless pull/load;
  - from Tasks 2–3: `pixelize`, `PixelizeOptions`, `parse_palette`, `build_a2`,
    `validate_a2`, `expand_blob47`, `EdgeStyle`, the exporters.
- Produces:
  - **Protocol.** `ImagePixelartModel` has `model_id`,
    `pixelize(image, opts: PixelizeOptions) -> PixelizeResult` and
    `autotile(request: AutotileRequest) -> AutotileResult`.
    - `AutotileRequest` is a dataclass with `tile_size`, `a2`, `fill`, `base`, `style`,
      `palette` and `name`.
    - `AutotileResult` is a dataclass with `a2: Image`, `blob47: Image`,
      `tiles: list[BlobTile]`, `godot_tres: str`, `tiled_tsx: str` and `metadata: dict`.
  - **Runtime.** `muse.modalities.image_pixelart.runtime:PixelTools`, aliased as `Model`
    in `models/pixel_tools.py`.
  - **`POST /v1/images/pixelize`.** Multipart fields exactly as in the spec. The
    response is JSON per the spec; with `response_format=png` it is raw `image/png`.
  - **`POST /v1/images/autotile`.** Multipart: `a2` file OR `fill` file (+ optional
    `base` file) plus style fields. The response is JSON per the spec; with
    `response_format=zip` it is raw `application/zip` with
    `Content-Disposition: attachment; filename="<name>.zip"`.
  - **DISCOVERY routes**, with `mcp_tool` names `muse_pixelize_image` and
    `muse_autotile`, `file_fields` (`["image"]`, and `["a2", "fill", "base"]`), and
    `media_fields`:
    - pixelize: `["data[].b64_json"]`;
    - autotile: `["files.a2.b64_json", "files.blob47.b64_json"]` plus `text_media_fields`
      `{"files.godot_tres": "text/plain", "files.tiled_tsx": "application/xml"}`.

    Check how `src/muse/mcp/generated.py` names saved files. If the saved names break the
    `.tres` → PNG reference, make the MCP tool default to `zip` (the generator saves
    binary responses as a file) and say so in the route's DISCOVERY summary.
  - **PROBE_DEFAULTS:** pixelize a synthetic 4× sprite (asserting `grid.cell == 4` inside
    the lambda) and autotile a synthetic fill at T=16.

- [ ] **Step 1: Write failing tests**

The route tests use the real `PixelTools` (it has no weights or heavy deps) behind a
`ModalityRegistry` + `create_app`, following `tests/modalities/image_upscale/test_routes.py`.

```python
def test_pixelize_json_envelope(client):
    r = client.post("/v1/images/pixelize", files={"image": png(upscaled_sprite(4))},
                    data={"model": "pixel-tools", "palette": "db32"})
    body = r.json(); assert r.status_code == 200 and body["object"] == "image.pixelize"
    assert body["grid"]["cell"] == 4 and body["size"] == [32, 32]

def test_pixelize_png_response(client): ... # content-type image/png, decodes, alpha binary
def test_pixelize_bad_palette_400(client): ... # code invalid_parameter
def test_pixelize_oversize_413(client, monkeypatch): ... # MUSE_PIXELART_MAX_INPUT_SIDE=64
def test_pixelize_preview_scale_bounds(client): ... # 0 and 17 -> 400
def test_autotile_from_fill_json(client): ... # files.a2/blob47 decode to (2T,3T)/(8T,6T); tres/tsx strings; layout 47
def test_autotile_from_a2_zip(client): ... # zip namelist == five names for name=grass
def test_autotile_wrong_a2_size_400(client): ... # Review Focus 5 at the HTTP layer; message names 96x144
def test_autotile_requires_exactly_one_source(client): ... # neither / both a2+fill -> 400
def test_autotile_tile_size_must_be_allowed(client): ... # 20 -> 400
def test_discovery_declares_tools(): ... # mcp_tool names, file_fields, media_fields as above
def test_pixel_tools_manifest(): # weights == "none", no hf_repo, device cpu, memory_gb 0.5
def test_probe_defaults_run(): PROBE_DEFAULTS["call"](PixelTools(model_id="pixel-tools", hf_repo="", local_dir=None))
```

Add one end-to-end test with an isolated catalog, `pull("pixel-tools")` (installs stubbed)
and then `load_backend`, proving the Task 1 seam works for the real script.

- [ ] **Step 2: Run, and verify the tests fail**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/image_pixelart tests/models/test_pixel_tools.py -v`
Expected: the new route/discovery/model tests FAIL.

- [ ] **Step 3: Implement.** Reuse the bounded upload helpers in
  `muse.modalities._common.uploads` and `image_generation/image_input.decode_image_file`
  (with `max_side`). Run the CPU work off the event loop the way the upscale route does
  (`run_native_offload` or `asyncio.to_thread`, whichever the builder finds is the
  pattern for CPU-only work).

- [ ] **Step 4: Run** the modality, models, mcp and core suites.

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/image_pixelart tests/models tests/mcp tests/core -q -m "not slow"`
Expected: all pass. This includes the discovery/capabilities/meta tests that scan every
modality and bundled script (pip-extras audit, runtime-helpers meta, capability keys).

- [ ] **Step 5: Commit** `feat(pixelart): image/pixelart modality with pixel-tools (pixelize, autotile)`

---

### Task 5: Klein LoRA support, `prompt_prefix`, curated `klein-pixel-art`

**Files:**
- Modify: `src/muse/modalities/image_generation/runtimes/reference.py`
- Modify: `src/muse/modalities/image_generation/hf.py` (`_resolve_lora`, about line 212)
- Modify: `src/muse/modalities/image_generation/routes.py` and/or `options.py`
  (`prompt_prefix`, and `lora_scale` reaching `edit()`)
- Modify: `src/muse/core/capability_keys.py` (add `prompt_prefix` to `PUBLIC_KEYS`)
- Modify: `src/muse/curated.yaml` (a new `klein-pixel-art` entry after `pixel-art-xl`)
- Test: `tests/modalities/image_generation/runtimes/test_reference.py`,
  `tests/modalities/image_generation/test_hf_plugin.py` (or the existing plugin test file),
  `tests/modalities/image_generation/test_routes*.py`, `tests/core/test_curated.py`

**Interfaces:**
- Produces:
  - **`DiffusersReferenceModel.__init__`** gains `lora_adapter: bool = False`,
    `base_model: str | None = None` and `lora_scale: float = 1.0`.
    - With `lora_adapter`, it loads the pipeline from `resolve_model_source(base_model)`,
      with the same not-pulled RuntimeError text pattern as `runtimes/diffusers.py`.
    - It calls `pipe.load_lora_weights(self._src, adapter_name="lora")` BEFORE
      `place_pipeline`.
  - **`generate(...)` and `edit(...)`** accept `lora_scale: float | None`. Under the
    route's inference lock they call `pipe.set_adapters(["lora"], adapter_weights=[scale])`
    when the effective scale differs from the last one applied.
  - **`_resolve_lora`:**
    - When the effective base (override or tag) names a klein repo (same name test as
      `_reference_family`, ignoring its `base` exclusion), it emits
      `backend_path = <reference runtime path>`, the reference capabilities
      (`pipeline_class`, `cpu_offload: model`, `device: cuda`, `size_multiple`,
      `max_pixels`, `max_reference_images`, `supports_reference_images`) plus
      `lora_adapter`, `lora_scale` and `base_model`, and pip extras =
      reference extras + `peft`.
    - With several top-level `.safetensors`, it picks `pytorch_lora_weights.safetensors`
      when present and downloads only that weight file plus `*.json`. Otherwise it keeps
      today's "exactly one" error.
  - **`options.apply_prompt_prefix(prompt: str, capabilities: dict) -> tuple[str, bool]`**
    prepends `capabilities["prompt_prefix"]` unless the prompt already contains it,
    case-insensitively. Generations and edits responses report
    `metadata.prompt_prefix_applied: bool` only when the capability exists.

- [ ] **Step 1: Write failing tests** (all mocked, following the existing tests for both
  runtimes)

```python
def test_reference_lora_loads_adapter_before_offload(mock_diffusers): # call order: from_pretrained(base) -> load_lora_weights -> place_pipeline
def test_reference_lora_missing_base_raises(...): # "is not pulled" message
def test_reference_lora_scale_per_request(...): # set_adapters called with [0.5] then not again for same scale
def test_reference_without_lora_unchanged(...): # no load_lora_weights / set_adapters calls
def test_resolve_lora_klein_base_routes_to_reference(fake_info_limbicnation):
    # tags base_model:adapter:black-forest-labs/FLUX.2-klein-4B; siblings with both safetensors
    r = _resolve_lora("Limbicnation/pixel-art-lora", info)
    assert r.backend_path.endswith("runtimes.reference:DiffusersReferenceModel")
    assert r.manifest["capabilities"]["lora_adapter"] is True
    assert "peft" in r.manifest["pip_extras"]
def test_resolve_lora_picks_diffusers_weight_name(...): # download allow_patterns == ["pytorch_lora_weights.safetensors", "*.json"]
def test_resolve_lora_sdxl_unchanged(...): # pixel-art-xl shape still -> diffusers runtime
def test_prompt_prefix_applied_and_reported(client): # prompt "a knight" -> backend sees "pixel art sprite, a knight"; metadata true
def test_prompt_prefix_not_duplicated(client): # "Pixel Art Sprite, knight" -> unchanged, metadata false
def test_lora_scale_reaches_edit(client): ...
def test_curated_klein_pixel_art_entry(): # find_curated("klein-pixel-art") fields == Global Constraints values
```

- [ ] **Step 2: Run, and verify the tests fail**

Run: `PYTHONPATH=$PWD/src pytest tests/modalities/image_generation tests/core/test_curated.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement** to the interfaces. Supervisor LoRA-from-base sizing
  (`cli_impl/supervisor.py` about lines 1998–2008) must cover the new entry; add a test
  there if it keys on the diffusers runtime path.

- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src pytest tests/ -q -m "not slow"`.
  Expected: all pass.

- [ ] **Step 5: Commit** `feat(images): LoRA adapters on the klein reference runtime; prompt_prefix; curated klein-pixel-art`

---

### Task 6: Bench script and docs

**Files:**
- Create: `scripts/bench/pixel_compare.py`, `scripts/bench/pixel_cases.yaml`
- Create: `docs/PIXELART.md`
- Modify:
  - `CLAUDE.md`: modality count and list; a paragraph for `image/pixelart`; the weightless
    model seam under Key modules; the klein LoRA note in the image/generation paragraph.
  - `README.md`, `docs/AGENT_DISCOVERY.md` (if it lists routes), `docs/MODEL_SCRIPTS.md`
    (document `weights: "none"`).
- Test: `tests/scripts/test_pixel_compare.py`

**Interfaces:**
- Consumes: the HTTP routes from Tasks 4–5 only (the script is an HTTP client like
  `scripts/bench/image_compare.py`; reuse its structure and helpers where they are
  importable).
- Produces: `python scripts/bench/pixel_compare.py --server URL --cases FILE --out DIR`.
  - It writes `DIR/results.json` and `DIR/index.html`.
  - Case kinds:
    - `sprite`: `prompt`, `model`, and `pixelize` (a dict of form fields, may list several
      variants);
    - `autotile`: `fill_prompt`, optional `base_prompt`, `model`, and `autotile` (form
      fields).
  - The HTML shows the raw image, the pixelized 1× and nearest preview, grid, palette and
    timings. For autotiles it shows the A2, the blob-47 sheet and a random 16×12 map
    painted from the blob tiles by mask, so seams are visible.
  - `pixel_cases.yaml` holds the spec's 8 sprite prompts (knight, mage, slime, goblin,
    chest, potion, sword icon, tavern sign), each with `palette: db32` and `palette:
    auto:16` variants, plus one autotile case: "seamless grass texture, top-down, pixel
    art" over "dirt path texture, top-down, pixel art".

- [ ] **Step 1: Write failing tests** with a fake `httpx` transport, following
  `tests/scripts/test_image_compare.py` if it exists:

```python
def test_runs_sprite_case_and_writes_outputs(tmp_path, fake_server): ...  # results.json has one row per pixelize variant
def test_autotile_case_paints_map(tmp_path, fake_server): ...           # index.html references the map png
def test_case_file_validation(): ...                                    # unknown kind -> SystemExit with message
```

- [ ] **Step 2: Run, and verify the tests fail**

Run: `PYTHONPATH=$PWD/src pytest tests/scripts/test_pixel_compare.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** the script and docs.

- [ ] **Step 4: Run the tests and the fast lane**

Run: `PYTHONPATH=$PWD/src pytest tests/scripts/test_pixel_compare.py -v`, then the fast lane.
Expected: all pass.

- [ ] **Step 5: Commit** `feat(bench): pixel_compare contact sheet; docs for pixel assets`

---

### Task 7: Live acceptance on the 3060

**Files:**
- Create: `docs/benchmarks/2026-10-28-pixel-assets-acceptance.md`
- Scratch only (not committed): the SDD workspace `live/` dir.

**Steps:**

- [ ] **Step 1: Isolated server.**
  - Catalog: `MUSE_CATALOG_DIR=~/mpx`. Keep the path short: espeak-ng is not involved
    here, but a short path keeps parity with earlier runs.
  - Pull `pixel-tools` and verify that no network download occurred (it must finish with
    no HF calls).
  - Pull `flux2-klein-4b` with `HF_HUB_CACHE=/home/spinoza/.muse/weights`, so `hf_seed`
    hardlinks the existing 15 GB instead of downloading.
  - Pull `klein-pixel-art`.
  - Start `muse serve --port 8011` from the worktree's `src`.
  - Check `nvidia-smi` free VRAM first. If the owner's :8000 server has a model resident
    on the GPU, wait or stop. **Never signal it.**
- [ ] **Step 2: Contact sheet.** Run `scripts/bench/pixel_compare.py --server
  http://127.0.0.1:8011 --cases scripts/bench/pixel_cases.yaml --out <workspace>/live/pixel`.
  Record grid detection, palette use and timings.
- [ ] **Step 3: Godot check.** Fetch a portable Godot 4 Linux binary (the official GitHub
  release, a stable 4.x; record the version and sha256) into the scratch dir. Load the
  exported `.tres` headless: a one-line GDScript `load("res://terrain.tres")` in a temp
  project, asserting the TileSet has 47 tiles with terrain peering. If it cannot run,
  record why and keep the structural checks.
- [ ] **Step 4: Existing art.** Download one CC0 RPG Maker-style A2 autotile sheet
  (OpenGameArt, CC0 licence recorded with its URL) into scratch. Run it through
  `/v1/images/autotile` with `a2=`. Include the blob map in the contact sheet.
- [ ] **Step 5: MCP.** `muse mcp --filter inference` against :8011, then call
  `muse_pixelize_image` and `muse_autotile` through the MCP client test harness (or the
  `mcp` CLI). Confirm files land in `mcp.output_dir` and that the `.tres` → PNG
  reference resolves.
- [ ] **Step 6: Weightless model.** `/v1/models` lists `pixel-tools` with `revision: null`.
  `/v1/capabilities?modality=image/pixelart` describes both routes. The response headers
  carry `X-Muse-Model-Artifacts` and no `X-Muse-Model-Revision`.
- [ ] **Step 7: Write the acceptance doc.** Cover numbers, misses (reported, bar not
  loosened), the contact-sheet path and the Godot result. Stop the :8011 server, then
  commit `docs(bench): pixel assets acceptance`.

---

### Task 8 (optional): Walk-sheet spike (go/no-go)

**Files:**
- Modify: `src/muse/curated.yaml` (`flux2-klein-base-4b`, `klein-pixel-walk`)
- Modify: `src/muse/modalities/image_generation/hf.py`. A curated entry may opt a `base`
  klein repo into the reference runtime via an explicit capability
  `reference_runtime: true`. With it, 28-step and guidance 4.0 presets apply; the default
  `base` exclusion is otherwise unchanged.
- Test: plugin and curated tests, as in Task 5.

**Steps:**

- [ ] **Step 1: Tests.** The opt-in capability routes a base repo to the reference
  runtime. Without it the exclusion holds. The curated entries carry the 4walk card's
  `prompt_prefix` template sentence (the builder copies it from the model card) and
  `default_size [512, 512]`.
- [ ] **Step 2: Implement.** Pull both into `~/mpx`; check disk (`df`) for about 16 GB
  first.
- [ ] **Step 3: Generate and score.** Generate 4 characters (knight, witch, goblin, elf
  ranger), then pixelize them with `sheet=4x4` and `palette=auto:24`. Add them to the
  contact sheet.
  - **Go** if ≥ 3 of 4 read as consistent 4-direction walk cycles. Record the verdict and
    reason in the acceptance doc.
  - **No-go**: revert the curated entries but keep the opt-in seam only if it is tested
    and harmless; otherwise revert it too.
- [ ] **Step 4: Commit** with a message stating go or no-go.
