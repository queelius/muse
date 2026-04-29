# `image/segmentation` modality design

**Date:** 2026-04-28
**Driver:** add the 14th muse modality, `image/segmentation`, exposing `POST /v1/images/segment` for promptable segmentation. SAM-2 family from Meta is the v0.26.0 target. CLIPSeg (text-prompted) is deferred until the bundled SAM-2 path is solid.

## Goal

1. Mount `POST /v1/images/segment` (multipart/form-data) on a new `image/segmentation` modality. Inputs: source image file plus a prompt-mode dispatch (auto, points, boxes, text). Behavior: list of binary masks plus per-mask metadata (score, bbox, area).
2. Bundle `facebook/sam2-hiera-tiny` (~40MB, Apache 2.0) as the default low-VRAM model, alias-mapped through `curated.yaml`.
3. New `SAM2Runtime` generic runtime in `image_segmentation/runtimes/sam2_runtime.py`, callable for any HF repo whose tags include `mask-generation` or `image-segmentation`.
4. New HF plugin sniffing the `mask-generation` / `image-segmentation` tags. Priority 110.
5. New `ImageSegmentationClient` HTTP client (multipart upload, JSON-encoded points/boxes).
6. New mask format design: every mask returned in either `png_b64` (default; portable, viewable) or `rle` (COCO-style run-length encoding; compact, standard for downstream tooling). Both representations round-trip cleanly.
7. Capability-gated mode dispatch (`supports_text_prompts`, `supports_point_prompts`, `supports_box_prompts`, `supports_automatic`); requests for unsupported modes return 400 before the runtime is invoked.

## Non-goals

- Text-prompted segmentation. The bundled SAM-2 backbone has no text encoder; text-mode is reserved at the wire level (returns 400 for SAM-2 family) and the runtime / curated entries can opt in via `supports_text_prompts: True` once a CLIPSeg-shaped runtime lands. Filed for v1.next.
- Mask2Former / closed-set semantic segmentation. The architecture is materially different (per-class label space + transformer decoder); deferred to a follow-up modality (`image/semantic-segmentation`?) or a second runtime under the same MIME tag once a clean shared dispatch is found.
- Video object segmentation. SAM-2 supports it natively but the wire shape (multi-frame video upload + per-frame mask out) is out of scope for v1.
- Mask post-processing (CRF refinement, hole filling). Out of scope.
- Polygon / spline mask formats. The RLE format covers compact transport; downstream tooling can convert to polygons.
- GUI / preview. Wire-only modality.

## Architecture

```
client.images.segment(image=src.read(), model="sam2-hiera-tiny", mode="auto")
   |
   v  HTTP POST /v1/images/segment, multipart/form-data
   |  fields: image (PNG/JPEG file), model, mode, prompt?, points?,
   |          boxes?, mask_format?, max_masks?
   v
routes.segment handler
   - decode_image_file(image) into PIL.Image
   - validate mode in {"auto", "points", "boxes", "text"}
   - capability gate: model must declare supports_<mode>=True for the mode
   - parse JSON-encoded points / boxes (mode-dependent); 400 on bad JSON
   - call backend.segment(img, mode=..., points=..., boxes=..., max_masks=...)
   - codec.encode_segmentation each MaskRecord into JSON dict (png_b64 or rle)
   v
SAM2Runtime.segment(image, mode="auto", ...)
   - lazy-load AutoModelForMaskGeneration + AutoProcessor once, cache as self._model/_processor
   - dispatch on mode:
       auto    -> Sam2AutomaticMaskGenerator (fallback: dense grid sampling)
       points  -> processor(images, input_points=...); model(...) -> masks
       boxes   -> processor(images, input_boxes=...);  model(...) -> masks
       text    -> raise CapabilityError (route already gated; defense in depth)
   - sort by score descending, truncate to max_masks
   - compute bbox + area from the binary mask via numpy
   v
SegmentationResult(masks=[MaskRecord(...)], image_size, mode, seed, metadata)
   v
codec.encode_segmentation -> JSON envelope
```

The MIME tag is `image/segmentation`; the URL is `/v1/images/segment`. The directory is `src/muse/modalities/image_segmentation/` (underscore in dir name; slash in MIME).

## Wire contract

`POST /v1/images/segment`, multipart/form-data:

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `image` | file (PNG/JPEG/WebP) | yes | - | size cap 10MB; max side cap MUSE_SEGMENTATION_MAX_INPUT_SIDE (default 2048) |
| `model` | str | no | None | bundled or pulled model id |
| `mode` | str | no | `"auto"` | one of {"auto","points","boxes","text"}; capability-gated |
| `prompt` | str | no | None | required when mode="text"; up to 4000 chars |
| `points` | str (JSON) | no | None | required when mode="points"; JSON list of [x, y] integers |
| `boxes` | str (JSON) | no | None | required when mode="boxes"; JSON list of [x1, y1, x2, y2] integers |
| `mask_format` | str | no | `"png_b64"` | one of {"png_b64", "rle"} |
| `max_masks` | int | no | 16 | 1 to 256; truncates returned masks |

Response:
```json
{
  "id": "seg-<uuid>",
  "model": "sam2-hiera-tiny",
  "mode": "auto",
  "image_size": [1024, 768],
  "masks": [
    {
      "index": 0,
      "score": 0.94,
      "mask": "<base64-PNG of binary mask, white=foreground>",
      "bbox": [100, 200, 200, 200],
      "area": 12450
    }
  ]
}
```

`image_size` is `[W, H]` (PIL convention; width first).
`bbox` is `[x, y, w, h]` with the top-left origin (numpy `np.where` convention; width first within the array semantics, but exposed as the standard COCO-bbox shape).
`area` is `int(mask.sum())`.

For `mask_format="rle"`, the per-mask `mask` field becomes a dict:
```json
"mask": {"size": [768, 1024], "counts": "iY:..."}
```
Note the `size` order is `[H, W]` per COCO RLE convention (rows first), distinct from `image_size` `[W, H]` (PIL convention). This mismatch is documented in `MaskRecord` and the codec docstring; clients are expected to follow the convention of the field they read.

## Module structure

```
src/muse/modalities/image_segmentation/
|-- __init__.py          # MODALITY="image/segmentation", build_router, exports, PROBE_DEFAULTS
|-- protocol.py          # SegmentationResult dataclass, MaskRecord, ImageSegmentationModel Protocol
|-- codec.py             # encode_segmentation, encode_mask_rle / decode_mask_rle, encode_mask_png
|-- routes.py            # POST /v1/images/segment (multipart, mode-aware)
|-- client.py            # ImageSegmentationClient (HTTP, multipart, JSON points/boxes)
|-- hf.py                # HF_PLUGIN sniffing mask-generation / image-segmentation
`-- runtimes/
    |-- __init__.py
    `-- sam2_runtime.py  # SAM2Runtime
```

`src/muse/models/sam2_hiera_tiny.py` is the bundled model script. It mirrors `bge_reranker_v2_m3.py`'s lazy-import pattern (module-level sentinels, `_ensure_deps()`, importlib-friendly tests).

## `MaskRecord` and `SegmentationResult` dataclasses

```python
@dataclass
class MaskRecord:
    """One segmentation mask plus provenance.

    `mask` is a numpy 2D bool / uint8 array shaped (H, W). The codec
    converts it to either a base64-encoded PNG or a COCO-style RLE
    dict at the wire layer.

    Note the H, W vs W, H convention split: numpy arrays index as
    [row, col] = [y, x] = [H, W]; PIL.Image.size returns (W, H).
    `bbox` is exposed in [x, y, w, h] order matching COCO bbox shape.
    """
    mask: Any
    score: float
    bbox: tuple[int, int, int, int]
    area: int


@dataclass
class SegmentationResult:
    masks: list[MaskRecord]
    image_size: tuple[int, int]   # (W, H) of input image, PIL convention
    mode: str                     # mode actually used by the runtime
    seed: int
    metadata: dict = field(default_factory=dict)
```

## Protocol

```python
@runtime_checkable
class ImageSegmentationModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def segment(
        self,
        image: Any,
        *,
        mode: str = "auto",
        prompt: str | None = None,
        points: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        max_masks: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SegmentationResult: ...
```

Duck-typed; backends only need to satisfy the structural shape.

## Runtime: `SAM2Runtime`

```python
class SAM2Runtime:
    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        max_masks: int = 64,
        supports_text_prompts: bool = False,
        supports_point_prompts: bool = True,
        supports_box_prompts: bool = True,
        supports_automatic: bool = True,
        **_: Any,
    ) -> None:
        # lazy-import torch + transformers
        # AutoModelForMaskGeneration.from_pretrained(local_dir or hf_repo, ...)
        # AutoProcessor.from_pretrained(...)
        # honor device + dtype

    def segment(self, image, *, mode="auto", points=None, boxes=None,
                max_masks=None, seed=None, **_) -> SegmentationResult:
        # dispatch by mode; compute bbox + area for each mask;
        # sort by score descending; truncate to max_masks
```

Mirrors `DiffusersUpscaleRuntime`'s structure: lazy-import sentinels at module scope, `_ensure_deps()` with short-circuit, `_select_device()` helper, the rest of the class. The runtime doesn't subclass anything; it satisfies the Protocol structurally.

The constructor reads `max_masks`, `supports_*` from manifest capabilities (registry merges them at `load_backend` time, same as upscale). The runtime keeps inference under `torch.inference_mode()` at call time so we never need to put the model into evaluation mode explicitly.

## Bundled `sam2_hiera_tiny.py`

```python
MANIFEST = {
    "model_id": "sam2-hiera-tiny",
    "modality": "image/segmentation",
    "hf_repo": "facebook/sam2-hiera-tiny",
    "description": "SAM-2 Hiera tiny: ~40MB promptable segmentation, Apache 2.0",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.43.0",
        "Pillow>=9.1.0",
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "max_masks": 64,
        "supports_text_prompts": False,
        "supports_point_prompts": True,
        "supports_box_prompts": True,
        "supports_automatic": True,
        "memory_gb": 0.8,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "preprocessor_config.json",
    ],
}
```

The `Model` class lazy-imports `torch` and `transformers.AutoModelForMaskGeneration`, mirrors the runtime's behavior, and is itself testable with importlib-shaped patches.

## Capability flags

| Flag | Default in resolver-pulled SAM family | Default in sam2-hiera-tiny |
|---|---|---|
| `max_masks` | 64 | 64 |
| `supports_text_prompts` | False (True only for clipseg-pattern) | False |
| `supports_point_prompts` | True | True |
| `supports_box_prompts` | True | True |
| `supports_automatic` | True | True |

## HF plugin

Priority 110. `_sniff` runs early since SAM-2 repos have a unique tag (`mask-generation`) that doesn't collide with the other priority-110+ plugins (chat/completion 100, image/animation 110, image/upscale 105). The exact priority value is set to 110 to claim the repo before image/upscale (105) considers it; even though the tag-set differs, defense-in-depth.

```python
def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "mask-generation" in tags or "image-segmentation" in tags
```

The `_resolve` infers per-pattern defaults:

| Repo pattern | max_masks | memory_gb | supports_text | supports_point | supports_box | supports_auto |
|---|---|---|---|---|---|---|
| `*sam2-hiera-tiny*` | 64 | 0.8 | False | True | True | True |
| `*sam2-hiera-base*` | 64 | 1.5 | False | True | True | True |
| `*sam2-hiera-large*` | 64 | 2.5 | False | True | True | True |
| `*sam-*` (original SAM family) | 64 | 1.5 | False | True | True | True |
| `*clipseg*` | 16 | 0.6 | True | False | False | False |
| fallback | 16 | 1.0 | False | True | True | True |

`_search` filters by `mask-generation` HF tag.

## Codec

```python
def encode_mask_png(mask: np.ndarray) -> bytes:
    """Encode a 2D bool/uint8 mask as a base64-able PNG (white=fg)."""

def encode_mask_rle(mask: np.ndarray) -> dict:
    """Encode a 2D bool/uint8 mask in COCO RLE format.

    Returns: {"size": [H, W], "counts": str}.

    Uses pycocotools.mask if available; falls back to a pure-Python
    RLE encoder otherwise. The COCO `counts` is a column-major run-
    length encoded sequence wrapped through their compact ASCII codec
    (range [0x30, 0x6F]). We honor that exact encoding so external
    tooling (pycocotools, FiftyOne) can round-trip our outputs.

def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict back into a 2D bool array.

    Reciprocal of encode_mask_rle. encode -> decode -> equal mask is
    a unit-tested invariant (round-trip property).

def encode_segmentation(
    result: SegmentationResult,
    *,
    mask_format: str,
) -> dict:
    """Project SegmentationResult into the wire JSON envelope.

    `mask_format` in {"png_b64", "rle"} dispatches per-mask encoding.
    The envelope adds `id`, `model` (caller fills these in the route),
    plus `mode`, `image_size`, `masks`.
```

The `_try_import_pycocotools()` helper mirrors the lazy-import pattern: it returns the `pycocotools.mask` module if importable, else None. Tests patch this directly to exercise both code paths.

## Multipart input decoding

Reuses `decode_image_file` from `muse.modalities.image_generation.image_input`. No need for a third copy. The route also runs the `MUSE_SEGMENTATION_MAX_INPUT_SIDE` cap after decode (default 2048 per side; environment-tunable). Oversize inputs return 400 with code `invalid_parameter`.

The cap rationale: SAM-2 internally resamples to 1024x1024 for inference, but extremely large inputs still consume RAM during decode. 2048 is a reasonable upper bound that leaves headroom for typical photo aspect ratios.

## PROBE_DEFAULTS

```python
def _make_probe_image():
    """Generate a small synthetic test image for `muse models probe`."""
    from PIL import Image
    return Image.new("RGB", (256, 256), (128, 128, 128))


PROBE_DEFAULTS = {
    "shape": "256x256, automatic mode, max_masks=4",
    "call": lambda m: m.segment(_make_probe_image(), mode="auto", max_masks=4),
}
```

## Curated entries

```yaml
- id: sam2-hiera-tiny
  bundled: true

- id: sam2-hiera-base-plus
  uri: hf://facebook/sam2-hiera-base-plus
  modality: image/segmentation
  size_gb: 0.15
  description: "SAM-2 Hiera base-plus: balanced quality, 150MB, Apache 2.0"
  capabilities:
    max_masks: 64
    memory_gb: 1.5

- id: sam2-hiera-large
  uri: hf://facebook/sam2-hiera-large
  modality: image/segmentation
  size_gb: 0.25
  description: "SAM-2 Hiera large: best quality, 225MB, Apache 2.0"
  capabilities:
    max_masks: 64
    memory_gb: 2.5
```

Mask2Former (closed-set semantic) deferred: the architecture differs (per-class output, no prompts) and the wire path (`/v1/images/segment` mode dispatch) doesn't extend cleanly. A second modality or an enriched route layer can host it later.

## Behavioral resilience

1. **Mode dispatch is capability-gated.** A POST with `mode="text"` against `sam2-hiera-tiny` (which has `supports_text_prompts=False`) returns 400 with `invalid_parameter` and message `"model 'sam2-hiera-tiny' does not support text-prompted segmentation"`. The capability is honored end-to-end so users see the constraint before paying for a forward pass.

2. **points / boxes are JSON strings in multipart.** The route parses them with `json.loads`. Bad JSON yields 400 `invalid_parameter`. The parsed value is validated against the expected shape (list of pairs / list of quads of integers); shape mismatches yield 400.

3. **max_masks guards SAM-2's automatic mode.** SAM-2 automatic can produce 100+ masks for a typical photo. After sorting by score descending, the runtime truncates to `max_masks`. The route layer caps the request to `[1, 256]`.

4. **PIL vs numpy axis order.** PIL.Image.size is `(W, H)`; numpy arrays are indexed `[row, col] = [y, x] = [H, W]`. `MaskRecord.mask` follows numpy convention; `image_size` follows PIL convention; `bbox` follows COCO `[x, y, w, h]`. Documented in protocol module and codec docstrings.

5. **pycocotools is optional.** If `pycocotools` is not installed, `encode_mask_rle` and `decode_mask_rle` fall back to a pure-Python implementation that produces and consumes the same `{"size": [H, W], "counts": str}` shape. The pure-Python encoder is byte-equivalent to pycocotools' output for any binary mask (verified by round-trip tests). Slower for large masks but no extra dep.

6. **bbox computation.** `np.where(mask)` finds nonzero pixels, then min/max each axis. Empty masks produce `(0, 0, 0, 0)` bbox and `area = 0`; the runtime drops empty masks before truncation.

7. **transformers >=4.43 required.** SAM-2 support landed in transformers 4.43. The runtime tests mock the entire model class, so they don't depend on the host having transformers 4.43; the per-model venv enforces the constraint at install time.

8. **`auto` mode without explicit prompts.** SAM-2 ships an automatic mask generator that samples a dense grid of point prompts internally. The runtime falls back to a `Sam2AutomaticMaskGenerator` if the transformers version exposes it; otherwise it samples a sparse 32x32 grid manually and aggregates masks with NMS. The bundled model declares `supports_automatic: True` to advertise the path.

## Modality `__init__.py` exports

```python
from muse.modalities.image_segmentation.client import ImageSegmentationClient
from muse.modalities.image_segmentation.protocol import (
    ImageSegmentationModel, MaskRecord, SegmentationResult,
)
from muse.modalities.image_segmentation.routes import build_router

MODALITY = "image/segmentation"

PROBE_DEFAULTS = {
    "shape": "256x256, automatic mode, max_masks=4",
    "call": lambda m: m.segment(_make_probe_image(), mode="auto", max_masks=4),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageSegmentationClient",
    "ImageSegmentationModel",
    "MaskRecord",
    "SegmentationResult",
]
```

## Migration / risk

- New modality directory; no impact on existing routes.
- HF plugin priority 110 sits above image/upscale (105) and image/animation (110). The unique `mask-generation` tag avoids collisions even at equal priority.
- New per-model venv when users `muse pull sam2-hiera-tiny`. Reuses the transformers stack; pip_extras is lean (no diffusers, no accelerate).
- `MUSE_SEGMENTATION_MAX_INPUT_SIDE` is opt-in env-tunable; default 2048.
- The mask format pair (PNG vs RLE) sets a precedent for future modalities that need either viewable or compact mask outputs.

## Test coverage

```
tests/modalities/image_segmentation/
|-- test_protocol.py      # MaskRecord + SegmentationResult shape, structural protocol check
|-- test_codec.py         # PNG + RLE encode/decode round-trip, both pycocotools paths
|-- test_routes.py        # POST /v1/images/segment (auto, points, boxes, text rejection)
|-- test_client.py        # ImageSegmentationClient (multipart upload, JSON points/boxes)
|-- test_hf_plugin.py     # _sniff and _resolve, including SAM-2 + clipseg + fallback
`-- runtimes/
    `-- test_sam2_runtime.py  # patched AutoModelForMaskGeneration

tests/models/test_sam2_hiera_tiny.py             # bundled-script tests

tests/cli_impl/test_e2e_image_segmentation.py    # @pytest.mark.slow
tests/integration/test_remote_image_segmentation.py # opt-in MUSE_REMOTE_SERVER
```

Specifically:

- `test_protocol.py`: SegmentationResult fields, MaskRecord fields, ImageSegmentationModel duck-typing.
- `test_codec.py`:
  - `encode_mask_png` produces valid PNG bytes (header check) and round-trips through PIL.
  - `encode_mask_rle` plus `decode_mask_rle` round-trip a random bool mask byte-for-byte.
  - `encode_mask_rle` produces a `{"size": [H, W], "counts": str}` shape.
  - With pycocotools patched to None, the pure-Python fallback still round-trips.
  - With pycocotools patched to a stub, both encode / decode delegate.
  - `encode_segmentation` envelope shape (id, model, mode, image_size, masks).
- `test_hf_plugin.py`:
  - `_sniff(sam2-hiera-tiny)` returns True (mask-generation tag).
  - `_sniff(clipseg)` returns True (image-segmentation tag).
  - `_sniff(stable-diffusion-2-1)` returns False.
  - `_resolve("facebook/sam2-hiera-tiny", ...)` produces correct capabilities.
  - `_resolve` for clipseg-pattern flips to `supports_text_prompts=True`.
  - `_resolve` fallback for unknown sam-shaped repos.
- `test_routes.py`:
  - happy path POST mode=auto returns 200 + envelope + masks.
  - mode=points with valid JSON list of pairs returns 200.
  - mode=boxes with valid JSON list of quads returns 200.
  - mode=text returns 400 (capability gate; sam2-hiera-tiny declines).
  - bad JSON in points returns 400.
  - shape mismatch in points returns 400 (e.g. triplets).
  - missing image returns 422.
  - empty image returns 400.
  - unknown model returns 404.
  - unsupported mask_format returns 400.
  - max_masks > 256 returns 400.
  - mask_format=rle returns dict masks; mask_format=png_b64 returns base64 string masks.
- `test_client.py`:
  - default base url + MUSE_SERVER override.
  - multipart shape (image + mode + JSON-encoded points/boxes).
  - decode response JSON.
  - non-200 raises RuntimeError.
- `runtimes/test_sam2_runtime.py`:
  - lazy-import sentinels.
  - constructor calls `AutoModelForMaskGeneration.from_pretrained` and `AutoProcessor.from_pretrained`.
  - segment(mode="auto") returns SegmentationResult; masks sorted by score desc; truncated to max_masks.
  - segment(mode="points") forwards points to processor.
  - segment(mode="boxes") forwards boxes to processor.
  - segment with empty mask is dropped.
  - bbox computation matches np.where bounds.
  - area computation matches mask.sum().
- `models/test_sam2_hiera_tiny.py`:
  - manifest required fields.
  - manifest pip_extras has torch + transformers + Pillow + numpy.
  - manifest capabilities advertise expected mode flags.
  - Model class loads via patched `AutoModelForMaskGeneration` + `AutoProcessor`.
  - segment returns SegmentationResult.
- `cli_impl/test_e2e_image_segmentation.py`: full TestClient flow with a fake backend that satisfies the protocol structurally.
- `integration/test_remote_image_segmentation.py`: protocol assertion on a real server, optional via MUSE_REMOTE_SERVER.

## Open questions

None blocking. Two follow-ups deferred:
- Whether `Sam2AutomaticMaskGenerator` is exposed by transformers 4.43 directly or only by the upstream `sam2` package; the runtime falls back to manual grid sampling either way.
- Whether to expose per-mask seed in the response (for deterministic re-runs); current design seeds at call level only.

## Out of scope (filed for later)

- CLIPSeg text-prompted segmentation runtime (separate runtime class). Track in v1.next.
- Mask2Former / closed-set semantic segmentation. Track in v1.next.
- Video object segmentation (SAM-2 supports it). Track in v1.next.
- CRF / morphological refinement post-processing. Future feature.
- Polygon mask format (alternative to RLE). Future feature.
- Tiled segmentation for very large inputs. Future feature.
- The `muse.modalities._common.uploads` shared module: with four multipart modalities (audio_transcription, image_generation, image_upscale, image_segmentation), the case for extraction is even stronger but still optional.
