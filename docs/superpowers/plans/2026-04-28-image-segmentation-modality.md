# Implementation plan: `image/segmentation` modality (v0.26.0)

**Date:** 2026-04-28
**Spec:** `docs/superpowers/specs/2026-04-28-image-segmentation-modality-design.md`
**Closes:** task #105

## Tasks (one commit each)

- **A** Protocol + codec (RLE encode/decode + PNG, pycocotools optional)
- **B** SAM2Runtime (per-mode dispatch + per-architecture extraction)
- **C** Routes + modality `__init__.py` (multipart, mode + capability gating)
- **D** ImageSegmentationClient
- **E** Bundled `sam2_hiera_tiny.py`
- **F** HF plugin
- **G** Curated entries (3) + slow e2e + integration tests
- **H** Documentation + v0.26.0 release

Each task ends with `pytest tests/ -q -m "not slow"`. Push at H only.

## Task A: protocol + codec

Files:
- `src/muse/modalities/image_segmentation/__init__.py` (skeleton; finalized in C)
- `src/muse/modalities/image_segmentation/protocol.py`
- `src/muse/modalities/image_segmentation/codec.py`
- `tests/modalities/image_segmentation/__init__.py`
- `tests/modalities/image_segmentation/test_protocol.py`
- `tests/modalities/image_segmentation/test_codec.py`

`protocol.py`:

```python
"""Muse image/segmentation modality protocol.

Defines ImageSegmentationModel (backend contract), MaskRecord (one
mask), and SegmentationResult (synthesis return). Backends produce a
list of masks per call, sorted by score descending.

Axis-order convention split (documented here so callers don't trip):
  - PIL.Image.size       -> (W, H)        used by `image_size`
  - numpy 2D arrays      -> [H, W]        used by `mask`
  - COCO bbox            -> [x, y, w, h]  used by `bbox`
  - COCO RLE size field  -> [H, W]        used at the wire layer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MaskRecord:
    mask: Any                                 # numpy 2D bool/uint8 (H, W)
    score: float
    bbox: tuple[int, int, int, int]           # (x, y, w, h)
    area: int


@dataclass
class SegmentationResult:
    masks: list[MaskRecord]
    image_size: tuple[int, int]               # (W, H), PIL convention
    mode: str
    seed: int
    metadata: dict = field(default_factory=dict)


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

`codec.py`:

```python
"""Mask encoding for the image/segmentation wire layer.

Two formats:
  - "png_b64": base64-encoded PNG (binary mask, white=foreground).
    Portable, viewable in any image tool. Larger payload.
  - "rle":     COCO RLE dict {"size": [H, W], "counts": str}. Compact.
    Standard format for downstream tooling (pycocotools, FiftyOne).

The pure-Python RLE encoder produces output byte-equivalent to
pycocotools' `encode` for any binary mask. pycocotools is preferred
when available (lazy-imported); the fallback ensures muse works
without it.

The codec is self-reciprocal: encode_mask_rle followed by
decode_mask_rle round-trips any mask exactly.
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import Any

from muse.modalities.image_segmentation.protocol import (
    MaskRecord, SegmentationResult,
)

logger = logging.getLogger(__name__)


def _try_import_pycocotools() -> Any | None:
    try:
        from pycocotools import mask as _m
        return _m
    except Exception as e:  # noqa: BLE001
        logger.debug("pycocotools unavailable; using pure-Python RLE: %s", e)
        return None


def _ensure_uint8(mask: Any) -> Any:
    import numpy as np
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D; got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = (arr.astype(bool)).astype(np.uint8)
    return arr


def encode_mask_png(mask: Any) -> bytes:
    """Encode a 2D bool/uint8 mask as PNG bytes (white=fg, black=bg)."""
    from PIL import Image
    arr = _ensure_uint8(mask)
    img = Image.fromarray(arr * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_mask_rle(mask: Any) -> dict:
    """Encode a 2D bool/uint8 mask in COCO RLE format.

    Returns {"size": [H, W], "counts": str}. Uses pycocotools.mask.encode
    if available; else a pure-Python encoder that is byte-equivalent.

    The COCO RLE counts string is column-major: traverse the mask in
    Fortran order, emit run lengths, and pack each integer into the
    [0x30, 0x6F] ASCII range with 5 bits per character (sign bit on
    first char).
    """
    import numpy as np
    arr = _ensure_uint8(mask)
    h, w = arr.shape
    pct = _try_import_pycocotools()
    if pct is not None:
        # pycocotools wants Fortran-ordered uint8.
        f_arr = np.asfortranarray(arr)
        rle = pct.encode(f_arr)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(h), int(w)], "counts": counts}
    # Pure-Python fallback. Same wire shape; same packing.
    runs = _binary_mask_to_uncompressed_rle(arr)
    counts = _pack_rle_counts(runs)
    return {"size": [int(h), int(w)], "counts": counts}


def decode_mask_rle(rle: dict) -> Any:
    """Reciprocal of encode_mask_rle; returns a (H, W) bool numpy array."""
    import numpy as np
    h, w = rle["size"]
    counts = rle["counts"]
    pct = _try_import_pycocotools()
    if pct is not None:
        rle_in = {"size": [int(h), int(w)], "counts": counts.encode("ascii")
                  if isinstance(counts, str) else counts}
        decoded = pct.decode(rle_in)
        return np.asarray(decoded, dtype=bool)
    # Pure-Python fallback.
    runs = _unpack_rle_counts(counts)
    arr = _uncompressed_rle_to_mask(runs, h, w)
    return arr.astype(bool)


def _binary_mask_to_uncompressed_rle(arr: Any) -> list[int]:
    """Run-length encode in column-major (Fortran) order.

    Returns a list of run lengths, alternating between 0-runs and
    1-runs, starting with 0-runs (per COCO convention).
    """
    import numpy as np
    flat = np.asfortranarray(arr).reshape(-1, order="F")
    runs: list[int] = []
    last = 0  # current value being counted; alternates 0,1,0,1,...
    count = 0
    for v in flat.tolist():
        if v == last:
            count += 1
        else:
            runs.append(count)
            last = 1 - last
            count = 1
    runs.append(count)
    return runs


def _uncompressed_rle_to_mask(runs: list[int], h: int, w: int) -> Any:
    import numpy as np
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run in runs:
        if val == 1:
            flat[pos:pos + run] = 1
        pos += run
        val = 1 - val
    return flat.reshape((h, w), order="F")


def _pack_rle_counts(runs: list[int]) -> str:
    """Pack a list of run lengths into the COCO compact ASCII format.

    Each integer is encoded as a base-32 little-endian sequence in the
    [0x30, 0x6F] range with bit 5 (0x20) signaling continuation. The
    leading character carries a sign bit; subsequent values are stored
    as differences from the run two positions earlier (RLE counts come
    in pairs of zero-then-one runs; the diff is taken against the
    same-parity neighbor).
    """
    out: list[int] = []
    for i, run in enumerate(runs):
        if i > 2:
            run = run - runs[i - 2]
        more = True
        first = True
        x = run
        while more:
            c = x & 0x1f
            x >>= 5
            if first:
                # Two's-complement sign bit lives in bit 4 of the first nibble.
                if (c & 0x10) != 0:
                    more = x != -1
                else:
                    more = x != 0
                first = False
            else:
                more = x != 0 if (c & 0x10) == 0 else x != -1
            if more:
                c |= 0x20
            c += 0x30
            out.append(c)
    return bytes(out).decode("ascii")


def _unpack_rle_counts(counts: str) -> list[int]:
    """Reciprocal of _pack_rle_counts."""
    runs: list[int] = []
    p = 0
    raw = counts.encode("ascii")
    while p < len(raw):
        x = 0
        k = 0
        more = True
        while more:
            c = raw[p] - 0x30
            x |= (c & 0x1f) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if not more and (c & 0x10) != 0:
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    return runs


def encode_segmentation(
    result: SegmentationResult,
    *,
    model_id: str,
    mask_format: str = "png_b64",
) -> dict:
    """Project SegmentationResult into the wire JSON envelope.

    Returns:
        {"id", "model", "mode", "image_size", "masks": [...]}.
    """
    if mask_format not in ("png_b64", "rle"):
        raise ValueError(
            f"mask_format must be 'png_b64' or 'rle'; got {mask_format!r}"
        )
    masks_out: list[dict] = []
    for idx, m in enumerate(result.masks):
        entry: dict = {
            "index": idx,
            "score": float(m.score),
            "bbox": [int(v) for v in m.bbox],
            "area": int(m.area),
        }
        if mask_format == "png_b64":
            entry["mask"] = base64.b64encode(encode_mask_png(m.mask)).decode("ascii")
        else:
            entry["mask"] = encode_mask_rle(m.mask)
        masks_out.append(entry)
    return {
        "id": f"seg-{uuid.uuid4().hex}",
        "model": model_id,
        "mode": result.mode,
        "image_size": [int(result.image_size[0]), int(result.image_size[1])],
        "masks": masks_out,
    }
```

Tests cover: PNG header round-trip, RLE round-trip with both pycocotools-on
and pycocotools-off paths (patch `_try_import_pycocotools`), envelope shape,
unsupported `mask_format` raises.

Skeleton `__init__.py`:

```python
"""Image segmentation modality (placeholder).

Final exports added in Task C once routes + client land.
"""
from muse.modalities.image_segmentation.protocol import (
    ImageSegmentationModel, MaskRecord, SegmentationResult,
)

MODALITY = "image/segmentation"

__all__ = [
    "MODALITY",
    "ImageSegmentationModel",
    "MaskRecord",
    "SegmentationResult",
]
```

Commit: `feat(image-seg): protocol + codec (PNG + RLE round-trip)`.

## Task B: SAM2Runtime

Files:
- `src/muse/modalities/image_segmentation/runtimes/__init__.py`
- `src/muse/modalities/image_segmentation/runtimes/sam2_runtime.py`
- `tests/modalities/image_segmentation/runtimes/__init__.py`
- `tests/modalities/image_segmentation/runtimes/test_sam2_runtime.py`

The runtime mirrors `DiffusersUpscaleRuntime`: lazy-import sentinels at module
scope, `_ensure_deps()` with short-circuit, `_select_device()` helper, the
class with structural protocol satisfaction.

Mode dispatch:
- `auto`: builds an internal grid of point prompts (32x32 by default),
  iterates the processor + model on batches of points, accumulates masks,
  applies a simple IoU-based NMS at iou_threshold=0.7, sorts by score.
- `points`: passes `input_points=[[points]]` to processor.
- `boxes`: passes `input_boxes=[[boxes]]` to processor.
- `text`: raises a `RuntimeError("text-prompted segmentation not supported by this model")`.
  The route layer gates capability already; runtime defense-in-depth.

For each model output, the runtime calls `processor.post_process_masks(...)`
to upsample to the original image size, then computes bbox + area:

```python
def _bbox_area(mask: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0), 0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), int(mask.sum())
```

Empty masks (area 0) are dropped before sorting + truncation.

The runtime keeps inference under `torch.inference_mode()` so we never need
to put the model into evaluation mode explicitly.

Tests mock `transformers.AutoModelForMaskGeneration` and
`transformers.AutoProcessor` entirely; no real weights loaded. Tests assert:
- constructor calls `from_pretrained` with the expected source.
- `segment(mode="points")` forwards `input_points` to the processor.
- `segment(mode="boxes")` forwards `input_boxes` to the processor.
- `segment(mode="text")` raises.
- masks sorted by score desc; truncated to max_masks.
- empty masks dropped.

Commit: `feat(image-seg): SAM2Runtime with auto/points/boxes mode dispatch`.

## Task C: routes + modality `__init__.py`

Files:
- `src/muse/modalities/image_segmentation/routes.py`
- `src/muse/modalities/image_segmentation/__init__.py` (update)
- `tests/modalities/image_segmentation/test_routes.py`

`routes.py` mirrors `image_upscale/routes.py`: multipart form fields, manual
validation, decode through `decode_image_file`, capability gate by mode,
then call backend.

Capability gate maps mode -> capability key:

```python
MODE_CAPABILITY = {
    "auto":   "supports_automatic",
    "points": "supports_point_prompts",
    "boxes":  "supports_box_prompts",
    "text":   "supports_text_prompts",
}
```

Validation discipline:
- mode in {"auto","points","boxes","text"} else 400.
- mask_format in {"png_b64","rle"} else 400.
- max_masks in [1, 256] else 400.
- mode=points: points required; JSON-decoded; non-empty list of pairs of ints.
- mode=boxes: boxes required; JSON-decoded; non-empty list of quads of ints.
- mode=text: prompt required; up to 4000 chars.
- capability gate: 400 if `MODE_CAPABILITY[mode]` is False in capabilities.

Input cap via `MUSE_SEGMENTATION_MAX_INPUT_SIDE` (default 2048).

The full `__init__.py` exports MODALITY, build_router, ImageSegmentationClient,
ImageSegmentationModel, MaskRecord, SegmentationResult, plus PROBE_DEFAULTS.

Tests: full happy-path matrix (auto, points, boxes), text-rejection, JSON
parse errors, shape errors, capability gating, oversize input, unknown
model 404, mask_format dispatch (PNG vs RLE).

Commit: `feat(image-seg): POST /v1/images/segment route + modality wiring`.

## Task D: ImageSegmentationClient

File: `src/muse/modalities/image_segmentation/client.py`. Plus
`tests/modalities/image_segmentation/test_client.py`.

Mirrors `ImageUpscaleClient`: multipart upload, JSON-encoded
points/boxes, MUSE_SERVER env support.

```python
class ImageSegmentationClient:
    def __init__(self, base_url=None, timeout=600.0): ...

    def segment(self, *, image, model=None, mode="auto", prompt=None,
                points=None, boxes=None, mask_format="png_b64",
                max_masks=16) -> dict:
        """Returns the parsed JSON envelope."""
```

`points` and `boxes` are Python lists; the client serializes via
`json.dumps` before POSTing as Form fields.

Tests: env var honored, default URL, multipart shape (image + form fields),
points/boxes JSON-encoded correctly, non-200 raises RuntimeError.

Commit: `feat(image-seg): ImageSegmentationClient (multipart + JSON points/boxes)`.

## Task E: bundled sam2_hiera_tiny.py

File: `src/muse/models/sam2_hiera_tiny.py`. Plus
`tests/models/test_sam2_hiera_tiny.py`.

Mirrors `bge_reranker_v2_m3.py`'s shape: lazy imports, MANIFEST,
`_select_device`, Model class with `_ensure_deps`. Loads
`AutoModelForMaskGeneration` and `AutoProcessor` from transformers.

The Model class delegates to the same dispatch logic the runtime uses, so
both code paths are kept consistent. (Internally, the bundled Model
imports `_segment_with` from the runtime module so we don't duplicate the
mode dispatch.)

Tests use importlib.import_module per call, patch
`muse.models.sam2_hiera_tiny.AutoModelForMaskGeneration` and
`AutoProcessor`, and exercise: manifest required fields, pip_extras list,
capabilities advertise expected mode flags, Model loads and segments.

Commit: `feat(image-seg): bundled sam2-hiera-tiny model script`.

## Task F: HF plugin

File: `src/muse/modalities/image_segmentation/hf.py`. Plus
`tests/modalities/image_segmentation/test_hf_plugin.py`.

Priority 110. `_sniff` returns True for tags including `mask-generation` or
`image-segmentation`. `_resolve` infers per-pattern capabilities; `_search`
filters by `mask-generation`.

Pattern table (pseudocode):

```python
def _infer_caps(repo_id: str) -> dict:
    rid = repo_id.lower()
    if "sam2-hiera-tiny" in rid:
        return {"max_masks": 64, "memory_gb": 0.8, ...}
    if "sam2-hiera-base" in rid:
        return {"max_masks": 64, "memory_gb": 1.5, ...}
    if "sam2-hiera-large" in rid:
        return {"max_masks": 64, "memory_gb": 2.5, ...}
    if "sam-" in rid or rid.endswith("/sam"):
        return {"max_masks": 64, "memory_gb": 1.5, ...}
    if "clipseg" in rid:
        return {"max_masks": 16, "memory_gb": 0.6,
                "supports_text_prompts": True,
                "supports_point_prompts": False,
                "supports_box_prompts": False,
                "supports_automatic": False, ...}
    return {"max_masks": 16, "memory_gb": 1.0, ...}
```

`_resolve` synthesizes the manifest with `backend_path` pointing at
`muse.modalities.image_segmentation.runtimes.sam2_runtime:SAM2Runtime` and
pip_extras `(torch>=2.1.0, transformers>=4.43.0, Pillow, numpy)`.

Tests: positive sniff for sam2-hiera-tiny tag, positive for clipseg
(image-segmentation tag), negative for stable-diffusion-2-1, resolve
shapes for each pattern, search filters tag.

Commit: `feat(image-seg): HF plugin sniffing mask-generation/image-segmentation`.

## Task G: curated entries + slow e2e + integration

Files:
- `src/muse/curated.yaml` (append three entries)
- `tests/cli_impl/test_e2e_image_segmentation.py` (slow)
- `tests/integration/test_remote_image_segmentation.py` (opt-in)
- `tests/integration/conftest.py` (add `segmentation_model` fixture)

Add curated entries:

```yaml
# ---------- image/segmentation (SAM-2) ----------

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

Slow e2e: full TestClient flow with FakeSegmenter that satisfies the
protocol structurally; covers each mode, both mask formats, the capability
gate, and the response envelope. Decorated with `@pytest.mark.slow`.

Integration test: opt-in via `MUSE_REMOTE_SERVER`; `segmentation_model`
fixture defaults to `sam2-hiera-tiny`. Tests are protocol-style assertions
that must always hold (test_protocol_*).

Commit: `feat(image-seg): curated entries + slow e2e + integration tests`.

## Task H: documentation + v0.26.0 release

Files:
- `pyproject.toml` (version bump to 0.26.0)
- `src/muse/__init__.py` (docstring update; add image/segmentation; bump v reference)
- `CLAUDE.md` (modality list + new section on mask format design)
- `README.md` (modality list + endpoint section)
- `git tag v0.26.0 && git push origin main v0.26.0`
- GitHub release via `gh release create`

CLAUDE.md update: add image/segmentation to the modality list, add a new
paragraph explaining the mask format design (PNG vs RLE) and the mode
dispatch with capability gating.

README.md update: bump count to 14 modalities, add image/segmentation row,
add a "Segmentation" subsection with curl examples for all three modes
plus a Python example.

Commit: `chore(release): v0.26.0`.

Tag, push, release.

## Acceptance

- 14 modalities discovered.
- `pytest tests/ -m "not slow"` green.
- `pytest tests/ -q` (slow lane) green.
- `muse models list` shows sam2-hiera-tiny / sam2-hiera-base-plus / sam2-hiera-large.
- `muse search sam2 --modality image/segmentation` returns SAM-2 family.
- `muse pull sam2-hiera-tiny` would create a venv with transformers + torch + Pillow + numpy.
- POST /v1/images/segment with valid auto / points / boxes payloads succeeds end-to-end.
- POST /v1/images/segment with mode=text returns 400 for sam2-hiera-tiny.
- v0.26.0 tagged, pushed, GitHub release published.
