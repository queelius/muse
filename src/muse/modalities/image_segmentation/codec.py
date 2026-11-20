"""Mask encoding for the image/segmentation wire layer.

Two formats:

  - ``"png_b64"``: base64-encoded PNG (binary mask, white=foreground).
    Portable, viewable in any image tool. Larger payload.
  - ``"rle"``: muse's own compact run-length format,
    ``{"size": [H, W], "counts": "p:<r0>,<r1>,..."}``. The runs are
    column-major (Fortran order, like COCO) and alternate 0-runs and
    1-runs starting with the leading 0-run, written as decimal integers
    after the ``p:`` marker. The ``size``/order conventions follow COCO,
    but the counts are NOT COCO's compressed string, so pycocotools and
    FiftyOne cannot decode it directly; ``decode_mask_rle`` (or splitting
    the counts on commas) can.

Encoding is vectorised with numpy: routes encode up to 256 masks per
request, and a per-pixel Python loop took ~0.5 s per 2048x2048 mask.

The PIL / numpy axis-order conventions are documented in
``muse.modalities.image_segmentation.protocol``.
"""
from __future__ import annotations

import base64
import io
import uuid
from typing import Any

from muse.modalities.image_segmentation.protocol import SegmentationResult


# Marks muse's run-length counts (see the module docstring).
_PURE_RLE_PREFIX = "p:"


def _ensure_uint8(mask: Any) -> Any:
    """Coerce a 2D bool / int / uint8 mask into a uint8 array of {0,1}."""
    import numpy as np
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D; got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(bool).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
    return arr


def encode_mask_png(mask: Any) -> bytes:
    """Encode a 2D bool / uint8 mask as PNG bytes.

    ``mask[y, x] = True`` becomes a white pixel; ``False`` becomes
    black. The output is a single-channel PNG.
    """
    from PIL import Image
    arr = _ensure_uint8(mask)
    img = Image.fromarray(arr * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_mask_rle(mask: Any) -> dict:
    """Encode a 2D bool / uint8 mask in muse's run-length format.

    Returns ``{"size": [H, W], "counts": "p:..."}``, reciprocal with
    ``decode_mask_rle``.
    """
    arr = _ensure_uint8(mask)
    h, w = arr.shape
    runs = _binary_mask_to_runs(arr)
    counts = _PURE_RLE_PREFIX + ",".join(map(str, runs))
    return {"size": [int(h), int(w)], "counts": counts}


def decode_mask_rle(rle: dict) -> Any:
    """Reciprocal of ``encode_mask_rle``; returns a ``(H, W)`` bool array.

    Raises ValueError for counts without the ``p:`` marker or runs that
    do not add up to ``H * W``.
    """
    h, w = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not counts.startswith(_PURE_RLE_PREFIX):
        raise ValueError(
            "RLE counts must use muse's 'p:' run-length format"
        )
    runs = [int(r) for r in counts[len(_PURE_RLE_PREFIX):].split(",") if r]
    if sum(runs) != h * w:
        raise ValueError(
            f"RLE runs cover {sum(runs)} pixels; size {h}x{w} needs {h * w}"
        )
    return _runs_to_binary_mask(runs, h, w).astype(bool)


def _binary_mask_to_runs(arr: Any) -> list[int]:
    """Run-length encode a binary mask in column-major (Fortran) order.

    Returns a list of run lengths, alternating between 0-runs and
    1-runs, starting with the count of leading zeros (per COCO
    convention, so a mask starting with 1 begins with a 0-length run).
    An all-zero mask produces a single-element list.
    """
    import numpy as np
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1, order="F")
    if flat.size == 0:
        return [0]
    # Indices where the value changes, as run boundaries.
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], edges, [flat.size]))
    runs = np.diff(bounds)
    if flat[0] == 1:
        runs = np.concatenate(([0], runs))
    return runs.tolist()


def _runs_to_binary_mask(runs: list[int], h: int, w: int) -> Any:
    """Reciprocal of ``_binary_mask_to_runs``."""
    import numpy as np
    lengths = np.asarray(runs, dtype=np.int64)
    values = np.arange(lengths.size, dtype=np.uint8) % 2
    flat = np.repeat(values, lengths)
    return flat.reshape((h, w), order="F")


def encode_segmentation(
    result: SegmentationResult,
    *,
    model_id: str,
    mask_format: str = "png_b64",
) -> dict:
    """Project a ``SegmentationResult`` into the wire JSON envelope.

    Returns:
        ``{"id", "model", "mode", "image_size", "masks": [...]}``.

    Raises:
        ValueError: when ``mask_format`` is unsupported.
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
            entry["mask"] = base64.b64encode(
                encode_mask_png(m.mask)
            ).decode("ascii")
        else:
            entry["mask"] = encode_mask_rle(m.mask)
        masks_out.append(entry)
    return {
        "id": f"seg-{uuid.uuid4().hex}",
        "model": model_id,
        "mode": result.mode,
        "image_size": [
            int(result.image_size[0]),
            int(result.image_size[1]),
        ],
        "masks": masks_out,
    }


__all__ = [
    "encode_mask_png",
    "encode_mask_rle",
    "decode_mask_rle",
    "encode_segmentation",
]
