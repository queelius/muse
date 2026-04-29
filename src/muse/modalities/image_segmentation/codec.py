"""Mask encoding for the image/segmentation wire layer.

Two formats:

  - ``"png_b64"``: base64-encoded PNG (binary mask, white=foreground).
    Portable, viewable in any image tool. Larger payload.
  - ``"rle"``: COCO RLE dict ``{"size": [H, W], "counts": str}``.
    Compact. Standard format for downstream tooling (pycocotools,
    FiftyOne).

Two RLE paths are supported:

  - When ``pycocotools.mask`` is importable, the codec delegates to it
    so external tooling can round-trip muse outputs byte-for-byte.
  - Otherwise the codec falls back to a pure-Python RLE encoder that
    follows the same wire shape (``{"size": [H, W], "counts": str}``)
    and is reciprocal with its own decoder. The pure-Python encoding
    is internally consistent (encode then decode equals identity); it
    is intentionally NOT cross-compatible with pycocotools' compact
    ASCII codec, which is intricate and undocumented outside their
    source. Cross-compatibility is the role of pycocotools when
    installed; the fallback is a self-contained safety net so muse
    works without the dep.

The PIL / numpy axis-order conventions are documented in
``muse.modalities.image_segmentation.protocol``.
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


# Marker that a codec output came from the pure-Python fallback.
# Decoders consult this prefix on counts to decide which decoder to
# use; pycocotools-encoded counts never start with this marker.
_PURE_RLE_PREFIX = "p:"


def _try_import_pycocotools() -> Any | None:
    """Lazy-import pycocotools.mask; return module or None.

    Tests patch this directly to exercise both code paths without a
    real pycocotools install.
    """
    try:
        from pycocotools import mask as _m
        return _m
    except Exception as e:  # noqa: BLE001
        logger.debug("pycocotools unavailable; using pure-Python RLE: %s", e)
        return None


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
    """Encode a 2D bool / uint8 mask in COCO RLE format.

    Returns ``{"size": [H, W], "counts": str}``. When pycocotools is
    available, delegates to ``pycocotools.mask.encode``. Otherwise
    falls back to the pure-Python encoder; the fallback is reciprocal
    with ``decode_mask_rle`` and produces output in the same wire
    shape but with a self-marking counts prefix.
    """
    import numpy as np
    arr = _ensure_uint8(mask)
    h, w = arr.shape
    pct = _try_import_pycocotools()
    if pct is not None:
        f_arr = np.asfortranarray(arr)
        rle = pct.encode(f_arr)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(h), int(w)], "counts": counts}
    runs = _binary_mask_to_runs(arr)
    counts = _PURE_RLE_PREFIX + ",".join(str(r) for r in runs)
    return {"size": [int(h), int(w)], "counts": counts}


def decode_mask_rle(rle: dict) -> Any:
    """Reciprocal of ``encode_mask_rle``; returns a ``(H, W)`` bool array.

    Dispatches by counts prefix:

      - ``"p:..."``: pure-Python format (this codec round-trip).
      - else: pycocotools-encoded; requires pycocotools to decode.
    """
    import numpy as np
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if counts.startswith(_PURE_RLE_PREFIX):
        runs = [int(r) for r in counts[len(_PURE_RLE_PREFIX):].split(",") if r]
        arr = _runs_to_binary_mask(runs, h, w)
        return arr.astype(bool)
    pct = _try_import_pycocotools()
    if pct is None:
        raise RuntimeError(
            "pycocotools is required to decode this RLE counts format; "
            "install pycocotools or re-encode with pure-Python encoder"
        )
    rle_in = {"size": [int(h), int(w)], "counts": counts.encode("ascii")}
    decoded = pct.decode(rle_in)
    return np.asarray(decoded, dtype=bool)


def _binary_mask_to_runs(arr: Any) -> list[int]:
    """Run-length encode a binary mask in column-major (Fortran) order.

    Returns a list of run lengths, alternating between 0-runs and
    1-runs, starting with the count of leading zeros (per COCO
    convention). An all-zero or all-one mask still produces a
    single-element list.
    """
    import numpy as np
    flat = np.asfortranarray(arr).reshape(-1, order="F")
    runs: list[int] = []
    last = 0
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


def _runs_to_binary_mask(runs: list[int], h: int, w: int) -> Any:
    """Reciprocal of ``_binary_mask_to_runs``."""
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
