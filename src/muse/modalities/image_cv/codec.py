"""Encoding for /v1/images/depth, /v1/images/keypoints, /v1/images/detect.

Pure functions: dataclass to OpenAI-shape envelope dict. Tested
without FastAPI.

The depth wire format has two encodings:

  png16:    16-bit grayscale PNG. Float depth values get linearly
            quantized into [0, 65535] using the array's actual min/max.
            Quantization step is range / 65536 (lossless to roughly
            4-5 significant decimal digits). Standard format; viewable
            in any image tool. Default.

  float32:  Raw little-endian float32 bytes (H * W * 4 bytes). Fully
            precise. Larger payload (~4x png16). Useful for downstream
            depth math.

Both encodings are returned base64-encoded along with the depth array's
min and max bounds so the client can recover the original units.
"""
from __future__ import annotations

import base64
import io
import uuid
from typing import Any

from muse.modalities.image_cv.protocol import (
    DepthResult,
    KeypointDetection,
    KeypointResult,
    ObjectDetection,
    ObjectDetectionResult,
)


# ---------- Depth ----------


def encode_depth_png16(depth: Any) -> tuple[bytes, float, float]:
    """Encode a (H, W) float numpy array as 16-bit grayscale PNG.

    Returns (png_bytes, min, max). The depth values are linearly
    rescaled into [0, 65535] using the actual array min/max so the
    full 16-bit range is used (avoiding wasted dynamic range when
    e.g. a relative-depth model emits values in [0, 0.3]).

    The wire envelope includes the min/max so the client can invert:
        depth = (decoded.astype(float) / 65535.0) * (max - min) + min
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(depth)
    lo = float(arr.min()) if arr.size else 0.0
    hi = float(arr.max()) if arr.size else 0.0
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    quantized = (norm * 65535.0).clip(0, 65535).astype(np.uint16)
    # PIL's "I;16" mode is 16-bit unsigned grayscale.
    img = Image.fromarray(quantized, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), lo, hi


def encode_depth_float32(depth: Any) -> tuple[bytes, float, float]:
    """Encode a (H, W) float array as raw little-endian float32 bytes.

    Returns (raw_bytes, min, max). No normalization; values are
    preserved bit-for-bit (after numpy's native float to f4 cast).
    """
    import numpy as np

    arr = np.asarray(depth, dtype="<f4")
    lo = float(arr.min()) if arr.size else 0.0
    hi = float(arr.max()) if arr.size else 0.0
    return arr.tobytes(), lo, hi


def encode_depth_envelope(
    result: DepthResult, *, response_format: str = "png16",
) -> dict[str, Any]:
    """Build the /v1/images/depth wire envelope.

    `response_format` is "png16" (default) or "float32". Other values
    raise ValueError so the route layer can surface as 400.
    """
    if response_format not in ("png16", "float32"):
        raise ValueError(
            f"response_format must be 'png16' or 'float32'; got {response_format!r}"
        )
    if response_format == "png16":
        raw, lo, hi = encode_depth_png16(result.depth)
    else:
        raw, lo, hi = encode_depth_float32(result.depth)
    width, height = result.image_size
    return {
        "id": f"depth-{uuid.uuid4().hex[:24]}",
        "model": result.model_id,
        "depth_map": base64.b64encode(raw).decode(),
        "format": response_format,
        "width": int(width),
        "height": int(height),
        "min_depth": lo,
        "max_depth": hi,
        "metric_depth": bool(result.metric_depth),
    }


# ---------- Keypoints ----------


def encode_keypoints_envelope(result: KeypointResult) -> dict[str, Any]:
    """Build the /v1/images/keypoints wire envelope."""
    detections_out = []
    for det in result.detections:
        detections_out.append({
            "bbox": list(det.bbox),
            "score": float(det.score),
            "keypoints": [
                {
                    "name": kp.name,
                    "x": float(kp.x),
                    "y": float(kp.y),
                    "score": float(kp.score),
                }
                for kp in det.keypoints
            ],
        })
    return {
        "id": f"kp-{uuid.uuid4().hex[:24]}",
        "model": result.model_id,
        "image_size": list(result.image_size),
        "detections": detections_out,
    }


# ---------- Object detection ----------


def encode_detections_envelope(result: ObjectDetectionResult) -> dict[str, Any]:
    """Build the /v1/images/detect wire envelope.

    Detections are assumed to be already sorted (the runtime sorts
    them; the codec just serializes).
    """
    detections_out = [
        {
            "bbox": list(det.bbox),
            "score": float(det.score),
            "label": det.label,
        }
        for det in result.detections
    ]
    return {
        "id": f"det-{uuid.uuid4().hex[:24]}",
        "model": result.model_id,
        "image_size": list(result.image_size),
        "detections": detections_out,
    }
