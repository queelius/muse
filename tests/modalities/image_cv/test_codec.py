"""Codec tests for image/cv: depth PNG16, depth float32, JSON envelopes."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from muse.modalities.image_cv.codec import (
    encode_depth_envelope,
    encode_depth_float32,
    encode_depth_png16,
    encode_detections_envelope,
    encode_keypoints_envelope,
)
from muse.modalities.image_cv.protocol import (
    DepthResult,
    Keypoint,
    KeypointDetection,
    KeypointResult,
    ObjectDetection,
    ObjectDetectionResult,
)


def test_encode_depth_png16_round_trip():
    """A normalized depth array round-trips through PNG16 with negligible
    quantization error (within 1/65535 of the dynamic range)."""
    rng = np.random.default_rng(seed=42)
    depth = rng.uniform(0.1, 0.9, size=(64, 80)).astype("float32")
    raw, lo, hi = encode_depth_png16(depth)

    # Decode and de-normalize to recover the original.
    img = Image.open(io.BytesIO(raw))
    assert img.mode == "I;16"
    quantized = np.array(img)
    recovered = (quantized.astype(float) / 65535.0) * (hi - lo) + lo

    # Quantization error: at most 1/65535 of (hi - lo).
    expected_eps = (hi - lo) / 65535.0
    assert np.max(np.abs(recovered - depth)) <= 2 * expected_eps


def test_encode_depth_png16_preserves_min_max():
    depth = np.array([[0.5, 1.5, 2.5], [3.0, 0.0, 1.0]], dtype="float32")
    _, lo, hi = encode_depth_png16(depth)
    assert lo == 0.0
    assert hi == 3.0


def test_encode_depth_png16_handles_constant_array():
    """When min == max, the normalization step would divide by zero;
    the encoder must short-circuit to a zeros array."""
    depth = np.full((4, 4), 0.5, dtype="float32")
    raw, lo, hi = encode_depth_png16(depth)
    assert lo == 0.5
    assert hi == 0.5
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)
    assert (arr == 0).all()


def test_encode_depth_float32_round_trip():
    """Raw float32 bytes preserve values exactly (modulo float32 cast)."""
    depth = np.array([[1.5, 2.0], [3.5, 4.0]], dtype="float32")
    raw, lo, hi = encode_depth_float32(depth)
    assert lo == 1.5
    assert hi == 4.0
    recovered = np.frombuffer(raw, dtype="<f4").reshape(2, 2)
    np.testing.assert_array_equal(recovered, depth)


def test_encode_depth_envelope_png16_default():
    arr = np.array([[0.5, 1.0], [0.0, 0.25]], dtype="float32")
    result = DepthResult(
        depth=arr, model_id="depth-anything-v2-small",
        image_size=(2, 2),
    )
    body = encode_depth_envelope(result)
    assert body["model"] == "depth-anything-v2-small"
    assert body["format"] == "png16"
    assert body["width"] == 2
    assert body["height"] == 2
    assert body["min_depth"] == 0.0
    assert body["max_depth"] == 1.0
    assert body["metric_depth"] is False
    assert body["id"].startswith("depth-")
    # Decode the base64 + PNG to check the bytes flow through.
    raw = base64.b64decode(body["depth_map"])
    Image.open(io.BytesIO(raw))  # raises if invalid


def test_encode_depth_envelope_float32():
    arr = np.array([[1.5, 2.5], [3.0, 4.0]], dtype="float32")
    result = DepthResult(
        depth=arr, model_id="zoedepth", image_size=(2, 2),
        metric_depth=True,
    )
    body = encode_depth_envelope(result, response_format="float32")
    assert body["format"] == "float32"
    assert body["metric_depth"] is True
    raw = base64.b64decode(body["depth_map"])
    recovered = np.frombuffer(raw, dtype="<f4").reshape(2, 2)
    np.testing.assert_array_equal(recovered, arr)


def test_encode_depth_envelope_invalid_format_raises():
    arr = np.zeros((2, 2), dtype="float32")
    result = DepthResult(depth=arr, model_id="m", image_size=(2, 2))
    with pytest.raises(ValueError, match="response_format"):
        encode_depth_envelope(result, response_format="unsupported")


def test_encode_depth_envelope_id_unique():
    arr = np.zeros((2, 2), dtype="float32")
    result = DepthResult(depth=arr, model_id="m", image_size=(2, 2))
    a = encode_depth_envelope(result)
    b = encode_depth_envelope(result)
    assert a["id"] != b["id"]


# ---------- Keypoints ----------


def test_encode_keypoints_envelope_shape():
    det = KeypointDetection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        score=0.95,
        keypoints=[
            Keypoint(name="nose", x=120.0, y=80.0, score=0.99),
            Keypoint(name="left_eye", x=100.0, y=70.0, score=0.97),
        ],
    )
    result = KeypointResult(
        detections=[det], model_id="vitpose", image_size=(640, 480),
    )
    body = encode_keypoints_envelope(result)
    assert body["model"] == "vitpose"
    assert body["image_size"] == [640, 480]
    assert body["id"].startswith("kp-")
    assert len(body["detections"]) == 1
    d = body["detections"][0]
    assert d["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert d["score"] == 0.95
    assert len(d["keypoints"]) == 2
    assert d["keypoints"][0] == {
        "name": "nose", "x": 120.0, "y": 80.0, "score": 0.99,
    }


def test_encode_keypoints_envelope_empty_detections():
    result = KeypointResult(
        detections=[], model_id="vp", image_size=(100, 100),
    )
    body = encode_keypoints_envelope(result)
    assert body["detections"] == []


# ---------- Object detection ----------


def test_encode_detections_envelope_shape():
    result = ObjectDetectionResult(
        detections=[
            ObjectDetection(bbox=(10, 20, 100, 200), score=0.9, label="cat"),
            ObjectDetection(bbox=(50, 60, 80, 90), score=0.7, label="dog"),
        ],
        model_id="detr-resnet-50",
        image_size=(640, 480),
    )
    body = encode_detections_envelope(result)
    assert body["model"] == "detr-resnet-50"
    assert body["image_size"] == [640, 480]
    assert body["id"].startswith("det-")
    assert len(body["detections"]) == 2
    d0 = body["detections"][0]
    assert d0["bbox"] == [10, 20, 100, 200]
    assert d0["label"] == "cat"
    assert d0["score"] == 0.9


def test_encode_detections_envelope_empty():
    result = ObjectDetectionResult(
        detections=[], model_id="m", image_size=(100, 100),
    )
    body = encode_detections_envelope(result)
    assert body["detections"] == []


def test_encode_detections_envelope_float_coercion():
    """Scores from numpy / torch tensors get coerced to plain Python float."""
    score_np = np.float32(0.95)
    result = ObjectDetectionResult(
        detections=[
            ObjectDetection(bbox=(0, 0, 1, 1), score=score_np, label="x"),
        ],
        model_id="m",
        image_size=(1, 1),
    )
    body = encode_detections_envelope(result)
    assert isinstance(body["detections"][0]["score"], float)
