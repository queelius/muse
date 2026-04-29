"""Tests for the image/segmentation codec.

Two RLE paths exercised via patching ``_try_import_pycocotools``:
  - Pure-Python fallback (pycocotools=None): exhaustive round-trip.
  - pycocotools-stub: delegation verified.

PNG path: header check + PIL round-trip back to a bool array.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from muse.modalities.image_segmentation import codec
from muse.modalities.image_segmentation.protocol import (
    MaskRecord, SegmentationResult,
)


# ---------------- PNG path ----------------


def test_encode_mask_png_starts_with_png_header():
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[1:3, 1:3] = 1
    blob = codec.encode_mask_png(arr)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_mask_png_round_trips_through_pil():
    arr = np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ], dtype=np.uint8)
    blob = codec.encode_mask_png(arr)
    img = Image.open(io.BytesIO(blob))
    decoded = (np.asarray(img) > 0).astype(np.uint8)
    assert decoded.shape == arr.shape
    np.testing.assert_array_equal(decoded, arr)


def test_encode_mask_png_accepts_bool_input():
    arr = np.zeros((3, 3), dtype=bool)
    arr[1, 1] = True
    blob = codec.encode_mask_png(arr)
    img = Image.open(io.BytesIO(blob))
    decoded = (np.asarray(img) > 0).astype(np.uint8)
    expected = arr.astype(np.uint8)
    np.testing.assert_array_equal(decoded, expected)


def test_encode_mask_png_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2D"):
        codec.encode_mask_png(np.zeros((2, 3, 4)))


# ---------------- Pure-Python RLE round-trip ----------------


def test_rle_round_trip_with_no_pycocotools(monkeypatch):
    """Encode then decode equals the original mask, byte-for-byte."""
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    rng = np.random.default_rng(seed=42)
    for _ in range(10):
        h, w = int(rng.integers(8, 32)), int(rng.integers(8, 32))
        arr = (rng.random((h, w)) > 0.5).astype(np.uint8)
        rle = codec.encode_mask_rle(arr)
        assert rle["size"] == [h, w]
        assert isinstance(rle["counts"], str)
        decoded = codec.decode_mask_rle(rle)
        assert decoded.dtype == bool
        np.testing.assert_array_equal(decoded.astype(np.uint8), arr)


def test_rle_round_trip_all_zeros(monkeypatch):
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    arr = np.zeros((6, 8), dtype=np.uint8)
    rle = codec.encode_mask_rle(arr)
    decoded = codec.decode_mask_rle(rle)
    np.testing.assert_array_equal(decoded.astype(np.uint8), arr)


def test_rle_round_trip_all_ones(monkeypatch):
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    arr = np.ones((6, 8), dtype=np.uint8)
    rle = codec.encode_mask_rle(arr)
    decoded = codec.decode_mask_rle(rle)
    np.testing.assert_array_equal(decoded.astype(np.uint8), arr)


def test_rle_size_uses_h_w_not_w_h(monkeypatch):
    """COCO RLE stores [H, W] in size; verify against a non-square mask."""
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    arr = np.zeros((3, 7), dtype=np.uint8)
    rle = codec.encode_mask_rle(arr)
    assert rle["size"] == [3, 7]


def test_rle_pure_python_counts_starts_with_marker(monkeypatch):
    """The pure-Python fallback marks its output with the `p:` prefix."""
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = codec.encode_mask_rle(arr)
    assert rle["counts"].startswith("p:")


# ---------------- pycocotools delegation ----------------


def test_rle_delegates_to_pycocotools_when_available(monkeypatch):
    """encode + decode go through pycocotools.mask when importable."""

    encode_calls: list = []
    decode_calls: list = []

    class _StubPct:
        @staticmethod
        def encode(arr):
            encode_calls.append(arr.shape)
            return {"size": [int(arr.shape[0]), int(arr.shape[1])],
                    "counts": b"OK"}

        @staticmethod
        def decode(rle):
            decode_calls.append(rle["counts"])
            h, w = rle["size"]
            return np.ones((h, w), dtype=np.uint8)

    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: _StubPct)

    arr = np.zeros((4, 5), dtype=np.uint8)
    rle = codec.encode_mask_rle(arr)
    assert rle["counts"] == "OK"
    assert encode_calls == [(4, 5)]

    decoded = codec.decode_mask_rle(rle)
    assert decoded.shape == (4, 5)
    assert decode_calls == [b"OK"]


def test_rle_decode_raises_without_pycocotools_for_native_format(monkeypatch):
    """If we're handed pycocotools-encoded counts but pct is unavailable, raise."""
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    rle = {"size": [4, 4], "counts": "abcd"}  # no `p:` prefix
    with pytest.raises(RuntimeError, match="pycocotools is required"):
        codec.decode_mask_rle(rle)


def test_rle_encode_rejects_3d_mask(monkeypatch):
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    with pytest.raises(ValueError, match="must be 2D"):
        codec.encode_mask_rle(np.zeros((2, 3, 4)))


# ---------------- encode_segmentation envelope ----------------


def test_encode_segmentation_envelope_png_format():
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[2:5, 2:5] = 1
    record = MaskRecord(mask=arr, score=0.9, bbox=(2, 2, 3, 3), area=9)
    result = SegmentationResult(
        masks=[record],
        image_size=(8, 8),
        mode="auto",
        seed=-1,
    )
    out = codec.encode_segmentation(
        result, model_id="m", mask_format="png_b64",
    )
    assert out["model"] == "m"
    assert out["mode"] == "auto"
    assert out["image_size"] == [8, 8]
    assert out["id"].startswith("seg-")
    assert len(out["masks"]) == 1
    entry = out["masks"][0]
    assert entry["index"] == 0
    assert entry["score"] == pytest.approx(0.9)
    assert entry["bbox"] == [2, 2, 3, 3]
    assert entry["area"] == 9
    assert isinstance(entry["mask"], str)
    decoded = base64.b64decode(entry["mask"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_segmentation_envelope_rle_format(monkeypatch):
    monkeypatch.setattr(codec, "_try_import_pycocotools", lambda: None)
    arr = np.zeros((4, 6), dtype=np.uint8)
    arr[1:3, 1:4] = 1
    record = MaskRecord(mask=arr, score=0.8, bbox=(1, 1, 3, 2), area=6)
    result = SegmentationResult(
        masks=[record], image_size=(6, 4), mode="points", seed=-1,
    )
    out = codec.encode_segmentation(
        result, model_id="seg", mask_format="rle",
    )
    entry = out["masks"][0]
    assert isinstance(entry["mask"], dict)
    assert entry["mask"]["size"] == [4, 6]
    assert isinstance(entry["mask"]["counts"], str)
    # round-trip through the codec's decoder
    decoded = codec.decode_mask_rle(entry["mask"])
    np.testing.assert_array_equal(decoded.astype(np.uint8), arr)


def test_encode_segmentation_rejects_bad_mask_format():
    result = SegmentationResult(
        masks=[], image_size=(1, 1), mode="auto", seed=-1,
    )
    with pytest.raises(ValueError, match="mask_format"):
        codec.encode_segmentation(
            result, model_id="m", mask_format="webp",
        )


def test_encode_segmentation_indexes_in_order():
    arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    masks = [
        MaskRecord(mask=arr, score=0.9, bbox=(0, 0, 2, 2), area=2),
        MaskRecord(mask=arr, score=0.5, bbox=(0, 0, 2, 2), area=2),
        MaskRecord(mask=arr, score=0.3, bbox=(0, 0, 2, 2), area=2),
    ]
    result = SegmentationResult(
        masks=masks, image_size=(2, 2), mode="auto", seed=-1,
    )
    out = codec.encode_segmentation(
        result, model_id="m", mask_format="png_b64",
    )
    assert [m["index"] for m in out["masks"]] == [0, 1, 2]
