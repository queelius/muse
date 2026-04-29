"""End-to-end: /v1/images/segment through FastAPI + codec correctly.

Uses a fake ImageSegmentationModel backend; no real weights. The fake
segmenter returns a single small mask which the codec encodes back
into either base64 PNG or COCO RLE end-to-end.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_segmentation import (
    MODALITY,
    MaskRecord,
    SegmentationResult,
    build_router,
)
from muse.modalities.image_segmentation.codec import decode_mask_rle


pytestmark = pytest.mark.slow


def _png_bytes(width=32, height=32, color=(5, 10, 15)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeSegmenter:
    """A fake segmenter that returns a fixed small mask."""

    model_id = "fake-seg"

    def segment(self, image, *, mode="auto", prompt=None,
                points=None, boxes=None, max_masks=None,
                seed=None, **kwargs):
        h, w = image.size[1], image.size[0]
        arr = np.zeros((h, w), dtype=bool)
        # Place a 4x4 box of foreground pixels at (1, 2)
        arr[1:5, 2:6] = True
        record = MaskRecord(
            mask=arr, score=0.92, bbox=(2, 1, 4, 4), area=16,
        )
        return SegmentationResult(
            masks=[record], image_size=image.size, mode=mode,
            seed=seed if seed is not None else -1,
            metadata={"prompt": prompt or ""},
        )


def _build_client(capabilities=None):
    if capabilities is None:
        capabilities = {
            "supports_automatic": True,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_text_prompts": False,
            "max_masks": 16,
        }
    reg = ModalityRegistry()
    reg.register(
        MODALITY, FakeSegmenter(),
        manifest={
            "model_id": "fake-seg",
            "modality": MODALITY,
            "capabilities": capabilities,
        },
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


@pytest.mark.timeout(10)
def test_segmentation_e2e_auto_png_b64():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_size"] == [64, 64]
    assert body["mode"] == "auto"
    assert len(body["masks"]) == 1
    decoded = base64.b64decode(body["masks"][0]["mask"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    out = Image.open(io.BytesIO(decoded))
    arr = np.asarray(out) > 0
    assert arr.shape == (64, 64)
    assert arr[1:5, 2:6].all()  # foreground region preserved


@pytest.mark.timeout(10)
def test_segmentation_e2e_auto_rle_round_trip():
    """RLE format round-trips: encode in route, decode in test, match input mask."""
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-seg", "mode": "auto",
            "mask_format": "rle",
        },
    )
    assert r.status_code == 200
    entry = r.json()["masks"][0]
    assert isinstance(entry["mask"], dict)
    decoded = decode_mask_rle(entry["mask"])
    expected = np.zeros((32, 32), dtype=bool)
    expected[1:5, 2:6] = True
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.timeout(10)
def test_segmentation_e2e_points_mode():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-seg", "mode": "points",
            "points": json.dumps([[10, 12]]),
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "points"


@pytest.mark.timeout(10)
def test_segmentation_e2e_boxes_mode():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-seg", "mode": "boxes",
            "boxes": json.dumps([[5, 5, 25, 25]]),
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "boxes"


@pytest.mark.timeout(10)
def test_segmentation_e2e_text_mode_capability_blocked():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-seg", "mode": "text",
            "prompt": "find the cat",
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert "text" in err["message"].lower()


@pytest.mark.timeout(10)
def test_segmentation_e2e_unknown_model_returns_404():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "nope", "mode": "auto"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.timeout(10)
def test_segmentation_e2e_envelope_carries_id_and_model():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(16, 16), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    body = r.json()
    assert body["id"].startswith("seg-")
    assert body["model"] == "fake-seg"


@pytest.mark.timeout(10)
def test_segmentation_e2e_mask_carries_bbox_and_area():
    client = _build_client()
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    entry = r.json()["masks"][0]
    assert entry["bbox"] == [2, 1, 4, 4]
    assert entry["area"] == 16
    assert entry["score"] == pytest.approx(0.92)
