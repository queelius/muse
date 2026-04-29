"""Tests for /v1/images/segment router."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_segmentation.protocol import (
    MaskRecord, SegmentationResult,
)
from muse.modalities.image_segmentation.routes import build_router


# ---------------- helpers ----------------


class RecordingSegmenter:
    """A fake segmenter that records calls and returns a fixed mask set."""

    def __init__(self, model_id="fake-seg"):
        self.model_id = model_id
        self.calls: list[dict] = []

    def segment(self, image, *, mode="auto", prompt=None,
                points=None, boxes=None, max_masks=None,
                seed=None, **kwargs):
        self.calls.append({
            "image": image,
            "mode": mode,
            "prompt": prompt,
            "points": points,
            "boxes": boxes,
            "max_masks": max_masks,
            "seed": seed,
            **kwargs,
        })
        # Return a single small mask so the codec has work to do.
        h, w = image.size[1], image.size[0]
        arr = np.zeros((h, w), dtype=bool)
        arr[1:3, 1:3] = True
        record = MaskRecord(
            mask=arr, score=0.92, bbox=(1, 1, 2, 2), area=4,
        )
        return SegmentationResult(
            masks=[record], image_size=image.size,
            mode=mode, seed=seed if seed is not None else -1,
            metadata={"prompt": prompt or ""},
        )


def _png_bytes(width=32, height=32, color=(50, 90, 120)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def model():
    return RecordingSegmenter()


def _build_client(model, capabilities=None):
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
        "image/segmentation", model,
        manifest={
            "model_id": model.model_id,
            "modality": "image/segmentation",
            "capabilities": capabilities,
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/segmentation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def client(model):
    return _build_client(model)


# ---------------- happy paths ----------------


def test_post_segment_auto_returns_envelope(client, model):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-seg"
    assert body["mode"] == "auto"
    assert body["image_size"] == [64, 64]
    assert len(body["masks"]) == 1
    assert body["id"].startswith("seg-")
    assert isinstance(body["masks"][0]["mask"], str)
    assert len(model.calls) == 1
    assert model.calls[0]["mode"] == "auto"


def test_post_segment_points_forwards_parsed_list(client, model):
    pts = [[10, 20], [30, 40]]
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={
            "model": "fake-seg", "mode": "points",
            "points": json.dumps(pts),
        },
    )
    assert r.status_code == 200, r.text
    assert model.calls[0]["mode"] == "points"
    assert model.calls[0]["points"] == [[10, 20], [30, 40]]


def test_post_segment_boxes_forwards_parsed_list(client, model):
    bxs = [[5, 5, 25, 25], [10, 10, 40, 40]]
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={
            "model": "fake-seg", "mode": "boxes",
            "boxes": json.dumps(bxs),
        },
    )
    assert r.status_code == 200
    assert model.calls[0]["mode"] == "boxes"
    assert model.calls[0]["boxes"] == [[5, 5, 25, 25], [10, 10, 40, 40]]


def test_post_segment_default_mask_format_is_png_b64(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 200
    import base64
    decoded = base64.b64decode(r.json()["masks"][0]["mask"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_post_segment_rle_mask_format_returns_dict(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(16, 16), "image/png")},
        data={
            "model": "fake-seg", "mode": "auto",
            "mask_format": "rle",
        },
    )
    assert r.status_code == 200
    entry = r.json()["masks"][0]
    assert isinstance(entry["mask"], dict)
    assert "size" in entry["mask"]
    assert "counts" in entry["mask"]


def test_post_segment_default_max_masks_passed_to_backend(client, model):
    client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert model.calls[0]["max_masks"] == 16


def test_post_segment_max_masks_request_passed_to_backend(client, model):
    client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={
            "model": "fake-seg", "mode": "auto", "max_masks": "5",
        },
    )
    assert model.calls[0]["max_masks"] == 5


# ---------------- mode validation ----------------


def test_post_segment_unknown_mode_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "weird"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert "mode" in err["message"].lower()


def test_post_segment_text_mode_capability_blocked(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "text", "prompt": "cat"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert "text" in err["message"].lower()


def test_post_segment_text_mode_requires_prompt(client):
    """Even before capability gating, missing prompt is a 400."""
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "text"},
    )
    assert r.status_code == 400
    assert "prompt" in r.json()["error"]["message"].lower()


def test_post_segment_text_mode_with_text_capable_model_succeeds(model):
    """A model that DOES support text segmentation accepts mode=text."""
    client = _build_client(model, capabilities={
        "supports_automatic": True,
        "supports_point_prompts": False,
        "supports_box_prompts": False,
        "supports_text_prompts": True,
    })
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "text", "prompt": "a cat"},
    )
    assert r.status_code == 200
    assert model.calls[-1]["prompt"] == "a cat"
    assert model.calls[-1]["mode"] == "text"


def test_post_segment_points_capability_disabled(model):
    client = _build_client(model, capabilities={
        "supports_automatic": True,
        "supports_point_prompts": False,
        "supports_box_prompts": True,
        "supports_text_prompts": False,
    })
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "points",
            "points": json.dumps([[1, 1]]),
        },
    )
    assert r.status_code == 400
    assert "point" in r.json()["error"]["message"].lower()


def test_post_segment_boxes_capability_disabled(model):
    client = _build_client(model, capabilities={
        "supports_automatic": True,
        "supports_point_prompts": True,
        "supports_box_prompts": False,
        "supports_text_prompts": False,
    })
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "boxes",
            "boxes": json.dumps([[0, 0, 1, 1]]),
        },
    )
    assert r.status_code == 400
    assert "box" in r.json()["error"]["message"].lower()


def test_post_segment_auto_capability_disabled(model):
    client = _build_client(model, capabilities={
        "supports_automatic": False,
        "supports_point_prompts": True,
        "supports_box_prompts": True,
        "supports_text_prompts": False,
    })
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 400
    assert "automatic" in r.json()["error"]["message"].lower()


# ---------------- JSON parsing of points/boxes ----------------


def test_post_segment_points_required_when_mode_is_points(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "points"},
    )
    assert r.status_code == 400
    assert "points" in r.json()["error"]["message"].lower()


def test_post_segment_points_bad_json_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "points",
            "points": "not json",
        },
    )
    assert r.status_code == 400
    assert "points" in r.json()["error"]["message"].lower()


def test_post_segment_points_bad_shape_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "points",
            "points": json.dumps([[1, 2, 3]]),  # triplet, not pair
        },
    )
    assert r.status_code == 400


def test_post_segment_boxes_bad_shape_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "boxes",
            "boxes": json.dumps([[1, 2]]),  # pair, not quad
        },
    )
    assert r.status_code == 400


def test_post_segment_points_empty_list_rejected(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "points",
            "points": json.dumps([]),
        },
    )
    assert r.status_code == 400


# ---------------- input validation ----------------


def test_post_segment_invalid_mask_format_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-seg", "mode": "auto",
            "mask_format": "webp",
        },
    )
    assert r.status_code == 400
    assert "mask_format" in r.json()["error"]["message"].lower()


def test_post_segment_max_masks_out_of_range_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-seg", "mode": "auto", "max_masks": "999"},
    )
    assert r.status_code == 400


def test_post_segment_empty_image_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", b"", "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_post_segment_malformed_image_returns_400(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", b"not an image", "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 400


def test_post_segment_oversize_input_returns_400(client, monkeypatch):
    monkeypatch.setenv("MUSE_SEGMENTATION_MAX_INPUT_SIDE", "16")
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-seg", "mode": "auto"},
    )
    assert r.status_code == 400
    assert (
        "too large" in r.json()["error"]["message"].lower()
        or "max" in r.json()["error"]["message"].lower()
    )


def test_post_segment_unknown_model_returns_404(client):
    r = client.post(
        "/v1/images/segment",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "no-such", "mode": "auto"},
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"
