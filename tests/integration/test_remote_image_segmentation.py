"""Integration tests for /v1/images/segment (opt-in via MUSE_REMOTE_SERVER).

Skipped automatically unless ``MUSE_REMOTE_SERVER`` is set AND the
server has the requested segmentation model loaded. Override the model
id via ``MUSE_SEGMENTATION_MODEL_ID``; default is ``sam2-hiera-tiny``.

Test naming:
  - ``test_protocol_*``: hard claims muse should always satisfy.
  - ``test_observe_*``: records a particular model's observed behavior;
    failures are notes, not regressions.
"""
from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image


pytestmark = pytest.mark.slow


def _png_bytes(width=128, height=128, color=(50, 90, 120)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_segment_auto_returns_envelope(remote_url, segmentation_model):
    """Hard claim: the segment route always returns a valid envelope."""
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={"model": segmentation_model, "mode": "auto", "max_masks": "4"},
        timeout=600.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "masks" in body
    assert "id" in body and body["id"].startswith("seg-")
    assert body["model"] == segmentation_model
    assert body["mode"] == "auto"
    assert isinstance(body["image_size"], list) and len(body["image_size"]) == 2


def test_protocol_segment_points_mode(remote_url, segmentation_model):
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={
            "model": segmentation_model, "mode": "points",
            "points": json.dumps([[64, 64]]),
        },
        timeout=600.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "points"


def test_protocol_segment_boxes_mode(remote_url, segmentation_model):
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={
            "model": segmentation_model, "mode": "boxes",
            "boxes": json.dumps([[20, 20, 100, 100]]),
        },
        timeout=600.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "boxes"


def test_protocol_segment_png_mask_is_valid_image(remote_url, segmentation_model):
    """When mask_format=png_b64, every mask decodes to a real PNG."""
    src = _png_bytes(64, 64)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={
            "model": segmentation_model, "mode": "auto", "max_masks": "2",
            "mask_format": "png_b64",
        },
        timeout=600.0,
    )
    assert r.status_code == 200
    for entry in r.json()["masks"]:
        decoded = base64.b64decode(entry["mask"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_protocol_segment_rle_mask_has_size_and_counts(remote_url, segmentation_model):
    """When mask_format=rle, every mask is a {"size": [H, W], "counts": str}."""
    src = _png_bytes(64, 64)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={
            "model": segmentation_model, "mode": "auto", "max_masks": "2",
            "mask_format": "rle",
        },
        timeout=600.0,
    )
    assert r.status_code == 200
    for entry in r.json()["masks"]:
        assert isinstance(entry["mask"], dict)
        assert "size" in entry["mask"]
        assert "counts" in entry["mask"]
        assert isinstance(entry["mask"]["size"], list)
        assert len(entry["mask"]["size"]) == 2


def test_protocol_segment_text_mode_blocked_for_sam2(remote_url, segmentation_model):
    """SAM-2 declares supports_text_prompts=False; the route must 400."""
    if segmentation_model not in (
        "sam2-hiera-tiny", "sam2-hiera-base-plus", "sam2-hiera-large",
    ):
        pytest.skip(f"text-mode capability gate test only for SAM-2; got {segmentation_model}")
    src = _png_bytes(64, 64)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={
            "model": segmentation_model, "mode": "text", "prompt": "cat",
        },
        timeout=60.0,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_protocol_segment_max_masks_caps_returned_count(remote_url, segmentation_model):
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={
            "model": segmentation_model, "mode": "auto", "max_masks": "3",
        },
        timeout=600.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["masks"]) <= 3


def test_observe_segment_masks_have_bbox_and_area(remote_url, segmentation_model):
    """Observation: SAM-2 returns non-degenerate masks for a typical photo.

    Kept as observe_* because some inputs (a uniform color) may produce
    no masks.
    """
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/segment",
        files={"image": ("src.png", src, "image/png")},
        data={"model": segmentation_model, "mode": "auto", "max_masks": "5"},
        timeout=600.0,
    )
    assert r.status_code == 200
    body = r.json()
    for entry in body["masks"]:
        assert "bbox" in entry and len(entry["bbox"]) == 4
        assert "area" in entry
        assert entry["area"] >= 0
