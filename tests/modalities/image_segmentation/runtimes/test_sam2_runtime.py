"""Tests for SAM2Runtime (transformers AutoModelForMaskGeneration wrapper).

The full mock surface mirrors the image_embedding test pattern:
patch the module-top sentinels (``torch``, ``AutoModelForMaskGeneration``,
``AutoProcessor``) so the real transformers package isn't required to
exercise the runtime's logic.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import muse.modalities.image_segmentation.runtimes.sam2_runtime as sam_mod
from muse.modalities.image_segmentation.protocol import (
    MaskRecord, SegmentationResult,
)
from muse.modalities.image_segmentation.runtimes.sam2_runtime import (
    SAM2Runtime,
    _bbox_area,
    _select_device,
    _segment_with,
)


# ---------------- helpers ----------------


class _FakePILImage:
    """Minimal PIL-like stand-in with a .size attribute."""

    def __init__(self, size):
        self.size = size


def _make_processor(masks_arr: np.ndarray, scores_arr: np.ndarray):
    """Build a processor mock that captures the kwargs and returns our masks.

    The processor's ``post_process_masks`` is stubbed to return
    ``[masks_arr]`` (one-item list, batch index 0).
    """
    proc = MagicMock()
    proc.captured_kwargs = []

    def _call(**kw):
        proc.captured_kwargs.append(kw)
        # Return a dict-shaped BatchEncoding stand-in.
        return {
            "pixel_values": MagicMock(),
            "original_sizes": [(masks_arr.shape[-2], masks_arr.shape[-1])],
            "reshaped_input_sizes": [(masks_arr.shape[-2], masks_arr.shape[-1])],
        }

    proc.side_effect = _call
    proc.post_process_masks.return_value = [masks_arr]
    return proc


def _make_model(masks_arr, scores_arr):
    """A model mock whose ``__call__`` returns pred_masks + iou_scores."""
    out = SimpleNamespace(
        pred_masks=masks_arr,
        iou_scores=scores_arr,
    )
    model = MagicMock(return_value=out)
    model.to.return_value = model
    return model


def _patched_torch():
    fake = MagicMock()
    fake.cuda.is_available.return_value = False
    fake.backends.mps.is_available.return_value = False
    fake.float16 = "float16"
    fake.float32 = "float32"
    fake.bfloat16 = "bfloat16"
    fake.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
    fake.inference_mode.return_value.__exit__ = MagicMock(return_value=None)
    return fake


def _build_runtime(*, model, processor, device="cpu",
                   max_masks=4, supports_text_prompts=False,
                   supports_point_prompts=True,
                   supports_box_prompts=True,
                   supports_automatic=True):
    """Construct a fully-mocked SAM2Runtime."""
    fake_amg = MagicMock()
    fake_amg.from_pretrained.return_value = model
    fake_ap = MagicMock()
    fake_ap.from_pretrained.return_value = processor
    fake_torch = _patched_torch()

    with patch.object(sam_mod, "AutoModelForMaskGeneration", fake_amg), \
            patch.object(sam_mod, "AutoProcessor", fake_ap), \
            patch.object(sam_mod, "torch", fake_torch):
        rt = SAM2Runtime(
            hf_repo="test/sam2",
            device=device,
            dtype="float16",
            model_id="sam2-test",
            max_masks=max_masks,
            supports_text_prompts=supports_text_prompts,
            supports_point_prompts=supports_point_prompts,
            supports_box_prompts=supports_box_prompts,
            supports_automatic=supports_automatic,
        )
    return rt, fake_amg, fake_ap


# ---------------- _bbox_area ----------------


def test_bbox_area_simple_rectangle():
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:5, 3:8] = 1
    bbox, area = _bbox_area(arr)
    assert bbox == (3, 2, 5, 3)
    assert area == 15


def test_bbox_area_empty_mask():
    arr = np.zeros((10, 10), dtype=np.uint8)
    bbox, area = _bbox_area(arr)
    assert bbox == (0, 0, 0, 0)
    assert area == 0


def test_bbox_area_full_mask():
    arr = np.ones((4, 5), dtype=np.uint8)
    bbox, area = _bbox_area(arr)
    assert bbox == (0, 0, 5, 4)
    assert area == 20


# ---------------- _select_device ----------------


def test_select_device_explicit_value_passes_through():
    fake = _patched_torch()
    with patch.object(sam_mod, "torch", fake):
        assert _select_device("cuda") == "cuda"
        assert _select_device("cpu") == "cpu"


def test_select_device_auto_falls_to_cpu_without_cuda():
    fake = _patched_torch()
    with patch.object(sam_mod, "torch", fake):
        assert _select_device("auto") == "cpu"


def test_select_device_auto_returns_cpu_when_torch_none():
    with patch.object(sam_mod, "torch", None):
        assert _select_device("auto") == "cpu"


# ---------------- constructor ----------------


def test_sam2_runtime_constructor_calls_from_pretrained():
    masks = np.zeros((1, 2, 8, 8), dtype=bool)
    scores = np.array([[0.9, 0.5]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, amg, ap = _build_runtime(model=model, processor=proc)
    amg.from_pretrained.assert_called_once_with("test/sam2", torch_dtype="float16")
    ap.from_pretrained.assert_called_once_with("test/sam2")
    assert rt.model_id == "sam2-test"
    assert rt.max_masks == 4


def test_sam2_runtime_raises_without_transformers():
    with patch.object(sam_mod, "AutoModelForMaskGeneration", None), \
            patch.object(sam_mod, "AutoProcessor", None), \
            patch.object(sam_mod, "_ensure_deps", lambda: None):
        with pytest.raises(RuntimeError, match="transformers is not installed"):
            SAM2Runtime(
                hf_repo="x", model_id="m",
            )


# ---------------- mode dispatch ----------------


def test_segment_points_forwards_input_points_to_processor():
    arr = np.zeros((8, 8), dtype=bool)
    arr[1:4, 1:4] = True
    masks = np.array([[arr]], dtype=bool)  # shape [1, 1, 8, 8]
    scores = np.array([[0.95]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc)
    img = _FakePILImage((8, 8))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="points", points=[[3, 3]])

    assert isinstance(result, SegmentationResult)
    assert result.mode == "points"
    assert result.image_size == (8, 8)
    assert len(result.masks) == 1
    assert result.masks[0].score == pytest.approx(0.95)
    # Verify input_points reached the processor.
    last_kw = proc.captured_kwargs[-1]
    assert last_kw.get("input_points") == [[[[3, 3]]]]


def test_segment_boxes_forwards_input_boxes_to_processor():
    arr = np.zeros((8, 8), dtype=bool)
    arr[2:6, 2:6] = True
    masks = np.array([[arr]], dtype=bool)
    scores = np.array([[0.85]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc)
    img = _FakePILImage((8, 8))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="boxes", boxes=[[1, 1, 6, 6]])

    last_kw = proc.captured_kwargs[-1]
    assert last_kw.get("input_boxes") == [[[1, 1, 6, 6]]]
    assert result.masks[0].score == pytest.approx(0.85)


def test_segment_auto_uses_grid_of_points():
    arr = np.zeros((16, 16), dtype=bool)
    arr[5:10, 5:10] = True
    masks = np.stack([arr for _ in range(4)])[None]  # shape [1, 4, 16, 16]
    scores = np.array([[0.9, 0.7, 0.5, 0.3]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc, max_masks=2)
    img = _FakePILImage((16, 16))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="auto")

    last_kw = proc.captured_kwargs[-1]
    pts = last_kw.get("input_points")
    assert pts is not None
    # Outer batch list -> per-image -> per-prompt-set -> coordinate.
    assert len(pts) == 1
    assert len(pts[0]) == 64  # 8x8 grid
    # Capped to max_masks=2
    assert len(result.masks) == 2
    assert result.masks[0].score >= result.masks[1].score


def test_segment_text_mode_raises():
    masks = np.zeros((1, 1, 4, 4), dtype=bool)
    scores = np.array([[0.5]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(
        model=model, processor=proc, supports_text_prompts=False,
    )
    img = _FakePILImage((4, 4))
    with pytest.raises(RuntimeError, match="text-prompted"):
        rt.segment(img, mode="text", prompt="cat")


def test_segment_points_disabled_raises_via_capability():
    masks = np.zeros((1, 1, 4, 4), dtype=bool)
    scores = np.array([[0.5]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(
        model=model, processor=proc, supports_point_prompts=False,
    )
    img = _FakePILImage((4, 4))
    with pytest.raises(RuntimeError, match="point-prompted"):
        rt.segment(img, mode="points", points=[[0, 0]])


# ---------------- mask extraction + sorting ----------------


def test_segment_drops_empty_masks():
    arr_full = np.zeros((4, 4), dtype=bool)
    arr_full[1:3, 1:3] = True
    arr_empty = np.zeros((4, 4), dtype=bool)
    masks = np.stack([arr_full, arr_empty])[None]  # [1, 2, 4, 4]
    scores = np.array([[0.9, 0.4]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc, max_masks=4)
    img = _FakePILImage((4, 4))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="points", points=[[1, 1]])

    assert len(result.masks) == 1  # empty mask dropped
    assert result.masks[0].area > 0


def test_segment_sorts_by_score_descending():
    a = np.zeros((4, 4), dtype=bool); a[1, 1] = True
    b = np.zeros((4, 4), dtype=bool); b[2, 2] = True
    c = np.zeros((4, 4), dtype=bool); c[3, 3] = True
    masks = np.stack([a, b, c])[None]  # [1, 3, 4, 4]
    scores = np.array([[0.3, 0.9, 0.6]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc, max_masks=10)
    img = _FakePILImage((4, 4))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="points", points=[[1, 1]])

    scores_out = [m.score for m in result.masks]
    assert scores_out == sorted(scores_out, reverse=True)


def test_segment_truncates_to_max_masks():
    """max_masks at request time overrides the runtime default."""
    arrs = []
    for k in range(5):
        a = np.zeros((4, 4), dtype=bool)
        a[k % 4, k % 4] = True
        arrs.append(a)
    masks = np.stack(arrs)[None]
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc, max_masks=10)
    img = _FakePILImage((4, 4))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="points", points=[[0, 0]], max_masks=2)

    assert len(result.masks) == 2


def test_segment_records_bbox_and_area():
    arr = np.zeros((8, 8), dtype=bool)
    arr[2:5, 3:7] = True  # 3 rows x 4 cols = 12 pixels
    masks = np.array([[arr]], dtype=bool)
    scores = np.array([[0.7]], dtype=np.float32)
    model = _make_model(masks, scores)
    proc = _make_processor(masks[0], scores[0])
    rt, _, _ = _build_runtime(model=model, processor=proc)
    img = _FakePILImage((8, 8))

    fake_torch = _patched_torch()
    with patch.object(sam_mod, "torch", fake_torch):
        result = rt.segment(img, mode="points", points=[[5, 3]])

    assert result.masks[0].bbox == (3, 2, 4, 3)
    assert result.masks[0].area == 12


# ---------------- _segment_with negative paths ----------------


def test_segment_with_unknown_mode_raises():
    proc = MagicMock()
    proc.return_value = {}
    model = MagicMock()
    img = _FakePILImage((4, 4))
    with pytest.raises(ValueError, match="unsupported mode"):
        _segment_with(
            model, proc, img,
            mode="weird", points=None, boxes=None,
            device="cpu", max_masks=8,
        )


def test_segment_with_points_required_for_points_mode():
    proc = MagicMock()
    model = MagicMock()
    img = _FakePILImage((4, 4))
    with pytest.raises(ValueError, match="points"):
        _segment_with(
            model, proc, img,
            mode="points", points=None, boxes=None,
            device="cpu", max_masks=8,
        )


def test_segment_with_boxes_required_for_boxes_mode():
    proc = MagicMock()
    model = MagicMock()
    img = _FakePILImage((4, 4))
    with pytest.raises(ValueError, match="boxes"):
        _segment_with(
            model, proc, img,
            mode="boxes", points=None, boxes=None,
            device="cpu", max_masks=8,
        )
