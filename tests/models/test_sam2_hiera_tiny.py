"""Tests for the bundled sam2_hiera_tiny script."""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.modalities.image_segmentation.protocol import SegmentationResult


def _get_module():
    """importlib.import_module on each call so module state is fresh."""
    return importlib.import_module("muse.models.sam2_hiera_tiny")


def test_manifest_required_fields():
    mod = _get_module()
    m = mod.MANIFEST
    assert m["model_id"] == "sam2-hiera-tiny"
    assert m["modality"] == "image/segmentation"
    assert m["hf_repo"] == "facebook/sam2-hiera-tiny"
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])
    assert any("Pillow" in x for x in m["pip_extras"])
    assert "numpy" in m["pip_extras"]


def test_manifest_capabilities_advertise_modes():
    mod = _get_module()
    caps = mod.MANIFEST["capabilities"]
    assert caps["supports_automatic"] is True
    assert caps["supports_point_prompts"] is True
    assert caps["supports_box_prompts"] is True
    assert caps["supports_text_prompts"] is False
    assert caps["max_masks"] == 64
    assert caps["device"] == "auto"
    assert "memory_gb" in caps


def test_manifest_allow_patterns_present():
    mod = _get_module()
    patterns = mod.MANIFEST["allow_patterns"]
    assert any("safetensors" in p for p in patterns)
    assert any("preprocessor_config" in p for p in patterns)


def test_model_class_exists():
    mod = _get_module()
    assert hasattr(mod, "Model")
    assert mod.Model.model_id == "sam2-hiera-tiny"


def test_model_construction_lazy_imports():
    """Model.__init__ should call _ensure_deps and read transformers
    classes via the module's sentinels so tests can patch them."""
    mod = _get_module()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_amg = MagicMock()
    fake_amg.from_pretrained.return_value = fake_model
    fake_proc = MagicMock()
    fake_ap = MagicMock()
    fake_ap.from_pretrained.return_value = fake_proc
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(mod, "AutoModelForMaskGeneration", fake_amg), \
            patch.object(mod, "AutoProcessor", fake_ap), \
            patch.object(mod, "torch", fake_torch):
        m = mod.Model(hf_repo="facebook/sam2-hiera-tiny", local_dir=None)
    assert m._device == "cpu"
    fake_amg.from_pretrained.assert_called_once()
    fake_ap.from_pretrained.assert_called_once()


def test_model_prefers_local_dir():
    mod = _get_module()
    fake_amg = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_amg.from_pretrained.return_value = fake_model
    fake_ap = MagicMock()
    fake_ap.from_pretrained.return_value = MagicMock()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(mod, "AutoModelForMaskGeneration", fake_amg), \
            patch.object(mod, "AutoProcessor", fake_ap), \
            patch.object(mod, "torch", fake_torch):
        mod.Model(hf_repo="facebook/sam2-hiera-tiny", local_dir="/tmp/sam")
    args, _ = fake_amg.from_pretrained.call_args
    assert args[0] == "/tmp/sam"


def test_model_raises_when_transformers_missing():
    """Stub _ensure_deps so it leaves transformers as None."""
    mod = _get_module()
    fake_torch = MagicMock()
    with patch.object(mod, "AutoModelForMaskGeneration", None), \
            patch.object(mod, "AutoProcessor", None), \
            patch.object(mod, "_ensure_deps", lambda: None), \
            patch.object(mod, "torch", fake_torch):
        with pytest.raises(RuntimeError, match="transformers"):
            mod.Model(hf_repo="facebook/sam2-hiera-tiny")


def test_segment_returns_segmentation_result():
    """A patched model + processor satisfy the dispatch contract."""
    mod = _get_module()
    sam_mod = importlib.import_module(
        "muse.modalities.image_segmentation.runtimes.sam2_runtime"
    )

    arr = np.zeros((8, 8), dtype=bool)
    arr[1:4, 2:5] = True
    masks_arr = np.array([[arr]], dtype=bool)  # [1, 1, 8, 8]
    scores_arr = np.array([[0.91]], dtype=np.float32)

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.return_value = SimpleNamespace(
        pred_masks=masks_arr, iou_scores=scores_arr,
    )
    fake_amg = MagicMock()
    fake_amg.from_pretrained.return_value = fake_model

    fake_proc = MagicMock()
    fake_proc.post_process_masks.return_value = [masks_arr[0]]

    def _proc_call(**kw):
        return {
            "pixel_values": MagicMock(),
            "original_sizes": [(8, 8)],
            "reshaped_input_sizes": [(8, 8)],
        }

    fake_proc.side_effect = _proc_call
    fake_ap = MagicMock()
    fake_ap.from_pretrained.return_value = fake_proc

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.inference_mode.return_value.__exit__ = MagicMock(return_value=None)

    class _FakeImage:
        size = (8, 8)

    with patch.object(mod, "AutoModelForMaskGeneration", fake_amg), \
            patch.object(mod, "AutoProcessor", fake_ap), \
            patch.object(mod, "torch", fake_torch), \
            patch.object(sam_mod, "torch", fake_torch):
        m = mod.Model(hf_repo="facebook/sam2-hiera-tiny", local_dir=None)
        result = m.segment(_FakeImage(), mode="points", points=[[3, 3]])

    assert isinstance(result, SegmentationResult)
    assert result.image_size == (8, 8)
    assert result.mode == "points"
    assert len(result.masks) == 1
    assert result.masks[0].score == pytest.approx(0.91)


def test_segment_text_mode_raises_capability_error():
    mod = _get_module()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_amg = MagicMock()
    fake_amg.from_pretrained.return_value = fake_model
    fake_proc = MagicMock()
    fake_ap = MagicMock()
    fake_ap.from_pretrained.return_value = fake_proc
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    class _FakeImage:
        size = (8, 8)

    with patch.object(mod, "AutoModelForMaskGeneration", fake_amg), \
            patch.object(mod, "AutoProcessor", fake_ap), \
            patch.object(mod, "torch", fake_torch):
        m = mod.Model(hf_repo="facebook/sam2-hiera-tiny", local_dir=None)
        with pytest.raises(RuntimeError, match="text-prompted"):
            m.segment(_FakeImage(), mode="text", prompt="cat")
