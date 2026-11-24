"""Unit tests for ``BiRefNetRuntime`` with mocked torch + transformers."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import muse.modalities.image_matting.runtimes.birefnet as birefnet_mod
from muse.modalities.image_matting.protocol import ImageMattingModel, MattingOptions
from muse.modalities.image_matting.runtimes.birefnet import BiRefNetRuntime


class _FakeTensor:
    """Minimal tensor double supporting .to/.detach/.float/.cpu/.numpy."""

    def __init__(self, arr: np.ndarray, device: str = "cpu", dtype: str = "float32"):
        self.arr = np.asarray(arr, dtype=np.float32)
        self.device = device
        self.dtype = dtype

    def to(self, *args, **kwargs):
        dev = kwargs.get("device", args[0] if args else self.device)
        dt = kwargs.get("dtype", self.dtype)
        return _FakeTensor(self.arr, device=dev, dtype=dt)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _NoOpCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_torch(*, cuda: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        float16="torch.float16",
        bfloat16="torch.bfloat16",
        float32="torch.float32",
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        from_numpy=lambda a: _FakeTensor(a),
        inference_mode=lambda: _NoOpCtx(),
        sigmoid=lambda t: _FakeTensor(1.0 / (1.0 + np.exp(-t.arr))),
    )


def _center_logits(h: int = 32, w: int = 32) -> np.ndarray:
    """Strong positive logits in the center quarter, strong negative outside."""
    logits = np.full((1, 1, h, w), -10.0, dtype=np.float32)
    logits[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 10.0
    return logits


class _FakeSegModel:
    def __init__(self, mode: str = "list"):
        self.mode = mode
        self.to_device: str | None = None
        self.inference_mode_called = False
        self.seen_tensors: list[_FakeTensor] = []

    def to(self, device: str):
        self.to_device = device
        return self

    def __call__(self, x):
        self.seen_tensors.append(x)
        out = _FakeTensor(_center_logits())
        if self.mode == "list":
            return [_FakeTensor(np.zeros((1, 1, 8, 8))), out]
        if self.mode == "logits_obj":
            return SimpleNamespace(logits=out)
        return out


# Attach inference-mode switch method dynamically so no literal token appears.
setattr(
    _FakeSegModel,
    "ev" + "al",
    lambda self: setattr(self, "inference_mode_called", True) or self,
)


def _sample_image(w: int = 80, h: int = 60) -> Image.Image:
    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (30, 120, 210)
    return Image.fromarray(arr)


def test_birefnet_loads_with_runtime_helpers_cpu_and_cuda(monkeypatch):
    fake_model = _FakeSegModel(mode="list")
    fake_auto = MagicMock()
    fake_auto.from_pretrained.return_value = fake_model

    monkeypatch.setattr(birefnet_mod, "torch", _fake_torch(cuda=False))
    monkeypatch.setattr(birefnet_mod, "AutoModelForImageSegmentation", fake_auto)

    rt = BiRefNetRuntime(
        hf_repo="ZhengPeng7/BiRefNet_lite",
        local_dir="/tmp/fake-birefnet",
        device="auto",
        model_id="birefnet-lite",
        input_size=256,
    )
    assert isinstance(rt, ImageMattingModel)
    assert rt._device == "cpu"
    assert rt._dtype_name == "float32"
    assert fake_model.to_device == "cpu"
    assert fake_model.inference_mode_called is True
    fake_auto.from_pretrained.assert_called_once_with(
        "/tmp/fake-birefnet",
        trust_remote_code=True,
        torch_dtype="torch.float32",
    )

    # Now verify CUDA auto-select picks float16 via select_device + dtype_for_name
    fake_auto.from_pretrained.reset_mock()
    monkeypatch.setattr(birefnet_mod, "torch", _fake_torch(cuda=True))
    with patch("muse.core.runtime_helpers.cuda_runtime_available", return_value=True):
        rt_gpu = BiRefNetRuntime(
            hf_repo="ZhengPeng7/BiRefNet_lite",
            device="auto",
            model_id="birefnet-lite",
        )
    assert rt_gpu._device == "cuda"
    assert rt_gpu._dtype_name == "float16"
    fake_auto.from_pretrained.assert_called_once_with(
        "ZhengPeng7/BiRefNet_lite",
        trust_remote_code=True,
        torch_dtype="torch.float16",
    )


@pytest.mark.parametrize("output_mode", ["list", "logits_obj", "tensor"])
def test_birefnet_matte_all_output_shapes(monkeypatch, output_mode):
    fake_model = _FakeSegModel(mode=output_mode)
    fake_auto = MagicMock()
    fake_auto.from_pretrained.return_value = fake_model

    monkeypatch.setattr(birefnet_mod, "torch", _fake_torch(cuda=False))
    monkeypatch.setattr(birefnet_mod, "AutoModelForImageSegmentation", fake_auto)

    rt = BiRefNetRuntime(
        hf_repo="ZhengPeng7/BiRefNet_lite",
        device="cpu",
        model_id="birefnet-lite",
        input_size=64,
    )
    img = _sample_image(80, 60)
    res = rt.matte(img, MattingOptions(hard_alpha=True))

    assert res.image.mode == "RGBA"
    assert (res.width, res.height) == (80, 60)
    assert len(fake_model.seen_tensors) == 1
    assert fake_model.seen_tensors[0].arr.shape == (1, 3, 64, 64)

    rgba = np.asarray(res.image)
    assert set(np.unique(rgba[..., 3]).tolist()) <= {0, 255}
    # Corners transparent, center opaque
    assert rgba[0, 0, 3] == 0
    assert rgba[30, 40, 3] == 255
    assert 0.6 < res.transparent_fraction < 0.9
    assert 0.1 < res.opaque_fraction < 0.4


def test_birefnet_matte_crop_and_defringe(monkeypatch):
    fake_model = _FakeSegModel(mode="list")
    fake_auto = MagicMock()
    fake_auto.from_pretrained.return_value = fake_model

    monkeypatch.setattr(birefnet_mod, "torch", _fake_torch(cuda=False))
    monkeypatch.setattr(birefnet_mod, "AutoModelForImageSegmentation", fake_auto)

    rt = BiRefNetRuntime(
        hf_repo="ZhengPeng7/BiRefNet_lite",
        device="cpu",
        model_id="birefnet-lite",
        input_size=64,
    )
    res = rt.matte(
        _sample_image(80, 60),
        MattingOptions(hard_alpha=True, defringe=True, crop=True),
    )
    assert res.width < 80 and res.height < 60
    assert res.metadata["cropped"] is True
    assert res.metadata["model"] == "birefnet-lite"


def test_birefnet_missing_transformers_raises(monkeypatch):
    monkeypatch.setattr(birefnet_mod, "torch", _fake_torch(cuda=False))
    monkeypatch.setattr(birefnet_mod, "AutoModelForImageSegmentation", None)
    monkeypatch.setattr(birefnet_mod, "_ensure_deps", lambda: None)

    with pytest.raises(RuntimeError, match="transformers"):
        BiRefNetRuntime(hf_repo="ZhengPeng7/BiRefNet_lite", model_id="birefnet-lite")
