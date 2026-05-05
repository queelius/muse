"""HFDepthRuntime: mocked-dep tests."""
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    orig = (mod.torch, mod.AutoModelForDepthEstimation, mod.AutoImageProcessor)
    yield
    mod.torch, mod.AutoModelForDepthEstimation, mod.AutoImageProcessor = orig


def _wire_runtime(mod, *, depth_array=None, depth_dim=3):
    """Install fake torch + AutoModel + AutoImageProcessor.

    Returns (processor, model_obj). The model's forward returns an
    object whose `predicted_depth` is a tensor-like with shape
    `depth_dim` (3 for (B, H, W); 4 for (B, 1, H, W)) backed by the
    given depth_array.
    """
    if depth_array is None:
        depth_array = np.zeros((1, 16, 16), dtype="float32")

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    fake_torch.float32 = np.dtype("float32")
    fake_torch.float16 = np.dtype("float16")

    # Predicted-depth tensor: needs .dim(), .squeeze(), .unsqueeze(),
    # .detach().cpu().to(...).numpy(), ability to be passed to
    # F.interpolate.
    pred = _make_depth_tensor(depth_array, dim=depth_dim)

    # outputs.predicted_depth = pred
    forward_outputs = MagicMock()
    forward_outputs.predicted_depth = pred

    model_obj = MagicMock()
    model_obj.return_value = forward_outputs
    model_obj.to = MagicMock(return_value=model_obj)
    model_obj.config = MagicMock(id2label={})

    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForDepthEstimation = model_factory

    # Processor returns a dict-like with tensors that have .to(device).
    encoded = {"pixel_values": _movable_tensor()}
    processor = MagicMock()
    processor.return_value = encoded

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=processor)
    mod.AutoImageProcessor = proc_factory

    # F.interpolate: bypass the real torch op by returning a tensor
    # whose .squeeze().squeeze().detach().cpu().to().numpy() chain
    # yields the depth_array reshaped to (H, W).
    interp_tensor = _make_resized_depth_tensor(depth_array)
    fake_torch.nn.functional.interpolate = MagicMock(return_value=interp_tensor)

    mod.torch = fake_torch
    return processor, model_obj


def _movable_tensor():
    """Build a tensor stand-in with a no-op .to(device)."""
    t = MagicMock()
    t.to = MagicMock(return_value=t)
    return t


def _make_depth_tensor(arr, *, dim):
    """Build a depth tensor mock that reports `dim` and squeeze chain.

    For dim=3: (B, H, W). squeeze(1) is a no-op.
    For dim=4: (B, 1, H, W). squeeze(1) drops the channel dim.

    The squeeze pattern in HFDepthRuntime is `if dim==4: squeeze(1)`
    then later `unsqueeze(1)` for F.interpolate. Use real numpy
    arrays via a thin wrapper so the math actually flows.
    """
    np_arr = np.asarray(arr, dtype="float32")
    t = MagicMock()
    t.dim = MagicMock(return_value=dim)
    if dim == 4:
        # squeeze(1) returns a (B, H, W) tensor.
        squeezed_3d = _make_depth_tensor(np_arr, dim=3)
        t.squeeze = MagicMock(return_value=squeezed_3d)
    elif dim == 3:
        t.squeeze = MagicMock(return_value=t)
    # unsqueeze(1) returns a (B, 1, H, W) tensor used for interpolate
    # input. We don't need to follow the chain because the interpolate
    # mock returns a fixed result.
    t.unsqueeze = MagicMock(return_value=t)
    t._np = np_arr
    return t


def _make_resized_depth_tensor(orig_arr):
    """Build the tensor returned by F.interpolate.

    The runtime calls .squeeze(1).squeeze(0) to get a (H, W) tensor,
    then .detach().cpu().to(torch.float32).numpy() to get a numpy
    array. We need to make .numpy() return a 2D float array of
    shape (H, W).
    """
    np_arr = np.asarray(orig_arr, dtype="float32")
    # Strip the leading batch dim if present.
    if np_arr.ndim == 3:
        np_arr = np_arr[0]
    elif np_arr.ndim == 4:
        np_arr = np_arr[0, 0]

    # Build the chain: .squeeze(1).squeeze(0).detach().cpu().to(dtype).numpy()
    final = MagicMock()
    final.numpy = MagicMock(return_value=np_arr)

    to_result = MagicMock()
    to_result.numpy = MagicMock(return_value=np_arr)

    cpu_result = MagicMock()
    cpu_result.to = MagicMock(return_value=to_result)

    detach_result = MagicMock()
    detach_result.cpu = MagicMock(return_value=cpu_result)

    squeeze1_result = MagicMock()
    squeeze1_result.detach = MagicMock(return_value=detach_result)

    squeeze0_result = MagicMock()
    squeeze0_result.squeeze = MagicMock(return_value=squeeze1_result)

    interp_tensor = MagicMock()
    interp_tensor.squeeze = MagicMock(return_value=squeeze0_result)
    return interp_tensor


def test_estimate_depth_returns_depth_result():
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    depth_arr = np.array([[[0.2, 0.4], [0.6, 0.8]]], dtype="float32")  # (1, 2, 2)
    _wire_runtime(mod, depth_array=depth_arr, depth_dim=3)

    runtime = mod.HFDepthRuntime(
        model_id="depth-test",
        hf_repo="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu",
    )
    image = MagicMock()
    image.size = (64, 32)  # PIL convention (W, H)
    result = runtime.estimate_depth(image)

    assert result.model_id == "depth-test"
    assert result.image_size == (64, 32)
    assert result.metric_depth is False  # default
    assert result.depth.shape == (2, 2)


def test_estimate_depth_handles_4d_predicted_depth():
    """Some heads return (B, 1, H, W); runtime squeezes the channel dim."""
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    arr = np.zeros((1, 1, 4, 4), dtype="float32")
    _wire_runtime(mod, depth_array=arr, depth_dim=4)
    runtime = mod.HFDepthRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (8, 8)
    result = runtime.estimate_depth(image)
    assert result.depth.shape == (4, 4)


def test_metric_depth_flag_propagates():
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    _wire_runtime(mod)
    runtime = mod.HFDepthRuntime(
        model_id="zoedepth", hf_repo="x", device="cpu", metric_depth=True,
    )
    image = MagicMock()
    image.size = (16, 16)
    result = runtime.estimate_depth(image)
    assert result.metric_depth is True


def test_local_dir_preferred_over_hf_repo():
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    _wire_runtime(mod)
    mod.HFDepthRuntime(
        model_id="m",
        hf_repo="repo-id",
        local_dir="/tmp/snapshot",
        device="cpu",
    )
    proc_call = mod.AutoImageProcessor.from_pretrained.call_args.args[0]
    model_call = mod.AutoModelForDepthEstimation.from_pretrained.call_args.args[0]
    assert proc_call == "/tmp/snapshot"
    assert model_call == "/tmp/snapshot"


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.AutoModelForDepthEstimation = MagicMock()
    mod.AutoImageProcessor = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFDepthRuntime(model_id="m", hf_repo="x", device="cpu")


def test_raises_when_transformers_not_installed(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_depth as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForDepthEstimation = None
    mod.AutoImageProcessor = None
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFDepthRuntime(model_id="m", hf_repo="x", device="cpu")
