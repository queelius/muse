"""HFKeypointRuntime: mocked-dep tests."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    orig = (
        mod.torch, mod.AutoModelForKeypointDetection, mod.AutoImageProcessor,
    )
    yield
    (
        mod.torch, mod.AutoModelForKeypointDetection, mod.AutoImageProcessor,
    ) = orig


def _movable():
    t = MagicMock()
    t.to = MagicMock(return_value=t)
    return t


def _wire_runtime(mod, *, processed=None, has_post_process=True, id2label=None):
    """Install fakes for AutoModel + AutoImageProcessor.

    `processed` is what processor.post_process_keypoint_detection returns
    (a list-of-lists structure shaped like the real HF API). When None,
    the runtime falls back to raw outputs.

    `has_post_process` toggles whether the processor exposes
    post_process_keypoint_detection.
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    fake_torch.float32 = "fp32-sentinel"

    forward_outputs = MagicMock()
    forward_outputs.keypoints = MagicMock()  # only used by fallback path

    model_obj = MagicMock()
    model_obj.return_value = forward_outputs
    model_obj.to = MagicMock(return_value=model_obj)
    if id2label is not None:
        model_obj.config = MagicMock(id2label=id2label)
    else:
        model_obj.config = MagicMock(id2label={})

    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForKeypointDetection = model_factory

    encoded = {"pixel_values": _movable()}
    processor = MagicMock()
    processor.return_value = encoded

    if has_post_process:
        if processed is None:
            processed = [[]]  # one image, no detections
        processor.post_process_keypoint_detection = MagicMock(
            return_value=processed,
        )
    else:
        # Drop the attribute so hasattr returns False.
        if hasattr(processor, "post_process_keypoint_detection"):
            del processor.post_process_keypoint_detection

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=processor)
    mod.AutoImageProcessor = proc_factory

    mod.torch = fake_torch
    return processor, model_obj


def _kp_dict(*, kps, scores, labels=None):
    """Build a processed-result dict whose tensors mock to-list."""
    kps_t = MagicMock()
    kps_t.detach.return_value.cpu.return_value.tolist.return_value = kps
    scores_t = MagicMock()
    scores_t.detach.return_value.cpu.return_value.tolist.return_value = scores
    out = {"keypoints": kps_t, "scores": scores_t}
    if labels is not None:
        labels_t = MagicMock()
        labels_t.detach.return_value.cpu.return_value.tolist.return_value = labels
        out["labels"] = labels_t
    return out


def test_detect_keypoints_returns_keypoint_result():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[
        _kp_dict(
            kps=[[100.0, 50.0], [120.0, 60.0]],
            scores=[0.99, 0.95],
        )
    ]]
    _wire_runtime(
        mod, processed=processed,
        id2label={0: "nose", 1: "left_eye"},
    )
    runtime = mod.HFKeypointRuntime(
        model_id="vp", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (640, 480)
    result = runtime.detect_keypoints(image)

    assert result.model_id == "vp"
    assert result.image_size == (640, 480)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.bbox == (0.0, 0.0, 640.0, 480.0)
    assert len(det.keypoints) == 2
    assert det.keypoints[0].name == "nose"
    assert det.keypoints[0].x == 100.0
    assert det.keypoints[0].score == 0.99


def test_detect_keypoints_threshold_filter():
    """Keypoints below the threshold get dropped."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[
        _kp_dict(
            kps=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            scores=[0.9, 0.2, 0.8],
        )
    ]]
    _wire_runtime(mod, processed=processed)
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image, threshold=0.5)
    # Only the 0.9 and 0.8 keypoints survive.
    assert len(result.detections) == 1
    assert len(result.detections[0].keypoints) == 2


def test_detect_keypoints_falls_back_to_index_when_no_id2label():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[
        _kp_dict(kps=[[5.0, 5.0]], scores=[0.99])
    ]]
    _wire_runtime(mod, processed=processed, id2label={})
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image)
    # Index 0 has no label, so name is "0".
    assert result.detections[0].keypoints[0].name == "0"


def test_detect_keypoints_typeerror_fallback_no_boxes():
    """Some processor.post_process_keypoint_detection signatures don't
    take `boxes=` and `threshold=`. The runtime catches TypeError and
    retries with the simpler signature."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[_kp_dict(kps=[[1.0, 1.0]], scores=[0.99])]]
    call_count = {"n": 0}

    def _post(outputs, **kwargs):
        call_count["n"] += 1
        if "boxes" in kwargs or "threshold" in kwargs:
            raise TypeError("unexpected keyword")
        return processed

    _wire_runtime(mod, processed=[[]])
    # Override post_process_keypoint_detection with our function.
    proc_factory = mod.AutoImageProcessor
    proc_obj = proc_factory.from_pretrained.return_value
    proc_obj.post_process_keypoint_detection = _post

    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image)
    assert call_count["n"] == 2  # first with boxes/threshold, then without
    assert len(result.detections) == 1


def test_detect_keypoints_empty_when_no_detections():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod, processed=[[]])
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image)
    assert result.detections == []


def test_raises_when_transformers_too_old(monkeypatch):
    """AutoModelForKeypointDetection arrived in transformers 4.46;
    older versions surface a clear RuntimeError."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForKeypointDetection = None
    mod.AutoImageProcessor = MagicMock()
    with pytest.raises(RuntimeError, match=">= 4.46"):
        mod.HFKeypointRuntime(model_id="m", hf_repo="x", device="cpu")


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.AutoModelForKeypointDetection = MagicMock()
    mod.AutoImageProcessor = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFKeypointRuntime(model_id="m", hf_repo="x", device="cpu")
