"""HFObjectDetectionRuntime: mocked-dep tests."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    orig = (
        mod.torch, mod.AutoModelForObjectDetection, mod.AutoImageProcessor,
    )
    yield
    (
        mod.torch, mod.AutoModelForObjectDetection, mod.AutoImageProcessor,
    ) = orig


def _movable():
    t = MagicMock()
    t.to = MagicMock(return_value=t)
    return t


def _wire_runtime(mod, *, scores, labels, boxes_xyxy, id2label=None):
    """Install fakes. The processor's post_process_object_detection
    returns one dict per image with scores, labels, boxes (xyxy)."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    fake_torch.float32 = "fp32"
    fake_torch.tensor = MagicMock(return_value=MagicMock())

    model_obj = MagicMock()
    model_obj.return_value = MagicMock()  # outputs
    model_obj.to = MagicMock(return_value=model_obj)
    model_obj.config = MagicMock(id2label=id2label or {})

    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForObjectDetection = model_factory

    scores_t = MagicMock()
    scores_t.detach.return_value.cpu.return_value.tolist.return_value = scores
    labels_t = MagicMock()
    labels_t.detach.return_value.cpu.return_value.tolist.return_value = labels
    boxes_t = MagicMock()
    boxes_t.detach.return_value.cpu.return_value.tolist.return_value = boxes_xyxy

    processed = [{"scores": scores_t, "labels": labels_t, "boxes": boxes_t}]
    encoded = {"pixel_values": _movable()}
    processor = MagicMock()
    processor.return_value = encoded
    processor.post_process_object_detection = MagicMock(return_value=processed)

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=processor)
    mod.AutoImageProcessor = proc_factory

    mod.torch = fake_torch
    return processor, model_obj


def test_detect_objects_returns_detection_result():
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    _wire_runtime(
        mod,
        scores=[0.9, 0.7],
        labels=[17, 18],  # 17="cat", 18="dog" in COCO
        boxes_xyxy=[[10.0, 20.0, 110.0, 220.0], [50.0, 60.0, 130.0, 150.0]],
        id2label={17: "cat", 18: "dog"},
    )
    runtime = mod.HFObjectDetectionRuntime(
        model_id="detr-test", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (640, 480)
    result = runtime.detect_objects(image)

    assert result.model_id == "detr-test"
    assert result.image_size == (640, 480)
    assert len(result.detections) == 2
    # Sorted by score desc; cat (0.9) first.
    assert result.detections[0].label == "cat"
    assert result.detections[0].score == 0.9
    # bbox converted from xyxy [10, 20, 110, 220] to xywh [10, 20, 100, 200].
    assert result.detections[0].bbox == (10.0, 20.0, 100.0, 200.0)


def test_detect_objects_sort_by_score_desc():
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    _wire_runtime(
        mod,
        scores=[0.5, 0.9, 0.7],
        labels=[1, 2, 3],
        boxes_xyxy=[[0.0]*4, [0.0]*4, [0.0]*4],
        id2label={1: "a", 2: "b", 3: "c"},
    )
    runtime = mod.HFObjectDetectionRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    result = runtime.detect_objects(MagicMock(size=(10, 10)))
    labels = [d.label for d in result.detections]
    assert labels == ["b", "c", "a"]


def test_detect_objects_max_detections_caps():
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    _wire_runtime(
        mod,
        scores=[0.9, 0.85, 0.8, 0.75, 0.7],
        labels=[1, 2, 3, 4, 5],
        boxes_xyxy=[[0.0]*4]*5,
        id2label={i: f"l{i}" for i in range(1, 6)},
    )
    runtime = mod.HFObjectDetectionRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    result = runtime.detect_objects(
        MagicMock(size=(10, 10)), max_detections=3,
    )
    assert len(result.detections) == 3


def test_detect_objects_label_index_fallback():
    """When id2label has no entry, fall back to stringified index."""
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    _wire_runtime(
        mod,
        scores=[0.9],
        labels=[42],
        boxes_xyxy=[[0.0, 0.0, 10.0, 10.0]],
        id2label={},
    )
    runtime = mod.HFObjectDetectionRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    result = runtime.detect_objects(MagicMock(size=(10, 10)))
    assert result.detections[0].label == "42"


def test_detect_objects_threshold_passed_to_processor():
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    processor, _ = _wire_runtime(
        mod,
        scores=[],
        labels=[],
        boxes_xyxy=[],
    )
    runtime = mod.HFObjectDetectionRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    runtime.detect_objects(MagicMock(size=(10, 10)), threshold=0.7)
    call_kwargs = processor.post_process_object_detection.call_args.kwargs
    assert call_kwargs["threshold"] == 0.7


def test_detect_objects_empty_when_no_detections():
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    _wire_runtime(mod, scores=[], labels=[], boxes_xyxy=[])
    runtime = mod.HFObjectDetectionRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    result = runtime.detect_objects(MagicMock(size=(10, 10)))
    assert result.detections == []


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.AutoModelForObjectDetection = MagicMock()
    mod.AutoImageProcessor = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFObjectDetectionRuntime(model_id="m", hf_repo="x", device="cpu")


def test_raises_when_transformers_not_installed(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_object_detection as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoModelForObjectDetection = None
    mod.AutoImageProcessor = None
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFObjectDetectionRuntime(model_id="m", hf_repo="x", device="cpu")
