"""Protocol + dataclass tests for image/cv."""
import numpy as np
import pytest

from muse.modalities.image_cv.protocol import (
    DepthEstimator,
    DepthResult,
    Keypoint,
    KeypointDetection,
    KeypointDetector,
    KeypointResult,
    ObjectDetection,
    ObjectDetectionResult,
    ObjectDetector,
)


def test_depth_result_construction():
    arr = np.zeros((32, 64), dtype="float32")
    r = DepthResult(depth=arr, model_id="m", image_size=(64, 32))
    assert r.depth.shape == (32, 64)
    assert r.image_size == (64, 32)
    assert r.metric_depth is False  # default


def test_depth_result_metric_flag():
    arr = np.zeros((4, 4))
    r = DepthResult(depth=arr, model_id="m", image_size=(4, 4), metric_depth=True)
    assert r.metric_depth is True


def test_keypoint_construction():
    kp = Keypoint(name="nose", x=320.0, y=240.0, score=0.99)
    assert kp.name == "nose"
    assert kp.x == 320.0
    assert kp.score == 0.99


def test_keypoint_detection_default_keypoints():
    """field(default_factory=list) avoids the mutable-default trap."""
    a = KeypointDetection(bbox=(0, 0, 100, 100), score=0.9)
    a.keypoints.append(Keypoint("nose", 50, 50, 0.9))
    b = KeypointDetection(bbox=(0, 0, 100, 100), score=0.9)
    assert b.keypoints == []


def test_keypoint_result_construction():
    det = KeypointDetection(
        bbox=(0, 0, 100, 100), score=0.95,
        keypoints=[Keypoint("nose", 50, 50, 0.99)],
    )
    r = KeypointResult(
        detections=[det], model_id="vp", image_size=(640, 480),
    )
    assert r.image_size == (640, 480)
    assert len(r.detections) == 1


def test_object_detection_construction():
    det = ObjectDetection(bbox=(10, 20, 100, 200), score=0.9, label="cat")
    assert det.label == "cat"
    assert det.bbox == (10, 20, 100, 200)


def test_object_detection_result_construction():
    r = ObjectDetectionResult(
        detections=[
            ObjectDetection(bbox=(0, 0, 10, 10), score=0.9, label="cat"),
        ],
        model_id="detr",
        image_size=(640, 480),
    )
    assert len(r.detections) == 1
    assert r.detections[0].label == "cat"


# ---------- Protocols (runtime_checkable) ----------


def test_depth_estimator_protocol_accepts_duck_type():
    class _Fake:
        def estimate_depth(self, image):
            return DepthResult(
                depth=np.zeros((1, 1)), model_id="fake", image_size=(1, 1),
            )
    assert isinstance(_Fake(), DepthEstimator)


def test_depth_estimator_protocol_rejects_missing_method():
    class _NoEstimate:
        pass
    assert not isinstance(_NoEstimate(), DepthEstimator)


def test_keypoint_detector_protocol_accepts_duck_type():
    class _Fake:
        def detect_keypoints(self, image, *, threshold=0.3):
            return KeypointResult(
                detections=[], model_id="fake", image_size=(1, 1),
            )
    assert isinstance(_Fake(), KeypointDetector)


def test_object_detector_protocol_accepts_duck_type():
    class _Fake:
        def detect_objects(self, image, *, threshold=0.5, max_detections=100):
            return ObjectDetectionResult(
                detections=[], model_id="fake", image_size=(1, 1),
            )
    assert isinstance(_Fake(), ObjectDetector)


def test_protocols_are_disjoint():
    """A model with only one of the three methods doesn't satisfy the
    other two protocols."""
    class _DepthOnly:
        def estimate_depth(self, image):
            return DepthResult(
                depth=np.zeros((1, 1)), model_id="x", image_size=(1, 1),
            )
    m = _DepthOnly()
    assert isinstance(m, DepthEstimator)
    assert not isinstance(m, KeypointDetector)
    assert not isinstance(m, ObjectDetector)
