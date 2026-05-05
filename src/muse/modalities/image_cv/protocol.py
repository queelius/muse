"""Protocol + dataclasses for image/cv (depth, keypoints, object detection).

Three primitives share this modality. Each has its own dataclass +
Protocol; the route layer dispatches based on capability flags. The
shared MIME tag (`image/cv`) groups them logically (CV pipelines tend
to combine all three) without forcing a single Protocol that doesn't
fit the divergent output shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------- Depth estimation ----------


@dataclass
class DepthResult:
    """Output of a depth estimation call.

    `depth` is a 2D float array shaped (H, W). Values are relative
    inverse depth in [0, 1] (the default for Depth-Anything family)
    or metric meters (ZoeDepth and other metric-depth models). The
    `metric_depth` flag tells the codec layer how to interpret the
    bounds.

    `image_size` is (W, H) in PIL convention; the depth array's shape
    is (H, W) in numpy convention. Yes, the conventions differ. Yes,
    that's annoying. The wire envelope reports `width` and `height`
    explicitly to avoid client confusion.
    """
    depth: Any  # numpy ndarray; not imported at module top to keep
                # discovery import cheap (numpy is a runtime dep).
    model_id: str
    image_size: tuple[int, int]  # (W, H)
    metric_depth: bool = False


@runtime_checkable
class DepthEstimator(Protocol):
    """Backend that estimates depth from one image."""

    def estimate_depth(self, image: Any) -> DepthResult:
        """Compute a depth map for one PIL.Image.

        Returns a DepthResult whose `depth` is a numpy float array of
        shape (H, W) where (W, H) matches `image.size`.
        """
        ...


# ---------- Keypoint detection ----------


@dataclass
class Keypoint:
    """One keypoint within a detected entity.

    `name` is the label from the model's id2label (e.g., 'nose',
    'left_eye'). When the model exposes no name table, names default
    to stringified indices ('0', '1', ...).
    """
    name: str
    x: float
    y: float
    score: float


@dataclass
class KeypointDetection:
    """One detected entity (person, animal, etc.) with its keypoints.

    `bbox` is [x, y, w, h] in COCO convention (matches image_segmentation,
    object_detection). `score` is the overall detection confidence;
    individual keypoint confidences live on each Keypoint.
    """
    bbox: tuple[float, float, float, float]
    score: float
    keypoints: list[Keypoint] = field(default_factory=list)


@dataclass
class KeypointResult:
    """Output of a keypoint detection call.

    `image_size` is (W, H) (PIL convention). `detections` may be
    empty when the model finds nothing. Detections are NOT pre-sorted
    (keypoint models typically have at most one detection per
    bbox-input anyway).
    """
    detections: list[KeypointDetection]
    model_id: str
    image_size: tuple[int, int]


@runtime_checkable
class KeypointDetector(Protocol):
    """Backend that detects keypoints in one image."""

    def detect_keypoints(
        self, image: Any, *, threshold: float = 0.3,
    ) -> KeypointResult:
        """Detect keypoints in one PIL.Image.

        `threshold` filters out per-keypoint scores below the cutoff.
        The Keypoint objects with score < threshold are dropped;
        an entity's `keypoints` list may shrink but the entity itself
        stays in `detections` as long as ANY keypoint passes.
        """
        ...


# ---------- Object detection ----------


@dataclass
class ObjectDetection:
    """One detected object with bbox and class label.

    `bbox` is [x, y, w, h] in COCO convention. `label` comes from
    the model's id2label (e.g., 'cat', 'dog'); never a numeric id.
    """
    bbox: tuple[float, float, float, float]
    score: float
    label: str


@dataclass
class ObjectDetectionResult:
    """Output of an object detection call.

    `detections` are sorted by score descending and capped at the
    request's `max_detections`.
    """
    detections: list[ObjectDetection]
    model_id: str
    image_size: tuple[int, int]


@runtime_checkable
class ObjectDetector(Protocol):
    """Backend that detects objects in one image."""

    def detect_objects(
        self,
        image: Any,
        *,
        threshold: float = 0.5,
        max_detections: int = 100,
    ) -> ObjectDetectionResult:
        """Detect objects in one PIL.Image.

        `threshold` filters out detections below the score cutoff.
        `max_detections` caps the returned list (after sorting by
        score desc).
        """
        ...
