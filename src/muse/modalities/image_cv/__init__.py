"""image/cv modality.

Three primitives share this MIME tag:

  POST /v1/images/depth      depth estimation (depth-anything, ZoeDepth, DPT)
  POST /v1/images/keypoints  keypoint / pose detection (ViTPose, SuperPoint)
  POST /v1/images/detect     object detection (DETR, YOLOS, RT-DETR)

Each model declares one of three capability flags
(`supports_depth`, `supports_keypoints`, `supports_detection`); the
route layer gates per-model. A model claiming none of the flags is
not callable from any image/cv route (returns 400 wrong_primitive).

Wire shapes diverge by primitive (depth = pixel-map output, keypoints
= structured detections with per-point coordinates, detect = bbox
list) but converge on multipart image upload, the per-modality
inference lock from v0.34.0, and the shared image-decode pipeline
from v0.34.0.

Bundled (one per primitive):
  - depth-anything-v2-small (relative depth, 25M, Apache 2.0)
  - vitpose-base-simple (COCO 17-keypoint pose, ~85M, Apache 2.0)
  - detr-resnet-50 (COCO 80-class detection, ~41M, Apache 2.0)

Curated additions cover ZoeDepth (metric depth), DPT-large (classic),
RT-DETR (faster), and depth-anything-v2-base (better quality).
"""
from muse.modalities.image_cv.client import (
    DepthClient,
    KeypointClient,
    ObjectDetectionClient,
)
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
from muse.modalities.image_cv.routes import build_router


MODALITY = "image/cv"

DISCOVERY = {
    "summary": (
        "Image analysis primitives. Estimates a depth map, human pose "
        "keypoints or detects objects in an image. Which routes a model "
        "serves depends on the model."
    ),
    "use_for": [
        "a depth map for parallax or 3D effects",
        "locating people, poses and objects in a frame",
    ],
    "not_for": ["segmentation masks (use image/segmentation)", "reading text (use image/ocr)"],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/images/depth",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_estimate_depth",
            "file_fields": ["image"],
            "media_fields": ["depth_map"],
            "example": {"model": "<model>", "response_format": "png16"},
        },
        {
            "method": "POST",
            "path": "/v1/images/keypoints",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_detect_keypoints",
            "file_fields": ["image"],
            "media_fields": [],
            "example": {"model": "<model>"},
        },
        {
            "method": "POST",
            "path": "/v1/images/detect",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_detect_objects",
            "file_fields": ["image"],
            "media_fields": [],
            "example": {"model": "<model>", "threshold": 0.5},
        },
    ],
}


def _probe_call(model):
    """Probe-default body: small white image, dispatch by capability.

    The probe worker reads the model's manifest capabilities to decide
    which primitive to exercise. We pick the first declared one in
    priority order: depth, keypoints, detection. Falls back to depth
    estimation against a trivial image when capabilities aren't
    introspectable from the model object alone.
    """
    from PIL import Image
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    if hasattr(model, "estimate_depth"):
        return model.estimate_depth(img)
    if hasattr(model, "detect_keypoints"):
        return model.detect_keypoints(img)
    if hasattr(model, "detect_objects"):
        return model.detect_objects(img)
    raise RuntimeError(
        f"image/cv probe: model {type(model).__name__} has no estimate_depth, "
        f"detect_keypoints, or detect_objects method"
    )


PROBE_DEFAULTS = {
    "shape": "1 small (64x64) white image, capability-dispatched",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    # Protocols
    "DepthEstimator",
    "KeypointDetector",
    "ObjectDetector",
    # Result dataclasses
    "DepthResult",
    "Keypoint",
    "KeypointDetection",
    "KeypointResult",
    "ObjectDetection",
    "ObjectDetectionResult",
    # Clients
    "DepthClient",
    "KeypointClient",
    "ObjectDetectionClient",
]
