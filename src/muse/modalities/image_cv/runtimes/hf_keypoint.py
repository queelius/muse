"""HFKeypointRuntime: keypoint / pose detection over AutoModelForKeypointDetection.

Wraps `transformers.AutoModelForKeypointDetection` (added in 4.46) +
`AutoImageProcessor`. Works for ViTPose-family checkpoints.

ViTPose-style models expect bounding boxes per pose-extraction target.
v1 of this runtime takes a single full-image bbox per call (the whole
image is treated as the entity to extract pose from). Multi-person
pose extraction needs a person detector first; that pipeline is
deferred to a future v0.X+1 (it's a separate concern from this
runtime).

Inference flow:

  1. processor(image, boxes=[[(0, 0, W, H)]], return_tensors='pt') ->
     batched pixel inputs
  2. model(**inputs) -> outputs
  3. processor.post_process_keypoint_detection(outputs,
        target_sizes=[(H, W)], threshold=...) -> list of dicts per box
  4. Map the dicts into KeypointDetection objects with bbox + score +
     per-keypoint name/x/y/score.

Some processors don't expose post_process_keypoint_detection (older
transformers versions, or bespoke processor classes). The runtime
detects this via hasattr and falls back to manual decoding from the
raw outputs tensor if needed.

Deferred imports follow the muse pattern.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.image_cv.protocol import (
    Keypoint,
    KeypointDetection,
    KeypointResult,
)


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForKeypointDetection: Any = None
AutoImageProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForKeypointDetection, AutoImageProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFKeypointRuntime torch unavailable: %s", e)
    if AutoModelForKeypointDetection is None:
        try:
            from transformers import AutoModelForKeypointDetection as _m
            AutoModelForKeypointDetection = _m
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFKeypointRuntime AutoModelForKeypointDetection unavailable: %s",
                e,
            )
    if AutoImageProcessor is None:
        try:
            from transformers import AutoImageProcessor as _p
            AutoImageProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFKeypointRuntime AutoImageProcessor unavailable: %s", e,
            )


class HFKeypointRuntime:
    """Generic keypoint detection runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        if AutoModelForKeypointDetection is None or AutoImageProcessor is None:
            raise RuntimeError(
                "transformers is not installed (or too old for "
                "AutoModelForKeypointDetection; need >= 4.46); run "
                "`muse pull` or install `transformers>=4.46.0` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        src = local_dir or hf_repo
        with LoadTimer(f"loading keypoint model from {src}", logger):
            self._processor = AutoImageProcessor.from_pretrained(src)
            self._model = AutoModelForKeypointDetection.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def detect_keypoints(
        self, image: Any, *, threshold: float = 0.3,
    ) -> KeypointResult:
        """Detect keypoints in one PIL.Image.

        v1: passes a single full-image bbox per call. The model's
        post-processing returns one detection's keypoints (the whole
        image, treated as one entity). Multi-person extraction is a
        future enhancement that needs a person detector upstream.
        """
        W, H = image.size
        # ViTPose expects boxes as a list of lists: outer list per
        # image, inner list per detection. One image, one bbox.
        boxes = [[[0.0, 0.0, float(W), float(H)]]]
        inputs = self._processor(
            image, boxes=boxes, return_tensors="pt",
        )
        inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        outputs = self._model(**inputs)

        detections: list[KeypointDetection] = []
        post = getattr(
            self._processor, "post_process_keypoint_detection", None,
        )
        if callable(post):
            try:
                processed = post(
                    outputs, target_sizes=[(H, W)],
                    boxes=boxes, threshold=threshold,
                )
            except TypeError:
                # Some signatures don't take `boxes` and/or `threshold`.
                processed = post(outputs, target_sizes=[(H, W)])
            # processed is a list (per image) of lists (per box) of
            # dicts {keypoints, scores, labels}. Pull the first image's
            # first box's results.
            if processed and processed[0]:
                first_image = processed[0]
                # Some processors flatten to a single dict, others to a list.
                first = first_image[0] if isinstance(first_image, list) else first_image
                detection = self._build_detection_from_processed(
                    first, bbox=(0.0, 0.0, float(W), float(H)),
                    threshold=threshold,
                )
                if detection.keypoints:
                    detections.append(detection)
        else:
            # Fallback: walk the raw outputs. Format is processor-
            # specific, but a common shape is outputs.keypoints of
            # shape (B, N, K, 3) where the last dim is (x, y, score).
            kp_tensor = getattr(outputs, "keypoints", None)
            if kp_tensor is not None and kp_tensor.dim() >= 3:
                detection = self._build_detection_from_raw(
                    kp_tensor[0], bbox=(0.0, 0.0, float(W), float(H)),
                    threshold=threshold,
                )
                if detection.keypoints:
                    detections.append(detection)

        return KeypointResult(
            detections=detections,
            model_id=self.model_id,
            image_size=(W, H),
        )

    def _build_detection_from_processed(
        self,
        processed: dict,
        *,
        bbox: tuple[float, float, float, float],
        threshold: float,
    ) -> KeypointDetection:
        """Convert a post-processed dict to a KeypointDetection."""
        kps_tensor = processed.get("keypoints")
        scores_tensor = processed.get("scores")
        labels_tensor = processed.get("labels")

        if kps_tensor is None or scores_tensor is None:
            return KeypointDetection(bbox=bbox, score=1.0, keypoints=[])

        kps = kps_tensor.detach().cpu().tolist() if hasattr(kps_tensor, "detach") else list(kps_tensor)
        scores = scores_tensor.detach().cpu().tolist() if hasattr(scores_tensor, "detach") else list(scores_tensor)
        labels = (
            labels_tensor.detach().cpu().tolist() if hasattr(labels_tensor, "detach")
            else (list(labels_tensor) if labels_tensor is not None else None)
        )

        # Build an id2label map from the model config when available.
        id2label = self._id2label()

        keypoints: list[Keypoint] = []
        for i, ((x, y), score) in enumerate(zip(kps, scores)):
            if float(score) < threshold:
                continue
            label_idx = labels[i] if labels else i
            name = id2label.get(int(label_idx), str(label_idx))
            keypoints.append(Keypoint(
                name=name, x=float(x), y=float(y), score=float(score),
            ))

        # Use the max keypoint score as the detection-level score for
        # ViTPose-style models that don't emit a separate detection score.
        det_score = float(max(scores)) if scores else 0.0
        return KeypointDetection(bbox=bbox, score=det_score, keypoints=keypoints)

    def _build_detection_from_raw(
        self,
        raw_keypoints: Any,
        *,
        bbox: tuple[float, float, float, float],
        threshold: float,
    ) -> KeypointDetection:
        """Fallback: decode keypoints from the raw outputs tensor.

        Accepts shape (N, K, 3) where last dim is (x, y, score), or
        (K, 3) for a single detection.
        """
        t = raw_keypoints
        if t.dim() == 3:
            # (N, K, 3): take the first (and assumed only) detection.
            t = t[0]
        # t is now (K, 3).
        rows = t.detach().cpu().tolist()
        id2label = self._id2label()
        keypoints: list[Keypoint] = []
        scores: list[float] = []
        for i, row in enumerate(rows):
            x, y, score = row[:3]
            if float(score) < threshold:
                continue
            name = id2label.get(i, str(i))
            keypoints.append(Keypoint(
                name=name, x=float(x), y=float(y), score=float(score),
            ))
            scores.append(float(score))
        det_score = max(scores) if scores else 0.0
        return KeypointDetection(bbox=bbox, score=det_score, keypoints=keypoints)

    def _id2label(self) -> dict[int, str]:
        """Pull the integer-keyed id2label from the model config.

        HF configs sometimes ship id2label with string keys (JSON
        round-trip artifact). Coerce keys to int so the runtime's
        lookups work without surprises.
        """
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return {}
        raw = getattr(cfg, "id2label", None) or {}
        out: dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out


def _select_device(device: str) -> str:
    return select_device(device, torch_module=torch)


def _resolve_dtype(dtype: str):
    return dtype_for_name(dtype, torch_module=torch)
