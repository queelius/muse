"""HFObjectDetectionRuntime: object detection over AutoModelForObjectDetection.

Wraps `transformers.AutoModelForObjectDetection` + `AutoImageProcessor`.
Works for DETR, YOLOS, RT-DETR, OWL-ViT (open-vocab), and other
HuggingFace object-detection checkpoints.

Inference flow:

  1. processor(images=image, return_tensors='pt') -> pixel_values
  2. model(**inputs) -> outputs (with logits + pred_boxes)
  3. processor.post_process_object_detection(outputs,
        target_sizes=[(H, W)], threshold=...) -> [{scores, labels, boxes}]
  4. Boxes come back in xyxy (x_min, y_min, x_max, y_max); we convert
     to COCO xywh for the wire shape (matches image_segmentation).
  5. Sort detections by score desc, cap at max_detections.

Deferred imports follow the muse pattern.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.image_cv.protocol import (
    ObjectDetection,
    ObjectDetectionResult,
)


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForObjectDetection: Any = None
AutoImageProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForObjectDetection, AutoImageProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFObjectDetectionRuntime torch unavailable: %s", e)
    if AutoModelForObjectDetection is None:
        try:
            from transformers import AutoModelForObjectDetection as _m
            AutoModelForObjectDetection = _m
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFObjectDetectionRuntime AutoModelForObjectDetection unavailable: %s",
                e,
            )
    if AutoImageProcessor is None:
        try:
            from transformers import AutoImageProcessor as _p
            AutoImageProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFObjectDetectionRuntime AutoImageProcessor unavailable: %s", e,
            )


class HFObjectDetectionRuntime:
    """Generic object detection runtime."""

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
        if AutoModelForObjectDetection is None or AutoImageProcessor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        src = local_dir or hf_repo
        with LoadTimer(f"loading detection model from {src}", logger):
            self._processor = AutoImageProcessor.from_pretrained(src)
            self._model = AutoModelForObjectDetection.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def detect_objects(
        self,
        image: Any,
        *,
        threshold: float = 0.5,
        max_detections: int = 100,
    ) -> ObjectDetectionResult:
        """Detect objects in one PIL.Image."""
        W, H = image.size
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        outputs = self._model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold,
        )
        # results[0] = {"scores": Tensor[N], "labels": Tensor[N],
        #               "boxes": Tensor[N, 4]} where boxes are xyxy
        if not results:
            return ObjectDetectionResult(
                detections=[], model_id=self.model_id, image_size=(W, H),
            )
        per_image = results[0]
        scores = per_image["scores"].detach().cpu().tolist()
        labels = per_image["labels"].detach().cpu().tolist()
        boxes = per_image["boxes"].detach().cpu().tolist()

        id2label = self._id2label()

        # Build detections, then sort by score desc, then cap.
        detections: list[ObjectDetection] = []
        for s, lbl, b in zip(scores, labels, boxes):
            x_min, y_min, x_max, y_max = (float(v) for v in b)
            w = x_max - x_min
            h = y_max - y_min
            label_name = id2label.get(int(lbl), str(int(lbl)))
            detections.append(ObjectDetection(
                bbox=(x_min, y_min, w, h),
                score=float(s),
                label=label_name,
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        if max_detections > 0:
            detections = detections[:max_detections]

        return ObjectDetectionResult(
            detections=detections,
            model_id=self.model_id,
            image_size=(W, H),
        )

    def _id2label(self) -> dict[int, str]:
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
