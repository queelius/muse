"""HFDepthRuntime: generic depth estimation over AutoModelForDepthEstimation.

Wraps `transformers.AutoModelForDepthEstimation` + `AutoImageProcessor`.
Works for the Depth-Anything family, DPT, ZoeDepth, and any other
HuggingFace depth-estimation checkpoint.

Inference flow:

  1. processor(image, return_tensors='pt') -> pixel_values
  2. model(**inputs) -> outputs with `predicted_depth` tensor
  3. F.interpolate to the original image size (depth heads typically
     output at lower resolution than input)
  4. cpu().numpy() -> (H, W) float array
  5. For relative depth (Depth-Anything default), the route layer
     reports the value range in `min_depth` / `max_depth` and the
     codec normalizes for PNG16. For metric depth (ZoeDepth), values
     are meters and pass through unchanged.

The `metric_depth: bool` constructor arg is informational; it doesn't
change the inference path, just the resulting DepthResult's flag.

Deferred imports follow the muse pattern: torch, AutoModelForDepthEstimation,
AutoImageProcessor as module-top sentinels. Tests patch them directly.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.image_cv.protocol import DepthResult


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForDepthEstimation: Any = None
AutoImageProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForDepthEstimation, AutoImageProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFDepthRuntime torch unavailable: %s", e)
    if AutoModelForDepthEstimation is None:
        try:
            from transformers import AutoModelForDepthEstimation as _m
            AutoModelForDepthEstimation = _m
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFDepthRuntime AutoModelForDepthEstimation unavailable: %s", e,
            )
    if AutoImageProcessor is None:
        try:
            from transformers import AutoImageProcessor as _p
            AutoImageProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFDepthRuntime AutoImageProcessor unavailable: %s", e,
            )


class HFDepthRuntime:
    """Generic depth estimation runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        metric_depth: bool = False,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        if AutoModelForDepthEstimation is None or AutoImageProcessor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        self.model_id = model_id
        self.metric_depth = bool(metric_depth)
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        src = local_dir or hf_repo
        with LoadTimer(f"loading depth model from {src}", logger):
            self._processor = AutoImageProcessor.from_pretrained(src)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def estimate_depth(self, image: Any) -> DepthResult:
        """Compute a depth map for one PIL.Image."""
        # PIL image_size is (W, H); the model output is (H, W) and we
        # need to interpolate it back to the source image's spatial size.
        W, H = image.size
        inputs = self._processor(image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)

        # outputs.predicted_depth is (B, H', W') for most depth heads.
        # Some checkpoints return (B, 1, H', W'); handle both.
        depth = outputs.predicted_depth
        if depth.dim() == 4:
            depth = depth.squeeze(1)  # (B, H', W')
        # F.interpolate expects (N, C, H, W); add a channel dim, resize,
        # squeeze it back out.
        resized = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).squeeze(0)  # (H, W)

        depth_np = resized.detach().cpu().to(torch.float32).numpy()
        return DepthResult(
            depth=depth_np,
            model_id=self.model_id,
            image_size=(W, H),
            metric_depth=self.metric_depth,
        )


# Thin delegators preserved for test imports (matches sibling runtimes;
# the meta-test in tests/core/test_runtime_helpers_meta.py flags
# re-implementations).
def _select_device(device: str) -> str:
    return select_device(device, torch_module=torch)


def _resolve_dtype(dtype: str):
    return dtype_for_name(dtype, torch_module=torch)
