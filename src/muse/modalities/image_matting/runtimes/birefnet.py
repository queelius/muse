"""``BiRefNetRuntime``: neural foreground matting over HuggingFace
``AutoModelForImageSegmentation`` models (BiRefNet, BiRefNet_lite, RMBG-2.0,
BEN2).

Deferred imports follow the muse pattern: ``torch`` and
``AutoModelForImageSegmentation`` stay as module-level ``None`` sentinels until
``_ensure_deps()`` lazy-imports them. Tests patch the sentinels directly;
``_ensure_deps`` short-circuits on non-None.

All device selection, dtype mapping, inference-mode switching, load timing,
and model source resolution delegate to ``muse.core.runtime_helpers``.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    resolve_model_source,
    select_device,
    set_inference_mode,
)
from muse.modalities.image_matting.protocol import MattingOptions, MattingResult

logger = logging.getLogger(__name__)

# Sentinels patched by tests; populated lazily by _ensure_deps at runtime.
torch: Any = None
AutoModelForImageSegmentation: Any = None

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_deps() -> None:
    global torch, AutoModelForImageSegmentation
    if torch is None:
        try:
            import torch as _t

            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("BiRefNetRuntime: torch unavailable: %s", e)
    if AutoModelForImageSegmentation is None:
        try:
            from transformers import AutoModelForImageSegmentation as _am

            AutoModelForImageSegmentation = _am
        except Exception as e:  # noqa: BLE001
            logger.debug("BiRefNetRuntime: transformers unavailable: %s", e)


def _extract_logits(outputs: Any) -> Any:
    """Extract the final mask logit tensor from BiRefNet / RMBG / HF outputs."""
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            raise RuntimeError("matting model returned an empty output sequence")
        return outputs[-1]
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    return outputs


def _to_prob_2d(logits: Any, torch_mod: Any) -> Any:
    """Convert raw logits tensor/array to a 2D float32 numpy array in [0, 1]."""
    import numpy as np

    if torch_mod is not None and hasattr(torch_mod, "sigmoid") and hasattr(logits, "detach"):
        probs = torch_mod.sigmoid(logits.detach())
        if hasattr(probs, "float"):
            probs = probs.float()
        if hasattr(probs, "cpu"):
            probs = probs.cpu()
        arr = probs.numpy() if hasattr(probs, "numpy") else np.asarray(probs)
    else:
        raw = logits
        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        raw_arr = np.asarray(raw, dtype=np.float32)
        arr = 1.0 / (1.0 + np.exp(-np.clip(raw_arr, -60.0, 60.0)))

    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"expected 2D matte prediction, got shape {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def _decontaminate_edges(rgba: Any, fringe_threshold: int) -> None:
    """Zero RGB on fully transparent pixels and clean bright semi-transparent
    1px edges using interior opaque neighbors."""
    import numpy as np

    alpha = rgba[..., 3]
    rgba[alpha == 0] = 0

    h, w = alpha.shape
    if h < 2 and w < 2:
        return
    semi = (alpha > 0) & (alpha < 255)
    if not semi.any():
        return

    rgb = rgba[..., :3].astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    halo = semi & (luma >= float(fringe_threshold))
    if not halo.any():
        return

    interior = alpha == 255
    if not interior.any():
        return

    sum_rgb = np.zeros((h, w, 3), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    int_f = interior.astype(np.float32)
    int_rgb = rgb * int_f[..., None]
    if h > 1:
        sum_rgb[1:, :] += int_rgb[:-1, :]
        cnt[1:, :] += int_f[:-1, :]
        sum_rgb[:-1, :] += int_rgb[1:, :]
        cnt[:-1, :] += int_f[1:, :]
    if w > 1:
        sum_rgb[:, 1:] += int_rgb[:, :-1]
        cnt[:, 1:] += int_f[:, :-1]
        sum_rgb[:, :-1] += int_rgb[:, 1:]
        cnt[:, :-1] += int_f[:, 1:]

    fix = halo & (cnt > 0)
    if fix.any():
        clean = np.rint(sum_rgb[fix] / cnt[fix, None]).clip(0, 255).astype(np.uint8)
        rgba[fix, :3] = clean


class BiRefNetRuntime:
    """Neural matting runtime backed by ``AutoModelForImageSegmentation``."""

    model_id: str

    def __init__(
        self,
        *,
        hf_repo: str = "ZhengPeng7/BiRefNet_lite",
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        model_id: str = "birefnet-lite",
        input_size: int = 1024,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForImageSegmentation is None:
            raise RuntimeError(
                "transformers is not installed; ensure the per-model "
                "venv has transformers>=4.40.0"
            )
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.local_dir = local_dir
        self._input_size = int(input_size)
        self._device = select_device(device, torch_module=torch)
        self._dtype_name = (
            ("float16" if self._device == "cuda" else "float32")
            if dtype == "auto"
            else dtype
        )
        self._torch_dtype = dtype_for_name(self._dtype_name, torch_module=torch)

        src = resolve_model_source(local_dir or hf_repo)
        logger.info(
            "loading matting model from %s (model_id=%s, device=%s, dtype=%s)",
            src,
            model_id,
            self._device,
            self._dtype_name,
        )

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._torch_dtype is not None:
            load_kwargs["torch_dtype"] = self._torch_dtype

        with LoadTimer(model_id, logger):
            self._model = AutoModelForImageSegmentation.from_pretrained(
                src, **load_kwargs
            )
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
            set_inference_mode(self._model)

    def matte(
        self,
        image: Any,
        opts: MattingOptions | None = None,
        **_: Any,
    ) -> MattingResult:
        """Predict an RGBA foreground matte for ``image``."""
        import numpy as np
        from PIL import Image

        opts = opts if opts is not None else MattingOptions()
        rgb_img = image.convert("RGB")
        orig_w, orig_h = rgb_img.size
        if orig_w < 1 or orig_h < 1:
            raise ValueError("image is empty")

        resized = rgb_img.resize(
            (self._input_size, self._input_size),
            Image.Resampling.BILINEAR,
        )
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(_IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        chw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])

        _torch = torch
        if _torch is not None and hasattr(_torch, "from_numpy"):
            tensor = _torch.from_numpy(chw)
            if hasattr(tensor, "to"):
                if self._torch_dtype is not None:
                    try:
                        tensor = tensor.to(device=self._device, dtype=self._torch_dtype)
                    except TypeError:
                        tensor = tensor.to(self._device)
                else:
                    tensor = tensor.to(self._device)
        else:
            tensor = chw

        if _torch is not None and hasattr(_torch, "inference_mode"):
            with _torch.inference_mode():
                outputs = self._model(tensor)
        elif _torch is not None and hasattr(_torch, "no_grad"):
            with _torch.no_grad():
                outputs = self._model(tensor)
        else:
            outputs = self._model(tensor)

        prob_2d = _to_prob_2d(_extract_logits(outputs), _torch)
        mask_u8 = np.rint(prob_2d * 255.0).clip(0, 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_u8)
        if mask_img.size != (orig_w, orig_h):
            mask_img = mask_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        alpha = np.asarray(mask_img, dtype=np.uint8)

        rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
        rgba = np.empty((orig_h, orig_w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb_arr
        rgba[..., 3] = alpha

        if opts.defringe:
            _decontaminate_edges(rgba, opts.fringe_threshold)
        else:
            rgba[rgba[..., 3] == 0] = 0

        if opts.hard_alpha:
            keep = (rgba[..., 3] >= opts.alpha_threshold) & (rgba[..., 3] > 0)
            rgba[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
            rgba[~keep] = 0

        cropped = False
        if opts.crop:
            ys, xs = np.where(rgba[..., 3] > 0)
            if ys.size > 0:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                if (y1 - y0, x1 - x0) != (orig_h, orig_w):
                    rgba = rgba[y0:y1, x0:x1].copy()
                    cropped = True

        out_h, out_w = rgba.shape[:2]
        total = int(rgba[..., 3].size)
        transparent_fraction = float((rgba[..., 3] == 0).sum() / total) if total else 0.0
        opaque_fraction = float((rgba[..., 3] == 255).sum() / total) if total else 0.0

        return MattingResult(
            image=Image.fromarray(rgba),
            width=out_w,
            height=out_h,
            transparent_fraction=transparent_fraction,
            opaque_fraction=opaque_fraction,
            metadata={
                "model": self.model_id,
                "device": self._device,
                "input_size": self._input_size,
                "cropped": cropped,
            },
        )
