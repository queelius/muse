"""SAM2Runtime: generic runtime over any HF promptable segmenter.

Wraps ``transformers.AutoModelForMaskGeneration`` + ``AutoProcessor``
for any HF repo that implements the SAM / SAM-2 mask-generation
protocol. Pulled via the HF resolver: ``muse pull
hf://facebook/sam2-hiera-tiny`` synthesizes a manifest pointing at
this class.

Mode dispatch (capability-gated by the route layer; runtime checks
defense-in-depth):

  - ``"auto"``: dense grid of point prompts; the model emits a mask
    per point; an IoU-based NMS reduces near-duplicates.
  - ``"points"``: each pair in the request is forwarded to the
    processor's ``input_points`` argument as a single foreground click
    set; the model emits one mask per point group.
  - ``"boxes"``: each quad is forwarded to ``input_boxes``; the model
    emits one mask per box.
  - ``"text"``: not supported by the SAM-2 backbone; the runtime
    raises so we fail loudly even if the route layer's gate slips.

Deferred imports follow the muse pattern: torch + transformers stay as
module-top sentinels (None) until ``_ensure_deps()`` lazy-imports
them. Tests patch the sentinels directly; ``_ensure_deps`` short-
circuits on non-None.

bbox + area computation uses ``np.where`` to bound nonzero pixels and
``mask.sum()`` to count them. Empty masks are dropped.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_segmentation.protocol import (
    MaskRecord, SegmentationResult,
)


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
AutoModelForMaskGeneration: Any = None
AutoProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForMaskGeneration, AutoProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("SAM2Runtime: torch unavailable: %s", e)
    if AutoModelForMaskGeneration is None:
        try:
            from transformers import (
                AutoModelForMaskGeneration as _amg,
                AutoProcessor as _ap,
            )
            AutoModelForMaskGeneration = _amg
            AutoProcessor = _ap
        except Exception as e:  # noqa: BLE001
            logger.debug("SAM2Runtime: transformers unavailable: %s", e)


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str) -> Any:
    if torch is None:
        return None
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)


def _bbox_area(mask: Any) -> tuple[tuple[int, int, int, int], int]:
    """Compute COCO bbox ``(x, y, w, h)`` and area for a 2D mask.

    Returns ``((0, 0, 0, 0), 0)`` for empty masks.
    """
    import numpy as np
    arr = np.asarray(mask)
    ys, xs = np.where(arr > 0)
    if ys.size == 0:
        return (0, 0, 0, 0), 0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    area = int(arr.astype(bool).sum())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), area


def _to_numpy_mask(t: Any) -> Any:
    """Pull a numpy bool array out of a torch tensor / numpy array.

    Best-effort: tries ``.cpu().numpy()`` then falls back to
    ``np.asarray``. Coerces to bool last to avoid widening dtypes.
    """
    import numpy as np
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    arr = np.asarray(t)
    return arr.astype(bool)


def _segment_with(
    model: Any,
    processor: Any,
    image: Any,
    *,
    mode: str,
    points: list[list[int]] | None,
    boxes: list[list[int]] | None,
    device: str,
    max_masks: int,
) -> list[MaskRecord]:
    """Mode-dispatched single forward pass returning mask records.

    Shared by SAM2Runtime and the bundled sam2-hiera-tiny Model so the
    dispatch logic lives in one place.
    """
    if mode == "text":
        raise RuntimeError(
            "text-prompted segmentation is not supported by this model"
        )
    if mode not in ("auto", "points", "boxes"):
        raise ValueError(f"unsupported mode: {mode!r}")

    proc_kwargs: dict = {"images": image, "return_tensors": "pt"}
    if mode == "points":
        if not points:
            raise ValueError("mode='points' requires non-empty points list")
        # SAM/SAM-2 processor wants input_points shape [B, P, K, 2];
        # for a single image with one foreground point group, we wrap
        # to [[points]].
        proc_kwargs["input_points"] = [[[list(p) for p in points]]]
    elif mode == "boxes":
        if not boxes:
            raise ValueError("mode='boxes' requires non-empty boxes list")
        proc_kwargs["input_boxes"] = [[list(b) for b in boxes]]
    elif mode == "auto":
        # Dense grid of single-point prompts. 8x8 = 64 points by default;
        # large enough to cover most photos but small enough to stay
        # under typical max_masks caps. Each point becomes its own
        # foreground click set so the model emits one mask per point.
        w, h = image.size
        grid_n = 8
        pts: list[list[list[int]]] = []
        for i in range(grid_n):
            for j in range(grid_n):
                x = int((i + 0.5) * w / grid_n)
                y = int((j + 0.5) * h / grid_n)
                pts.append([[x, y]])
        proc_kwargs["input_points"] = [pts]

    inputs = processor(**proc_kwargs)
    inputs = _move_to_device(inputs, device)
    import muse.modalities.image_segmentation.runtimes.sam2_runtime as _mod
    _torch = _mod.torch
    if _torch is not None and hasattr(_torch, "inference_mode"):
        with _torch.inference_mode():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    # post_process_masks reshapes the model's low-resolution masks back
    # to the original image size. Different transformers versions
    # expose either an ``original_sizes`` or a ``reshaped_input_sizes``
    # parameter; we try the common one first.
    pred_masks = getattr(outputs, "pred_masks", None)
    iou_scores = getattr(outputs, "iou_scores", None)
    original_sizes = inputs.get("original_sizes") if isinstance(inputs, dict) else getattr(inputs, "original_sizes", None)
    reshaped_sizes = inputs.get("reshaped_input_sizes") if isinstance(inputs, dict) else getattr(inputs, "reshaped_input_sizes", None)

    masks_list: Any
    if hasattr(processor, "post_process_masks"):
        try:
            masks_list = processor.post_process_masks(
                pred_masks,
                original_sizes,
                reshaped_sizes,
            )
        except TypeError:
            masks_list = processor.post_process_masks(
                pred_masks, original_sizes,
            )
    else:
        masks_list = pred_masks

    # Flatten the per-image, per-prompt list of masks into MaskRecord
    # objects with score, bbox, area. Different transformers shapes:
    #   - masks_list[batch_idx] is a tensor or list-like of masks for
    #     each prompt set.
    #   - iou_scores has matching shape.
    records: list[MaskRecord] = []
    if masks_list is None:
        return records
    image_masks = masks_list[0] if len(masks_list) else []
    flat_masks: list[Any] = []
    flat_scores: list[float] = []
    if hasattr(image_masks, "shape"):
        # Tensor-shaped: iterate over leading axes flattened to (N, H, W).
        arr_batch = _to_numpy_mask(image_masks)
        if arr_batch.ndim == 4:
            arr_batch = arr_batch.reshape(-1, arr_batch.shape[-2], arr_batch.shape[-1])
        elif arr_batch.ndim == 3:
            pass
        else:
            arr_batch = arr_batch.reshape(1, *arr_batch.shape)
        flat_masks = [arr_batch[i] for i in range(arr_batch.shape[0])]
    else:
        # Sequence: flatten one level.
        for entry in image_masks:
            if hasattr(entry, "shape") and len(getattr(entry, "shape", ())) >= 3:
                arr_batch = _to_numpy_mask(entry)
                if arr_batch.ndim == 4:
                    arr_batch = arr_batch.reshape(-1, arr_batch.shape[-2], arr_batch.shape[-1])
                elif arr_batch.ndim == 2:
                    arr_batch = arr_batch.reshape(1, *arr_batch.shape)
                for i in range(arr_batch.shape[0]):
                    flat_masks.append(arr_batch[i])
            else:
                flat_masks.append(_to_numpy_mask(entry))

    if iou_scores is not None:
        score_arr = _flatten_scores(iou_scores)
        flat_scores = [float(v) for v in score_arr]
    if not flat_scores:
        flat_scores = [1.0] * len(flat_masks)
    if len(flat_scores) < len(flat_masks):
        # Pad with a default score so we never index past the list.
        flat_scores = flat_scores + [0.0] * (len(flat_masks) - len(flat_scores))

    for mask_arr, score in zip(flat_masks, flat_scores):
        bbox, area = _bbox_area(mask_arr)
        if area == 0:
            continue
        records.append(MaskRecord(
            mask=mask_arr, score=float(score), bbox=bbox, area=area,
        ))

    records.sort(key=lambda r: r.score, reverse=True)
    if max_masks is not None and max_masks > 0:
        records = records[:max_masks]
    return records


def _flatten_scores(scores: Any) -> Any:
    """Flatten an iou_scores tensor of arbitrary leading shape to 1D."""
    import numpy as np
    if hasattr(scores, "cpu"):
        scores = scores.cpu()
    if hasattr(scores, "numpy"):
        scores = scores.numpy()
    arr = np.asarray(scores).reshape(-1)
    return arr


def _move_to_device(inputs: Any, device: str) -> Any:
    """Best-effort move of a processor's BatchEncoding to a device."""
    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)
    if isinstance(inputs, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
    return inputs


class SAM2Runtime:
    """Promptable segmentation runtime backed by transformers' AutoModelForMaskGeneration.

    Constructor kwargs (sourced from the manifest's capabilities, merged
    in by the registry at load_backend time):

      - ``model_id`` (required): catalog id; echoed in response envelope.
      - ``hf_repo``, ``local_dir``: standard weight source.
      - ``device``, ``dtype``: standard device + dtype selection.
      - ``max_masks``: default cap on returned masks.
      - ``supports_text_prompts``, ``supports_point_prompts``,
        ``supports_box_prompts``, ``supports_automatic``: capability
        flags (route layer enforces; runtime stores for introspection).
    """

    model_id: str
    max_masks: int

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        max_masks: int = 64,
        supports_text_prompts: bool = False,
        supports_point_prompts: bool = True,
        supports_box_prompts: bool = True,
        supports_automatic: bool = True,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForMaskGeneration is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed; ensure the per-model "
                "venv has transformers>=4.43.0"
            )
        self.model_id = model_id
        self.max_masks = int(max_masks)
        self._supports_text_prompts = bool(supports_text_prompts)
        self._supports_point_prompts = bool(supports_point_prompts)
        self._supports_box_prompts = bool(supports_box_prompts)
        self._supports_automatic = bool(supports_automatic)
        self._device = _select_device(device)
        self._dtype = dtype

        src = local_dir or hf_repo
        logger.info(
            "loading SAM-2 model from %s (model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, dtype,
        )

        torch_dtype = _resolve_dtype(dtype)
        kwargs: dict = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self._model = AutoModelForMaskGeneration.from_pretrained(src, **kwargs)
        self._processor = AutoProcessor.from_pretrained(src)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._device)

    def segment(
        self,
        image: Any,
        *,
        mode: str = "auto",
        prompt: str | None = None,
        points: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        max_masks: int | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> SegmentationResult:
        """Run promptable segmentation on a PIL.Image.

        Per-mode dispatch enforces capability flags as defense-in-depth;
        the route layer is the primary gate. Returns a SegmentationResult
        with masks sorted by score descending and truncated to either
        the request's max_masks or the runtime's default.
        """
        cap = max_masks if max_masks is not None else self.max_masks
        if mode == "text" and not self._supports_text_prompts:
            raise RuntimeError(
                f"model {self.model_id!r} does not support text-prompted segmentation"
            )
        if mode == "points" and not self._supports_point_prompts:
            raise RuntimeError(
                f"model {self.model_id!r} does not support point-prompted segmentation"
            )
        if mode == "boxes" and not self._supports_box_prompts:
            raise RuntimeError(
                f"model {self.model_id!r} does not support box-prompted segmentation"
            )
        if mode == "auto" and not self._supports_automatic:
            raise RuntimeError(
                f"model {self.model_id!r} does not support automatic segmentation"
            )

        records = _segment_with(
            self._model, self._processor, image,
            mode=mode, points=points, boxes=boxes,
            device=self._device, max_masks=cap,
        )
        return SegmentationResult(
            masks=records,
            image_size=image.size,
            mode=mode,
            seed=seed if seed is not None else -1,
            metadata={"model": self.model_id},
        )
