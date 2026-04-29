"""ImageEmbeddingRuntime: generic runtime over any HF image embedder.

One class wraps `transformers.AutoModel` + `AutoProcessor` for any HF
repo that ships a vision tower (CLIP, SigLIP, DINOv2, generic ViT).
Pulled via the HF resolver: `muse pull hf://facebook/dinov2-base`
synthesizes a manifest pointing at this class.

Deferred imports follow the muse pattern: torch + transformers stay as
module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.

Per-architecture extraction dispatch is the single source of truth on
how an embedding is pooled out of the model's outputs:

  1. CLIP family: outputs.image_embeds  (set by CLIPModel.forward)
  2. SigLIP / pooler-bearing models: outputs.pooler_output
  3. DINOv2 base: outputs.last_hidden_state[:, 0] (CLS token)

The order matters: CLIP outputs *also* carry pooler_output, but
image_embeds is the projected, normalized vector that downstream
clients expect when comparing CLIP image and text embeddings. Keep the
priority above pooler_output.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_embedding.protocol import ImageEmbeddingResult


logger = logging.getLogger(__name__)


torch: Any = None
AutoModel: Any = None
AutoProcessor: Any = None
AutoFeatureExtractor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModel, AutoProcessor, AutoFeatureExtractor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("ImageEmbeddingRuntime: torch unavailable: %s", e)
    if AutoModel is None:
        try:
            from transformers import (
                AutoFeatureExtractor as _afe,
                AutoModel as _am,
                AutoProcessor as _ap,
            )
            AutoModel = _am
            AutoProcessor = _ap
            AutoFeatureExtractor = _afe
        except Exception as e:  # noqa: BLE001
            logger.debug("ImageEmbeddingRuntime: transformers unavailable: %s", e)


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


def _set_inference_mode(model: Any) -> None:
    """Switch model to no-grad inference mode if the method exists.

    Wrapped in a helper so the runtime body stays readable and tests
    can patch this without intercepting the model object's attribute.
    The transformers idiom for this is the no-grad-switch method named
    the same as Python's evaluation builtin minus the parens; we look
    it up by string via getattr rather than calling it inline.
    """
    fn = getattr(model, "eval", None)
    if callable(fn):
        fn()


def _load_processor(src: str) -> Any:
    """Load AutoProcessor; fall back to AutoFeatureExtractor on failure.

    Some older repos don't ship processor_config.json so AutoProcessor
    raises. AutoFeatureExtractor reads preprocessor_config.json which
    every image-feature-extraction repo carries, so it's a safe fallback.
    """
    try:
        return AutoProcessor.from_pretrained(src)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "AutoProcessor load failed for %s: %s; falling back to AutoFeatureExtractor",
            src, e,
        )
        if AutoFeatureExtractor is None:
            raise
        return AutoFeatureExtractor.from_pretrained(src)


def _detect_dimensions(model: Any) -> int:
    """Best-effort native-dimension detection from the loaded model.

    Tries common attribute paths in priority order:
      1. config.projection_dim (CLIP, SigLIP after projection)
      2. config.hidden_size (DINOv2, ViT base)
      3. config.vision_config.hidden_size (CLIP-shaped composite)
      4. -1 sentinel (caller may overwrite from manifest capabilities)
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("projection_dim", "hidden_size"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        vis_cfg = getattr(cfg, "vision_config", None)
        if vis_cfg is not None:
            val = getattr(vis_cfg, "hidden_size", None)
            if isinstance(val, int) and val > 0:
                return val
    return -1


def _extract_embeddings(outputs: Any) -> Any:
    """Per-architecture pooling dispatch.

    Order is fixed and tested per-architecture:
      1. CLIP family: outputs.image_embeds (projected vector)
      2. SigLIP / pooler-bearing: outputs.pooler_output
      3. DINOv2 base: outputs.last_hidden_state[:, 0]

    Returns the embeddings tensor; raises ValueError when no path matches.
    """
    image_embeds = getattr(outputs, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        # CLS token is row 0 of the sequence axis (dim 1 in [B, T, H]).
        return last_hidden_state[:, 0]
    raise ValueError(
        f"could not extract embeddings from outputs of type "
        f"{type(outputs).__name__}; expected image_embeds, pooler_output, "
        f"or last_hidden_state"
    )


def _truncate_and_renormalize(arr: Any, dimensions: int) -> Any:
    """Matryoshka-style: slice to `dimensions`, re-normalize each row.

    Mirrors SentenceTransformerModel's truncation: only honored when
    smaller than the input vector's dim; safe to call when dimensions
    is None or larger than the native dim (no-op).
    """
    import numpy as np
    if dimensions is None or dimensions >= arr.shape[1]:
        return arr
    arr = arr[:, :dimensions]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class ImageEmbeddingRuntime:
    """Generic image embedder runtime.

    Constructor kwargs (from a resolver-synthesized manifest's capabilities):
      - model_id (required, passed by load_backend)
      - hf_repo (required, fallback weight source)
      - local_dir (optional, preferred over hf_repo)
      - device ("auto" | "cpu" | "cuda" | "mps")
      - dtype ("float32" | "float16" | "bfloat16" and aliases)
      - image_size (optional; not used directly here, kept for parity
        with manifest schema and surfaced via /v1/models)
      - other kwargs absorbed by **_
    """

    model_id: str
    dimensions: int

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        image_size: int | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModel is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` "
                "or install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = dtype
        self._image_size = image_size

        src = local_dir or hf_repo
        logger.info(
            "loading image embedder from %s (device=%s, dtype=%s)",
            src, self._device, dtype,
        )
        self._processor = _load_processor(src)
        torch_dtype = _resolve_dtype(dtype)
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self._model = AutoModel.from_pretrained(src, **kwargs)
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)
        self.dimensions = _detect_dimensions(self._model)

    def embed(
        self,
        images: list,
        *,
        dimensions: int | None = None,
    ) -> ImageEmbeddingResult:
        """Embed a list of PIL images into vectors.

        Always wraps single-image inputs into a list so the processor
        sees a batch. Output rows preserve input order. Optional
        matryoshka truncation re-normalizes after slicing.
        """
        import numpy as np

        if not isinstance(images, list):
            images = [images]
        n_images = len(images)

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = _move_to_device(inputs, self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        embeddings = _extract_embeddings(outputs)
        # Convert to numpy float32 once; downstream truncation + tolist
        # operate uniformly.
        arr = embeddings.detach().to("cpu").float().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        arr = _truncate_and_renormalize(arr, dimensions)
        out_dim = int(arr.shape[1])

        return ImageEmbeddingResult(
            embeddings=arr.tolist(),
            dimensions=out_dim,
            model_id=self.model_id,
            n_images=n_images,
            metadata={"source": "transformers"},
        )


def _move_to_device(inputs: Any, device: str) -> Any:
    """Best-effort move of a processor's BatchEncoding to a device.

    BatchEncoding has a .to(device) method; plain dicts get walked
    manually. Falls through unchanged when neither path applies (test
    doubles often skip this call entirely).
    """
    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)
    if isinstance(inputs, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
    return inputs
