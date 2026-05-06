"""HFVision2SeqRuntime: generic OCR runtime over AutoModelForVision2Seq.

Wraps `transformers.AutoModelForVision2Seq` + `AutoProcessor`. Works
for TrOCR, Nougat, TexTeller, GOT-OCR, and any vision-encoder + text-
decoder pair. The processor's per-family preprocessing
(TrOCRProcessor, NougatProcessor) is dispatched via AutoProcessor.

Deferred imports follow the muse pattern: torch, AutoModelForVision2Seq,
AutoProcessor as module-top sentinels populated by _ensure_deps().
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.

Image preprocessor fallback chain (v0.41.2+):
  1. AutoProcessor (unified): TrOCR, Nougat, repos with
     preprocessor_config.json. Returns multimodal processor.
  2. AutoImageProcessor (split): repos that ship preprocessor_config.json
     but no unified config. Pairs with AutoTokenizer.
  3. _DerivedImageProcessor (split, v0.41.2+): repos with neither
     preprocessor file. Reads encoder hints (num_channels, image_size)
     from config.json and builds preprocessing accordingly. Handles
     TexTeller (1-channel grayscale at 448x448) without hardcoding.
  4. ViTImageProcessor with defaults: last-resort, ImageNet RGB 224x224.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.image_ocr.protocol import OcrResult


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForVision2Seq: Any = None
AutoProcessor: Any = None
AutoImageProcessor: Any = None
AutoTokenizer: Any = None
_LAST_IMPORT_ERROR: Exception | None = None


def _ensure_deps() -> None:
    """Lazy-import torch + transformers Auto* classes.

    Vision2Seq family naming history:
      - transformers <= 4.x: `AutoModelForVision2Seq` is the canonical
        class for vision-encoder + text-decoder models (TrOCR, Nougat,
        TexTeller).
      - transformers 5.0 renamed it to `AutoModelForImageTextToText`
        to match HF's tag taxonomy. The old name is gone in 5.x.

    Try the new name first (forward-looking), fall back to the old
    name (covers 4.x venvs). Either one populates the module sentinel.

    The last-import-error is recorded on the module so the constructor
    can surface a useful message instead of the generic "transformers
    is not installed" line, which lies when transformers IS installed
    but the specific class isn't.
    """
    global torch, AutoModelForVision2Seq, AutoProcessor
    global AutoImageProcessor, AutoTokenizer, _LAST_IMPORT_ERROR
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFVision2SeqRuntime torch unavailable: %s", e)
    if AutoModelForVision2Seq is None:
        last_err: Exception | None = None
        try:
            from transformers import AutoModelForImageTextToText as _m
            AutoModelForVision2Seq = _m
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.debug(
                "HFVision2SeqRuntime AutoModelForImageTextToText unavailable: %s", e,
            )
            try:
                from transformers import AutoModelForVision2Seq as _m
                AutoModelForVision2Seq = _m
            except Exception as e2:  # noqa: BLE001
                last_err = e2
                logger.debug(
                    "HFVision2SeqRuntime AutoModelForVision2Seq unavailable: %s",
                    e2,
                )
        if AutoModelForVision2Seq is None:
            _LAST_IMPORT_ERROR = last_err
    if AutoProcessor is None:
        try:
            from transformers import AutoProcessor as _p
            AutoProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("HFVision2SeqRuntime AutoProcessor unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if AutoImageProcessor is None:
        try:
            from transformers import AutoImageProcessor as _ip
            AutoImageProcessor = _ip
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFVision2SeqRuntime AutoImageProcessor unavailable: %s", e,
            )
    if AutoTokenizer is None:
        try:
            from transformers import AutoTokenizer as _tk
            AutoTokenizer = _tk
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFVision2SeqRuntime AutoTokenizer unavailable: %s", e,
            )


def _read_encoder_preprocess_hints(src: str) -> dict:
    """Read encoder preprocessing hints from a model's `config.json`.

    Returns a dict with any of `num_channels`, `image_size` (int or
    list), `image_mean`, `image_std` that the encoder config exposes.
    Empty dict when config.json is missing or unreadable.

    Vision-encoder-decoder configs nest the encoder's hyperparams
    under either `encoder` (the canonical layout) or directly at the
    top level (some older repos). We check both.

    Used as the fallback when neither AutoProcessor nor
    AutoImageProcessor can synthesize a preprocessor from the repo
    files: derive enough preprocessing parameters from the model's
    own config to build a `_DerivedImageProcessor`. Handles repos
    like TexTeller (1-channel grayscale at 448x448) without
    hardcoding model-specific paths.
    """
    config_path = Path(src) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("could not read %s: %s", config_path, e)
        return {}
    enc = cfg.get("encoder") or cfg
    out: dict = {}
    for key in ("num_channels", "image_size", "image_mean", "image_std"):
        if key in enc:
            out[key] = enc[key]
    return out


class _DerivedImageProcessor:
    """Minimal image preprocessor derived from a model's config.json.

    Produces a `BatchFeature`-compatible object with `pixel_values`
    tensor of shape `(1, num_channels, H, W)`. Mimics the
    `AutoImageProcessor` interface that the runtime's inference path
    expects: callable with `image` + `return_tensors="pt"`, returning
    an object that supports `.to(device)` (returns a BatchFeature
    with all tensors moved) and unpacks via `**inputs` for the
    model's `generate(...)`.

    Used when the repo lacks both `preprocessor_config.json` and
    `feature_extractor_config.json`, so AutoImageProcessor and
    AutoFeatureExtractor both raise. We synthesize preprocessing
    from `config.json`'s encoder hyperparams.

    Conventions:
      - `num_channels=1`: convert PIL to grayscale ("L" mode);
        output tensor shape (1, 1, H, W).
      - `num_channels=3` (default): convert to RGB; (1, 3, H, W).
      - `image_size` is the side length (int) or a (h, w) pair.
      - Pixel range normalized to [-1, 1] via mean=0.5, std=0.5
        per-channel (the canonical ViT default for grayscale and
        the most common for RGB OCR encoders). Override via
        `image_mean` / `image_std` from config.json when the encoder
        declares them.
    """

    def __init__(
        self,
        *,
        num_channels: int = 3,
        image_size: int | tuple[int, int] = 224,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        self.num_channels = int(num_channels)
        if isinstance(image_size, (list, tuple)):
            self.height, self.width = int(image_size[0]), int(image_size[1])
        else:
            self.height = self.width = int(image_size)
        # Default to ViT canonical symmetric [-1, 1] scaling; matches
        # what most encoders without a shipped preprocessor were trained on.
        self.image_mean = image_mean or [0.5] * self.num_channels
        self.image_std = image_std or [0.5] * self.num_channels

    def __call__(self, image: Any, *, return_tensors: str = "pt") -> Any:
        """Preprocess a PIL image into a model-ready BatchFeature.

        Output has shape `(1, num_channels, H, W)` and supports
        `.to(device)` for the runtime's `inputs.to(self._device)` chain.
        """
        target_mode = "L" if self.num_channels == 1 else "RGB"
        if not hasattr(image, "convert"):
            raise TypeError(
                f"_DerivedImageProcessor expected a PIL Image; got {type(image)!r}"
            )
        from PIL import Image as _PILImage
        image = image.convert(target_mode).resize(
            (self.width, self.height), _PILImage.Resampling.BICUBIC,
        )

        import numpy as np
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if self.num_channels == 1:
            arr = arr[..., None]  # PIL "L" gives (H, W); add channel axis.
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        arr = ((arr - mean) / std).transpose(2, 0, 1)[None, :, :, :]

        if return_tensors != "pt":
            return {"pixel_values": arr}
        if torch is None:
            raise RuntimeError(
                "torch is not available; "
                "_DerivedImageProcessor requires torch for tensor output"
            )
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature({"pixel_values": torch.from_numpy(arr)})


def _build_image_processor_fallback(*, src: str, model_id: str) -> Any:
    """Three-tier image-processor fallback for the split-component path.

    See module docstring for the full ladder. Raises RuntimeError if
    none of the tiers can produce a usable processor.
    """
    # Tier 1: AutoImageProcessor.from_pretrained
    try:
        return AutoImageProcessor.from_pretrained(src)
    except Exception as e_auto:  # noqa: BLE001
        logger.debug(
            "AutoImageProcessor.from_pretrained(%s) failed: %s",
            src, e_auto,
        )

    # Tier 2: _DerivedImageProcessor from config.json hints
    hints = _read_encoder_preprocess_hints(src)
    if hints:
        try:
            proc = _DerivedImageProcessor(
                num_channels=int(hints.get("num_channels", 3)),
                image_size=hints.get("image_size", 224),
                image_mean=hints.get("image_mean"),
                image_std=hints.get("image_std"),
            )
            logger.info(
                "Loaded %s with _DerivedImageProcessor "
                "(num_channels=%s image_size=%s; repo lacks "
                "preprocessor_config.json, derived from config.json)",
                model_id, hints.get("num_channels"), hints.get("image_size"),
            )
            return proc
        except Exception as e_derived:  # noqa: BLE001
            logger.debug(
                "_DerivedImageProcessor failed for %s: %s",
                src, e_derived,
            )

    # Tier 3: ViTImageProcessor defaults
    try:
        from transformers import ViTImageProcessor
        proc = ViTImageProcessor()
        logger.warning(
            "Loaded %s with ViTImageProcessor defaults "
            "(no preprocessor_config.json AND no usable encoder hints "
            "in config.json; output may be wrong for non-RGB-224 models)",
            model_id,
        )
        return proc
    except Exception as e_vit:  # noqa: BLE001
        raise RuntimeError(
            f"Cannot load image processor for {src!r}: tried "
            f"AutoImageProcessor, _DerivedImageProcessor (config.json "
            f"hints: {hints!r}), and ViTImageProcessor defaults; all "
            f"failed. Last error: {e_vit}"
        ) from e_vit


class HFVision2SeqRuntime:
    """Generic vision-encoder + text-decoder OCR runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        max_new_tokens: int = 512,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        if AutoModelForVision2Seq is None or AutoProcessor is None:
            # Don't lie: transformers might be installed but missing the
            # specific class. Surface the underlying ImportError so the
            # operator can diagnose (e.g. "transformers 5.x dropped
            # AutoModelForVision2Seq; pin transformers<5.0 OR upgrade
            # muse to a version that imports AutoModelForImageTextToText").
            try:
                import transformers as _t
                version = getattr(_t, "__version__", "unknown")
                detail = (
                    f"transformers {version} is installed but neither "
                    f"`AutoModelForImageTextToText` (transformers 5.x) "
                    f"nor `AutoModelForVision2Seq` (transformers 4.x) "
                    f"could be imported"
                )
            except Exception:  # noqa: BLE001
                detail = (
                    "transformers package is not importable in this venv"
                )
            if _LAST_IMPORT_ERROR is not None:
                detail += f" (last error: {_LAST_IMPORT_ERROR})"
            raise RuntimeError(
                f"{detail}; run `muse models refresh {model_id}` or "
                f"install transformers into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        self._default_max_new_tokens = max_new_tokens
        src = local_dir or hf_repo
        with LoadTimer(f"loading vision2seq from {src}", logger):
            # Modern multimodal repos (TrOCR, Nougat) ship a
            # `preprocessor_config.json` so AutoProcessor returns a
            # unified processor that handles both image preprocessing
            # AND tokenization. Older repos (TexTeller) have only
            # `tokenizer_config.json`, so AutoProcessor falls back to
            # AutoTokenizer; calling `tokenizer(image)` would then fail
            # at inference with a confusing "text input must be of type
            # str" error.
            #
            # Detect the split case by looking for the standard image-side
            # attributes (`image_processor` or legacy `feature_extractor`)
            # on the loaded processor. If absent, load
            # `AutoImageProcessor` + `AutoTokenizer` separately and
            # leave `_processor=None` so the inference path uses the
            # split components.
            unified = AutoProcessor.from_pretrained(src)
            has_image_side = (
                hasattr(unified, "image_processor")
                or hasattr(unified, "feature_extractor")
            )
            if has_image_side:
                self._processor = unified
                self._image_processor = None
                self._tokenizer = None
            else:
                # Split path: `unified` is actually a tokenizer.
                self._processor = None
                self._tokenizer = unified
                if AutoImageProcessor is None:
                    raise RuntimeError(
                        "AutoImageProcessor not available; cannot load "
                        f"{src!r} (the repo lacks preprocessor_config.json "
                        "so the split-component path is required, but "
                        "AutoImageProcessor failed to import). Run "
                        f"`muse models refresh {model_id}`."
                    )
                self._image_processor = _build_image_processor_fallback(
                    src=src, model_id=model_id,
                )
            self._model = AutoModelForVision2Seq.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def ocr(
        self,
        image: Any,
        *,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        num_beams: int = 1,
    ) -> OcrResult:
        """Extract text from one PIL.Image.

        With prompt=None, the model uses its default decoder start
        token (TrOCR, Nougat-without-task-tag). With prompt set, the
        runtime tokenizes it as decoder_input_ids so models like
        Nougat can receive the task hint.
        """
        max_new = max_new_tokens or self._default_max_new_tokens
        # Two paths depending on whether the repo shipped a unified
        # multimodal processor (TrOCR, Nougat) or only a tokenizer
        # (TexTeller and other older repos that lack
        # preprocessor_config.json). The constructor decided which
        # branch we're in by inspecting the AutoProcessor return.
        if self._processor is not None:
            inputs = self._processor(
                image, return_tensors="pt",
            ).to(self._device)
            tokenizer = self._processor.tokenizer
            decoder = self._processor
        else:
            inputs = self._image_processor(
                image, return_tensors="pt",
            ).to(self._device)
            tokenizer = self._tokenizer
            decoder = self._tokenizer
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "num_beams": num_beams,
        }
        # Track the decoder prompt length so completion_tokens reflects
        # only the newly generated tokens. Without prompt, the model
        # uses a single bos/decoder_start token; with prompt, the
        # tokenized prompt prepends.
        prompt_len = 1
        if prompt is not None:
            tok_ids = tokenizer(
                prompt, return_tensors="pt",
            ).input_ids.to(self._device)
            gen_kwargs["decoder_input_ids"] = tok_ids
            prompt_len = int(tok_ids.shape[-1])

        outputs = self._model.generate(**inputs, **gen_kwargs)
        # Subtract the prompt length so completion_tokens reports
        # newly-generated tokens, not the full sequence (mirrors the
        # OpenAI usage.completion_tokens semantics).
        completion_tokens = max(0, int(outputs.shape[-1]) - prompt_len)

        decoded = decoder.batch_decode(
            outputs, skip_special_tokens=True,
        )[0]
        # post_process_generation is a Nougat-specific helper that strips
        # control tokens and normalizes whitespace. Lives on the unified
        # multimodal processor; in the split-component path it never
        # exists (TexTeller and similar legacy repos don't ship it).
        post = (
            getattr(self._processor, "post_process_generation", None)
            if self._processor is not None
            else None
        )
        if callable(post):
            try:
                # Some processors take fix_markdown=True; pass it when
                # supported and ignore TypeError if not.
                text = post(decoded, fix_markdown=True)
            except TypeError:
                text = post(decoded)
        else:
            text = decoded

        return OcrResult(
            text=text,
            model_id=self.model_id,
            completion_tokens=completion_tokens,
        )


# Thin delegators preserved for test imports (matches sibling runtimes;
# the meta-test in tests/core/test_runtime_helpers_meta.py flags
# re-implementations).
def _select_device(device: str) -> str:
    return select_device(device, torch_module=torch)


def _resolve_dtype(dtype: str):
    return dtype_for_name(dtype, torch_module=torch)
