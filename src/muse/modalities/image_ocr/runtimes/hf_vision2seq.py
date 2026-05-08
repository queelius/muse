"""HFVision2SeqRuntime: generic OCR runtime over AutoModelForVision2Seq.

Wraps `transformers.AutoModelForVision2Seq` + `AutoProcessor`. Works
for TrOCR, Nougat, TexTeller, GOT-OCR, and any vision-encoder + text-
decoder pair. The processor's per-family preprocessing
(TrOCRProcessor, NougatProcessor) is dispatched via AutoProcessor.

Deferred imports follow the muse pattern: torch, AutoModelForVision2Seq,
AutoProcessor as module-top sentinels populated by _ensure_deps().
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.

Image preprocessor fallback chain (v0.42.1+):
  1. AutoProcessor (unified): TrOCR, Nougat, repos with
     preprocessor_config.json. Returns multimodal processor.
  2. Split components: AutoImageProcessor + AutoTokenizer. The
     AutoImageProcessor side delegates to muse.core.image_preprocessing
     .build_image_processor for the four-tier dispatch (override-first,
     auto, encoder-hints-derived, fail-loud).
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.image_preprocessing import (
    ImageProcessorError, build_image_processor,
)
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
        image_processor_overrides: dict | None = None,
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
                self._image_processor = build_image_processor(
                    src, overrides=image_processor_overrides, model_id=model_id,
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
