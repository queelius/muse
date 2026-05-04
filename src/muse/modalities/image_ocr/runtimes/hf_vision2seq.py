"""HFVision2SeqRuntime: generic OCR runtime over AutoModelForVision2Seq.

Wraps `transformers.AutoModelForVision2Seq` + `AutoProcessor`. Works
for TrOCR, Nougat, TexTeller, GOT-OCR, and any vision-encoder + text-
decoder pair. The processor's per-family preprocessing
(TrOCRProcessor, NougatProcessor) is dispatched via AutoProcessor.

Deferred imports follow the muse pattern: torch, AutoModelForVision2Seq,
AutoProcessor as module-top sentinels populated by _ensure_deps().
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.image_ocr.protocol import OcrResult


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForVision2Seq: Any = None
AutoProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForVision2Seq, AutoProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFVision2SeqRuntime torch unavailable: %s", e)
    if AutoModelForVision2Seq is None:
        try:
            from transformers import AutoModelForVision2Seq as _m
            AutoModelForVision2Seq = _m
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFVision2SeqRuntime AutoModelForVision2Seq unavailable: %s", e,
            )
    if AutoProcessor is None:
        try:
            from transformers import AutoProcessor as _p
            AutoProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("HFVision2SeqRuntime AutoProcessor unavailable: %s", e)


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
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        self._default_max_new_tokens = max_new_tokens
        src = local_dir or hf_repo
        with LoadTimer(f"loading vision2seq from {src}", logger):
            self._processor = AutoProcessor.from_pretrained(src)
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
        inputs = self._processor(image, return_tensors="pt").to(self._device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "num_beams": num_beams,
        }
        if prompt is not None:
            tok_ids = self._processor.tokenizer(
                prompt, return_tensors="pt",
            ).input_ids.to(self._device)
            gen_kwargs["decoder_input_ids"] = tok_ids

        outputs = self._model.generate(**inputs, **gen_kwargs)
        completion_tokens = int(outputs.shape[-1])

        decoded = self._processor.batch_decode(
            outputs, skip_special_tokens=True,
        )[0]
        # post_process_generation is a Nougat-specific helper that strips
        # control tokens and normalizes whitespace. Older transformers
        # versions don't have it; absent processor handles fall through.
        post = getattr(self._processor, "post_process_generation", None)
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
