"""HFVisionLanguageModel: generic VLM runtime over AutoModelForImageTextToText.

Wraps `transformers.AutoModelForImageTextToText` + `AutoProcessor`. Works
for SmolVLM, Qwen2-VL, LLaVA-1.5/1.6, Idefics3, Pixtral, MiniCPM-V, and any
other vision-language model on the Hub that exposes the
image-text-to-text architecture.

Vision class naming history:
  - transformers <= 4.45: `AutoModelForVision2Seq` was the canonical class.
  - transformers >= 4.46: `AutoModelForImageTextToText` is the canonical
    class for VLMs (image-text-to-text), distinct from
    AutoModelForVision2Seq which now targets older OCR-style models.

Try the new name first, fall back to the old name. Either populates the
module sentinel.

Deferred imports follow the muse pattern: torch + transformers stay as
module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.chat_completion.protocol import (
    ChatChoice, ChatChunk, ChatResult,
)


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForImageTextToText: Any = None
AutoModelForVision2Seq: Any = None
AutoProcessor: Any = None
TextIteratorStreamer: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForImageTextToText, AutoModelForVision2Seq
    global AutoProcessor, TextIteratorStreamer
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFVisionLanguageModel torch unavailable: %s", e)
    if AutoModelForImageTextToText is None and AutoModelForVision2Seq is None:
        try:
            from transformers import AutoModelForImageTextToText as _m
            AutoModelForImageTextToText = _m
        except Exception:  # noqa: BLE001
            try:
                from transformers import AutoModelForVision2Seq as _m
                AutoModelForVision2Seq = _m
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "HFVisionLanguageModel: vision-text class unavailable: %s",
                    e,
                )
    if AutoProcessor is None:
        try:
            from transformers import AutoProcessor as _p
            AutoProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFVisionLanguageModel AutoProcessor unavailable: %s", e,
            )
    if TextIteratorStreamer is None:
        try:
            from transformers import TextIteratorStreamer as _ts
            TextIteratorStreamer = _ts
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFVisionLanguageModel TextIteratorStreamer unavailable: %s",
                e,
            )


class HFVisionLanguageModel:
    """Generic vision-language model runtime."""

    model_id: str
    supports_vision: bool = True
    supports_tools: bool = False

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp16",
        supports_multi_image: bool = False,
        max_new_tokens: int = 512,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        model_class = AutoModelForImageTextToText or AutoModelForVision2Seq
        if model_class is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed or too old; needs >= 4.46 "
                "for AutoModelForImageTextToText. Run "
                f"`muse models refresh {model_id}`."
            )
        self.model_id = model_id
        self.supports_multi_image = bool(supports_multi_image)
        self._device = select_device(device, torch_module=torch)
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._default_max_new_tokens = max_new_tokens
        src = local_dir or hf_repo
        with LoadTimer(f"loading VLM from {src}", logger):
            self._processor = AutoProcessor.from_pretrained(src)
            self._model = model_class.from_pretrained(
                src, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        text, images = _prepare_inputs(messages, self._processor)
        max_new = kwargs.get("max_tokens") or self._default_max_new_tokens

        proc_kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self._processor(**proc_kwargs).to(self._device)

        prompt_len = int(inputs["input_ids"].shape[-1])

        seed = kwargs.get("seed")
        if seed is not None and torch is not None:
            torch.manual_seed(int(seed))

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new}
        for key in ("temperature", "top_p"):
            if kwargs.get(key) is not None:
                gen_kwargs[key] = kwargs[key]

        outputs = self._model.generate(**inputs, **gen_kwargs)
        # outputs shape is [batch, total_seq_len]; outputs[0] is the full
        # generated sequence for the first (only) batch element.
        # Slice off the prompt to get completion tokens only.
        completion_ids = outputs[0][prompt_len:]
        text_out = self._processor.batch_decode(
            [completion_ids], skip_special_tokens=True,
        )[0].strip()

        total_tokens = int(outputs.shape[-1])
        completion_tokens = total_tokens - prompt_len
        now = int(time.time())
        return ChatResult(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            model_id=self.model_id,
            created=now,
            choices=[ChatChoice(
                index=0,
                message={"role": "assistant", "content": text_out},
                finish_reason="stop",
            )],
            usage={
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

    def chat_stream(
        self, messages: list[dict], **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        text, images = _prepare_inputs(messages, self._processor)
        max_new = kwargs.get("max_tokens") or self._default_max_new_tokens

        proc_kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self._processor(**proc_kwargs).to(self._device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        seed = kwargs.get("seed")
        if seed is not None and torch is not None:
            torch.manual_seed(int(seed))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new, "streamer": streamer,
        }
        for key in ("temperature", "top_p"):
            if kwargs.get(key) is not None:
                gen_kwargs[key] = kwargs[key]

        generate_inputs = {**inputs, **gen_kwargs}

        def _generate_with_cleanup() -> None:
            try:
                self._model.generate(**generate_inputs)
            finally:
                streamer.end()

        thread = threading.Thread(target=_generate_with_cleanup, daemon=True)
        thread.start()

        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
        for token in streamer:
            yield ChatChunk(
                id=chunk_id,
                model_id=self.model_id,
                created=int(time.time()),
                choice_index=0,
                delta={"content": token},
                finish_reason=None,
            )
        thread.join()
        yield ChatChunk(
            id=chunk_id,
            model_id=self.model_id,
            created=int(time.time()),
            choice_index=0,
            delta={},
            finish_reason="stop",
        )


def _prepare_inputs(messages: list[dict], processor: Any) -> tuple[str, list]:
    """Walk message content; return (chat_template_text, images_list).

    Image parts arrive as {type: image, image: <PIL>} (the route layer
    rewrites image_url -> image after decoding). Text parts stay as
    strings or {type: text, text: ...} dicts. The processor's
    apply_chat_template handles the model-specific multimodal template.

    Raises ValueError if a still-undecoded image_url part reaches us
    (means the route layer didn't run; fail loud).
    """
    images: list = []
    template_messages: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            template_messages.append({"role": msg["role"], "content": content})
            continue
        new_content: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            ptype = part.get("type")
            if ptype == "image":
                images.append(part["image"])
                new_content.append({"type": "image"})
            elif ptype == "text":
                new_content.append({"type": "text", "text": part["text"]})
            elif ptype == "image_url":
                raise ValueError(
                    "image_url part reached runtime; "
                    "route layer did not decode it"
                )
            else:
                new_content.append(part)
        template_messages.append(
            {"role": msg["role"], "content": new_content},
        )
    text = processor.apply_chat_template(
        template_messages, add_generation_prompt=True, tokenize=False,
    )
    return text, images
