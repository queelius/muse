"""TranslationRuntime: generic runtime over any HF seq2seq translation model.

One class serves every bundled/curated translation model
(m2m100, nllb, opus-mt, madlad) via per-family dispatch on the HF repo
id. NOT a reuse of text_summarization's BartSeq2SeqRuntime: translation
generation needs per-family language-token plumbing (forced BOS
tokens, FLORES-200 mapping, fixed-pair validation, target-language
prefixing) the summarizer must not carry.

Deferred imports follow the muse pattern: torch + transformers stay as
module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from muse.core.runtime_helpers import (
    dtype_for_name,
    select_device,
    set_inference_mode,
)
from muse.modalities.text_translation.protocol import (
    TranslationResult,
    UnsupportedLanguageError,
)
from muse.modalities.text_translation.runtimes.nllb_codes import ISO_TO_FLORES


logger = logging.getLogger(__name__)


torch: Any = None
AutoModelForSeq2SeqLM: Any = None
AutoTokenizer: Any = None


# Case-insensitive substring markers, first match wins. Order matters
# only in that none of these markers overlap today; if a future family
# name could collide with an existing marker, put the more specific
# pattern first (mirrors the chat_formats.yaml precedent).
_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    ("m2m100", "m2m100"),
    ("nllb", "nllb"),
    ("opus-mt", "opus_mt"),
    ("madlad", "madlad"),
)


_MADLAD_TARGET_TOKEN_RE = re.compile(r"^<2([A-Za-z_-]+)>$")


def _family_for(hf_repo: str) -> str:
    """Classify an HF repo id into one of the four supported families.

    Case-insensitive substring match on the repo id. Raises ValueError
    when no marker matches; the constructor decides whether an unknown
    repo falls back to opus_mt semantics (when a declared pair is
    supplied) or propagates the error.
    """
    lowered = hf_repo.lower()
    for marker, family in _FAMILY_MARKERS:
        if marker in lowered:
            return family
    raise ValueError(f"unknown translation family for repo {hf_repo!r}")


def _ensure_deps() -> None:
    global torch, AutoModelForSeq2SeqLM, AutoTokenizer
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("TranslationRuntime: torch unavailable: %s", e)
    if AutoModelForSeq2SeqLM is None:
        try:
            from transformers import (
                AutoModelForSeq2SeqLM as _amfsl,
                AutoTokenizer as _atok,
            )
            AutoModelForSeq2SeqLM = _amfsl
            AutoTokenizer = _atok
        except Exception as e:  # noqa: BLE001
            logger.debug("TranslationRuntime: transformers unavailable: %s", e)


def _resolve_dtype(dtype: str) -> Any:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return dtype_for_name(dtype, torch)


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


def _set_inference_mode(model: Any) -> None:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    set_inference_mode(model)


class TranslationRuntime:
    """Generic seq2seq translation runtime with per-family dispatch."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        num_beams: int = 4,
        source_language: str | None = None,
        target_language: str | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` "
                "or install `transformers` into this venv"
            )
        self.model_id = model_id
        self._num_beams = num_beams
        self._declared_pair = (
            (source_language, target_language)
            if source_language and target_language
            else None
        )

        try:
            self._family = _family_for(hf_repo)
        except ValueError:
            if self._declared_pair is not None:
                self._family = "opus_mt"
            else:
                raise

        self._device = _select_device(device)
        self._dtype = dtype

        src = local_dir or hf_repo
        logger.info(
            "loading %s translation model from %s (device=%s, dtype=%s)",
            self._family, src, self._device, dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        torch_dtype = _resolve_dtype(dtype)
        kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self._model = AutoModelForSeq2SeqLM.from_pretrained(src, **kwargs)
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)

    def translate(
        self, texts: list[str], *, source: str, target: str
    ) -> TranslationResult:
        if self._family == "m2m100":
            return self._translate_m2m100(texts, source=source, target=target)
        if self._family == "nllb":
            return self._translate_nllb(texts, source=source, target=target)
        if self._family == "opus_mt":
            return self._translate_opus_mt(texts, source=source, target=target)
        if self._family == "madlad":
            return self._translate_madlad(texts, source=source, target=target)
        raise ValueError(f"unhandled translation family {self._family!r}")  # pragma: no cover

    def supported_languages(self) -> dict[str, list[str]]:
        if self._family == "m2m100":
            return self._supported_m2m100()
        if self._family == "nllb":
            return self._supported_nllb()
        if self._family == "opus_mt":
            return self._supported_opus_mt()
        if self._family == "madlad":
            return self._supported_madlad()
        raise ValueError(f"unhandled translation family {self._family!r}")  # pragma: no cover

    # -- per-family translate --

    def _translate_m2m100(
        self, texts: list[str], *, source: str, target: str
    ) -> TranslationResult:
        self._tokenizer.src_lang = source
        forced_bos_token_id = self._tokenizer.get_lang_id(target)
        return self._generate(texts, forced_bos_token_id=forced_bos_token_id)

    def _translate_nllb(
        self, texts: list[str], *, source: str, target: str
    ) -> TranslationResult:
        flores_src = ISO_TO_FLORES.get(source)
        if flores_src is None:
            raise UnsupportedLanguageError(source, sorted(ISO_TO_FLORES))
        flores_tgt = ISO_TO_FLORES.get(target)
        if flores_tgt is None:
            raise UnsupportedLanguageError(target, sorted(ISO_TO_FLORES))
        self._tokenizer.src_lang = flores_src
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(flores_tgt)
        return self._generate(texts, forced_bos_token_id=forced_bos_token_id)

    def _translate_opus_mt(
        self, texts: list[str], *, source: str, target: str
    ) -> TranslationResult:
        if self._declared_pair is None or (source, target) != self._declared_pair:
            supported = (
                {self._declared_pair[0]: [self._declared_pair[1]]}
                if self._declared_pair else {}
            )
            raise UnsupportedLanguageError(f"{source}->{target}", supported)
        return self._generate(texts)

    def _translate_madlad(
        self, texts: list[str], *, source: str, target: str
    ) -> TranslationResult:
        prefixed = [f"<2{target}> {t}" for t in texts]
        return self._generate(prefixed)

    # -- per-family supported_languages --

    def _supported_m2m100(self) -> dict[str, list[str]]:
        codes = sorted(self._tokenizer.lang_code_to_id.keys())
        return {c: [x for x in codes if x != c] for c in codes}

    def _supported_nllb(self) -> dict[str, list[str]]:
        codes = sorted(ISO_TO_FLORES.keys())
        return {c: [x for x in codes if x != c] for c in codes}

    def _supported_opus_mt(self) -> dict[str, list[str]]:
        if self._declared_pair is None:
            return {}
        src, tgt = self._declared_pair
        return {src: [tgt]}

    def _supported_madlad(self) -> dict[str, list[str]]:
        codes = self._madlad_target_codes()
        return {c: [x for x in codes if x != c] for c in codes}

    def _madlad_target_codes(self) -> list[str]:
        """Best-effort scan of the tokenizer vocab for <2xx> target tokens."""
        vocab = self._tokenizer.get_vocab()
        codes = set()
        for token in vocab.keys():
            m = _MADLAD_TARGET_TOKEN_RE.match(token)
            if m:
                codes.add(m.group(1))
        return sorted(codes)

    # -- shared batch generation --

    def _generate(
        self, texts: list[str], *, forced_bos_token_id: int | None = None
    ) -> TranslationResult:
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self._device)
        longest = int(input_ids.shape[-1])
        max_new_tokens = min(1024, 2 * longest + 16)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": self._num_beams,
        }
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        output_ids = self._model.generate(input_ids, **gen_kwargs)
        texts_out = [
            self._tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        return TranslationResult(texts=texts_out)
