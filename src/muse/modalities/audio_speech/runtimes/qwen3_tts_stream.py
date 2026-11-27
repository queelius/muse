"""Streaming speech decode for qwen-tts 0.1.1's 12 Hz tokenizer.

The 12 Hz speech decoder (``Qwen3TTSTokenizerV2Decoder``) is causal
throughout: a quantizer and causal ``pre_conv``, an 8-layer causal
transformer at the frame rate, then a conv stack (``upsample`` and
``decoder`` blocks, then a clamp) that turns each frame into 1920 samples.
:class:`SplitDecoder` runs the two halves separately:
``conv_stack(front(codes))`` is bit-identical to ``decoder(codes)``.

Streaming cannot chunk the decoder as upstream's ``chunked_decode`` does
(decode ``[start - L, end)``, drop the context): the transformer's memory is
long, and at 25 frames of context that reaches only 13 dB SNR against the
one-pass audio. Splitting it works: the front is causal, so its output for a
frame never changes once computed, and the conv stack's receptive field is
short (about 41 dB against the one-pass audio at 4 frames of left context,
flat from 6). :class:`StreamDecoder` runs the front over every frame so far
(the reference clip's in front, in clone mode) and the conv stack over the
new frames plus ``context`` frames before them. The result sits at the
decoder's own floor, ~36-37 dB against one pass: decoding a prefix of the
same codes in one pass differs from the full decode by as much (bf16
rounding that depends on the sequence length).

torch is passed in (``torch=``) so the module imports without it.
"""
from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_QWEN_TTS = frozenset({"0.1.1"})
# Frames of left context for the conv stack: 4 reach ~41 dB against the
# one-pass audio, 6 and more the plateau (41.4 dB, RTX 3060, bf16).
CONV_CONTEXT = 6


def _qwen_tts_version() -> str | None:
    try:
        return importlib.metadata.version("qwen-tts")
    except importlib.metadata.PackageNotFoundError:
        return None


class SplitDecoder:
    """A 12 Hz speech decoder as its causal front and its conv stack."""

    def __init__(self, decoder: Any, *, torch: Any, context: int = CONV_CONTEXT,
                 device: Any = None) -> None:
        self.decoder = decoder
        self.torch = torch
        self.context = int(context)
        self.upsample = int(decoder.total_upsample)
        self.device = device

    def front(self, codes: Any) -> Any:
        """Codes ``[1, quantizers, frames]`` -> transformer output
        ``[1, channels, frames]``: upstream's forward up to the conv stack."""
        dec = self.decoder
        hidden = dec.quantizer.decode(codes)
        hidden = dec.pre_conv(hidden).transpose(1, 2)
        hidden = dec.pre_transformer(inputs_embeds=hidden).last_hidden_state
        return hidden.permute(0, 2, 1)

    def conv_stack(self, hidden: Any) -> Any:
        """Front output ``[1, channels, frames]`` -> audio ``[1, 1, samples]``."""
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def window(self, hidden: Any, start: int, end: int) -> np.ndarray:
        """Audio for frames ``[start, end)`` of a front output, decoding
        ``context`` frames before ``start`` and dropping their samples."""
        lo = max(0, start - self.context)
        wav = self.conv_stack(hidden[..., lo:end])[0, 0][(start - lo) * self.upsample:]
        return wav.to(self.torch.float32).cpu().numpy()

    def decode(self, codes: Any, *, skip: int = 0) -> np.ndarray:
        """A whole line ``[frames, quantizers]`` in one call, without the
        first ``skip`` frames' audio (the reference clip in clone mode),
        whose conv stack is not run."""
        with self.torch.inference_mode():
            codes = self.torch.clamp(codes, min=0).transpose(0, 1)[None]
            return self.window(self.front(codes), skip, codes.shape[-1])

    def stream(self, ref_codes: Any = None) -> "StreamDecoder":
        return StreamDecoder(self, ref_codes)


class StreamDecoder:
    """One line's incremental decode: :meth:`push` frames as the talker makes
    them, :meth:`decode` the ones not yet decoded. Clone mode seeds it with
    the reference clip's frames, which reach the front but never the output."""

    def __init__(self, split: SplitDecoder, ref_codes: Any = None) -> None:
        self.split = split
        self._frames: list = []
        if ref_codes is not None and len(ref_codes):
            self._frames.append(ref_codes)
        self._total = sum(int(f.shape[0]) for f in self._frames)
        self._decoded = self._total  # frames whose audio has been returned
        self._codes: Any = None

    @property
    def pending(self) -> int:
        return self._total - self._decoded

    def push(self, frames: Any) -> None:
        """Frames of codes ``[n, quantizers]`` (the talker's ``[1, groups]``)."""
        self._frames.append(frames)
        self._total += int(frames.shape[0])

    def decode(self) -> np.ndarray:
        """float32 audio for the frames pushed since the last call."""
        start, end = self._decoded, self._total
        if end == start:
            return np.zeros(0, dtype=np.float32)
        torch = self.split.torch
        with torch.inference_mode():
            parts = ([] if self._codes is None else [self._codes]) + self._frames
            device = parts[-1].device
            self._codes = torch.cat([p.to(device) for p in parts], dim=0)
            self._frames = []
            codes = torch.clamp(self._codes, min=0).transpose(0, 1)[None]
            audio = self.split.window(self.split.front(codes), start, end)
        self._decoded = end
        return audio


def split_decoder(tts: Any, *, torch: Any, context: int = CONV_CONTEXT) -> SplitDecoder | None:
    """A :class:`SplitDecoder` over ``tts``'s 12 Hz speech decoder, or None
    for another qwen-tts version, tokenizer or module layout."""
    version = _qwen_tts_version()
    if version not in SUPPORTED_QWEN_TTS:
        logger.info("qwen-tts %s: streaming speech decode not available (checked: %s)",
                    version, ", ".join(sorted(SUPPORTED_QWEN_TTS)))
        return None
    try:
        tokenizer = tts.model.speech_tokenizer
        model = tokenizer.model
        if model.get_model_type() != "qwen3_tts_tokenizer_12hz":
            return None
        dec = model.decoder
        ok = (callable(dec.quantizer.decode) and callable(dec.pre_conv)
              and callable(dec.pre_transformer) and int(dec.total_upsample) > 0
              and all(callable(block) for blocks in dec.upsample for block in blocks)
              and all(callable(block) for block in dec.decoder)
              and int(dec.config.num_quantizers) > 0)
    except (AttributeError, TypeError, ValueError):
        ok = False
    if not ok:
        logger.warning("unexpected qwen-tts speech decoder layout; streaming not available")
        return None
    return SplitDecoder(dec, torch=torch, context=context, device=getattr(tokenizer, "device", None))
