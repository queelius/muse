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
# Padded shapes (CUDA graphs): the front runs on codes padded at the end to a
# bucket length (64 frames, doubling, up to MAX_FRONT_FRAMES; eager beyond),
# the conv stack on widths rounded up to CONV_WIDTH_STEP frames. Both halves
# are causal, so padding at the end leaves the real frames' outputs alone.
FRONT_BUCKET = 64
MAX_FRONT_FRAMES = 2048
CONV_WIDTH_STEP = 4
# Distinct graphs kept per decoder; beyond this, decode eagerly.
MAX_GRAPHS = 24


def _qwen_tts_version() -> str | None:
    try:
        return importlib.metadata.version("qwen-tts")
    except importlib.metadata.PackageNotFoundError:
        return None


class SplitDecoder:
    """A 12 Hz speech decoder as its causal front and its conv stack."""

    def __init__(self, decoder: Any, *, torch: Any, context: int = CONV_CONTEXT,
                 device: Any = None, use_cuda_graphs: bool = False,
                 padded: bool = False) -> None:
        self.decoder = decoder
        self.torch = torch
        self.context = int(context)
        self.upsample = int(decoder.total_upsample)
        self.device = device
        # Streaming runs each half as a CUDA graph per padded shape (the
        # front eagerly costs ~23 ms of kernel launches whatever its length,
        # graphed ~3 ms). ``padded`` alone pads without graphs (tests).
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.padded = bool(padded) or self.use_cuda_graphs
        self._graphs: dict[tuple, tuple[Any, Any, Any]] = {}
        self._pool: Any = None

    def front(self, codes: Any) -> Any:
        """Codes ``[streams, quantizers, frames]`` -> transformer output
        ``[streams, channels, frames]``: upstream's forward up to the conv
        stack. Rows are independent (one length; pad shorter rows at the end)."""
        dec = self.decoder
        hidden = dec.quantizer.decode(codes)
        hidden = dec.pre_conv(hidden).transpose(1, 2)
        hidden = dec.pre_transformer(inputs_embeds=hidden).last_hidden_state
        return hidden.permute(0, 2, 1)

    def conv_stack(self, hidden: Any) -> Any:
        """Front output ``[streams, channels, frames]`` -> audio
        ``[streams, 1, samples]``; rows independent."""
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def window(self, hidden: Any, start: int, end: int, *, padded: bool = False) -> np.ndarray:
        """Audio for frames ``[start, end)`` of a front output, decoding
        ``context`` frames before ``start`` and dropping their samples."""
        lo = max(0, start - self.context)
        part = hidden[..., lo:end]
        wav = self._conv(part) if padded else self.conv_stack(part)
        wav = wav[0, 0][(start - lo) * self.upsample:(end - lo) * self.upsample]
        return wav.to(self.torch.float32).cpu().numpy()

    def decode(self, codes: Any, *, skip: int = 0) -> np.ndarray:
        """A whole line ``[frames, quantizers]`` in one call, without the
        first ``skip`` frames' audio (the reference clip in clone mode),
        whose conv stack is not run. Eager, unpadded."""
        with self.torch.inference_mode():
            codes = self.torch.clamp(codes, min=0).transpose(0, 1)[None]
            return self.window(self.front(codes), skip, codes.shape[-1])

    def stream_window(self, codes: Any, start: int, end: int) -> np.ndarray:
        """Streaming: the front over ``codes`` ``[1, quantizers, end]`` (padded
        and graphed where enabled), then audio for frames ``[start, end)``."""
        if self.padded and end <= MAX_FRONT_FRAMES:
            bucket = FRONT_BUCKET
            while bucket < end:
                bucket *= 2
            hidden = self._shaped("front", self.front, codes, bucket)[..., :end]
        else:
            hidden = self.front(codes)
        return self.window(hidden, start, end, padded=self.padded)

    # -- padded shapes and CUDA graphs ----------------------------------------------

    def _conv(self, hidden: Any) -> Any:
        width = -(-hidden.shape[-1] // CONV_WIDTH_STEP) * CONV_WIDTH_STEP
        return self._shaped("conv", self.conv_stack, hidden, width)

    def _shaped(self, kind: str, fn: Any, x: Any, size: int) -> Any:
        """``fn(x)`` with ``x`` zero-padded at the end of its last axis to
        ``size``; on CUDA one graph per shape. The result is the graph's
        output buffer: use it before the next call."""
        if x.shape[-1] < size:
            x = self.torch.nn.functional.pad(x, (0, size - x.shape[-1]))
        if not (self.use_cuda_graphs and getattr(x, "is_cuda", False)):
            return fn(x)
        key = (kind, tuple(x.shape), x.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= MAX_GRAPHS:
                return fn(x)
            try:
                entry = self._capture(fn, x)
            except Exception as exc:
                if type(exc).__name__ == "OutOfMemoryError":
                    raise
                logger.warning("Qwen3-TTS speech decoder graph capture failed; "
                               "streaming decodes eagerly", exc_info=True)
                self.use_cuda_graphs = False
                self._graphs.clear()
                self._pool = None
                return fn(x)
            self._graphs[key] = entry
        graph, static_in, static_out = entry
        static_in.copy_(x)
        graph.replay()
        return static_out

    def _capture(self, fn: Any, x: Any) -> tuple[Any, Any, Any]:
        torch = self.torch
        static_in = x.clone()
        side = torch.cuda.Stream(device=x.device)
        side.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm cuDNN / allocator before capture
                fn(static_in)
        torch.cuda.current_stream(x.device).wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = fn(static_in)
        logger.info("captured Qwen3-TTS speech decoder CUDA graph for %s", tuple(x.shape))
        return graph, static_in, static_out

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
            audio = self.split.stream_window(codes, start, end)
        self._decoded = end
        return audio


def split_decoder(tts: Any, *, torch: Any, context: int = CONV_CONTEXT,
                  use_cuda_graphs: bool = False) -> SplitDecoder | None:
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
    return SplitDecoder(dec, torch=torch, context=context, use_cuda_graphs=use_cuda_graphs,
                        device=getattr(tokenizer, "device", None))
