"""Several Qwen3-TTS lines per 12 Hz frame (qwen-tts 0.1.1, 12 Hz checkpoints).

:class:`BatchedTalker` keeps one decode state with a leading row dimension.
A request joins at a frame boundary: its prefill runs alone, upstream's exact
path as today, into a free row. Each frame then advances every live row by
one 12 Hz frame (code predictor, codec-embedding sum, one talker token,
transformers' processors, sampling); each row has its own position, text
cursor, repetition set, stop flag and RoPE angle, and leaves at its own EOS.
Live rows are kept in ``[0, live)`` (a leaving row's slot takes the last
row), and a frame runs the smallest bucket (1, 2, 4, ... rows) that holds
them; padding rows are idle. On CUDA each (bucket, KV capacity, parameters)
is one CUDA graph.

The one-row bucket is :meth:`FastTalker._frame` itself on a view of row 0:
a lone line runs today's frame, kernels and numerics. Wider buckets run a
masked whole-cache frame in torch ops (per-row positions); up to
``INT8_MAX_ROWS`` rows they read the int8 layers through the Triton GEMV,
wider ones the float weights kept beside them (``keep_float``), through
cuBLAS: the int8 GEMV stops scaling past ~4 rows (measured, RTX 3060).

:class:`Engine` owns the GPU on one thread: it prepares requests (qwen-tts
prompt building), admits them into rows, runs frames, decodes streams' chunks
and whole lines, and paces: when every stream is more than ``lead_seconds``
of audio ahead of its playback it waits instead of generating. It imports no
web framework; muse's route is one user of it.

torch is passed in (``torch=``) so the module imports without it.
"""
from __future__ import annotations

import collections
import contextlib
import logging
import math
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from muse.modalities.audio_speech.runtimes import qwen3_tts_fast as fast

logger = logging.getLogger(__name__)

BUCKETS = (1, 2, 4, 8, 16, 32)
# Widest bucket that never reads the kept float weights. Wider buckets run in
# fast.float_weights(), where each product of more than fast.GEMV_ROWS rows
# reads float weights (cuBLAS) and smaller ones stay on the int8 GEMV: at 4
# rows the talker and most predictor steps are int8, the predictor's first
# step (2 positions a row, 8 rows) is float.
INT8_MAX_ROWS = 2
CAPACITY = 512  # first KV capacity (tokens); doubles when a row needs more
# Batched prefill: the prompt padded at the end to a multiple of this many
# tokens, one CUDA graph per padded length (causal: the real positions are
# unchanged), on the kept float weights when there are any.
PREFILL_STEP = 32
MAX_GRAPHS = 48
# Deadline-driven chunks: a playing line's pending frames are sent once one
# more frame plus a decode would not fit in its remaining audio (minus this);
# its first chunk is sized so the second can be at least as long.
SAFETY_SECONDS = 0.03
FIRST_MARGIN = 1.5
EMA = 0.2  # weight of the newest sample in the frame / decode time estimates
# The wait for "one more frame" is the slowest of the last WAIT_WINDOW loop
# iterations: an iteration in which other lines decode is several times an
# idle one, and a line that waits must survive it (an average was late by
# 10-30 ms at 8-16 speakers).
WAIT_WINDOW = 32


def first_chunk_frames(frame_est: float, frame_seconds: float, first_frames: int,
                       chunk_frames: int) -> int:
    """A first chunk of c frames plays for c * T; the next one needs f per
    frame to generate, so it can be as long as the first when
    c >= f / (T - f). With FIRST_MARGIN, at least ``first_frames``, at most
    ``chunk_frames`` (and the cap once the engine is slower than real time)."""
    if frame_est >= frame_seconds:
        return chunk_frames
    need = math.ceil(FIRST_MARGIN * frame_est / (frame_seconds - frame_est) - 1e-9)
    return max(first_frames, min(chunk_frames, need))


# Mean all-in frame time, as a share of a frame's audio, from which a first
# chunk is sized for the slowest recent frame (fewest decodes) instead of the
# mean (fastest first audio). Below it there is headroom for small chunks.
HEADROOM = 0.5


def first_chunk_size(*, mean: float, wait: float, decode: float, frame_seconds: float,
                     first: int, chunk: int, safety: float = SAFETY_SECONDS) -> int:
    """A line's first chunk, in frames. With headroom (mean all-in frame time
    under HEADROOM of a frame's audio): the larger of the growth term on the
    mean and a floor that covers one slow frame plus a decode; near real
    time: :func:`first_chunk_frames` on the slowest recent frame."""
    if mean >= HEADROOM * frame_seconds:
        return first_chunk_frames(max(wait, mean), frame_seconds, first, chunk)
    grow = math.ceil(FIRST_MARGIN * mean / (frame_seconds - mean) - 1e-9)
    floor = math.ceil((wait + decode + safety) / frame_seconds - 1e-9) + 1
    return max(first, min(chunk, max(grow, floor)))


def least_chunk_frames(mean: float, frame_seconds: float, chunk: int) -> int:
    """The smallest chunk a line already behind its playback may send at mean
    all-in frame time ``mean``: behind real time every line is due every
    frame, and 1-2-frame chunks (each a full decode) keep the engine there;
    c >= 1.5 f / (T - f) lets the next chunk be generated while this one
    plays. Lines not yet late go out on their deadline whatever they have."""
    if mean >= 0.9 * frame_seconds:
        return chunk
    return max(1, min(chunk, math.ceil(FIRST_MARGIN * mean / (frame_seconds - mean) - 1e-9)))


def emission_due(*, pending: int, sent: bool, lead: float, first: int, chunk: int,
                 frame_est: float, decode_est: float, least: int = 1,
                 safety: float = SAFETY_SECONDS) -> bool:
    """Whether a streaming line should decode and send its pending frames now:
    at ``chunk`` frames; before its first chunk, at ``first`` frames; after
    it, when its audio ahead of playback (``lead``, client buffer included)
    would not outlast one more frame plus a decode."""
    if pending <= 0:
        return False
    if pending >= chunk:
        return True
    if not sent:
        return pending >= first
    if lead < 0:  # already behind: a bigger chunk, not one decode per frame
        return pending >= least
    return lead - safety < frame_est + decode_est


class GrowthRefused(RuntimeError):
    """Growing the decode state (``kind``: "rows" or "capacity") ran out of
    memory; the state and its rows are unchanged."""

    def __init__(self, kind: str, cause: BaseException) -> None:
        super().__init__(f"Qwen3-TTS batched talker could not grow its {kind}: {cause}")
        self.kind, self.cause = kind, cause


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError"


def bucket_for(rows: int) -> int:
    for b in BUCKETS:
        if b >= rows:
            return b
    raise ValueError(f"{rows} rows exceeds the widest bucket {BUCKETS[-1]}")


class _State:
    """Decode state for ``rows`` rows; CUDA graphs capture these tensors'
    addresses, so rows are reset and moved in place."""

    def __init__(self, ft: Any, like: Any, rows: int, capacity: int) -> None:
        torch = ft.torch
        self.rows, self.capacity = rows, capacity
        device = like.device
        shape = (rows, ft.kv_heads, capacity, ft.head_dim)
        self.keys = [like.new_zeros(shape) for _ in range(ft.layers)]
        self.values = [like.new_zeros(shape) for _ in range(ft.layers)]
        self.positions = torch.arange(capacity, device=device)
        probe = like.new_zeros(1, capacity, ft.head_dim)
        cos, sin = ft.talker.model.rotary_emb(
            probe, self.positions[None, None].expand(3, 1, capacity))
        self.cos, self.sin = cos[0, 0], sin[0, 0]
        self.trailing = like.new_zeros(rows, capacity, ft.hidden)
        self.pos = torch.zeros(rows, dtype=torch.long, device=device)
        self.n = torch.zeros(rows, dtype=torch.long, device=device)
        self.token = torch.zeros(rows, dtype=torch.long, device=device)
        self.hidden = like.new_zeros(rows, 1, ft.hidden)
        self.unfinished = torch.zeros(rows, dtype=torch.bool, device=device)
        self.seen = torch.zeros(rows, ft.vocab, dtype=torch.bool, device=device)
        self.live = torch.zeros(rows, dtype=torch.bool, device=device)
        self.arange = torch.arange(rows, device=device)
        self._views: dict[int, _RowView] = {}

    def scalars(self) -> list:
        return [self.pos, self.n, self.token, self.hidden, self.unfinished, self.seen, self.live]

    def view(self, row: int) -> "_RowView":
        if row not in self._views:
            self._views[row] = _RowView(self, row)
        return self._views[row]

    def matches(self, like: Any) -> bool:
        return self.trailing.device == like.device and self.trailing.dtype == like.dtype


class _RowView:
    """Row ``r`` of a :class:`_State` with :class:`fast._TalkerState`'s
    attributes, so FastTalker's prefill, ``_choose`` and one-row frame run on
    it unchanged (every tensor is a view; writes land in the row)."""

    def __init__(self, st: _State, r: int) -> None:
        self.capacity = st.capacity
        self.keys = [k[r:r + 1] for k in st.keys]
        self.values = [v[r:r + 1] for v in st.values]
        self.positions, self.cos, self.sin = st.positions, st.cos, st.sin
        self.trailing = st.trailing[r:r + 1]
        self.pos, self.n, self.token = st.pos[r:r + 1], st.n[r:r + 1], st.token[r:r + 1]
        self.hidden = st.hidden[r:r + 1]
        self.unfinished, self.seen = st.unfinished[r:r + 1], st.seen[r:r + 1]

    def scalars(self) -> list:
        return [self.pos, self.n, self.token, self.hidden, self.unfinished, self.seen]


class _Scratch:
    """Batch-1 K/V for one padded prefill (FastTalker._layers' exact path
    reads ``keys`` / ``values`` / ``cos`` / ``sin``)."""

    def __init__(self, keys: list, values: list, cos: Any, sin: Any) -> None:
        self.keys, self.values, self.cos, self.sin = keys, values, cos, sin


class _Row:
    """Host bookkeeping for one live row."""

    __slots__ = ("request", "produced", "unfinished", "pos", "text")

    def __init__(self, request: Any, length: int, text: int) -> None:
        self.request = request
        self.produced = 1
        self.unfinished = True
        self.pos = length
        self.text = text  # trailing text rows written; appendable


class BatchedTalker:
    """Rows of up to ``max_rows`` lines sharing each frame (one thread)."""

    def __init__(self, ft: Any, *, max_rows: int, use_cuda_graphs: bool = False,
                 padded_prefill: bool | None = None) -> None:
        if not 1 <= max_rows <= BUCKETS[-1]:
            raise ValueError(f"max_rows must be 1..{BUCKETS[-1]}")
        self.ft = ft
        self.torch = ft.torch
        self.max_rows = int(max_rows)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.st: _State | None = None
        self.rows: list[_Row] = []
        self.buckets_used: set[int] = set()
        self._graphs: dict[tuple, tuple[Any, Any]] = {}
        self._pool: Any = None
        self._moved: tuple[int, int] | None = None
        self.padded_prefill = use_cuda_graphs if padded_prefill is None else bool(padded_prefill)
        self.prefill_lengths: set[int] = set()

    # -- rows ----------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self.st.capacity if self.st is not None else 0

    @property
    def live(self) -> int:
        return len(self.rows)

    def params(self) -> Any:
        return self.rows[0].request.params if self.rows else None

    def accepts(self, req: Any) -> bool:
        """Room for ``req``, and its parameters match the live rows'."""
        return len(self.rows) < self.max_rows and (not self.rows or req.params == self.params())

    def join(self, req: Any) -> int:
        """Prefill ``req`` into the next free row (upstream's exact path, as
        FastTalker.start); returns the row."""
        torch = self.torch
        embeds, trailing, pad = req.inputs_embeds, req.trailing_text_hidden, req.tts_pad_embed
        length, text = embeds.shape[1], trailing.shape[1]
        padded = -(-length // PREFILL_STEP) * PREFILL_STEP if self.padded_prefill else length
        st = self._ready(embeds, max(padded, text + 1, length + 1))
        r = len(self.rows)
        if r >= st.rows:  # rows grow by bucket, so VRAM follows the live streams
            st = self._rebuild(bucket_for(r + 1), st.capacity)
        view = st.view(r)
        with torch.no_grad():
            for k, v in zip(view.keys, view.values):
                k.zero_()
                v.zero_()
            view.trailing[:, :text] = trailing
            view.trailing[:, text:] = pad
            view.seen.zero_()
            view.unfinished.fill_(True)
            view.n.zero_()
            hidden = self._prefill(embeds, view, padded)
            view.hidden.copy_(hidden[:, -1:])
            view.pos.fill_(length)
            self.ft._choose(self.ft.talker.codec_head(hidden)[:, -1].float(), view, req.params)
            st.live[r] = True
        row = _Row(req, length, text)
        row.unfinished = bool(view.unfinished)
        self.rows.append(row)
        return r

    def _prefill(self, embeds: Any, view: Any, padded: int) -> Any:
        """Upstream's exact layer stack over the prompt, K/V into ``view``'s
        row; returns the hidden states of the real positions."""
        length = embeds.shape[1]
        if padded == length and not self.padded_prefill:
            return self.ft._layers(embeds, view, start=0, masked=False)
        self.prefill_lengths.add(padded)
        st = self.st
        key = ("prefill", padded, embeds.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._graphs[key] = self._prefill_entry(embeds, padded)
        graph, x, scratch, out = entry
        x.zero_()
        x[:, :length] = embeds
        if graph is not None:
            graph.replay()
        else:
            out = self._prefill_run(x, scratch)
        for dst, src in zip(view.keys + view.values, scratch.keys + scratch.values):
            dst[:, :, :length] = src[:, :, :length]
        return out[:, :length]

    def _prefill_entry(self, like: Any, padded: int) -> tuple:
        torch, ft, st = self.torch, self.ft, self.st
        x = like.new_zeros(1, padded, ft.hidden)
        shape = (1, ft.kv_heads, padded, ft.head_dim)
        scratch = _Scratch([like.new_zeros(shape) for _ in range(ft.layers)],
                           [like.new_zeros(shape) for _ in range(ft.layers)], st.cos, st.sin)
        if not (self.use_cuda_graphs and getattr(like, "is_cuda", False)):
            return None, x, scratch, None
        side = torch.cuda.Stream(device=like.device)
        side.wait_stream(torch.cuda.current_stream(like.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._prefill_run(x, scratch)
        torch.cuda.current_stream(like.device).wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self._prefill_run(x, scratch)
        logger.info("captured Qwen3-TTS batched prefill CUDA graph (%d tokens)", padded)
        return graph, x, scratch, out

    def _prefill_run(self, x: Any, scratch: Any) -> Any:
        weights = fast.float_weights() if self._has_float() else contextlib.nullcontext()
        with weights:
            return self.ft._layers(x, scratch, start=0, masked=False)

    def _has_float(self) -> bool:
        fused = self.ft._fused
        return bool(fused) and bool(fused[0].float)

    def finished(self, r: int) -> bool:
        row = self.rows[r]
        return not row.unfinished or row.produced >= row.request.max_new_tokens

    def leave(self, r: int) -> None:
        """Free row ``r``: the last live row moves into its slot."""
        st = self.st
        last = len(self.rows) - 1
        with self.torch.no_grad():
            if r != last:
                for t in st.keys + st.values:
                    t[r].copy_(t[last])
                st.trailing[r].copy_(st.trailing[last])
                for t in st.scalars():
                    t[r].copy_(t[last])
                self.rows[r] = self.rows[last]
            st.live[last] = False
            st.unfinished[last] = False
        self.rows.pop()
        self._moved = (last, r) if r != last else None

    def moved(self, r: int) -> int:
        """Where row ``r`` lives after the last :meth:`leave`."""
        if self._moved is not None and r == self._moved[0]:
            return self._moved[1]
        return r

    def frame(self) -> Any:
        """One frame for every live row: codes ``[live, groups]`` (row order),
        the graph's buffer on CUDA (copy what you keep). Call only with no
        finished rows."""
        k = len(self.rows)
        if not k:
            raise RuntimeError("no live rows")
        st = self.st
        if max(row.pos for row in self.rows) >= st.capacity:
            st = self._grow(2 * st.capacity)
        bucket = bucket_for(k)
        p = self.rows[0].request.params
        self.buckets_used.add(bucket)
        with self.torch.no_grad():
            codes = self._run(bucket, p)[:k]
        for row in self.rows:
            row.produced += 1
            row.pos += 1
        flags = st.unfinished[:k].tolist()  # the frame's one host sync
        for row, flag in zip(self.rows, flags):
            row.unfinished = bool(flag)
        return codes

    # -- frames ----------------------------------------------------------------------

    def _run(self, bucket: int, p: Any) -> Any:
        st = self.st
        if not (self.use_cuda_graphs and st.trailing.is_cuda):
            return self._frame(bucket, p)
        key = (bucket, st.capacity, p)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= MAX_GRAPHS:
                return self._frame(bucket, p)
            entry = self._graphs[key] = self._capture(bucket, p)
        graph, codes = entry
        graph.replay()
        return codes

    def _capture(self, bucket: int, p: Any) -> tuple[Any, Any]:
        torch = self.torch
        st = self.st
        device = st.trailing.device
        saved = [t.clone() for t in st.scalars()]
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm; the KV slots written are rewritten before read
                self._frame(bucket, p)
                for live, copy in zip(st.scalars(), saved):
                    live.copy_(copy)
        torch.cuda.current_stream(device).wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            codes = self._frame(bucket, p)
        logger.info("captured Qwen3-TTS batched talker CUDA graph (%d rows, KV %d)",
                    bucket, st.capacity)
        return graph, codes

    def _frame(self, bucket: int, p: Any) -> Any:
        if bucket == 1:  # today's one-row frame, on row 0
            return self.ft._frame(self.st.view(0), p, start=0, masked=True)
        weights = (fast.float_weights() if bucket > INT8_MAX_ROWS
                   else contextlib.nullcontext())
        with weights:
            return self._frame_rows(bucket, p)

    def _frame_rows(self, R: int, p: Any) -> Any:
        """FastTalker._frame for rows ``[0, R)`` with per-row positions."""
        torch, ft, st = self.torch, self.ft, self.st
        token = st.token[:R]
        last = ft.talker.model.codec_embedding(token[:, None])
        pair = torch.cat((st.hidden[:R], last), dim=1)
        sub = ft.cp._decode(pair, p.sub)
        codes = torch.cat((token[:, None], sub), dim=-1)
        embeds = [last] + [emb(sub[..., i:i + 1])
                           for i, emb in enumerate(ft.cp.cp.model.codec_embedding)]
        x = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        cursor = (st.n[:R] - 1).clamp(min=0, max=st.capacity - 1)
        x = x + st.trailing[:R].gather(1, cursor[:, None, None].expand(R, 1, ft.hidden))
        hidden = self._layers_rows(x, R)
        st.hidden[:R].copy_(hidden)
        live = st.live[:R]
        st.pos[:R].add_(live.long())
        scores = fast._proj(ft.talker.codec_head, hidden, torch)[:, -1].float()
        self._choose_rows(scores, R, p)
        return codes

    def _layers_rows(self, x: Any, R: int) -> Any:
        torch, ft, st = self.torch, self.ft, self.st
        sdpa = torch.nn.functional.scaled_dot_product_attention
        heads, kv_heads, dim = ft.heads, ft.kv_heads, ft.head_dim
        pos = st.pos[:R]
        cos = st.cos.index_select(0, pos)[:, None, None, :]
        sin = st.sin.index_select(0, pos)[:, None, None, :]
        allowed = (st.positions[None, :] <= pos[:, None])[:, None, None, :]
        rows = st.arange[:R]
        layers = ft.talker.model.layers
        if ft._fused is None:
            ft._fused = fast._fused_layers(layers, ft.int8, torch)
        h = fast._norm(layers[0].input_layernorm, x)
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            q, k, v = ft._fused[i].project(layer, h, torch)
            q = attn.q_norm(q.view(R, 1, heads, dim)).transpose(1, 2)
            k = attn.k_norm(k.view(R, 1, kv_heads, dim)).transpose(1, 2)
            q = q * cos + fast._rotate_half(q, torch) * sin
            k = k * cos + fast._rotate_half(k, torch) * sin
            v = v.view(R, 1, kv_heads, dim).transpose(1, 2)
            keys, values = st.keys[i][:R], st.values[i][:R]
            keys[rows, :, pos] = k[:, :, 0]
            values[rows, :, pos] = v[:, :, 0]
            out = sdpa(q, keys, values, attn_mask=allowed, scale=attn.scaling,
                       enable_gqa=heads > kv_heads)
            out = out.transpose(1, 2).reshape(R, 1, -1)
            x, post = fast._add_norm(x, fast._proj(attn.o_proj, out, torch),
                                     layer.post_attention_layernorm)
            following = (layers[i + 1].input_layernorm if i + 1 < len(layers)
                         else ft.talker.model.norm)
            x, h = fast._add_norm(x, ft._fused[i].mlp(layer, post, torch), following)
        return h

    def _choose_rows(self, scores: Any, R: int, p: Any) -> None:
        """FastTalker._choose per row."""
        torch, ft, st = self.torch, self.ft, self.st
        eos_mask, suppress_mask = ft._token_masks(scores.device, p)
        seen, n = st.seen[:R], st.n[:R]
        if p.repetition_penalty != 1.0:
            rp = p.repetition_penalty
            scores = torch.where(seen, torch.where(scores < 0, scores * rp, scores / rp), scores)
        if p.min_new > 0:
            scores = scores.masked_fill(eos_mask[None] & (n < p.min_new)[:, None], float("-inf"))
        if suppress_mask is not None:
            scores = scores.masked_fill(suppress_mask[None], float("-inf"))
        do_sample, temperature, top_k, top_p = p.sample
        token = fast.sample_next(scores, torch=torch, do_sample=do_sample,
                                 temperature=temperature, top_k=top_k, top_p=top_p)
        unfinished = st.unfinished[:R]
        token = torch.where(unfinished, token, p.eos)
        unfinished.logical_and_(token != p.eos)
        seen.scatter_(1, token[:, None], True)
        st.token[:R].copy_(token)
        n.add_(st.live[:R].long())

    # -- capacity --------------------------------------------------------------------

    def _ready(self, like: Any, need: int) -> _State:
        st = self.st
        if st is None or not st.matches(like):
            if self.rows:
                raise RuntimeError("a line of another dtype / device cannot join live rows")
            capacity = CAPACITY
            while capacity < need:
                capacity *= 2
            self._drop_graphs()
            try:
                self.st = _State(self.ft, like, 1, capacity)
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                self._empty_cache(like)
                raise GrowthRefused("capacity", exc) from exc
            return self.st
        if st.capacity < need:
            capacity = st.capacity
            while capacity < need:
                capacity *= 2
            return self._grow(capacity)
        return st

    def _grow(self, capacity: int) -> _State:
        return self._rebuild(self.st.rows, capacity)

    def _rebuild(self, rows: int, capacity: int) -> _State:
        """A state with ``rows`` rows and ``capacity`` KV slots holding the
        current one's contents. Graphs captured the old buffers: dropped."""
        old = self.st
        self._drop_graphs()
        try:
            new = _State(self.ft, old.trailing, rows, capacity)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            self._empty_cache(old.trailing)
            raise GrowthRefused("rows" if rows > old.rows else "capacity", exc) from exc
        self.st = None
        size, n = old.capacity, old.rows
        with self.torch.no_grad():
            for dst, src in zip(new.keys + new.values, old.keys + old.values):
                dst[:n, :, :size] = src
            new.trailing[:n, :size] = old.trailing
            new.trailing[:n, size:] = old.trailing[:, size - 1:]
            for dst, src in zip(new.scalars(), old.scalars()):
                dst[:n].copy_(src)
        del old
        self.st = new
        logger.info("Qwen3-TTS batched talker state: %d rows, KV %d tokens", rows, capacity)
        return new

    def _drop_graphs(self) -> None:
        self._graphs.clear()
        self._pool = None

    def _empty_cache(self, like: Any) -> None:
        if getattr(like, "is_cuda", False):
            self.torch.cuda.empty_cache()

    def warm(self, req: Any, rows: int, prefill_lengths: tuple = ()) -> None:
        """Allocate the state for ``rows`` rows (its bucket) and capture the
        frame graph of every bucket up to it, by joining ``req`` ``rows``
        times; then capture the padded prefill for ``prefill_lengths``. Leaves
        no rows. Only with no live rows."""
        if self.rows:
            raise RuntimeError("warm() needs an idle talker")
        with self.torch.no_grad():
            for _ in range(min(rows, self.max_rows)):
                self.join(req)
                self.frame()
                self.frame()
            while self.rows:
                self.leave(len(self.rows) - 1)
            for length in prefill_lengths:
                key = ("prefill", length, req.inputs_embeds.dtype)
                if self.padded_prefill and key not in self._graphs:
                    self._ready(req.inputs_embeds, length)
                    self._graphs[key] = self._prefill_entry(req.inputs_embeds, length)

    def reset(self) -> None:
        """Forget every row and graph (after a failure)."""
        self.rows.clear()
        self.st = None
        self._drop_graphs()


# -- engine ------------------------------------------------------------------------------


class EngineUnavailable(RuntimeError):
    """The engine cannot take this request (failed earlier, closed, or the
    request is not one the fast talker handles); use the one-at-a-time path."""


_DONE = object()


class Handle:
    """One request in the engine. Iterate it for audio chunks (streams), or
    call :meth:`result` for the whole line; :meth:`cancel` frees its row at
    the next frame. ``priority`` is carried for the scheduler (unused yet)."""

    def __init__(self, prepare: Callable[[], Any], *, stream: bool, priority: int) -> None:
        self.prepare = prepare
        self.stream = bool(stream)
        self.priority = int(priority)
        self.cancelled = threading.Event()
        self._items: queue.Queue = queue.Queue()
        self.submitted = time.perf_counter()

    def cancel(self) -> None:
        self.cancelled.set()

    def _put(self, item: Any) -> None:
        self._items.put(item)

    def __iter__(self):
        while True:
            item = self._items.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def result(self, timeout: float | None = None) -> Any:
        item = self._items.get(timeout=timeout)
        if isinstance(item, BaseException):
            raise item
        return item


class _Line:
    """A live row's request: its handle, decoder or frames, pacing clock."""

    def __init__(self, handle: Handle, request: Any, decoder: Any) -> None:
        self.handle = handle
        self.request = request
        self.decoder = decoder
        self.frames: list = []
        self.parts: list = []    # a whole line's audio, decoded chunk by chunk
        self.first = 0           # frames in this line's first chunk (load-aware)
        self.audio_s = 0.0       # audio handed out so far
        self.first_at: float | None = None
        self.error: BaseException | None = None


class Engine:
    """One thread owning a model's GPU for batched Qwen3-TTS.

    ``decoder`` is a :class:`qwen3_tts_stream.SplitDecoder` (``.stream(ref)``)
    for streams; ``decode_whole(codes, ref_codes) -> audio`` decodes a
    finished line for :meth:`Handle.result` (the one-call path's decode)."""

    def __init__(self, ft: Any, *, decoder: Any, decode_whole: Callable[[Any, Any], Any],
                 torch: Any, max_rows: int, first_frames: int = 2, chunk_frames: int = 16,
                 lead_seconds: float | None = None, frame_seconds: float = 0.08,
                 use_cuda_graphs: bool = False, buffer_seconds: float = 0.0,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.talker = BatchedTalker(ft, max_rows=max_rows, use_cuda_graphs=use_cuda_graphs)
        self.ft = ft
        self.torch = torch
        self.decoder = decoder
        self.decode_whole = decode_whole
        self.first_frames, self.chunk_frames = int(first_frames), int(chunk_frames)
        self.lead_seconds = lead_seconds
        self.frame_seconds = float(frame_seconds)
        self.buffer_seconds = float(buffer_seconds)
        self._clock = clock
        self.frame_est = 0.0   # all-in seconds per loop frame (frame + takes), EMA
        self.decode_est = 0.0  # seconds per chunk decode, EMA
        self._iterations: collections.deque = collections.deque(maxlen=WAIT_WINDOW)
        self.bucket_est: dict[int, float] = {}  # all-in seconds per frame, by bucket (EMA)
        self.failed = False
        self.frames_run = 0
        self.prefill_seconds = 0.0  # host-measured prefill time (synced), for reports
        # Wall seconds the engine thread spent per kind of work (each ends in a
        # host sync, so it approximates GPU time), and rows summed over frames.
        self.stats = {"prefill_s": 0.0, "frame_s": 0.0, "decode_s": 0.0, "idle_s": 0.0,
                      "frames": 0, "row_frames": 0}
        self._pending: list[Handle] = []
        self._jobs: list[tuple[Callable[[], Any], queue.Queue]] = []
        self._lines: list[_Line] = []
        self._cv = threading.Condition()
        self._closed = False
        self.thread_name = f"qwen3-tts-engine-{id(self):x}"
        self._thread = threading.Thread(target=self._loop, name=self.thread_name, daemon=True)
        self._thread.start()

    # -- API ---------------------------------------------------------------------------

    def submit(self, prepare: Callable[[], Any], *, stream: bool, priority: int = 0) -> Handle:
        """Queue a request; ``prepare()`` runs on the engine thread and returns
        a TalkerRequest (None: not one the fast talker handles)."""
        with self._cv:
            if self.failed or self._closed:
                raise EngineUnavailable("the batched engine is off")
            handle = Handle(prepare, stream=stream, priority=priority)
            self._pending.append(handle)
            self._cv.notify()
        return handle

    def run_on_engine(self, fn: Callable[[], Any], timeout: float | None = None) -> Any:
        """Run ``fn`` on the engine thread between frames (GPU work that must
        not overlap the engine's); returns its result or raises its error."""
        if threading.current_thread() is self._thread:
            return fn()
        done: queue.Queue = queue.Queue()
        with self._cv:
            if self._closed:
                raise EngineUnavailable("the batched engine is closed")
            self._jobs.append((fn, done))
            self._cv.notify()
        ok, value = done.get(timeout=timeout)
        if not ok:
            raise value
        return value

    def warm(self, rows: int, prepare: Callable[[], Any],
             prefill_lengths: tuple = (), timeout: float | None = None) -> None:
        """Allocate ``rows`` rows and capture the frame graphs up to their
        bucket (and the padded prefill for ``prefill_lengths``) now, instead of
        on the first joins, which would stall live lines. ``prepare()`` gives
        a representative request (sampling parameters key the graphs)."""
        def work():
            req = prepare()
            if req is None:
                raise EngineUnavailable("warm-up request not handled by the fast talker")
            self.talker.warm(req, rows, prefill_lengths)
        self.run_on_engine(work, timeout=timeout)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()
        self._thread.join(timeout=30)

    # -- loop --------------------------------------------------------------------------

    def _loop(self) -> None:
        with self.torch.no_grad():
            self._serve()

    def _serve(self) -> None:
        while True:
            with self._cv:
                while not (self._closed or self._jobs or self._lines
                           or (self._pending and not self.failed)):
                    self._cv.wait()
                if self._closed:
                    self._fail_all(EngineUnavailable("the batched engine closed"))
                    return
                jobs, self._jobs = self._jobs, []
            for fn, done in jobs:
                try:
                    done.put((True, fn()))
                except BaseException as exc:  # noqa: BLE001 - handed to the caller
                    done.put((False, exc))
            try:
                self._admit()
                self._drop_finished()
                if not self._lines:
                    continue
                t0 = time.perf_counter()
                if self._ahead():
                    self.stats["idle_s"] += time.perf_counter() - t0
                    continue
                c0 = self._clock()
                try:
                    codes = self.talker.frame()
                except GrowthRefused as refused:
                    self._overflow(refused)
                    continue
                t1 = time.perf_counter()
                self.frames_run += 1
                self.stats["frames"] += 1
                self.stats["row_frames"] += len(self._lines)
                self.stats["frame_s"] += t1 - t0
                rows = codes.clone()  # one copy for every row's frame
                for r, line in enumerate(self._lines):
                    self._take(line, rows[r:r + 1])
                self.stats["decode_s"] += time.perf_counter() - t1
                took = self._clock() - c0
                self.frame_est += EMA * (took - self.frame_est)
                self._iterations.append(took)
                b = bucket_for(len(rows))
                self.bucket_est[b] = self.bucket_est.get(b, took) + EMA * (
                    took - self.bucket_est.get(b, took))
                self._drop_finished()
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ == "OutOfMemoryError":
                    logger.warning("Qwen3-TTS batched engine out of memory", exc_info=True)
                else:
                    logger.warning("Qwen3-TTS batched engine failed; the model serves one "
                                   "request at a time from now on", exc_info=True)
                self._fail_all(exc)

    def _admit(self) -> None:
        joined: list[_Line] = []
        try:
            self._admit_pass(joined)
        finally:
            if joined:
                # a burst (several joins in one pass) is sized for the bucket
                # it lands in; a single join from the running mean
                mean = self.frame_est
                if len(joined) > 1:
                    mean = max(mean, self.bucket_est.get(bucket_for(self.talker.live), 0.0))
                first = first_chunk_size(mean=mean, wait=max(self.frame_wait, mean),
                                         decode=self.decode_est, frame_seconds=self.frame_seconds,
                                         first=self.first_frames, chunk=self.chunk_frames)
                for line in joined:
                    line.first = first

    def _admit_pass(self, joined: list) -> None:
        with self._cv:
            pending = list(self._pending)
        for handle in pending:
            if handle.cancelled.is_set():
                self._unqueue(handle)
                handle._put(_DONE if handle.stream else EngineUnavailable("cancelled"))
                continue
            if self.talker.live >= self.talker.max_rows:
                return
            req = getattr(handle, "request", None)
            if req is None:
                try:
                    req = handle.request = handle.prepare()
                except Exception as exc:  # noqa: BLE001 - the request's error, not the engine's
                    self._unqueue(handle)
                    handle._put(exc)
                    continue
                if req is None:
                    self._unqueue(handle)
                    handle._put(EngineUnavailable("not handled by the fast talker"))
                    continue
            if not self.talker.accepts(req):
                continue  # other parameters: waits for the live rows to drain
            sync = getattr(self.torch, "cuda", None)
            on_cuda = getattr(req.inputs_embeds, "is_cuda", False)
            t0 = time.perf_counter()
            try:
                self.talker.join(req)
            except GrowthRefused as refused:
                if refused.kind == "rows" and self.talker.live:
                    # no memory for more rows: cap here; the line waits for a free row
                    self.talker.max_rows = self.talker.live
                    logger.warning("Qwen3-TTS batched engine capped at %d rows (out of "
                                   "memory growing the state)", self.talker.live)
                    return
                self._unqueue(handle)
                handle._put(refused.cause)
                continue
            self._unqueue(handle)
            if on_cuda:
                sync.synchronize()
            self.prefill_seconds += time.perf_counter() - t0
            self.stats["prefill_s"] += time.perf_counter() - t0
            # whole lines decode chunk by chunk too: one long decode at their
            # end would stall every stream (8 at once: ~1 s on the 0.6B)
            dec = self.decoder.stream(req.ref_codes) if self.decoder is not None else None
            line = _Line(handle, req, dec)
            line.first = self.first_frames
            self._lines.append(line)
            joined.append(line)

    def _overflow(self, refused: GrowthRefused) -> None:
        """A live line needs more KV than memory allows: fail just the lines
        at the end of the cache, keep the rest."""
        talker = self.talker
        logger.warning("Qwen3-TTS batched engine: out of memory growing the KV cache; "
                       "failing the lines that need it", exc_info=refused.cause)
        for r, line in enumerate(self._lines):
            if talker.rows[r].pos >= talker.capacity:
                line.error = refused.cause
        self._drop_finished()

    def _unqueue(self, handle: Handle) -> None:
        with self._cv:
            if handle in self._pending:
                self._pending.remove(handle)

    def _ahead(self) -> bool:
        """Pace: every line has audio out and is more than ``lead_seconds``
        ahead of its playback. Waits a little and returns True if so."""
        if self.lead_seconds is None:
            return False
        now = self._clock()
        leads = []
        for line in self._lines:
            if line.first_at is None or not line.handle.stream:
                return False
            leads.append(line.audio_s + line.decoder.pending * self.frame_seconds
                         - (now - line.first_at))
        spare = min(leads) - self.lead_seconds
        if spare <= 0:
            return False
        with self._cv:
            if not (self._pending or self._jobs or self._closed):
                self._cv.wait(timeout=min(spare, 0.05))
        return True

    @property
    def frame_wait(self) -> float:
        """Seconds one more frame may take: the slowest recent loop iteration."""
        return max(self._iterations) if self._iterations else self.frame_est

    def lead(self, line: _Line, now: float | None = None) -> float:
        """Seconds of audio this line has handed out beyond its playback
        position, client buffer included (0 before its first chunk)."""
        if line.first_at is None:
            return 0.0
        now = self._clock() if now is None else now
        return line.audio_s - (now - line.first_at) + self.buffer_seconds

    def _take(self, line: _Line, frame: Any) -> None:
        if not line.handle.stream and line.decoder is not None:
            line.decoder.push(frame)
            if line.decoder.pending >= self.chunk_frames:
                line.parts.append(self._decode(line))
            return
        if line.handle.stream:
            line.decoder.push(frame)
            if emission_due(pending=line.decoder.pending, sent=line.first_at is not None,
                            lead=self.lead(line), first=line.first, chunk=self.chunk_frames,
                            frame_est=self.frame_wait, decode_est=self.decode_est,
                            least=least_chunk_frames(self.frame_est, self.frame_seconds,
                                                     self.chunk_frames)):
                self._emit(line)
        else:
            line.frames.append(frame)

    def _decode(self, line: _Line) -> Any:
        c0 = self._clock()
        audio = line.decoder.decode()
        self.decode_est += EMA * ((self._clock() - c0) - self.decode_est)
        return audio

    def _emit(self, line: _Line) -> None:
        frames = line.decoder.pending
        audio = self._decode(line)
        if len(audio):
            if line.first_at is None:
                line.first_at = self._clock()
            line.audio_s += frames * self.frame_seconds
            line.handle._put(audio)

    def _drop_finished(self) -> None:
        talker = self.talker
        r = 0
        while r < len(self._lines):
            line = self._lines[r]
            cancelled = line.handle.cancelled.is_set()
            if not (cancelled or line.error is not None or talker.finished(r)):
                r += 1
                continue
            if line.error is not None:
                line.handle._put(line.error)
            elif not cancelled:
                if talker.rows[r].unfinished:
                    logger.warning("Qwen3-TTS line reached its frame limit (%d) without "
                                   "ending; stopped there", line.request.max_new_tokens - 1)
                self._finish(line)
            elif line.handle.stream:
                line.handle._put(_DONE)
            else:
                line.handle._put(EngineUnavailable("cancelled"))
            talker.leave(r)
            last = len(self._lines) - 1
            self._lines[r] = self._lines[last]
            self._lines.pop()

    def _finish(self, line: _Line) -> None:
        handle = line.handle
        if handle.stream:
            if line.decoder.pending:
                self._emit(line)
            handle._put(_DONE)
            return
        if line.decoder is not None:
            if line.decoder.pending:
                line.parts.append(self._decode(line))
            parts = [p for p in line.parts if len(p)]
            handle._put(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32))
            return
        torch = self.torch
        groups = line.request.inputs_embeds.new_zeros(0).long()
        codes = (torch.cat(line.frames, dim=0) if line.frames
                 else groups.reshape(0, int(self.ft.cp.steps) + 1))
        handle._put(self.decode_whole(codes, line.request.ref_codes))

    def _fail_all(self, exc: BaseException) -> None:
        with self._cv:
            self.failed = True
            pending, self._pending = self._pending, []
        for line in self._lines:
            line.handle._put(exc)
        self._lines = []
        for handle in pending:
            handle._put(exc if not isinstance(exc, EngineUnavailable)
                        else EngineUnavailable(str(exc)))
        self.talker.reset()
