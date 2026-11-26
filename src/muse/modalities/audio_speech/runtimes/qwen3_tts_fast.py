"""Fast decoding for qwen-tts 0.1.1 (Qwen3-TTS 12 Hz checkpoints).

Two drop-ins replace qwen-tts's two nested transformers ``generate()`` loops:
:class:`FastCodePredictor` (below) and :class:`FastTalker` (the outer loop,
which on CUDA runs each 12 Hz frame, code predictor included, as one CUDA
graph; see its docstring). :func:`install` and :func:`install_talker` patch
them in.

For every 12 Hz codec frame, qwen-tts's talker calls
``talker.code_predictor.generate(...)``: a full transformers ``generate()``
loop that predicts the frame's 15 remaining codebook tokens from a fixed
2-token prefill. The Python overhead of ``generate()`` (validation, logits
processors, a dynamic KV cache, unused hidden states) dominated synthesis:
about 155 ms per frame, 70% of wall time, on an RTX 3060.

:class:`FastCodePredictor` replaces that call with a fixed, unrolled 15-step
decoder over a preallocated KV cache. It reuses the predictor's own modules
(projections, RMSNorms, MLPs, rotary table, per-step embeddings and heads)
and reimplements only the attention wiring and the cache. Step ``k`` reads
``codec_embedding[k - 1]`` and ``lm_head[k]``, so the loop is unrolled rather
than one generic step. On CUDA the whole 15-step decode, sampling included,
is captured once per (batch shape, sampling parameters) as a CUDA graph and
replayed.

Sampling follows transformers' order (temperature, then top-k, then top-p)
and draws from the same distribution; the random stream differs (Gumbel-max
via exponential noise, which stays graph-capturable). Greedy decoding picks
the same argmax as upstream.

:func:`install` patches the instance's ``generate`` only for a qwen-tts
version and module layout it was checked against. Any other call shape
delegates to the original ``generate``, and an unexpected failure logs once,
switches the fast path off for the process and delegates.

torch is passed in (``torch=``) so the module imports without it.
"""
from __future__ import annotations

import functools
import importlib.metadata
import logging
from types import SimpleNamespace
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

SUPPORTED_QWEN_TTS = frozenset({"0.1.1"})
# Generated tokens beyond the 2-token prefill: positions 0..(2 + steps - 2).
_PREFILL = 2
# Distinct (shape, dtype, sampling) graphs kept; beyond this, decode eagerly.
MAX_GRAPHS = 16
# transformers' GenerationConfig defaults, used when a caller passes None.
_DEFAULTS = {"do_sample": False, "temperature": 1.0, "top_k": 50, "top_p": 1.0}
# generate() kwargs the fast path accepts and ignores (the caller only reads
# ``.sequences``).
_IGNORED_KWARGS = frozenset({"output_hidden_states", "return_dict_in_generate"})


def _qwen_tts_version() -> str | None:
    try:
        return importlib.metadata.version("qwen-tts")
    except importlib.metadata.PackageNotFoundError:
        return None


def sample_next(logits: Any, *, torch: Any, do_sample: bool, temperature: float,
                top_k: int, top_p: float) -> Any:
    """Next token per row of ``logits`` (``[batch, vocab]``), like HF generate."""
    if not do_sample:
        return logits.float().argmax(-1)
    kernels = _kernels(logits)
    if (kernels is not None and logits.dim() == 2 and (top_p is None or top_p >= 1.0)
            and temperature is not None and temperature > 0):
        # one kernel: temperature, top-k and the Gumbel-max draw on torch's noise
        noise = torch.empty(logits.shape, device=logits.device).exponential_(1)
        return kernels.sample_top_k(logits, noise, temperature, top_k or logits.shape[-1])
    logits = logits.float()
    if temperature is not None and temperature != 1.0:
        logits = logits / temperature
    neg_inf = float("-inf")
    if top_k:
        k = min(int(top_k), logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, neg_inf)
    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=False, dim=-1)
        drop = ordered.softmax(-1).cumsum(-1) <= (1 - top_p)
        drop[..., -1:] = False  # always keep the most likely token
        logits = logits.masked_fill(drop.scatter(-1, order, drop), neg_inf)
    probs = logits.softmax(-1)
    # Gumbel-max with exponential noise: argmax(p / E), E ~ Exp(1), samples
    # from p without a host sync, so it can be captured in a CUDA graph.
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return (probs / noise).argmax(-1)


def _rotate_half(x: Any, torch: Any) -> Any:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _fuse_linears(linears: list, torch: Any) -> Any:
    """One weight holding ``linears``' weights row-stacked, or None (biases).

    Each module's weight becomes a view into it, so memory does not grow and
    the original modules (upstream and the exact path) compute as before."""
    if any(getattr(lin, "bias", None) is not None for lin in linears):
        return None
    fused = torch.cat([lin.weight.detach() for lin in linears], dim=0)
    parts = fused.split([lin.weight.shape[0] for lin in linears], dim=0)
    for lin, part in zip(linears, parts):
        lin.weight = torch.nn.Parameter(part, requires_grad=False)
    return fused


_KERNELS: Any = None  # the Triton kernels module, False when unavailable


def _kernels(x: Any) -> Any:
    """:mod:`.qwen3_tts_triton` for a CUDA tensor, or None (CPU, no Triton)."""
    global _KERNELS
    if not getattr(x, "is_cuda", False):
        return None
    if _KERNELS is None:
        try:
            from muse.modalities.audio_speech.runtimes import qwen3_tts_triton
            _KERNELS = qwen3_tts_triton
        except Exception:  # no triton, or it failed to import
            logger.info("Triton unavailable; Qwen3-TTS fast path uses torch ops")
            _KERNELS = False
    return _KERNELS or None


def _norm(module: Any, x: Any) -> Any:
    """An RMSNorm module applied to ``x``: one Triton kernel on CUDA."""
    k = _kernels(x)
    return k.rms_norm(module, x) if k is not None else module(x)


def quantize_rows(weight: Any, torch: Any) -> tuple[Any, Any]:
    """Weight-only int8 for ``weight`` ``[n, k]``: ``(q, scale)`` with
    ``q * scale[:, None]`` the nearest point of each row's symmetric grid
    (the row's largest magnitude maps to 127). ``q`` int8, ``scale`` fp32."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127
    q = torch.round(w / scale[:, None]).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _linear(x: Any, weight: Any, torch: Any, quant: Any = None) -> Any:
    """``F.linear(x, weight)``; a few-row product is a Triton GEMV on CUDA,
    over ``quant`` (``weight`` from :func:`quantize_rows`) when given."""
    k = _kernels(x)
    if k is None:
        return torch.nn.functional.linear(x, weight)
    return k.linear(x, weight, quant)


def _bias_free(module: Any, torch: Any) -> bool:
    return isinstance(module, torch.nn.Linear) and module.bias is None


def _proj(module: Any, x: Any, torch: Any, quant: Any = None) -> Any:
    """``module(x)`` for a Linear; bias-free ones go through :func:`_linear`."""
    if not _bias_free(module, torch):
        return module(x)
    return _linear(x, module.weight, torch, quant)


def _add_norm(x: Any, delta: Any, module: Any) -> tuple[Any, Any]:
    """``(x + delta, module(x + delta))``: one Triton kernel on CUDA."""
    k = _kernels(x)
    if k is not None:
        return k.add_rms_norm(x, delta, module)
    x = x + delta
    return x, module(x)


def _norm_rope(module: Any, x: Any, cos: Any, sin: Any, torch: Any) -> Any:
    """Per-head RMSNorm then RoPE: ``x`` ``[batch, tokens, heads, dim]``,
    ``cos`` / ``sin`` ``[tokens, dim]``; returns ``[batch, heads, tokens, dim]``.
    One Triton kernel on CUDA."""
    k = _kernels(x)
    if k is not None:
        return k.rms_norm_rope(module, x, cos, sin).transpose(1, 2)
    y = module(x).transpose(1, 2)
    return y * cos + _rotate_half(y, torch) * sin


class _FusedLayer:
    """A decoder layer's q/k/v and gate/up projections as single GEMMs, and
    (after :meth:`quantize`) int8 copies of its projections.

    Use :meth:`of`: one per layer, shared by every fast decoder over it."""

    @classmethod
    def of(cls, layer: Any, torch: Any) -> "_FusedLayer":
        fused = layer.__dict__.get("_muse_fused")
        if fused is None:
            fused = layer.__dict__["_muse_fused"] = cls(layer, torch)
        return fused

    def __init__(self, layer: Any, torch: Any) -> None:
        attn, mlp = layer.self_attn, layer.mlp
        projections = [attn.q_proj, attn.k_proj, attn.v_proj]
        self.sizes = [p.weight.shape[0] for p in projections]
        self.qkv = _fuse_linears(projections, torch)
        act = type(getattr(mlp, "act_fn", None)).__name__
        self.gate_up = (_fuse_linears([mlp.gate_proj, mlp.up_proj], torch)
                        if act in ("SiLU", "SiLUActivation") else None)
        self.int8: dict[str, tuple[Any, Any]] = {}

    def quantize(self, layer: Any, torch: Any) -> None:
        """Make the int8 copies (:func:`quantize_rows`) the ``int8`` calls
        below read; the float weights stay for every other path."""
        if self.int8:
            return
        weights = {"qkv": self.qkv, "gate_up": self.gate_up}
        if _bias_free(layer.self_attn.o_proj, torch):
            weights["o"] = layer.self_attn.o_proj.weight
        if self.gate_up is not None and _bias_free(layer.mlp.down_proj, torch):
            weights["down"] = layer.mlp.down_proj.weight
        self.int8 = {name: quantize_rows(w, torch)
                     for name, w in weights.items() if w is not None}

    def _quant(self, name: str, int8: bool) -> Any:
        return self.int8.get(name) if int8 else None

    def project(self, layer: Any, h: Any, torch: Any, int8: bool = False) -> tuple:
        if self.qkv is None:
            attn = layer.self_attn
            return attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)
        return _linear(h, self.qkv, torch, self._quant("qkv", int8)).split(self.sizes, dim=-1)

    def attn_out(self, layer: Any, x: Any, torch: Any, int8: bool = False) -> Any:
        return _proj(layer.self_attn.o_proj, x, torch, self._quant("o", int8))

    def mlp(self, layer: Any, x: Any, torch: Any, int8: bool = False) -> Any:
        if self.gate_up is None:
            return layer.mlp(x)
        gate_up = _linear(x, self.gate_up, torch, self._quant("gate_up", int8))
        kernels = _kernels(gate_up)
        if kernels is not None:
            return _proj(layer.mlp.down_proj, kernels.silu_mul(gate_up), torch,
                         self._quant("down", int8))
        gate, up = gate_up.chunk(2, dim=-1)
        return _proj(layer.mlp.down_proj, torch.nn.functional.silu(gate) * up, torch,
                     self._quant("down", int8))


def _fused_layers(layers: Any, like: Any, int8: bool, torch: Any) -> list:
    """Each layer's :class:`_FusedLayer`, quantized for an ``int8`` decoder
    where the int8 GEMV exists (``like`` on CUDA with Triton)."""
    fused = [_FusedLayer.of(layer, torch) for layer in layers]
    if int8 and _kernels(like) is not None:
        for one, layer in zip(fused, layers):
            one.quantize(layer, torch)
    return fused


class FastCodePredictor:
    """Drop-in for ``code_predictor.generate`` on the talker's call shape."""

    def __init__(self, code_predictor: Any, *, torch: Any, use_cuda_graphs: bool = False,
                 upstream: Any = None, int8: bool = False) -> None:
        self.cp = code_predictor
        self.int8 = bool(int8)  # the layers' GEMVs read int8 weights (CUDA)
        self.torch = torch
        self.upstream = upstream if upstream is not None else code_predictor.generate
        cfg = code_predictor.config
        self.steps = int(cfg.num_code_groups) - 1
        self.heads = int(cfg.num_attention_heads)
        self.kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.length = _PREFILL + self.steps - 1  # last token is never fed back
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.disabled = False
        self._rope: dict[tuple, tuple[Any, Any]] = {}
        self._graphs: dict[tuple, tuple[Any, Any, Any]] = {}
        self._pool: Any = None
        self._fused: list | None = None
        gen_cfg = getattr(code_predictor, "generation_config", None)
        self._defaults = {k: getattr(gen_cfg, k, v) if gen_cfg is not None else v
                          for k, v in _DEFAULTS.items()}

    # -- entry point -----------------------------------------------------------

    def generate(self, inputs_embeds: Any = None, max_new_tokens: Any = None,
                 do_sample: Any = None, top_p: Any = None, top_k: Any = None,
                 temperature: Any = None, **kwargs: Any) -> Any:
        call = dict(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                    do_sample=do_sample, top_p=top_p, top_k=top_k,
                    temperature=temperature, **kwargs)
        if not self._handles(inputs_embeds, max_new_tokens, kwargs):
            return self.upstream(**call)
        params = (
            bool(self._pick("do_sample", do_sample)),
            float(self._pick("temperature", temperature)),
            int(self._pick("top_k", top_k) or 0),
            float(self._pick("top_p", top_p)),
        )
        try:
            with self.torch.no_grad():
                sequences = self._run(inputs_embeds, params)
        except Exception as exc:
            if type(exc).__name__ == "OutOfMemoryError":
                raise
            logger.warning(
                "fast Qwen3-TTS code predictor failed; using qwen-tts generate() "
                "for the rest of this process", exc_info=True)
            self.disabled = True
            return self.upstream(**call)
        return SimpleNamespace(sequences=sequences)

    def _handles(self, embeds: Any, max_new_tokens: Any, kwargs: dict) -> bool:
        return (
            not self.disabled
            and embeds is not None
            and getattr(embeds, "dim", lambda: 0)() == 3
            and embeds.shape[1] == _PREFILL
            and max_new_tokens == self.steps
            and set(kwargs) <= _IGNORED_KWARGS
        )

    def _pick(self, name: str, value: Any) -> Any:
        return self._defaults[name] if value is None else value

    def _run(self, embeds: Any, params: tuple) -> Any:
        if self.use_cuda_graphs and embeds.is_cuda:
            return self._run_graph(embeds, params)
        return self._decode(embeds, params)

    # -- CUDA graphs -----------------------------------------------------------

    def _run_graph(self, embeds: Any, params: tuple) -> Any:
        key = (tuple(embeds.shape), embeds.dtype, embeds.device, params)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= MAX_GRAPHS:
                return self._decode(embeds, params)
            entry = self._capture(embeds, params)
            self._graphs[key] = entry
        graph, static_in, static_out = entry
        static_in.copy_(embeds)
        graph.replay()
        return static_out.clone()

    def _capture(self, embeds: Any, params: tuple) -> tuple[Any, Any, Any]:
        torch = self.torch
        static_in = embeds.clone()
        side = torch.cuda.Stream(device=embeds.device)
        side.wait_stream(torch.cuda.current_stream(embeds.device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm cuBLAS / allocator before capture
                self._decode(static_in, params)
        torch.cuda.current_stream(embeds.device).wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = self._decode(static_in, params)
        logger.info("captured Qwen3-TTS code-predictor CUDA graph for batch %d",
                    embeds.shape[0])
        return graph, static_in, static_out

    # -- decoder ---------------------------------------------------------------

    def _decode(self, embeds: Any, params: tuple) -> Any:
        torch, cp = self.torch, self.cp
        do_sample, temperature, top_k, top_p = params
        batch = embeds.shape[0]
        layers = cp.model.layers
        shape = (batch, self.kv_heads, self.length, self.head_dim)
        keys = [embeds.new_zeros(shape) for _ in layers]
        values = [embeds.new_zeros(shape) for _ in layers]
        cos, sin = self._rope_table(embeds)

        def next_token(step: int, x: Any, start: int) -> Any:
            hidden = self._forward(x, start, keys, values, cos, sin)
            return sample_next(_proj(cp.lm_head[step], hidden, torch), torch=torch,
                               do_sample=do_sample, temperature=temperature,
                               top_k=top_k, top_p=top_p)

        token = next_token(0, cp.small_to_mtp_projection(embeds), 0)
        tokens = [token]
        for step in range(1, self.steps):
            fed = cp.model.codec_embedding[step - 1](token)[:, None]
            token = next_token(step, cp.small_to_mtp_projection(fed), _PREFILL + step - 1)
            tokens.append(token)
        return torch.stack(tokens, dim=1)

    def _forward(self, x: Any, start: int, keys: list, values: list,
                 cos: Any, sin: Any) -> Any:
        torch = self.torch
        sdpa = torch.nn.functional.scaled_dot_product_attention
        batch, n, _ = x.shape
        end = start + n
        c, s = cos[start:end], sin[start:end]
        rep = self.heads // self.kv_heads
        kernels = _kernels(x) if batch == 1 else None
        if kernels is not None:  # first and last slots, as device tensors for kernels
            first = self._positions(x)[start:start + 1]
            last = self._positions(x)[end - 1:end]
        layers = self.cp.model.layers
        if self._fused is None:
            self._fused = _fused_layers(layers, x, self.int8, torch)
        h = _norm(layers[0].input_layernorm, x)
        for i, layer in enumerate(layers):
            attn, fused = layer.self_attn, self._fused[i]
            q, k, v = fused.project(layer, h, torch, self.int8)
            q = _norm_rope(attn.q_norm, q.view(batch, n, self.heads, self.head_dim), c, s, torch)
            k = k.view(batch, n, self.kv_heads, self.head_dim)
            v = v.view(batch, n, self.kv_heads, self.head_dim)
            if kernels is not None:  # K and V straight into slots start..end-1
                kernels.norm_rope_store(attn.k_norm, k, v, c, s, keys[i], values[i], first)
            else:
                keys[i][:, :, start:end] = _norm_rope(attn.k_norm, k, c, s, torch)
                values[i][:, :, start:end] = v.transpose(1, 2)
            if kernels is not None and n == 1:
                out = kernels.decode_attention(q, keys[i], values[i], last, attn.scaling)
            else:
                out = sdpa(q, keys[i][:, :, :end], values[i][:, :, :end], is_causal=n > 1,
                           scale=attn.scaling, enable_gqa=rep > 1)
            out = out.transpose(1, 2).reshape(batch, n, -1)
            x, post = _add_norm(x, fused.attn_out(layer, out, torch, self.int8),
                                layer.post_attention_layernorm)
            following = (layers[i + 1].input_layernorm if i + 1 < len(layers)
                         else self.cp.model.norm)
            x, h = _add_norm(x, fused.mlp(layer, post, torch, self.int8), following)
        return h[:, -1]

    def _positions(self, like: Any) -> Any:
        """0..length-1 on ``like``'s device (cached: graphs hold its address)."""
        table = self._rope.get(("positions", like.device))
        if table is None:
            table = self._rope[("positions", like.device)] = self.torch.arange(
                self.length, device=like.device)
        return table

    def _rope_table(self, like: Any) -> tuple[Any, Any]:
        """cos / sin for positions 0..length-1 from the model's own rotary module."""
        key = (like.device, like.dtype)
        table = self._rope.get(key)
        if table is None:
            positions = self.torch.arange(self.length, device=like.device)[None]
            probe = like.new_zeros(1, self.length, self.head_dim)
            cos, sin = self.cp.model.rotary_emb(probe, positions)
            table = self._rope[key] = (cos[0], sin[0])
        return table


# -- talker -------------------------------------------------------------------

# First KV capacity (tokens) of the talker cache; doubles when a line needs more.
TALKER_CAPACITY = 512
# talker.generate() kwargs qwen-tts 0.1.1 passes (Qwen3TTSForConditionalGeneration.generate).
_TALKER_KWARGS = frozenset({
    "inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed",
    "max_new_tokens", "min_new_tokens", "do_sample", "top_k", "top_p", "temperature",
    "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature",
    "eos_token_id", "repetition_penalty", "suppress_tokens",
    "output_hidden_states", "return_dict_in_generate",
})
# GenerationConfig fields that would add logits processors or change decoding
# beyond what FastTalker implements; each must keep its default.
_TALKER_FIXED_CONFIG = (
    "num_beams", "num_beam_groups", "penalty_alpha", "guidance_scale",
    "no_repeat_ngram_size", "encoder_no_repeat_ngram_size", "bad_words_ids",
    "force_words_ids", "sequence_bias", "forced_bos_token_id", "forced_eos_token_id",
    "begin_suppress_tokens", "exponential_decay_length_penalty", "remove_invalid_values",
    "renormalize_logits", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff",
    "watermarking_config", "prefill_chunk_size", "stop_strings", "diversity_penalty",
    "encoder_repetition_penalty", "low_memory",
)


class _TalkerParams(NamedTuple):
    """Per-request decoding parameters; hashable, so they key the CUDA graphs."""

    sample: tuple          # (do_sample, temperature, top_k, top_p)
    repetition_penalty: float
    min_new: int
    eos: int
    suppress: tuple        # sorted token ids
    sub: tuple             # code predictor's (do_sample, temperature, top_k, top_p)


class _TalkerState:
    """Preallocated decode state for batch 1. CUDA graphs capture these
    tensors' addresses, so a request resets them in place."""

    def __init__(self, fast: "FastTalker", like: Any, capacity: int) -> None:
        torch = fast.torch
        self.capacity = capacity
        device = like.device
        shape = (1, fast.kv_heads, capacity, fast.head_dim)
        self.keys = [like.new_zeros(shape) for _ in range(fast.layers)]
        self.values = [like.new_zeros(shape) for _ in range(fast.layers)]
        self.positions = torch.arange(capacity, device=device)
        probe = like.new_zeros(1, capacity, fast.head_dim)
        cos, sin = fast.talker.model.rotary_emb(
            probe, self.positions[None, None].expand(3, 1, capacity))
        # Text-only positions are equal across the three M-RoPE sections, so
        # the multimodal rotation reduces to section 0's.
        self.cos, self.sin = cos[0, 0], sin[0, 0]
        self.trailing = like.new_zeros(1, capacity, fast.hidden)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.n = torch.zeros(1, dtype=torch.long, device=device)
        self.token = torch.zeros(1, dtype=torch.long, device=device)
        self.hidden = like.new_zeros(1, 1, fast.hidden)
        self.unfinished = torch.ones(1, dtype=torch.bool, device=device)
        self.seen = torch.zeros(1, fast.vocab, dtype=torch.bool, device=device)

    def scalars(self) -> list:
        return [self.pos, self.n, self.token, self.hidden, self.unfinished, self.seen]

    def matches(self, like: Any) -> bool:
        return self.trailing.device == like.device and self.trailing.dtype == like.dtype


class FastTalker:
    """Drop-in for ``talker.generate`` on qwen-tts's batch-1 call.

    Prefill runs eagerly (its length varies per request). Each later step is
    one 12 Hz frame: the 15-token code predictor, the summed codec
    embeddings plus that step's trailing text, one talker token over a static
    KV cache, transformers' logits processors (repetition penalty, min new
    tokens, suppressed tokens) and sampling. On CUDA the frame is one CUDA
    graph per (cache capacity, parameters); the host syncs once per frame on
    the stop flag.

    Without graphs, attention reads exactly the live keys, as upstream does,
    so greedy decoding reproduces upstream. A graphed frame attends over the
    whole cache with a mask, which is equal in exact arithmetic but rounds
    differently.
    """

    def __init__(self, talker: Any, code_predictor: FastCodePredictor, *, torch: Any,
                 use_cuda_graphs: bool = False, upstream: Any = None,
                 int8: bool = False) -> None:
        self.talker = talker
        self.int8 = bool(int8)  # graphed frames' layer GEMVs read int8 weights
        self.cp = code_predictor
        self.torch = torch
        self.upstream = upstream if upstream is not None else talker.generate
        cfg = talker.config
        self.layers = int(cfg.num_hidden_layers)
        self.heads = int(cfg.num_attention_heads)
        self.kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.hidden = int(cfg.hidden_size)
        self.vocab = int(cfg.vocab_size)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.disabled = False
        self._state: _TalkerState | None = None
        self._graphs: dict[tuple, tuple[Any, Any]] = {}
        self._pool: Any = None
        self._masks: dict[tuple, tuple[Any, Any]] = {}
        self._fused: list | None = None
        gen_cfg = getattr(talker, "generation_config", None)
        self._defaults = {k: getattr(gen_cfg, k, v) if gen_cfg is not None else v
                          for k, v in _DEFAULTS.items()}

    # -- entry point -----------------------------------------------------------

    def generate(self, **kwargs: Any) -> Any:
        params = self._params(kwargs)
        if params is None:
            return self.upstream(**kwargs)
        try:
            with self.torch.no_grad():
                entries = self._generate(
                    kwargs["inputs_embeds"], kwargs["trailing_text_hidden"],
                    kwargs["tts_pad_embed"], params, int(kwargs["max_new_tokens"]))
        except Exception as exc:
            if type(exc).__name__ == "OutOfMemoryError":
                raise
            logger.warning(
                "fast Qwen3-TTS talker failed; using qwen-tts generate() "
                "for the rest of this process", exc_info=True)
            self.disabled = True
            self._state = None
            self._drop_graphs()
            return self.upstream(**kwargs)
        return SimpleNamespace(hidden_states=tuple(entries))

    def _params(self, kw: dict) -> _TalkerParams | None:
        """Decoding parameters, or None when the call is not the one handled."""
        if self.disabled or not set(kw) <= _TALKER_KWARGS:
            return None
        embeds, trailing, pad = (kw.get("inputs_embeds"), kw.get("trailing_text_hidden"),
                                 kw.get("tts_pad_embed"))
        mask, eos, max_new = kw.get("attention_mask"), kw.get("eos_token_id"), kw.get("max_new_tokens")
        if not (_is_batch1(embeds, self.hidden) and embeds.shape[1] >= 2
                and _is_batch1(trailing, self.hidden) and _is_batch1(pad, self.hidden)
                and pad.shape[1] == 1):
            return None
        if kw.get("output_hidden_states") is False or kw.get("return_dict_in_generate") is False:
            return None
        if not isinstance(eos, int) or isinstance(eos, bool) or not isinstance(max_new, int) or max_new < 1:
            return None
        if mask is not None and (tuple(mask.shape) != tuple(embeds.shape[:2]) or not bool(mask.all())):
            return None
        sample = (
            bool(self._pick("do_sample", kw.get("do_sample"))),
            float(self._pick("temperature", kw.get("temperature"))),
            int(self._pick("top_k", kw.get("top_k")) or 0),
            float(self._pick("top_p", kw.get("top_p"))),
        )
        sub = (
            bool(self.cp._pick("do_sample", kw.get("subtalker_dosample"))),
            float(self.cp._pick("temperature", kw.get("subtalker_temperature"))),
            int(self.cp._pick("top_k", kw.get("subtalker_top_k")) or 0),
            float(self.cp._pick("top_p", kw.get("subtalker_top_p"))),
        )
        penalty = kw.get("repetition_penalty")
        suppress = tuple(sorted({int(t) for t in kw.get("suppress_tokens") or ()}))
        if any(not 0 <= t < self.vocab for t in suppress) or not 0 <= eos < self.vocab:
            return None
        return _TalkerParams(sample, 1.0 if penalty is None else float(penalty),
                             int(kw.get("min_new_tokens") or 0), eos, suppress, sub)

    def _pick(self, name: str, value: Any) -> Any:
        return self._defaults[name] if value is None else value

    # -- generation loop -------------------------------------------------------

    def _generate(self, embeds: Any, trailing: Any, pad: Any, p: _TalkerParams,
                  max_new: int) -> list:
        """The ``hidden_states`` tuple HF generate returns, as qwen-tts reads
        it: entry 0 is ``((prefill hidden,), None)``, entry k is
        ``((hidden_k,), codec ids of frame k-1)``."""
        length, text = embeds.shape[1], trailing.shape[1]
        st = self._ready_state(embeds, max(length, text) + 1)
        st.trailing[:, :text] = trailing
        st.trailing[:, text:] = pad
        st.seen.zero_()
        st.unfinished.fill_(True)
        st.n.zero_()
        hidden = self._layers(embeds, st, start=0, masked=False)
        st.hidden.copy_(hidden[:, -1:])
        st.pos.fill_(length)
        self._choose(self.talker.codec_head(hidden)[:, -1].float(), st, p)
        entries = [((st.hidden.clone(),), None)]
        pos, produced = length, 1
        while produced < max_new and bool(st.unfinished):
            if pos >= st.capacity:
                st = self._grow(st, 2 * st.capacity)
            codes = self._step(st, p, pos)
            entries.append(((st.hidden.clone(),), codes.clone()))
            pos, produced = pos + 1, produced + 1
        return entries

    def _step(self, st: _TalkerState, p: _TalkerParams, pos: int) -> Any:
        if not (self.use_cuda_graphs and st.trailing.is_cuda):
            return self._frame(st, p, start=pos, masked=False)
        key = (st.capacity, p)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= MAX_GRAPHS:
                return self._frame(st, p, start=pos, masked=False)
            entry = self._graphs[key] = self._capture(st, p, pos)
        graph, codes = entry
        graph.replay()
        return codes

    def _capture(self, st: _TalkerState, p: _TalkerParams, pos: int) -> tuple[Any, Any]:
        torch = self.torch
        device = st.trailing.device
        saved = [t.clone() for t in st.scalars()]
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm cuBLAS / allocator; the KV slot written is unread
                self._frame(st, p, start=pos, masked=True)
                for live, copy in zip(st.scalars(), saved):
                    live.copy_(copy)
        torch.cuda.current_stream(device).wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            codes = self._frame(st, p, start=pos, masked=True)
        logger.info("captured Qwen3-TTS talker CUDA graph (KV capacity %d)", st.capacity)
        return graph, codes

    # -- one frame -------------------------------------------------------------

    def _frame(self, st: _TalkerState, p: _TalkerParams, *, start: int, masked: bool) -> Any:
        """Expand the last talker token into a frame's 16 codes, then predict
        the next talker token. Returns the frame's codes ``[1, groups]``."""
        torch = self.torch
        token = st.token[:, None]
        last = self.talker.model.codec_embedding(token)
        pair = torch.cat((st.hidden, last), dim=1)
        sub = self.cp._decode(pair, p.sub) if masked else self.cp._run(pair, p.sub)
        # Built now: ``token`` views st.token, which _choose overwrites.
        codes = torch.cat((token, sub), dim=-1)
        embeds = [last] + [emb(sub[..., i:i + 1])
                           for i, emb in enumerate(self.cp.cp.model.codec_embedding)]
        x = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        # Step g = n - 1 adds trailing text row g; rows past the text hold the pad.
        x = x + st.trailing.index_select(1, (st.n - 1).clamp(max=st.capacity - 1))
        hidden = self._layers(x, st, start=start, masked=masked)
        st.hidden.copy_(hidden)
        st.pos.add_(1)
        head = self.talker.codec_head
        scores = _proj(head, hidden, torch) if masked else head(hidden)
        self._choose(scores[:, -1].float(), st, p)
        return codes

    def _choose(self, scores: Any, st: _TalkerState, p: _TalkerParams) -> None:
        """transformers' processors in its order, then sampling; updates the
        token, stop flag, repetition set and token count in place."""
        torch = self.torch
        eos_mask, suppress_mask = self._token_masks(scores.device, p)
        if p.repetition_penalty != 1.0:
            rp = p.repetition_penalty
            scores = torch.where(st.seen, torch.where(scores < 0, scores * rp, scores / rp), scores)
        if p.min_new > 0:
            scores = scores.masked_fill(eos_mask & (st.n < p.min_new), float("-inf"))
        if suppress_mask is not None:
            scores = scores.masked_fill(suppress_mask, float("-inf"))
        do_sample, temperature, top_k, top_p = p.sample
        token = sample_next(scores, torch=torch, do_sample=do_sample,
                            temperature=temperature, top_k=top_k, top_p=top_p)
        token = torch.where(st.unfinished, token, p.eos)
        st.unfinished.logical_and_(token != p.eos)
        st.seen.scatter_(1, token[:, None], True)
        st.token.copy_(token)
        st.n.add_(1)

    def _token_masks(self, device: Any, p: _TalkerParams) -> tuple[Any, Any]:
        key = (device, p.eos, p.suppress)
        masks = self._masks.get(key)
        if masks is None:
            ids = self.torch.arange(self.vocab, device=device)
            suppress = None
            if p.suppress:
                suppress = self.torch.isin(ids, self.torch.tensor(p.suppress, device=device))
            masks = self._masks[key] = (ids == p.eos, suppress)
        return masks

    def _layers(self, x: Any, st: _TalkerState, *, start: int, masked: bool) -> Any:
        """Talker decoder layers plus final norm over ``x`` (``[1, n, hidden]``)
        at cache positions ``start..``, writing K/V into the cache.

        ``masked`` (a graphed frame, ``n == 1``) reads the position from
        ``st.pos`` and attends over the whole cache with a mask; otherwise
        attention reads exactly the first ``start + n`` keys, as upstream."""
        torch = self.torch
        sdpa = torch.nn.functional.scaled_dot_product_attention
        batch, n, _ = x.shape
        rep = self.heads // self.kv_heads
        kernels = _kernels(x) if masked else None
        if masked:
            cos = st.cos.index_select(0, st.pos)
            sin = st.sin.index_select(0, st.pos)
            allowed = None if kernels is not None else (st.positions <= st.pos)[None, None, None]
        else:
            end = start + n
            cos, sin = st.cos[start:end][None, None], st.sin[start:end][None, None]
        layers = self.talker.model.layers
        if masked and self._fused is None:
            self._fused = _fused_layers(layers, x, self.int8, torch)
        if masked:
            h = _norm(layers[0].input_layernorm, x)
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            if masked:  # graphed path: fused projections, norms and RoPE
                q, k, v = self._fused[i].project(layer, h, torch, self.int8)
                q = _norm_rope(attn.q_norm, q.view(batch, n, self.heads, self.head_dim),
                               cos, sin, torch)
                k = k.view(batch, n, self.kv_heads, self.head_dim)
                v = v.view(batch, n, self.kv_heads, self.head_dim)
                if kernels is not None:  # K and V go straight to slot pos;
                    kernels.norm_rope_store(attn.k_norm, k, v, cos, sin, st.keys[i],
                                            st.values[i], st.pos)
                    # attention reads only the live keys 0..pos
                    out = kernels.decode_attention(q, st.keys[i], st.values[i], st.pos,
                                                   attn.scaling)
                else:
                    st.keys[i].index_copy_(2, st.pos, _norm_rope(attn.k_norm, k, cos, sin, torch))
                    st.values[i].index_copy_(2, st.pos, v.transpose(1, 2))
                    out = sdpa(q, st.keys[i], st.values[i], attn_mask=allowed,
                               scale=attn.scaling, enable_gqa=rep > 1)
            else:  # exact path: the upstream modules, as upstream calls them
                h = layer.input_layernorm(x)
                q, k, v = attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)
                q = attn.q_norm(q.view(batch, n, self.heads, self.head_dim)).transpose(1, 2)
                k = attn.k_norm(k.view(batch, n, self.kv_heads, self.head_dim)).transpose(1, 2)
                q = q * cos + _rotate_half(q, torch) * sin
                k = k * cos + _rotate_half(k, torch) * sin
                v = v.view(batch, n, self.kv_heads, self.head_dim).transpose(1, 2)
                st.keys[i][:, :, start:end] = k
                st.values[i][:, :, start:end] = v
                out = sdpa(q, st.keys[i][:, :, :end], st.values[i][:, :, :end],
                           is_causal=n > 1, scale=attn.scaling, enable_gqa=rep > 1)
            out = out.transpose(1, 2).reshape(batch, n, -1)
            if masked:
                fused = self._fused[i]
                x, post = _add_norm(x, fused.attn_out(layer, out, torch, self.int8),
                                    layer.post_attention_layernorm)
                following = (layers[i + 1].input_layernorm if i + 1 < len(layers)
                             else self.talker.model.norm)
                x, h = _add_norm(x, fused.mlp(layer, post, torch, self.int8), following)
            else:
                x = x + attn.o_proj(out)
                x = x + layer.mlp(layer.post_attention_layernorm(x))
        return h if masked else self.talker.model.norm(x)

    # -- cache capacity --------------------------------------------------------

    def _ready_state(self, like: Any, need: int) -> _TalkerState:
        st = self._state
        if st is None or not st.matches(like):
            self._drop_graphs()
            capacity = TALKER_CAPACITY
            while capacity < need:
                capacity *= 2
            st = self._state = _TalkerState(self, like, capacity)
        elif st.capacity < need:
            capacity = st.capacity
            while capacity < need:
                capacity *= 2
            st = self._grow(st, capacity)
        return st

    def _drop_graphs(self) -> None:
        # A pool is released once every graph using it is gone; capturing into
        # the stale handle fails an allocator assert, so start a fresh pool.
        self._graphs.clear()
        self._pool = None

    def _grow(self, old: _TalkerState, capacity: int) -> _TalkerState:
        """A bigger cache holding ``old``'s contents. Graphs captured the old
        buffers, so they are dropped and recaptured on demand."""
        new = _TalkerState(self, old.trailing, capacity)
        size = old.capacity
        for dst, src in zip(new.keys + new.values, old.keys + old.values):
            dst[:, :, :size] = src
        new.trailing[:, :size] = old.trailing
        new.trailing[:, size:] = old.trailing[:, size - 1:]  # the last row is the pad
        for dst, src in zip(new.scalars(), old.scalars()):
            dst.copy_(src)
        self._drop_graphs()
        self._state = new
        logger.info("Qwen3-TTS talker KV capacity grown to %d tokens", capacity)
        return new


def _is_batch1(t: Any, hidden: int) -> bool:
    return (getattr(t, "dim", lambda: 0)() == 3 and t.shape[0] == 1
            and t.shape[-1] == hidden)


def _supported_layout(cp: Any) -> bool:
    try:
        cfg = cp.config
        model = cp.model
        layers = list(model.layers)
        steps = int(cfg.num_code_groups) - 1
        attn = layers[0].self_attn
        return (
            len(layers) == int(cfg.num_hidden_layers)
            and all(t == "full_attention" for t in cfg.layer_types)
            and int(cfg.num_attention_heads) % int(cfg.num_key_value_heads) == 0
            and len(cp.lm_head) == steps
            and len(model.codec_embedding) == steps
            and callable(cp.small_to_mtp_projection)
            and callable(model.rotary_emb)
            and all(hasattr(attn, name) for name in (
                "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "scaling"))
            and all(hasattr(layer, name) for layer in layers for name in (
                "input_layernorm", "post_attention_layernorm", "mlp"))
        )
    except (AttributeError, TypeError, ValueError, IndexError):
        return False


def install(tts: Any, *, torch: Any, use_cuda_graphs: bool = False,
            int8: bool = False) -> bool:
    """Patch ``tts.model.talker.code_predictor.generate``; True when active.
    ``int8``: the layers read int8 weights on CUDA (output heads stay float)."""
    version = _qwen_tts_version()
    if version not in SUPPORTED_QWEN_TTS:
        logger.info("qwen-tts %s: fast code predictor not installed (checked: %s)",
                    version, ", ".join(sorted(SUPPORTED_QWEN_TTS)))
        return False
    cp = getattr(getattr(getattr(tts, "model", None), "talker", None),
                 "code_predictor", None)
    if cp is None or not _supported_layout(cp):
        logger.warning("unexpected qwen-tts code predictor layout; "
                       "fast code predictor not installed")
        return False
    current = getattr(cp, "generate", None)
    if isinstance(getattr(current, "__self__", None), FastCodePredictor):
        return True
    fast = FastCodePredictor(cp, torch=torch, use_cuda_graphs=use_cuda_graphs,
                             upstream=current, int8=int8)
    cp.generate = fast.generate
    return True


def _supported_talker(talker: Any) -> bool:
    try:
        cfg = talker.config
        model = talker.model
        layers = list(model.layers)
        attn = layers[0].self_attn
        gen_cfg = talker.generation_config
        defaults = type(gen_cfg)()
        return (
            len(layers) == int(cfg.num_hidden_layers)
            and getattr(cfg, "sliding_window", None) is None
            and int(cfg.num_attention_heads) % int(cfg.num_key_value_heads) == 0
            and all(callable(getattr(obj, name, None)) for obj, name in (
                (talker, "codec_head"), (model, "norm"), (model, "rotary_emb"),
                (model, "codec_embedding")))
            and all(hasattr(attn, name) for name in (
                "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "scaling"))
            and all(hasattr(layer, name) for layer in layers for name in (
                "input_layernorm", "post_attention_layernorm", "mlp"))
            and all(getattr(gen_cfg, name, None) == getattr(defaults, name, None)
                    for name in _TALKER_FIXED_CONFIG)
        )
    except (AttributeError, TypeError, ValueError, IndexError):
        return False


def install_talker(tts: Any, *, torch: Any, use_cuda_graphs: bool = False,
                   int8: bool = False) -> bool:
    """Patch ``tts.model.talker.generate``; True when active. Needs the fast
    code predictor (:func:`install`) first. ``int8`` as :func:`install`, for
    the graphed frames (prefill reads the float weights, as upstream)."""
    version = _qwen_tts_version()
    if version not in SUPPORTED_QWEN_TTS:
        logger.info("qwen-tts %s: fast talker not installed (checked: %s)",
                    version, ", ".join(sorted(SUPPORTED_QWEN_TTS)))
        return False
    talker = getattr(getattr(tts, "model", None), "talker", None)
    cp_fast = getattr(getattr(getattr(talker, "code_predictor", None), "generate", None),
                      "__self__", None)
    if not isinstance(cp_fast, FastCodePredictor):
        logger.info("fast talker needs the fast code predictor; not installed")
        return False
    if not _supported_talker(talker):
        logger.warning("unexpected qwen-tts talker layout or generation config; "
                       "fast talker not installed")
        return False
    current = getattr(talker, "generate", None)
    if isinstance(getattr(current, "__self__", None), FastTalker):
        return True
    fast = FastTalker(talker, cp_fast, torch=torch, use_cuda_graphs=use_cuda_graphs,
                      upstream=current, int8=int8)
    talker.generate = fast.generate
    return True


# -- speech decoder -------------------------------------------------------------


def _snake_forward(module: Any, upstream: Any, x: Any) -> Any:
    """SnakeBeta's forward: one Triton kernel on CUDA (bit-identical), else
    the module's own."""
    kernels = _kernels(x)
    if kernels is None or x.dim() != 3 or x.shape[1] != module.alpha.numel():
        return upstream(x)
    return kernels.snake_beta(module, x)


def install_decoder(tts: Any, *, torch: Any) -> bool:
    """Route the speech decoder's SnakeBeta activations (about a third of its
    GPU time, as seven elementwise passes each) through one kernel; True when
    any are routed. Independent of the talker and code predictor patches."""
    version = _qwen_tts_version()
    if version not in SUPPORTED_QWEN_TTS:
        logger.info("qwen-tts %s: fast speech decoder not installed (checked: %s)",
                    version, ", ".join(sorted(SUPPORTED_QWEN_TTS)))
        return False
    model = getattr(getattr(getattr(tts, "model", None), "speech_tokenizer", None),
                    "model", None)
    if not callable(getattr(model, "modules", None)):
        return False
    snakes = [m for m in model.modules() if type(m).__name__ == "SnakeBeta"
              and all(hasattr(m, name) for name in ("alpha", "beta", "no_div_by_zero"))]
    for snake in snakes:
        if "_muse_upstream_forward" not in vars(snake):
            upstream = snake.forward
            vars(snake)["_muse_upstream_forward"] = upstream
            vars(snake)["forward"] = functools.partial(_snake_forward, snake, upstream)
    return bool(snakes)
