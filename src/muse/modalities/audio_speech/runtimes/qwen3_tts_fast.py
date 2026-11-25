"""Fast code predictor for qwen-tts 0.1.1 (Qwen3-TTS 12 Hz checkpoints).

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

import importlib.metadata
import logging
from types import SimpleNamespace
from typing import Any

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
    logits = logits.float()
    if not do_sample:
        return logits.argmax(-1)
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


class FastCodePredictor:
    """Drop-in for ``code_predictor.generate`` on the talker's call shape."""

    def __init__(self, code_predictor: Any, *, torch: Any,
                 use_cuda_graphs: bool = False, upstream: Any = None) -> None:
        self.cp = code_predictor
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
            return sample_next(cp.lm_head[step](hidden), torch=torch,
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
        c, s = cos[start:end][None, None], sin[start:end][None, None]
        rep = self.heads // self.kv_heads
        for i, layer in enumerate(self.cp.model.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_norm(attn.q_proj(h).view(batch, n, self.heads, self.head_dim))
            k = attn.k_norm(attn.k_proj(h).view(batch, n, self.kv_heads, self.head_dim))
            v = attn.v_proj(h).view(batch, n, self.kv_heads, self.head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * c + _rotate_half(q, torch) * s
            k = k * c + _rotate_half(k, torch) * s
            keys[i][:, :, start:end] = k
            values[i][:, :, start:end] = v
            k_all = keys[i][:, :, :end]
            v_all = values[i][:, :, :end]
            if rep > 1:
                k_all = k_all.repeat_interleave(rep, dim=1)
                v_all = v_all.repeat_interleave(rep, dim=1)
            out = sdpa(q, k_all, v_all, is_causal=n > 1, scale=attn.scaling)
            x = x + attn.o_proj(out.transpose(1, 2).reshape(batch, n, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.cp.model.norm(x)[:, -1]

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


def install(tts: Any, *, torch: Any, use_cuda_graphs: bool = False) -> bool:
    """Patch ``tts.model.talker.code_predictor.generate``; True when active."""
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
                             upstream=current)
    cp.generate = fast.generate
    return True
