"""Triton kernels for the graphed Qwen3-TTS decode path (CUDA only).

Imported lazily by :mod:`.qwen3_tts_fast` when CUDA and Triton are present;
importing this module requires both. Each kernel reproduces one upstream
module's arithmetic in fewer launches (a per-frame decode runs thousands of
tiny kernels, so launch count, not FLOPs, dominates).
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, heads, stride_t, stride_h, n, eps,
                     BLOCK: tl.constexpr):
    # One program per row. Row r is (token r // heads, head r % heads), so a
    # per-head norm reads a strided view (q / k sliced from a fused QKV GEMM).
    row = tl.program_id(0)
    base = (row // heads) * stride_t + (row % heads) * stride_h
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n
    # Qwen3TTSRMSNorm: normalise in fp32, round to the input dtype, then
    # multiply by the weight (a bf16 product, i.e. fp32 rounded once).
    xn = (x * tl.math.rsqrt(var + eps)).to(y_ptr.dtype.element_ty)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (w.to(tl.float32) * xn.to(tl.float32)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * n + cols, y, mask=mask)


def rms_norm(module: Any, x: Any) -> Any:
    """``module(x)`` for a Qwen3TTSRMSNorm over the last dim, as one kernel.

    ``x`` may be ``[..., n]`` contiguous, or ``[batch, tokens, heads, n]``
    with unit stride on the last dim. Returns a contiguous tensor."""
    n = x.shape[-1]
    if x.stride(-1) != 1:
        x = x.contiguous()
    if x.dim() == 4:
        batch, tokens, heads, _ = x.shape
        if batch != 1 and x.stride(0) != tokens * x.stride(1):
            x = x.contiguous()
        stride_t, stride_h = x.stride(1), x.stride(2)
        rows = batch * tokens * heads
    else:
        x = x.contiguous()
        heads, stride_t, stride_h = 1, n, 0
        rows = x.numel() // n
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rms_norm_kernel[(rows,)](x, module.weight, y, heads, stride_t, stride_h, n,
                              float(module.variance_epsilon),
                              BLOCK=triton.next_power_of_2(n))
    return y


@triton.jit
def _norm_rope_row(x_ptr, w_ptr, cos_ptr, sin_ptr, n, eps, dt, BLOCK: tl.constexpr):
    # One head's RMSNorm then rotary embedding (x_ptr / cos_ptr / sin_ptr at
    # the row), rounding where upstream does: norm -> dtype, * weight ->
    # dtype, then (y * cos) and (rotate_half(y) * sin) each -> dtype, and
    # their sum -> dtype.
    cols = tl.arange(0, BLOCK)
    half = n // 2
    mask = cols < n
    perm = tl.where(cols < half, cols + half, cols - half)
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + perm, mask=mask, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / n + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    wp = tl.load(w_ptr + perm, mask=mask, other=0.0).to(tl.float32)
    y = (w * (x * inv).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    yp = (wp * (xp * inv).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    rot = tl.where(cols < half, -yp, yp)
    c = tl.load(cos_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    return ((y * c).to(dt).to(tl.float32) + (rot * s).to(dt).to(tl.float32)).to(dt)


@triton.jit
def _rms_norm_rope_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, heads, tokens,
                          stride_t, stride_h, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    t = row // heads
    cols = tl.arange(0, BLOCK)
    out = _norm_rope_row(x_ptr + t * stride_t + (row % heads) * stride_h, w_ptr,
                         cos_ptr + (t % tokens) * n, sin_ptr + (t % tokens) * n, n, eps,
                         y_ptr.dtype.element_ty, BLOCK)
    tl.store(y_ptr + row * n + cols, out, mask=cols < n)


@triton.jit
def _norm_rope_store_kernel(k_ptr, v_ptr, w_ptr, cos_ptr, sin_ptr, keys_ptr, values_ptr,
                            pos_ptr, heads, stride_t, stride_h, stride_vt, stride_vh,
                            stride_ch, stride_cs, n, eps, BLOCK: tl.constexpr):
    # Token t, kv head h: rope(norm(k)) and v into cache slot pos + t.
    row = tl.program_id(0)
    t = row // heads
    h = row % heads
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    slot = h * stride_ch + (tl.load(pos_ptr) + t) * stride_cs
    out = _norm_rope_row(k_ptr + t * stride_t + h * stride_h, w_ptr, cos_ptr + t * n,
                         sin_ptr + t * n, n, eps, keys_ptr.dtype.element_ty, BLOCK)
    tl.store(keys_ptr + slot + cols, out, mask=mask)
    v = tl.load(v_ptr + t * stride_vt + h * stride_vh + cols, mask=mask)
    tl.store(values_ptr + slot + cols, v, mask=mask)


def rms_norm_rope(module: Any, x: Any, cos: Any, sin: Any) -> Any:
    """``rope(module(x))`` for per-head ``x`` ``[batch, tokens, heads, n]``
    (unit last-dim stride) and ``cos`` / ``sin`` ``[tokens, n]``; one kernel.
    Returns a contiguous ``[batch, tokens, heads, n]``."""
    batch, tokens, heads, n = x.shape
    if x.stride(-1) != 1 or (batch != 1 and x.stride(0) != tokens * x.stride(1)):
        x = x.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rms_norm_rope_kernel[(batch * tokens * heads,)](
        x, module.weight, cos, sin, y, heads, tokens, x.stride(1), x.stride(2), n,
        float(module.variance_epsilon), BLOCK=triton.next_power_of_2(n))
    return y


def norm_rope_store(module: Any, k: Any, v: Any, cos: Any, sin: Any, keys: Any,
                    values: Any, pos: Any) -> None:
    """Write ``rope(module(k))`` and ``v`` (``[1, tokens, kv_heads, dim]``, unit
    last-dim stride) for token ``t`` into slot ``pos + t`` of ``keys`` /
    ``values`` (``[1, kv_heads, capacity, dim]``); ``pos`` is a 1-element
    device tensor and ``cos`` / ``sin`` are ``[tokens, dim]``. One kernel."""
    _, tokens, heads, n = k.shape
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    _norm_rope_store_kernel[(tokens * heads,)](
        k, v, module.weight, cos, sin, keys, values, pos, heads, k.stride(1), k.stride(2),
        v.stride(1), v.stride(2), keys.stride(1), keys.stride(2), n,
        float(module.variance_epsilon), BLOCK=triton.next_power_of_2(n))


@triton.jit
def _decode_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, pos_ptr, stride_qh,
                             stride_kh, stride_kc, group, scale,
                             D: tl.constexpr, BLOCK: tl.constexpr):
    # One query token per head over keys 0..pos (read on device, so a CUDA
    # graph replays it at any length); online softmax in fp32.
    h = tl.program_id(0)
    kvh = h // group
    length = tl.load(pos_ptr) + 1
    d = tl.arange(0, D)
    q = tl.load(q_ptr + h * stride_qh + d).to(tl.float32) * scale
    m = tl.full((), float("-inf"), tl.float32)
    total = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    base = kvh * stride_kh
    for start in range(0, length, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < length
        ptrs = base + offs[:, None] * stride_kc + d[None, :]
        k = tl.load(k_ptr + ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.where(valid, tl.sum(k * q[None, :], axis=1), float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m - m_new)
        v = tl.load(v_ptr + ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        total = total * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    tl.store(o_ptr + h * D + d, (acc / total).to(o_ptr.dtype.element_ty))


def decode_attention(q: Any, keys: Any, values: Any, pos: Any, scale: float) -> Any:
    """Attention of one query token ``q`` ``[1, heads, 1, dim]`` over cache
    slots ``0..pos`` of ``keys`` / ``values`` ``[1, kv_heads, capacity, dim]``
    (``pos`` a 1-element device tensor). Returns ``[1, heads, 1, dim]``."""
    _, heads, _, dim = q.shape
    kv_heads = keys.shape[1]
    if keys.stride(-1) != 1 or values.stride() != keys.stride():
        keys, values = keys.contiguous(), values.contiguous()
    if q.stride(-1) != 1:
        q = q.contiguous()
    out = torch.empty((1, heads, 1, dim), dtype=q.dtype, device=q.device)
    _decode_attention_kernel[(heads,)](
        q, keys, values, out, pos, q.stride(1), keys.stride(1), keys.stride(2),
        heads // kv_heads, float(scale), D=dim, BLOCK=64)
    return out


@triton.jit
def _silu_mul_kernel(gu_ptr, y_ptr, n, BLOCK: tl.constexpr):
    # silu(gate) * up over a fused [gate | up] row, rounding where upstream
    # does (the activation's output, then the product).
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dt = y_ptr.dtype.element_ty
    g = tl.load(gu_ptr + row * 2 * n + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * n + n + offs, mask=mask, other=0.0).to(tl.float32)
    act = (g / (1.0 + tl.exp(-g))).to(dt).to(tl.float32)
    tl.store(y_ptr + row * n + offs, (act * u).to(dt), mask=mask)


def silu_mul(gate_up: Any) -> Any:
    """``silu(gate) * up`` for ``gate_up`` ``[..., 2 * n]`` = ``[gate | up]``."""
    gate_up = gate_up.contiguous()
    n = gate_up.shape[-1] // 2
    rows = gate_up.numel() // (2 * n)
    y = torch.empty((*gate_up.shape[:-1], n), dtype=gate_up.dtype, device=gate_up.device)
    block = 1024
    _silu_mul_kernel[(rows, triton.cdiv(n, block))](gate_up, y, n, BLOCK=block)
    return y


@triton.jit
def _add_rms_norm_kernel(x_ptr, d_ptr, w_ptr, xo_ptr, y_ptr, n, eps,
                         BLOCK: tl.constexpr):
    # Residual add (rounded to the dtype, as upstream's add) followed by the
    # next RMSNorm; writes both the new residual and its normalised form.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    dt = y_ptr.dtype.element_ty
    x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
    xs = (x + d).to(dt)
    tl.store(xo_ptr + row * n + cols, xs, mask=mask)
    x32 = xs.to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x32 * x32, axis=0) / n + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (w * (x32 * inv).to(dt).to(tl.float32)).to(dt)
    tl.store(y_ptr + row * n + cols, y, mask=mask)


def add_rms_norm(x: Any, delta: Any, module: Any) -> tuple[Any, Any]:
    """``(x + delta, module(x + delta))`` for an RMSNorm over the last dim."""
    x, delta = x.contiguous(), delta.contiguous()
    n = x.shape[-1]
    out = torch.empty_like(x)
    y = torch.empty_like(x)
    _add_rms_norm_kernel[(x.numel() // n,)](x, delta, module.weight, out, y, n,
                                            float(module.variance_epsilon),
                                            BLOCK=triton.next_power_of_2(n))
    return out, y


@triton.jit
def _gemv_kernel(w_ptr, x_ptr, y_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # y = W @ x for one row x: BLOCK_N outputs per program, fp32 accumulation,
    # one rounding at the end (as cuBLAS). Weights stream through once, so
    # they are marked evict-first to leave L2 to the activations.
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(0, K, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None],
                    other=0.0, eviction_policy="evict_first")
        x = tl.load(x_ptr + cols)
        acc += w.to(tl.float32) * x.to(tl.float32)[None, :]
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _gemv_rows_kernel(w_ptr, x_ptr, y_ptr, M, N, K, ROWS: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # As _gemv_kernel for 2..4 rows of x: each weight tile is read once and
    # applied to every row (a two-token step costs about one GEMV, where
    # cuBLAS's small-M GEMM costs 10-25% more).
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    m = tl.arange(0, ROWS)
    mmask = m < M
    acc = tl.zeros((ROWS, BLOCK_N, BLOCK_K), tl.float32)
    for start in range(0, K, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None],
                    other=0.0, eviction_policy="evict_first")
        x = tl.load(x_ptr + m[:, None] * K + cols[None, :], mask=mmask[:, None], other=0.0)
        acc += w.to(tl.float32)[None, :, :] * x.to(tl.float32)[:, None, :]
    tl.store(y_ptr + m[:, None] * N + rows[None, :],
             tl.sum(acc, axis=2).to(y_ptr.dtype.element_ty),
             mask=mmask[:, None] & rmask[None, :])


# (BLOCK_N, BLOCK_K, num_warps, num_stages) by input width, tuned on an
# RTX 3060 for the Qwen3-TTS decode shapes; other widths use the default.
_GEMV_CONFIGS = {1024: (2, 512, 2, 3), 2048: (4, 256, 4, 3), 3072: (8, 128, 8, 3)}
_GEMV_DEFAULT = (4, 256, 4, 3)
_GEMV_ROWS_CONFIGS = {1024: (2, 512, 2, 3), 2048: (4, 256, 2, 2), 3072: (4, 256, 2, 3)}
_GEMV_ROWS_DEFAULT = (2, 256, 2, 3)
_GEMV_MAX_ROWS = 4  # more rows than this are a GEMM: F.linear


def linear(x: Any, weight: Any) -> Any:
    """``F.linear(x, weight)`` (no bias). Up to four rows of ``x`` use a
    Triton GEMV, which beats cuBLAS on these bandwidth-bound shapes;
    anything else is ``F.linear``."""
    n, kdim = weight.shape
    m = x.numel() // kdim if x.shape[-1] == kdim else 0
    table, default = ((_GEMV_CONFIGS, _GEMV_DEFAULT) if m == 1
                      else (_GEMV_ROWS_CONFIGS, _GEMV_ROWS_DEFAULT))
    block_n, block_k, warps, stages = table.get(kdim, default)
    if not 0 < m <= _GEMV_MAX_ROWS or kdim % block_k or weight.stride() != (kdim, 1):
        return torch.nn.functional.linear(x, weight)
    y = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, block_n),)
    if m == 1:
        _gemv_kernel[grid](weight, x.contiguous(), y, n, kdim, BLOCK_N=block_n,
                           BLOCK_K=block_k, num_warps=warps, num_stages=stages)
    else:
        _gemv_rows_kernel[grid](weight, x.contiguous(), y, m, n, kdim,
                                ROWS=triton.next_power_of_2(m), BLOCK_N=block_n,
                                BLOCK_K=block_k, num_warps=warps, num_stages=stages)
    return y


@triton.jit
def _sample_top_k_kernel(logits_ptr, noise_ptr, out_ptr, vocab, stride,
                         temperature, top_k, BLOCK: tl.constexpr):
    # One row per program: scale by the temperature, keep the top_k largest
    # (ties with the k-th kept, as HF's processor), then Gumbel-max with the
    # given Exp(1) noise: argmax(p / E) == argmax(scaled - log E) over kept.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < vocab
    raw = tl.load(logits_ptr + row * stride + cols, mask=valid, other=float("-inf"))
    scaled = raw.to(tl.float32) / temperature
    # k-th largest by binary search on order-preserving integer keys
    bits = scaled.to(tl.int32, bitcast=True)
    keys = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64)
    lo = tl.full((), -2147483648, tl.int64)  # count(keys >= lo) >= top_k always
    hi = tl.full((), 2147483648, tl.int64)   # count(keys >= hi) == 0 < top_k
    for _ in range(33):
        mid = (lo + hi) // 2
        enough = tl.sum((keys >= mid).to(tl.int32), axis=0) >= top_k
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid)
    kept = valid & (keys >= lo)
    noise = tl.load(noise_ptr + row * vocab + cols, mask=valid, other=1.0)
    score = tl.where(kept, scaled - tl.log(tl.maximum(noise, 1e-10)), float("-inf"))
    tl.store(out_ptr + row, tl.argmax(score, axis=0).to(tl.int64))


def sample_top_k(logits: Any, noise: Any, temperature: float, top_k: int) -> Any:
    """Token per row of ``logits`` ``[batch, vocab]``: HF's temperature and
    top-k processors, then a draw by Gumbel-max with Exp(1) ``noise``
    (same shape, fp32). Returns int64 ``[batch]``."""
    batch, vocab = logits.shape
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    out = torch.empty(batch, dtype=torch.int64, device=logits.device)
    _sample_top_k_kernel[(batch,)](logits, noise.contiguous(), out, vocab, logits.stride(0),
                                   float(temperature),
                                   min(int(top_k), vocab), BLOCK=triton.next_power_of_2(vocab),
                                   num_warps=4)
    return out
