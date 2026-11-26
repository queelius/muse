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
def _rms_norm_rope_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, heads, tokens,
                          stride_t, stride_h, n, eps, BLOCK: tl.constexpr):
    # Per-head RMSNorm then rotary embedding, rounding where upstream does:
    # norm -> dtype, * weight -> dtype, then (y * cos) and
    # (rotate_half(y) * sin) each -> dtype, and their sum -> dtype.
    row = tl.program_id(0)
    t = row // heads
    base = t * stride_t + (row % heads) * stride_h
    cols = tl.arange(0, BLOCK)
    half = n // 2
    mask = cols < n
    perm = tl.where(cols < half, cols + half, cols - half)
    dt = y_ptr.dtype.element_ty
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + base + perm, mask=mask, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / n + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    wp = tl.load(w_ptr + perm, mask=mask, other=0.0).to(tl.float32)
    y = (w * (x * inv).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    yp = (wp * (xp * inv).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    rot = tl.where(cols < half, -yp, yp)
    pos = (t % tokens) * n
    c = tl.load(cos_ptr + pos + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + pos + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * c).to(dt).to(tl.float32) + (rot * s).to(dt).to(tl.float32)
    tl.store(y_ptr + row * n + cols, out.to(dt), mask=mask)


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
