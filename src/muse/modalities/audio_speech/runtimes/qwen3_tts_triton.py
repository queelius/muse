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
