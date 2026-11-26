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
