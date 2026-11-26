"""Triton kernels for the graphed Qwen3-TTS path; CUDA + Triton only."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from muse.modalities.audio_speech.runtimes import qwen3_tts_triton as k  # noqa: E402


class RMSNorm(torch.nn.Module):  # Qwen3TTSRMSNorm
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(n))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_matches_module_on_rows_and_strided_heads(dtype):
    torch.manual_seed(0)
    norm = RMSNorm(128).to("cuda", dtype)
    hidden = RMSNorm(1024).to("cuda", dtype)
    fused = torch.randn(1, 3, 4096, device="cuda", dtype=dtype)
    heads = fused[..., :2048].view(1, 3, 16, 128)  # q sliced from a fused QKV
    x = torch.randn(1, 3, 1024, device="cuda", dtype=dtype)
    with torch.no_grad():
        for mod, inp in ((norm, heads), (hidden, x)):
            want, got = mod(inp), k.rms_norm(mod, inp)
            assert got.shape == want.shape
            tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
            assert torch.allclose(got.float(), want.float(), atol=tol, rtol=tol)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_rope_matches_norm_then_rope(dtype):
    torch.manual_seed(1)
    norm = RMSNorm(128).to("cuda", dtype)
    fused = torch.randn(1, 3, 4096, device="cuda", dtype=dtype)
    heads = fused[..., 2048:3072].view(1, 3, 8, 128)
    ang = torch.randn(3, 128, device="cuda")
    cos, sin = ang.cos().to(dtype), ang.sin().to(dtype)
    with torch.no_grad():
        y = norm(heads)
        want = y * cos[None, :, None] + _rotate_half(y) * sin[None, :, None]
        got = k.rms_norm_rope(norm, heads, cos, sin)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert got.shape == want.shape
    assert torch.allclose(got.float(), want.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("pos", [0, 63, 64, 200])
def test_decode_attention_matches_masked_sdpa(pos):
    torch.manual_seed(2)
    q = torch.randn(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    keys = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
    keys[:, :, pos + 1:] = 1e4  # stale slots past the live length must not leak in
    p = torch.tensor([pos], device="cuda")
    allowed = (torch.arange(512, device="cuda") <= pos)[None, None, None]
    with torch.no_grad():
        want = torch.nn.functional.scaled_dot_product_attention(
            q, keys, values, attn_mask=allowed, scale=128 ** -0.5, enable_gqa=True)
        got = k.decode_attention(q, keys, values, p, 128 ** -0.5)
    assert got.shape == want.shape
    assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_silu_mul_matches_torch(dtype):
    torch.manual_seed(3)
    gu = torch.randn(1, 2, 6144, device="cuda", dtype=dtype)
    gate, up = gu.chunk(2, dim=-1)
    with torch.no_grad():
        want = torch.nn.functional.silu(gate) * up
        got = k.silu_mul(gu)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert got.shape == want.shape
    assert torch.allclose(got.float(), want.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_rms_norm_matches_add_then_norm(dtype):
    torch.manual_seed(4)
    norm = RMSNorm(1024).to("cuda", dtype)
    x = torch.randn(1, 2, 1024, device="cuda", dtype=dtype)
    d = torch.randn(1, 2, 1024, device="cuda", dtype=dtype)
    with torch.no_grad():
        want_x = x + d
        want_y = norm(want_x)
        got_x, got_y = k.add_rms_norm(x, d, norm)
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.equal(got_x, want_x)
    assert torch.allclose(got_y.float(), want_y.float(), atol=tol, rtol=tol)
