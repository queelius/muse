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
