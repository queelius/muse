"""Split speech decoder for streaming (``runtimes/qwen3_tts_stream.py``).

A tiny stand-in mirrors the module layout of qwen-tts 0.1.1's
``Qwen3TTSTokenizerV2Decoder``: a quantizer, a causal ``pre_conv``, a causal
``pre_transformer`` (here a running mean, so every frame depends on all the
frames before it, as the real transformer's long memory does), then
``upsample`` and ``decoder`` blocks with a short receptive field and the final
clamp; the conv stack reaches 8 samples back (within 3 frames). Streaming must
reproduce the one-pass decode of the same codes.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from muse.modalities.audio_speech.runtimes import qwen3_tts_stream as stream  # noqa: E402

Q = 4          # quantizers (codebook groups)
BINS = 11      # codebook size
DIM = 6        # quantizer output
LATENT = 5     # pre_conv / transformer width
UP = 3         # total upsampling, samples per frame


class CausalConv(nn.Module):
    """Left-padded Conv1d: output t sees inputs t-k+1..t (like qwen-tts's)."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k)
        self.pad = k - 1

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.books = nn.ModuleList([nn.Embedding(BINS, DIM) for _ in range(Q)])

    def decode(self, codes):  # [1, Q, T] -> [1, DIM, T]
        return sum(book(codes[:, i]) for i, book in enumerate(self.books)).transpose(1, 2)


class RunningMean(nn.Module):
    """Causal with unbounded memory: frame t is the mean of frames 0..t."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(LATENT, LATENT)

    def forward(self, inputs_embeds=None):
        x = inputs_embeds
        n = torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
        return SimpleNamespace(last_hidden_state=self.proj(x.cumsum(1) / n))


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose1d(LATENT, LATENT, UP, UP)

    def forward(self, x):
        return self.conv(x)


class FakeSpeechDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=Q)
        self.total_upsample = UP
        self.quantizer = Quantizer()
        self.pre_conv = CausalConv(DIM, LATENT, 3)
        self.pre_transformer = RunningMean()
        self.upsample = nn.ModuleList([nn.ModuleList([Upsample(), CausalConv(LATENT, LATENT, 4)])])
        self.decoder = nn.ModuleList([CausalConv(LATENT, LATENT, 3), nn.Tanh(),
                                      CausalConv(LATENT, 1, 4)])

    def forward(self, codes):  # upstream's Qwen3TTSTokenizerV2Decoder.forward
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


class FakeTokenizerModel(nn.Module):
    def __init__(self, kind="qwen3_tts_tokenizer_12hz"):
        super().__init__()
        self.decoder = FakeSpeechDecoder()
        self.kind = kind

    def get_model_type(self):
        return self.kind


def _tts(model=None):
    return SimpleNamespace(model=SimpleNamespace(speech_tokenizer=SimpleNamespace(
        model=model if model is not None else FakeTokenizerModel(), device="cpu")))


@pytest.fixture
def decoder():
    torch.manual_seed(0)
    dec = FakeSpeechDecoder()
    with torch.no_grad():
        for p in dec.parameters():
            p.mul_(2.0)
    return dec


def _codes(frames, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, BINS, (frames, Q), generator=g)


def _one_pass(dec, codes):
    with torch.no_grad():
        return dec(codes.T[None])[0, 0].float().numpy()


def test_front_then_conv_stack_is_the_decoder(decoder):
    split = stream.SplitDecoder(decoder, torch=torch)
    codes = _codes(20).T[None]
    with torch.no_grad():
        assert torch.equal(split.conv_stack(split.front(codes)), decoder(codes))
    assert split.upsample == UP


@pytest.mark.parametrize("ref_frames", [0, 7], ids=["design", "clone"])
@pytest.mark.parametrize("schedule", [[1] * 30, [4, 8, 8, 8, 2], [30]])
def test_streamed_chunks_equal_the_one_pass_decode(decoder, ref_frames, schedule):
    split = stream.SplitDecoder(decoder, torch=torch, context=3)
    ref = _codes(ref_frames, seed=2) if ref_frames else None
    gen = _codes(30)
    want = _one_pass(decoder, gen if ref is None else torch.cat((ref, gen)))[ref_frames * UP:]
    sd = split.stream(ref)
    chunks, at = [], 0
    for size in schedule:
        for frame in gen[at:at + size]:
            sd.push(frame[None])
        at += size
        assert sd.pending == size
        chunk = sd.decode()
        assert isinstance(chunk, np.ndarray) and chunk.dtype == np.float32
        assert len(chunk) == size * UP and sd.pending == 0
        chunks.append(chunk)
    got = np.concatenate(chunks)
    assert len(got) == len(want) == 30 * UP
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_conv_stack_sees_only_the_new_frames_plus_context(decoder, monkeypatch):
    split = stream.SplitDecoder(decoder, torch=torch, context=3)
    seen = []
    real = split.conv_stack
    monkeypatch.setattr(split, "conv_stack", lambda h: seen.append(h.shape[-1]) or real(h))
    sd = split.stream(_codes(7, seed=2))
    for size in (4, 8, 3):
        for frame in _codes(size, seed=size):
            sd.push(frame[None])
        sd.decode()
    assert seen == [4 + 3, 8 + 3, 3 + 3]


def test_decode_with_nothing_pending_is_empty(decoder):
    sd = stream.SplitDecoder(decoder, torch=torch).stream()
    assert len(sd.decode()) == 0
    sd.push(_codes(3))  # several frames at once
    assert sd.pending == 3 and len(sd.decode()) == 3 * UP


def test_whole_line_decode_skips_the_reference_conv_stack(decoder, monkeypatch):
    split = stream.SplitDecoder(decoder, torch=torch, context=3)
    ref, gen = _codes(7, seed=2), _codes(12)
    codes = torch.cat((ref, gen))
    want = _one_pass(decoder, codes)[7 * UP:]
    seen = []
    real = split.conv_stack
    monkeypatch.setattr(split, "conv_stack", lambda h: seen.append(h.shape[-1]) or real(h))
    got = split.decode(codes, skip=7)
    assert seen == [12 + 3]
    np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(split.decode(codes), _one_pass(decoder, codes), atol=1e-6)


def test_split_decoder_installs_only_on_the_checked_layout(monkeypatch):
    monkeypatch.setattr(stream, "_qwen_tts_version", lambda: "0.1.1")
    split = stream.split_decoder(_tts(), torch=torch)
    assert isinstance(split, stream.SplitDecoder) and split.device == "cpu"
    assert stream.split_decoder(_tts(FakeTokenizerModel("qwen3_tts_tokenizer_25hz")),
                                torch=torch) is None
    broken = FakeTokenizerModel()
    del broken.decoder.pre_transformer
    assert stream.split_decoder(_tts(broken), torch=torch) is None
    assert stream.split_decoder(SimpleNamespace(), torch=torch) is None
    monkeypatch.setattr(stream, "_qwen_tts_version", lambda: "0.2.0")
    assert stream.split_decoder(_tts(), torch=torch) is None
