"""Fast Qwen3-TTS code predictor (``runtimes/qwen3_tts_fast.py``).

A tiny randomly initialised stand-in mirrors the attribute layout of
qwen-tts 0.1.1's ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``
(per-step ``codec_embedding`` / ``lm_head``, a ``small_to_mtp_projection``,
Qwen3 attention with q/k RMSNorm and grouped KV heads). The fast static-KV
decoder is checked against an independent full-recompute reference written
here, so a KV-cache, RoPE or per-step-weight bug changes the tokens.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from muse.modalities.audio_speech.runtimes import qwen3_tts_fast as fast  # noqa: E402

GROUPS = 16          # num_code_groups: 1 talker code + 15 predicted codes
STEPS = GROUPS - 1
EMBED = 12           # talker hidden size (codec_embedding output dim)
HIDDEN = 8           # predictor hidden size
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 6
VOCAB = 32
LAYERS = 2


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(1.0 + 0.1 * torch.randn(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.scaling = HEAD_DIM ** -0.5
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, 16, bias=False)
        self.up_proj = nn.Linear(HIDDEN, 16, bias=False)
        self.down_proj = nn.Linear(16, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.mlp = MLP()


class Rotary(nn.Module):
    """Default RoPE, same formula as Qwen3TTSRotaryEmbedding."""

    def __init__(self):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x, position_ids):
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(LAYERS)])
        self.norm = RMSNorm(HIDDEN)
        self.rotary_emb = Rotary()
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(VOCAB, EMBED) for _ in range(STEPS)])

    def get_input_embeddings(self):
        return self.codec_embedding


class FakeCodePredictor(nn.Module):
    def __init__(self, layer_types=None):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=LAYERS, num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
            num_code_groups=GROUPS, vocab_size=VOCAB,
            layer_types=layer_types or ["full_attention"] * LAYERS,
            _attn_implementation="sdpa",
        )
        self.model = Inner()
        self.lm_head = nn.ModuleList(
            [nn.Linear(HIDDEN, VOCAB, bias=False) for _ in range(STEPS)])
        self.small_to_mtp_projection = nn.Linear(EMBED, HIDDEN)
        self.upstream_calls: list[dict] = []

    def generate(self, **kw):  # the upstream HF generate this replaces
        self.upstream_calls.append(kw)
        b = kw["inputs_embeds"].shape[0]
        return SimpleNamespace(sequences=torch.zeros(b, STEPS, dtype=torch.long))


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def reference_logits(cp, embeds, tokens):
    """Teacher-forced logits for each of the 15 steps, recomputing the whole
    sequence every step (no cache) with eager causal attention."""
    out = []
    seq = list(embeds.unbind(1))
    for step in range(STEPS):
        if step > 0:
            seq.append(cp.model.codec_embedding[step - 1](tokens[:, step - 1]))
        x = cp.small_to_mtp_projection(torch.stack(seq, dim=1))
        b, n, _ = x.shape
        pos = torch.arange(n)[None].expand(b, n)
        cos, sin = cp.model.rotary_emb(x, pos)
        cos, sin = cos[:, None], sin[:, None]
        for layer in cp.model.layers:
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(b, n, HEADS, HEAD_DIM)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(b, n, KV_HEADS, HEAD_DIM)).transpose(1, 2)
            v = a.v_proj(h).view(b, n, KV_HEADS, HEAD_DIM).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
            w = (q @ k.transpose(-1, -2)) * a.scaling
            w = w.masked_fill(torch.ones(n, n).triu(1).bool(), float("-inf"))
            att = (w.softmax(-1) @ v).transpose(1, 2).reshape(b, n, -1)
            x = x + a.o_proj(att)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = cp.model.norm(x)
        out.append(cp.lm_head[step](x[:, -1]).float())
    return out


def _gen(cp, embeds, **kw):
    args = dict(inputs_embeds=embeds, max_new_tokens=STEPS, do_sample=False,
                top_p=1.0, top_k=50, temperature=0.9,
                output_hidden_states=True, return_dict_in_generate=True)
    args.update(kw)
    return fast.FastCodePredictor(cp, torch=torch).generate(**args)


@pytest.fixture
def cp():
    torch.manual_seed(0)
    model = FakeCodePredictor()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)  # sharper logits so greedy ties are impossible
    return model


@pytest.fixture
def embeds():
    torch.manual_seed(1)
    return torch.randn(3, 2, EMBED)


def test_greedy_matches_full_recompute_reference(cp, embeds):
    with torch.no_grad():
        seq = _gen(cp, embeds).sequences
        assert seq.shape == (3, STEPS) and seq.dtype == torch.long
        ref = reference_logits(cp, embeds, seq)
    for step, logits in enumerate(ref):
        assert torch.equal(seq[:, step], logits.argmax(-1)), step
    assert cp.upstream_calls == []


def test_top_k_one_sampling_is_greedy(cp, embeds):
    with torch.no_grad():
        greedy = _gen(cp, embeds).sequences
        sampled = _gen(cp, embeds, do_sample=True, top_k=1, temperature=0.7).sequences
    assert torch.equal(greedy, sampled)


def test_sampled_tokens_stay_inside_top_k(cp, embeds):
    torch.manual_seed(2)
    with torch.no_grad():
        for _ in range(5):
            seq = _gen(cp, embeds, do_sample=True, top_k=3, temperature=1.5).sequences
            ref = reference_logits(cp, embeds, seq)
            for step, logits in enumerate(ref):
                allowed = logits.topk(3, dim=-1).indices
                assert (allowed == seq[:, step:step + 1]).any(-1).all(), step


def test_sampler_follows_softmax_with_temperature():
    torch.manual_seed(3)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]]).expand(40000, 4)
    toks = fast.sample_next(logits, torch=torch, do_sample=True,
                            temperature=0.5, top_k=0, top_p=1.0)
    freq = torch.bincount(toks, minlength=4).float() / toks.numel()
    expected = (logits[0] / 0.5).softmax(-1)
    assert torch.allclose(freq, expected, atol=0.01)


def test_sampler_top_p_drops_the_tail_like_hf():
    torch.manual_seed(4)
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()[None].expand(20000, 4)
    toks = fast.sample_next(logits, torch=torch, do_sample=True,
                            temperature=1.0, top_k=0, top_p=0.75)
    counts = torch.bincount(toks, minlength=4)
    # HF TopP removes tokens whose ascending cumulative prob <= 1 - top_p.
    assert counts[2] == 0 and counts[3] == 0
    assert abs(counts[0].item() / 20000 - 0.5 / 0.8) < 0.02


def test_greedy_ignores_temperature_and_top_k():
    logits = torch.tensor([[0.0, 3.0, 1.0], [5.0, 0.0, 0.0]])
    toks = fast.sample_next(logits, torch=torch, do_sample=False,
                            temperature=0.1, top_k=1, top_p=0.5)
    assert toks.tolist() == [1, 0]


@pytest.mark.parametrize("kw", [
    {"max_new_tokens": 3},
    {"inputs_embeds": torch.zeros(1, 3, EMBED)},
])
def test_unexpected_shapes_delegate_to_upstream(cp, kw):
    out = _gen(cp, torch.zeros(1, 2, EMBED), **kw)
    assert len(cp.upstream_calls) == 1
    assert out.sequences.shape == (1, STEPS)


def test_fast_path_failure_falls_back_to_upstream_and_stays_off(cp, embeds, monkeypatch):
    pred = fast.FastCodePredictor(cp, torch=torch)

    def boom(*a, **k):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(pred, "_run", boom)
    kw = dict(inputs_embeds=embeds, max_new_tokens=STEPS, do_sample=False,
              top_p=1.0, top_k=50, temperature=0.9)
    pred.generate(**kw)
    pred.generate(**kw)
    assert len(cp.upstream_calls) == 2
    assert pred.disabled


def _tts(cp):
    return SimpleNamespace(model=SimpleNamespace(talker=SimpleNamespace(code_predictor=cp)))


def test_install_patches_generate_on_supported_version(cp, embeds, monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    assert fast.install(_tts(cp), torch=torch) is True
    assert isinstance(cp.generate.__self__, fast.FastCodePredictor)
    with torch.no_grad():
        cp.generate(inputs_embeds=embeds, max_new_tokens=STEPS, do_sample=False,
                    top_p=1.0, top_k=50, temperature=0.9)
    assert cp.upstream_calls == []
    assert fast.install(_tts(cp), torch=torch) is True  # idempotent
    assert not isinstance(cp.generate.__self__.upstream.__self__, fast.FastCodePredictor)


@pytest.mark.parametrize("version", ["0.1.2", "0.2.0", None])
def test_install_refuses_unsupported_qwen_tts(cp, monkeypatch, version):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: version)
    assert fast.install(_tts(cp), torch=torch) is False
    assert "generate" not in vars(cp)


def test_install_refuses_unexpected_structure(monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    sliding = FakeCodePredictor(layer_types=["full_attention", "sliding_attention"])
    assert fast.install(_tts(sliding), torch=torch) is False
    assert fast.install(SimpleNamespace(), torch=torch) is False
