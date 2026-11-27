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
    def __init__(self, dim=HIDDEN):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.scaling = HEAD_DIM ** -0.5
        self.q_proj = nn.Linear(dim, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(dim, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(dim, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, dim, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)


class MLP(nn.Module):
    def __init__(self, dim=HIDDEN):
        super().__init__()
        self.gate_proj = nn.Linear(dim, 16, bias=False)
        self.up_proj = nn.Linear(dim, 16, bias=False)
        self.down_proj = nn.Linear(16, dim, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, dim=HIDDEN):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)
        self.self_attn = Attention(dim)
        self.post_attention_layernorm = RMSNorm(dim)
        self.mlp = MLP(dim)


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
        x = reference_stack(cp.model, x, cos[:, None], sin[:, None])
        out.append(cp.lm_head[step](x[:, -1]).float())
    return out


def reference_stack(model, x, cos, sin):
    """Decoder layers plus final norm, full causal attention, no cache."""
    b, n, _ = x.shape
    for layer in model.layers:
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
    return model.norm(x)


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


# -- talker ---------------------------------------------------------------------
#
# A tiny stand-in for qwen-tts 0.1.1's Qwen3TTSTalkerForConditionalGeneration:
# talker layers at width EMBED (the predictor's input width), a codec
# embedding and head over its own vocabulary, three-section M-RoPE inputs and
# the FakeCodePredictor above. reference_talker re-implements HF generate's
# semantics for this model (full recompute every step, transformers'
# repetition-penalty / min-new-tokens / suppress-tokens processors, greedy).

TVOCAB = 40
EOS = 39
SUPPRESS = (30, 31, 32, 33)


class TalkerRotary(Rotary):
    """position_ids ``[3, batch, n]`` -> cos / sin ``[3, batch, n, head_dim]``."""

    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class TalkerInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(EMBED) for _ in range(LAYERS)])
        self.norm = RMSNorm(EMBED)
        self.rotary_emb = TalkerRotary()
        self.codec_embedding = nn.Embedding(TVOCAB, EMBED)


class GenConfig:
    """A GenerationConfig whose defaults are all None (like a fresh one)."""


class FakeTalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=LAYERS, num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, hidden_size=EMBED,
            vocab_size=TVOCAB, num_code_groups=GROUPS, sliding_window=None)
        self.model = TalkerInner()
        self.codec_head = nn.Linear(EMBED, TVOCAB, bias=False)
        self.code_predictor = FakeCodePredictor()
        self.generation_config = GenConfig()
        self.upstream_calls: list[dict] = []

    def generate(self, **kw):  # the upstream HF generate this replaces
        self.upstream_calls.append(kw)
        return SimpleNamespace(hidden_states=())


def reference_predictor(cp, pair):
    """Greedy 15 codes, one full recompute per step."""
    seq, tokens = list(pair.unbind(1)), []
    for step in range(STEPS):
        if step > 0:
            seq.append(cp.model.codec_embedding[step - 1](tokens[-1]))
        x = cp.small_to_mtp_projection(torch.stack(seq, dim=1))
        b, n, _ = x.shape
        cos, sin = cp.model.rotary_emb(x, torch.arange(n)[None].expand(b, n))
        x = reference_stack(cp.model, x, cos[:, None], sin[:, None])
        tokens.append(cp.lm_head[step](x[:, -1]).argmax(-1))
    return torch.stack(tokens, dim=1)


def reference_talker(talker, inputs_embeds, trailing_text_hidden, tts_pad_embed, *,
                     max_new, min_new=2, penalty=1.05, suppress=SUPPRESS):
    """(codes per frame ``[1, GROUPS]``, hidden per step ``[1, 1, EMBED]``)."""
    embeds, trailing, pad = inputs_embeds, trailing_text_hidden, tts_pad_embed
    model, cp = talker.model, talker.code_predictor
    rows = list(embeds[0])
    generated, codes, hiddens = [], [], []

    def forward():
        x = torch.stack(rows)[None]
        n = x.shape[1]
        cos, sin = model.rotary_emb(x, torch.arange(n)[None, None].expand(3, 1, n))
        return reference_stack(model, x, cos[0][:, None], sin[0][:, None])

    def choose(hidden):
        s = talker.codec_head(hidden)[:, -1].float()
        if generated and penalty != 1.0:
            ids = torch.tensor([generated])
            g = s.gather(1, ids)
            s = s.scatter(1, ids, torch.where(g < 0, g * penalty, g / penalty))
        if len(generated) < min_new:
            s[:, EOS] = float("-inf")
        s[:, list(suppress)] = float("-inf")
        return int(s.argmax(-1))

    hidden = forward()
    hiddens.append(hidden[:, -1:])
    generated.append(choose(hidden))
    while len(generated) < max_new and generated[-1] != EOS:
        tok = torch.tensor([[generated[-1]]])
        last = model.codec_embedding(tok)
        sub = reference_predictor(cp, torch.cat((hiddens[-1], last), dim=1))
        parts = [last] + [cp.model.codec_embedding[i](sub[:, i:i + 1]) for i in range(STEPS)]
        step = len(generated) - 1
        text = trailing[0, step] if step < trailing.shape[1] else pad[0, 0]
        rows.append(torch.cat(parts, dim=1).sum(1)[0] + text)
        codes.append(torch.cat((tok, sub), dim=-1))
        hidden = forward()
        hiddens.append(hidden[:, -1:])
        generated.append(choose(hidden))
    return codes, hiddens


@pytest.fixture
def talker():
    torch.manual_seed(5)
    model = FakeTalker()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    return model


@pytest.fixture
def prompt():
    torch.manual_seed(6)
    return dict(inputs_embeds=torch.randn(1, 5, EMBED),
                trailing_text_hidden=torch.randn(1, 3, EMBED),
                tts_pad_embed=torch.randn(1, 1, EMBED))


def _talker_kwargs(prompt, **kw):
    args = dict(prompt, attention_mask=torch.ones(1, prompt["inputs_embeds"].shape[1],
                                                  dtype=torch.long),
                max_new_tokens=12, min_new_tokens=2, do_sample=False, top_k=50,
                top_p=1.0, temperature=0.9, subtalker_dosample=False,
                subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
                eos_token_id=EOS, repetition_penalty=1.05, suppress_tokens=list(SUPPRESS),
                output_hidden_states=True, return_dict_in_generate=True)
    args.update(kw)
    return args


def _fast_talker(talker, *, masked=False, monkeypatch=None, int8=False):
    cp = fast.FastCodePredictor(talker.code_predictor, torch=torch, int8=int8)
    ft = fast.FastTalker(talker, cp, torch=torch, int8=int8)
    if masked:  # run the CUDA-graph frame (masked whole-cache attention) eagerly
        monkeypatch.setattr(ft, "_step", lambda st, p, pos: ft._frame(st, p, start=pos, masked=True))
    return ft


def _split(result):
    """What qwen-tts reads from talker.generate: frame codes and hiddens."""
    codes = [entry[-1] for entry in result.hidden_states if entry[-1] is not None]
    hiddens = [entry[0][-1][:, -1:] for entry in result.hidden_states]
    return codes, hiddens


def _assert_matches_reference(result, ref):
    codes, hiddens = _split(result)
    ref_codes, ref_hiddens = ref
    assert len(codes) == len(ref_codes) and len(hiddens) == len(ref_hiddens)
    for i, (got, want) in enumerate(zip(codes, ref_codes)):
        assert torch.equal(got, want), i
    for i, (got, want) in enumerate(zip(hiddens, ref_hiddens)):
        assert torch.allclose(got, want, atol=1e-4), i


@pytest.mark.parametrize("masked", [False, True], ids=["exact", "masked"])
def test_talker_greedy_matches_reference(talker, prompt, masked, monkeypatch):
    ft = _fast_talker(talker, masked=masked, monkeypatch=monkeypatch)
    with torch.no_grad():
        result = ft.generate(**_talker_kwargs(prompt))
        ref = reference_talker(talker, **prompt, max_new=12)
    assert len(result.hidden_states) == 12 and result.hidden_states[0][-1] is None
    _assert_matches_reference(result, ref)
    assert talker.upstream_calls == [] and talker.code_predictor.upstream_calls == []


@pytest.mark.parametrize("masked", [False, True], ids=["exact", "masked"])
def test_talker_cache_grows_mid_line(talker, prompt, masked, monkeypatch):
    monkeypatch.setattr(fast, "TALKER_CAPACITY", 4)
    ft = _fast_talker(talker, masked=masked, monkeypatch=monkeypatch)
    with torch.no_grad():
        result = ft.generate(**_talker_kwargs(prompt, max_new_tokens=10))
        ref = reference_talker(talker, **prompt, max_new=10)
    _assert_matches_reference(result, ref)
    # 4 -> 8 to hold the 5-token prefill, -> 16 at position 8 (prefill + 9 fed).
    assert ft._state.capacity == 16


def test_talker_reuses_state_across_lines(talker, prompt):
    ft = _fast_talker(talker)
    short = dict(prompt, inputs_embeds=prompt["inputs_embeds"][:, :3])
    with torch.no_grad():
        ft.generate(**_talker_kwargs(prompt, max_new_tokens=15))
        result = ft.generate(**_talker_kwargs(short, max_new_tokens=6))
        ref = reference_talker(talker, **short, max_new=6)
    _assert_matches_reference(result, ref)


def test_talker_eos_waits_for_min_new_tokens_then_stops(talker, prompt):
    with torch.no_grad():
        talker.codec_head.weight[EOS] = 50 * talker.codec_head.weight[0].sign()
        talker.model.norm.weight.abs_()
    ft = _fast_talker(talker)
    with torch.no_grad():
        result = ft.generate(**_talker_kwargs(prompt, min_new_tokens=3))
        ref = reference_talker(talker, **prompt, max_new=12, min_new=3)
    _assert_matches_reference(result, ref)
    assert len(result.hidden_states) == 4  # tokens 0-2 held back, token 3 is EOS


def test_talker_repetition_penalty_and_suppression(talker, prompt):
    ft = _fast_talker(talker)
    with torch.no_grad():
        plain = _split(ft.generate(**_talker_kwargs(prompt, repetition_penalty=1.0,
                                                    suppress_tokens=[])))[0]
        result = ft.generate(**_talker_kwargs(prompt, repetition_penalty=4.0,
                                              suppress_tokens=[7, 8, 9]))
        ref = reference_talker(talker, **prompt, max_new=12, penalty=4.0, suppress=(7, 8, 9))
    _assert_matches_reference(result, ref)
    firsts = [int(c[0, 0]) for c in _split(result)[0]]
    assert not {7, 8, 9} & set(firsts)
    assert firsts != [int(c[0, 0]) for c in plain]  # the penalty changed the line


def test_talker_sampling_respects_top_k(talker, prompt):
    torch.manual_seed(7)
    ft = _fast_talker(talker)
    with torch.no_grad():
        greedy = _split(ft.generate(**_talker_kwargs(prompt)))[0]
        top1 = _split(ft.generate(**_talker_kwargs(prompt, do_sample=True, top_k=1,
                                                   subtalker_dosample=True,
                                                   subtalker_top_k=1)))[0]
    assert all(torch.equal(a, b) for a, b in zip(greedy, top1)) and len(greedy) == len(top1)


@pytest.mark.parametrize("change", [
    {"inputs_embeds": torch.zeros(2, 5, EMBED)},
    {"attention_mask": torch.tensor([[0, 1, 1, 1, 1]])},
    {"eos_token_id": [EOS]},
    {"streamer": object()},
    {"return_dict_in_generate": False},
])
def test_talker_unhandled_calls_delegate(talker, prompt, change):
    ft = _fast_talker(talker)
    ft.generate(**_talker_kwargs(prompt, **change))
    assert len(talker.upstream_calls) == 1


def test_talker_failure_falls_back_and_stays_off(talker, prompt, monkeypatch):
    ft = _fast_talker(talker)

    def boom(*a, **k):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(ft, "_layers", boom)
    ft.generate(**_talker_kwargs(prompt))
    ft.generate(**_talker_kwargs(prompt))
    assert len(talker.upstream_calls) == 2 and ft.disabled


def _talker_tts(talker):
    return SimpleNamespace(model=SimpleNamespace(talker=talker))


def test_install_talker_needs_the_fast_predictor(talker, prompt, monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    tts = _talker_tts(talker)
    assert fast.install_talker(tts, torch=torch) is False
    assert fast.install(tts, torch=torch) is True
    assert fast.install_talker(tts, torch=torch) is True
    assert isinstance(talker.generate.__self__, fast.FastTalker)
    with torch.no_grad():
        talker.generate(**_talker_kwargs(prompt))
    assert talker.upstream_calls == []
    assert fast.install_talker(tts, torch=torch) is True  # idempotent
    assert not isinstance(talker.generate.__self__.upstream.__self__, fast.FastTalker)


def test_install_talker_refuses_extra_generation_processors(talker, monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    tts = _talker_tts(talker)
    fast.install(tts, torch=torch)
    talker.generation_config.no_repeat_ngram_size = 3
    assert fast.install_talker(tts, torch=torch) is False
    assert "generate" not in vars(talker)


def test_install_talker_refuses_unsupported_qwen_tts(talker, monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    tts = _talker_tts(talker)
    fast.install(tts, torch=torch)
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.2.0")
    assert fast.install_talker(tts, torch=torch) is False


# -- speech decoder ---------------------------------------------------------------


class SnakeBeta(nn.Module):  # name and attributes of qwen-tts's 12 Hz decoder activation
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(0.3 * torch.randn(channels))
        self.beta = nn.Parameter(0.3 * torch.randn(channels))
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)


def _decoder_tts(*modules):
    return SimpleNamespace(model=SimpleNamespace(
        speech_tokenizer=SimpleNamespace(model=nn.Sequential(*modules))))


def test_install_decoder_routes_snake_activations_and_keeps_their_output(monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    snakes = [SnakeBeta(4), SnakeBeta(4)]
    tts = _decoder_tts(nn.Conv1d(4, 4, 1), *snakes)
    x = torch.randn(1, 4, 9)
    want = [s(x) for s in snakes]
    assert fast.install_decoder(tts, torch=torch) is True
    assert all("forward" in vars(s) for s in snakes)
    with torch.no_grad():  # CPU: the upstream forward, unchanged
        assert all(torch.equal(s(x), w) for s, w in zip(snakes, want))
    patched = [vars(s)["forward"] for s in snakes]
    assert fast.install_decoder(tts, torch=torch) is True  # idempotent
    assert [vars(s)["forward"] for s in snakes] == patched


def test_install_decoder_sends_cuda_input_to_the_kernel(monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    seen = []
    kernels = SimpleNamespace(snake_beta=lambda m, x: seen.append(m) or "kernel")
    monkeypatch.setattr(fast, "_kernels", lambda x: kernels)
    snake = SnakeBeta(4)
    fast.install_decoder(_decoder_tts(snake), torch=torch)
    assert snake(torch.randn(1, 4, 9)) == "kernel" and seen == [snake]
    with pytest.raises(RuntimeError):  # channel mismatch: upstream's forward, and its error
        snake(torch.randn(1, 3, 9))
    assert seen == [snake]


@pytest.mark.parametrize("version", ["0.1.2", None])
def test_install_decoder_refuses_unsupported_qwen_tts_or_layouts(monkeypatch, version):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: version)
    assert fast.install_decoder(_decoder_tts(SnakeBeta(4)), torch=torch) is False
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    assert fast.install_decoder(_decoder_tts(nn.Conv1d(4, 4, 1)), torch=torch) is False
    assert fast.install_decoder(SimpleNamespace(), torch=torch) is False


# -- int8 weights ---------------------------------------------------------------


def test_quantize_rows_rounds_each_row_to_its_own_int8_grid():
    torch.manual_seed(0)
    w = torch.randn(6, 40) * torch.tensor([1e-3, 0.02, 1.0, 5.0, 0.3, 0.0])[:, None]
    q, scale = fast.quantize_rows(w.to(torch.bfloat16), torch)
    w = w.to(torch.bfloat16).float()
    assert q.dtype == torch.int8 and q.shape == w.shape
    assert scale.dtype == torch.float32 and scale.shape == (6,)
    assert bool(torch.isfinite(scale).all())
    assert q.abs().amax(1).tolist() == [127] * 5 + [0]  # full range per row; zero row
    err = (q.float() * scale[:, None] - w).abs()
    assert bool((err <= scale[:, None] * 0.5001).all())  # the nearest grid point


def test_quantize_rows_is_the_same_past_one_chunk_of_rows():
    torch.manual_seed(1)
    w = torch.randn(2500, 6)
    q, scale = fast.quantize_rows(w, torch)
    want_scale = w.abs().amax(1) / 127
    assert torch.equal(scale, want_scale)
    assert torch.equal(q, torch.round(w / want_scale[:, None]).to(torch.int8))


def _with_silu(model):
    """qwen-tts's MLPs name their activation; the fused gate/up needs it."""
    for layer in model.layers:
        layer.mlp.act_fn = nn.SiLU()


def _projections(layer):
    attn, mlp = layer.self_attn, layer.mlp
    return [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
            mlp.gate_proj, mlp.up_proj, mlp.down_proj]


def _spy_linears(monkeypatch):
    calls = []
    real = fast._linear

    def spy(x, weight, torch_, quant=None):
        calls.append((weight, quant))
        return real(x, weight, torch_, quant)
    monkeypatch.setattr(fast, "_linear", spy)
    return calls


def test_quantize_stores_a_layer_as_int8_only_and_its_modules_still_compute(cp):
    _with_silu(cp.model)
    layer = cp.model.layers[0]
    modules = _projections(layer)
    torch.manual_seed(3)
    xs = [torch.randn(2, 7, m.in_features) for m in modules]
    floats = [m.weight.detach().clone() for m in modules]
    with torch.no_grad():
        want = [m(x) for m, x in zip(modules, xs)]
        fused = fast._FusedLayer.of(layer, torch)
        fused.quantize(layer, torch)
        got = [m(x) for m, x in zip(modules, xs)]  # as upstream calls them
    assert set(fused.int8) == {"qkv", "o", "gate_up", "down"}
    assert fused.qkv is None and fused.gate_up is None  # the float copies are gone
    for m, w, a, b, x in zip(modules, floats, want, got, xs):
        assert m.weight is None and m.bias is None
        q, scale = vars(m)["_muse_int8"]
        assert q.dtype == torch.int8 and q.shape == w.shape
        dequant = q.float() * scale[:, None]
        assert torch.allclose(dequant, w, atol=float(scale.max()) * 0.51)
        assert torch.allclose(b, nn.functional.linear(x, dequant), atol=1e-5)
        assert torch.allclose(b, a, atol=0.1) and not torch.equal(b, a)
    before = dict(fused.int8)
    fused.quantize(layer, torch)  # idempotent
    assert all(fused.int8[n][0] is before[n][0] for n in before)


def test_fused_layers_are_quantized_only_for_an_int8_decoder_with_kernels(cp, monkeypatch):
    _with_silu(cp.model)
    layers = cp.model.layers
    assert not any(f.int8 for f in fast._fused_layers(layers, True, torch))  # CPU: no int8 GEMV
    monkeypatch.setattr(fast, "_kernels", lambda t: object())
    assert not any(f.int8 for f in fast._fused_layers(layers, False, torch))
    assert all(set(f.int8) == {"qkv", "o", "gate_up", "down"}
               for f in fast._fused_layers(layers, True, torch))


def _int8_decoders(talker, monkeypatch):
    """Decoders built as on CUDA with Triton: their layers stored as int8."""
    _with_silu(talker.model)
    _with_silu(talker.code_predictor.model)
    with monkeypatch.context() as m:
        m.setattr(fast, "_kernels", lambda t: object())
        cp = fast.FastCodePredictor(talker.code_predictor, torch=torch, int8=True)
        ft = fast.FastTalker(talker, cp, torch=torch, int8=True)
    for model in (talker.model, talker.code_predictor.model):
        assert all(m.weight is None for layer in model.layers for m in _projections(layer))
    return cp, ft


def test_int8_decoders_on_cpu_keep_the_float_weights(talker):
    _with_silu(talker.model)
    cp = fast.FastCodePredictor(talker.code_predictor, torch=torch, int8=True)
    fast.FastTalker(talker, cp, torch=torch, int8=True)
    for model in (talker.model, talker.code_predictor.model):
        assert all(m.weight is not None for layer in model.layers for m in _projections(layer))


def test_int8_predictor_matches_a_reference_over_the_same_int8_layers(talker, embeds,
                                                                      monkeypatch):
    dec, _ = _int8_decoders(talker, monkeypatch)
    cp = talker.code_predictor
    calls = _spy_linears(monkeypatch)
    with torch.no_grad():
        seq = dec.generate(inputs_embeds=embeds, max_new_tokens=STEPS, do_sample=False,
                           top_p=1.0, top_k=50, temperature=0.9).sequences
        seen = list(calls)
        ref = reference_logits(cp, embeds, seq)
    for step, logits in enumerate(ref):
        assert torch.equal(seq[:, step], logits.argmax(-1)), step
    assert cp.upstream_calls == []
    # layers read int8 (no float weight left); the output heads stay float
    heads = {id(h.weight) for h in cp.lm_head}
    head = [quant for weight, quant in seen if weight is not None and id(weight) in heads]
    body = [(weight, quant) for weight, quant in seen
            if weight is None or id(weight) not in heads]
    assert len(head) == STEPS and all(quant is None for quant in head)
    assert body and all(weight is None and quant[0].dtype == torch.int8
                        for weight, quant in body)


@pytest.mark.parametrize("masked", [False, True], ids=["exact", "masked"])
def test_int8_talker_matches_a_reference_over_the_same_int8_layers(talker, prompt, masked,
                                                                   monkeypatch):
    _, ft = _int8_decoders(talker, monkeypatch)
    if masked:
        monkeypatch.setattr(ft, "_step",
                            lambda st, p, pos: ft._frame(st, p, start=pos, masked=True))
    with torch.no_grad():
        result = ft.generate(**_talker_kwargs(prompt))
        ref = reference_talker(talker, **prompt, max_new=12)
    _assert_matches_reference(result, ref)
    assert talker.upstream_calls == [] and talker.code_predictor.upstream_calls == []
    assert talker.codec_head.weight is not None
    assert all(h.weight is not None for h in talker.code_predictor.lm_head)


def test_installs_pass_int8_to_the_decoders(talker, monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    tts = _talker_tts(talker)
    assert fast.install(tts, torch=torch, int8=True) is True
    assert fast.install_talker(tts, torch=torch, int8=True) is True
    assert talker.code_predictor.generate.__self__.int8 is True
    assert talker.generate.__self__.int8 is True
    other = FakeTalker()
    tts = _talker_tts(other)
    fast.install(tts, torch=torch)
    fast.install_talker(tts, torch=torch)
    assert other.code_predictor.generate.__self__.int8 is False
    assert other.generate.__self__.int8 is False


# -- prepare (capture) and stepper -------------------------------------------------


def _stepped(ft, kwargs, ref_codes=None):
    """Frames from start / step, as a streaming caller drives them."""
    req = ft.request(kwargs, ref_codes=ref_codes)
    assert req is not None
    run = ft.start(req)
    frames = []
    while (codes := ft.step(run)) is not None:
        frames.append(codes)
    return req, run, frames


@pytest.mark.parametrize("masked", [False, True], ids=["exact", "masked"])
def test_stepper_frames_equal_generate(talker, prompt, masked, monkeypatch):
    ft = _fast_talker(talker, masked=masked, monkeypatch=monkeypatch)
    with torch.no_grad():
        want = _split(ft.generate(**_talker_kwargs(prompt)))[0]
        _, run, got = _stepped(ft, _talker_kwargs(prompt))
        ref = reference_talker(talker, **prompt, max_new=12)[0]
    assert len(got) == len(want) == len(ref) == 11 and run.done
    for a, b, c in zip(got, want, ref):
        assert torch.equal(a, b) and torch.equal(a, c)
    assert ft.step(run) is None  # finished runs stay finished
    assert talker.upstream_calls == []


@pytest.mark.parametrize("masked", [False, True], ids=["exact", "masked"])
def test_stepper_grows_the_cache_mid_line(talker, prompt, masked, monkeypatch):
    monkeypatch.setattr(fast, "TALKER_CAPACITY", 4)
    ft = _fast_talker(talker, masked=masked, monkeypatch=monkeypatch)
    with torch.no_grad():
        _, run, got = _stepped(ft, _talker_kwargs(prompt, max_new_tokens=10))
        ref = reference_talker(talker, **prompt, max_new=10)[0]
    assert len(got) == len(ref) and all(torch.equal(a, b) for a, b in zip(got, ref))
    assert ft._state.capacity == 16 and run.state is ft._state


def test_stepper_frames_are_copies(talker, prompt, monkeypatch):
    """A graphed frame returns its static output buffer; step must clone it."""
    ft = _fast_talker(talker)
    buffer = torch.zeros(1, GROUPS, dtype=torch.long)

    def fake_step(st, p, pos):
        ft._frame(st, p, start=pos, masked=False)
        buffer.fill_(pos)
        return buffer
    monkeypatch.setattr(ft, "_step", fake_step)
    with torch.no_grad():
        _, _, got = _stepped(ft, _talker_kwargs(prompt, max_new_tokens=4))
    assert [int(c[0, 0]) for c in got] == [5, 6, 7]


def test_stepper_stops_at_eos_like_generate(talker, prompt):
    with torch.no_grad():
        talker.codec_head.weight[EOS] = 50 * talker.codec_head.weight[0].sign()
        talker.model.norm.weight.abs_()
    ft = _fast_talker(talker)
    with torch.no_grad():
        want = _split(ft.generate(**_talker_kwargs(prompt, min_new_tokens=3)))[0]
        _, run, got = _stepped(ft, _talker_kwargs(prompt, min_new_tokens=3))
    assert len(got) == len(want) == 3 and run.done
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(int(c[0, 0]) != EOS for c in got)


def test_starting_a_run_retires_the_previous_one(talker, prompt):
    ft = _fast_talker(talker)
    with torch.no_grad():
        first = ft.start(ft.request(_talker_kwargs(prompt)))
        ft.step(first)
        second = ft.start(ft.request(_talker_kwargs(prompt)))
        with pytest.raises(RuntimeError, match="retired"):
            ft.step(first)
        assert ft.step(second) is not None
        ft.generate(**_talker_kwargs(prompt))  # the one-call path retires runs too
        with pytest.raises(RuntimeError, match="retired"):
            ft.step(second)


def test_request_is_plain_data_and_none_for_unhandled_calls(talker, prompt):
    ft = _fast_talker(talker)
    ref = torch.zeros(3, GROUPS, dtype=torch.long)
    req = ft.request(_talker_kwargs(prompt, max_new_tokens=7), ref_codes=ref)
    assert req.inputs_embeds is prompt["inputs_embeds"]
    assert req.trailing_text_hidden is prompt["trailing_text_hidden"]
    assert req.tts_pad_embed is prompt["tts_pad_embed"]
    assert req.max_new_tokens == 7 and req.ref_codes is ref
    assert req.params.eos == EOS and req.params.suppress == SUPPRESS
    assert ft.request(_talker_kwargs(prompt, streamer=object())) is None
    ft.disabled = True
    assert ft.request(_talker_kwargs(prompt)) is None


def _upstream_entry(talker, kwargs, after):
    """Stand-in for qwen-tts's generate_voice_*: builds the talker's kwargs,
    calls talker.generate, then post-processes (``after``)."""
    talker.generate(**kwargs)
    after.append("post-processed")
    return "wavs"


def test_capture_returns_the_talker_kwargs_and_skips_the_rest(talker, prompt, monkeypatch):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    tts = _talker_tts(talker)
    fast.install(tts, torch=torch)
    fast.install_talker(tts, torch=torch)
    ft = talker.generate.__self__
    after = []
    kwargs = _talker_kwargs(prompt)
    with torch.no_grad():
        got = ft.capture(lambda: _upstream_entry(talker, kwargs, after))
        assert got == kwargs and after == []
        assert ft._state is None  # nothing ran on the GPU
        # outside capture the same entry point generates as before
        assert _upstream_entry(talker, kwargs, after) == "wavs" and after == ["post-processed"]
    assert talker.upstream_calls == []


def test_capture_unwinds_through_broad_except_clauses(talker, prompt):
    """The unwind is not an Exception, so an ``except Exception`` between the
    entry point and talker.generate cannot swallow it."""
    ft = _fast_talker(talker)
    kwargs = _talker_kwargs(prompt)

    def guarded():
        try:
            ft.generate(**kwargs)
        except Exception:  # noqa: BLE001
            return "swallowed"
    assert ft.capture(guarded) == kwargs


def test_capture_is_per_thread(talker, prompt):
    import threading

    ft = _fast_talker(talker)
    kwargs = _talker_kwargs(prompt)
    other: dict = {}
    entered, release = threading.Event(), threading.Event()

    def inside():
        entered.set()
        release.wait(5)
        ft.generate(**kwargs)

    def elsewhere():
        entered.wait(5)
        with torch.no_grad():
            other["result"] = ft.generate(**kwargs)
        release.set()

    worker = threading.Thread(target=elsewhere)
    worker.start()
    with torch.no_grad():
        got = ft.capture(inside)
    worker.join(5)
    assert got == kwargs
    assert len(other["result"].hidden_states) == 12  # generated, not captured


def test_capture_without_a_talker_call_raises_and_clears(talker, prompt):
    ft = _fast_talker(talker)
    with pytest.raises(RuntimeError, match="talker.generate"):
        ft.capture(lambda: None)
    with pytest.raises(ValueError):
        ft.capture(lambda: (_ for _ in ()).throw(ValueError("upstream refused")))
    with torch.no_grad():  # not left in capture mode
        assert len(ft.generate(**_talker_kwargs(prompt)).hidden_states) == 12


def test_disable_drops_state_and_graphs(talker, prompt):
    ft = _fast_talker(talker)
    with torch.no_grad():
        ft.generate(**_talker_kwargs(prompt))
    ft._graphs[("k",)] = ("graph", None)
    ft.disable("test")
    assert ft.disabled and ft._state is None and ft._graphs == {}
    assert ft.request(_talker_kwargs(prompt)) is None
