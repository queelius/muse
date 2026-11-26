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


def _fast_talker(talker, *, masked=False, monkeypatch=None):
    cp = fast.FastCodePredictor(talker.code_predictor, torch=torch)
    ft = fast.FastTalker(talker, cp, torch=torch)
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
