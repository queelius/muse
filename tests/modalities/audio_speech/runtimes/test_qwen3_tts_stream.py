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
    def __init__(self, quantizers=Q, bins=BINS):
        super().__init__()
        self.books = nn.ModuleList([nn.Embedding(bins, DIM) for _ in range(quantizers)])

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
    def __init__(self, quantizers=Q, bins=BINS):
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=quantizers)
        self.total_upsample = UP
        self.quantizer = Quantizer(quantizers, bins)
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
    def __init__(self, kind="qwen3_tts_tokenizer_12hz", **decoder):
        super().__init__()
        self.decoder = FakeSpeechDecoder(**decoder)
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


def _codes(frames, seed=1, quantizers=Q, bins=BINS):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, bins, (frames, quantizers), generator=g)


def _one_pass(dec, codes):
    with torch.no_grad():
        return dec(codes.T[None])[0, 0].float().numpy()


def test_front_then_conv_stack_is_the_decoder(decoder):
    split = stream.SplitDecoder(decoder, torch=torch)
    codes = _codes(20).T[None]
    with torch.no_grad():
        assert torch.equal(split.conv_stack(split.front(codes)), decoder(codes))
    assert split.upsample == UP


@pytest.mark.parametrize("padded", [False, True])
def test_front_and_conv_stack_take_several_streams_at_once(decoder, padded):
    """A leading batch dimension: row b of a batched call is that stream's
    own call (stage 2 decodes the streams that are due together)."""
    split = stream.SplitDecoder(decoder, torch=torch, padded=padded)
    rows = torch.stack([_codes(20, seed=s).T for s in (1, 2, 3)])  # [3, Q, 20]
    with torch.no_grad():
        front = split.front(rows)
        wav = split.conv_stack(front)
        assert front.shape[0] == 3 and wav.shape[:2] == (3, 1)
        for b in range(3):
            one = split.front(rows[b:b + 1])
            assert torch.allclose(front[b:b + 1], one, atol=1e-6)
            assert torch.allclose(wav[b:b + 1], split.conv_stack(one), atol=1e-6)
        shaped = split._shaped("front", split.front, rows, 64)
        assert shaped.shape == (3, front.shape[1], 64)
        assert torch.allclose(shaped[..., :20], front, atol=1e-6)


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


@pytest.mark.parametrize("ref_frames", [0, 7], ids=["design", "clone"])
def test_padded_shapes_stream_the_same_audio(decoder, ref_frames, monkeypatch):
    """With CUDA graphs the front runs on codes padded at the end to a bucket
    length and the conv stack on widths rounded up; both halves are causal,
    so the real frames' audio is unchanged. Checked eagerly here."""
    split = stream.SplitDecoder(decoder, torch=torch, context=3, padded=True)
    fronts, convs = [], []
    real_front, real_conv = split.front, split.conv_stack
    monkeypatch.setattr(split, "front", lambda c: fronts.append(c.shape[-1]) or real_front(c))
    monkeypatch.setattr(split, "conv_stack", lambda h: convs.append(h.shape[-1]) or real_conv(h))
    ref = _codes(ref_frames, seed=2) if ref_frames else None
    gen = _codes(70)
    want = _one_pass(decoder, gen if ref is None else torch.cat((ref, gen)))[ref_frames * UP:]
    sd = split.stream(ref)
    chunks, at = [], 0
    for size in (3, 16, 16, 16, 16, 3):
        sd.push(gen[at:at + size])
        at += size
        chunks.append(sd.decode())
        assert len(chunks[-1]) == size * UP
    np.testing.assert_allclose(np.concatenate(chunks), want, atol=1e-6)
    assert all(f in (64, 128) for f in fronts) and fronts[-1] == 128
    assert all(w % stream.CONV_WIDTH_STEP == 0 for w in convs)


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


# -- Qwen3TTSRuntime.synthesize_stream ------------------------------------------------

from muse.modalities.audio_speech.protocol import AudioChunk  # noqa: E402
from muse.modalities.audio_speech.runtimes import qwen3_tts as rt  # noqa: E402
from muse.modalities.audio_speech.runtimes import qwen3_tts_fast as fast  # noqa: E402

from .test_qwen3_tts_fast import (  # noqa: E402
    EMBED, EOS, GROUPS, TVOCAB, FakeTalker, _talker_kwargs,
)

REF_FRAMES = 7
MAX_NEW = 30  # 29 frames: the fake talker never picks EOS greedily
TOL = 1e-5  # fp32 rounding of the prefix-length running mean


class FakeQwen:
    """qwen_tts.Qwen3TTSModel over the fake talker and speech decoder: builds
    the talker's kwargs, calls ``talker.generate``, keeps the frames before
    the first EOS, prepends the reference codes in clone mode, decodes in one
    pass and cuts the reference's share of the audio, as qwen-tts 0.1.1 does."""

    instances: list = []

    def __init__(self):
        torch.manual_seed(5)
        talker = FakeTalker()
        torch.manual_seed(0)
        speech = FakeTokenizerModel(quantizers=GROUPS, bins=TVOCAB)
        with torch.no_grad():
            for p in [*talker.parameters(), *speech.parameters()]:
                p.mul_(3.0)
        self.model = SimpleNamespace(talker=talker, speech_tokenizer=SimpleNamespace(
            model=speech, device="cpu", decode=self._decode))
        self.ref_codes = _codes(REF_FRAMES, seed=2, quantizers=GROUPS, bins=TVOCAB)
        self.decoded: list = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.instances.append(cls())
        return cls.instances[-1]

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        return [SimpleNamespace(ref_code=self.ref_codes, icl_mode=True)]

    def _kwargs(self, text, salt):
        g = torch.Generator().manual_seed(len(text) * 7 + salt)
        return _talker_kwargs(dict(inputs_embeds=torch.randn(1, 5, EMBED, generator=g),
                                   trailing_text_hidden=torch.randn(1, 3, EMBED, generator=g),
                                   tts_pad_embed=torch.randn(1, 1, EMBED, generator=g)),
                              max_new_tokens=MAX_NEW)

    def _decode(self, items):
        """qwen-tts's speech_tokenizer.decode: [{audio_codes: [frames, groups]}]
        -> ([float32 audio], sample rate), one pass."""
        full = items[0]["audio_codes"]
        with torch.inference_mode():
            wav = self.model.speech_tokenizer.model.decoder(full.T[None].clamp(min=0))
        return [wav[0, 0].float().numpy()], 24000

    def _run(self, kwargs, ref):
        result = self.model.talker.generate(**kwargs)
        frames = [h[-1] for h in result.hidden_states if h[-1] is not None]
        codes = torch.cat(frames) if frames else torch.zeros(0, GROUPS, dtype=torch.long)
        stop = (codes[:, 0] == EOS).nonzero()
        if len(stop):
            codes = codes[:int(stop[0, 0])]
        self.decoded.append(codes)
        full = codes if ref is None else torch.cat((ref, codes))
        wav = self.model.speech_tokenizer.decode([{"audio_codes": full}])[0][0]
        if ref is not None:
            wav = wav[int(len(ref) / max(len(full), 1) * len(wav)):]
        return [wav], 24000

    def _limited(self, kwargs, max_new_tokens):
        if max_new_tokens is not None:  # qwen-tts passes it through to the talker
            kwargs["max_new_tokens"] = min(MAX_NEW, int(max_new_tokens))
        return kwargs

    def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens=None):
        return self._run(self._limited(self._kwargs(text, 0), max_new_tokens),
                         voice_clone_prompt[0].ref_code)

    def generate_voice_design(self, text, instruct, language, max_new_tokens=None):
        return self._run(self._limited(self._kwargs(text, len(instruct)), max_new_tokens), None)


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setattr(fast, "_qwen_tts_version", lambda: "0.1.1")
    monkeypatch.setattr(stream, "_qwen_tts_version", lambda: "0.1.1")
    monkeypatch.setenv("MUSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(rt, "QwenTTS", FakeQwen)
    monkeypatch.setattr(rt, "torch", torch)
    FakeQwen.instances = []

    def build(mode, device="cuda", **kw):
        model_id = "qwen3-tts-0.6b-base" if mode == "clone" else "qwen3-tts-1.7b-voicedesign"
        return rt.Qwen3TTSRuntime(hf_repo="Qwen/x", device=device, model_id=model_id,
                                  mode=mode, **kw)
    return build


def _talker_of(m):
    return m._tts.model.talker.generate.__self__


def _chunks(m, text="A line of speech.", **kw):
    chunks = list(m.synthesize_stream(text, **kw))
    assert chunks and all(isinstance(c, AudioChunk) and c.sample_rate == 24000 for c in chunks)
    assert all(c.audio.dtype == np.float32 for c in chunks)
    return chunks


@pytest.mark.parametrize("mode", ["clone", "design"])
def test_stream_equals_the_one_chunk_audio(make, mode, monkeypatch):
    m = make(mode)
    pushed = []
    real_push = stream.StreamDecoder.push
    monkeypatch.setattr(stream.StreamDecoder, "push",
                        lambda self, f: pushed.append(f.clone()) or real_push(self, f))
    want = m.synthesize("A line of speech.").audio
    chunks = _chunks(m)
    assert [len(c.audio) // UP for c in chunks] == [2, 4, 8, 15]  # default: doubles from 2, up to 16
    got = np.concatenate([c.audio for c in chunks])
    assert len(got) == len(want) == 29 * UP
    np.testing.assert_allclose(got, want, atol=TOL)
    codes = FakeQwen.instances[0].decoded
    assert len(codes) == 1  # streaming never ran upstream's decode
    assert torch.equal(torch.cat(pushed), codes[0])  # the same codes, token for token
    assert m._tts.model.talker.upstream_calls == []


def test_chunk_schedule_comes_from_the_constructor(make):
    m = make("clone", stream_first_frames=2, stream_chunk_frames=5)
    assert [len(c.audio) // UP for c in _chunks(m)] == [2, 4, 5, 5, 5, 5, 3]


@pytest.mark.parametrize("kw", [{"stream_first_frames": 0}, {"stream_chunk_frames": -1}])
def test_chunk_schedule_must_be_positive(make, kw):
    with pytest.raises(ValueError):
        make("clone", **kw)


def test_instructions_reach_the_stream(make):
    m = make("design")
    a = np.concatenate([c.audio for c in _chunks(m, instructions="A gravelly old wizard.")])
    b = np.concatenate([c.audio for c in _chunks(m, instructions="   ")])
    want_a = m.synthesize("A line of speech.", instructions="A gravelly old wizard.").audio
    want_b = m.synthesize("A line of speech.").audio  # blank: the default instructions
    np.testing.assert_allclose(a, want_a, atol=TOL)
    np.testing.assert_allclose(b, want_b, atol=TOL)
    assert not np.allclose(want_a, want_b)


def test_unknown_voice_raises_before_any_audio(make):
    from muse.modalities.audio_speech.protocol import UnknownVoiceError

    with pytest.raises(UnknownVoiceError):
        next(make("clone").synthesize_stream("Hi.", voice="nobody"))


def test_cpu_streams_one_chunk(make):
    m = make("clone", device="cpu")
    chunks = _chunks(m)
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0].audio, m.synthesize("A line of speech.").audio)


def test_disabled_or_unhandled_talker_streams_one_chunk(make, monkeypatch):
    m = make("clone")
    with monkeypatch.context() as patch:
        patch.setattr(fast.FastTalker, "request", lambda self, kw, ref_codes=None: None)
        assert len(_chunks(m)) == 1
    _talker_of(m).disabled = True
    assert len(_chunks(m)) == 1


def test_without_the_fast_talker_streams_one_chunk(make):
    m = make("clone", fast_talker=False)
    assert len(_chunks(m)) == 1


def _boom(*a, **k):
    raise RuntimeError("boom")


def test_failure_before_the_first_chunk_falls_back_to_one_chunk(make, monkeypatch):
    m = make("clone")
    ft = _talker_of(m)
    monkeypatch.setattr(ft, "step", _boom)
    chunks = list(m.synthesize_stream("A line of speech."))
    assert len(chunks) == 1 and ft.disabled  # the rest of the process uses upstream
    assert len(m._tts.model.talker.upstream_calls) == 1


def test_failure_after_a_chunk_ends_the_stream(make, monkeypatch):
    m = make("clone", stream_first_frames=4)
    ft = _talker_of(m)
    real, calls = ft.step, []

    def flaky(run):
        calls.append(1)
        if len(calls) > 6:
            raise RuntimeError("boom")
        return real(run)
    monkeypatch.setattr(ft, "step", flaky)
    gen = m.synthesize_stream("A line of speech.")
    assert len(next(gen).audio) == 4 * UP
    with pytest.raises(RuntimeError, match="boom"):
        next(gen)
    assert ft.disabled


def test_decoder_failure_falls_back_and_turns_streaming_off(make, monkeypatch):
    m = make("clone")
    with monkeypatch.context() as patch:
        patch.setattr(stream.StreamDecoder, "decode", _boom)
        assert len(_chunks(m)) == 1
    assert len(_chunks(m)) == 1  # off for the process
    assert not _talker_of(m).disabled  # the talker itself was fine


def test_oom_is_not_swallowed(make, monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    def oom(run):
        raise OutOfMemoryError()
    m = make("clone")
    ft = _talker_of(m)
    monkeypatch.setattr(ft, "step", oom)
    with pytest.raises(OutOfMemoryError):
        list(m.synthesize_stream("A line of speech."))
    assert not ft.disabled


def test_a_closed_stream_leaves_the_model_usable(make):
    m = make("clone")
    want = m.synthesize("A line of speech.").audio
    gen = m.synthesize_stream("A line of speech.")
    next(gen)
    gen.close()  # the client went away
    got = np.concatenate([c.audio for c in _chunks(m)])
    np.testing.assert_allclose(got, want, atol=TOL)
    np.testing.assert_allclose(m.synthesize("A line of speech.").audio, want, atol=TOL)
