"""Decima ONNX adapter, against a fake decima-ai runtime."""
from __future__ import annotations

import json
import math
import unicodedata
from unittest.mock import patch

import numpy as np
import pytest

import muse.modalities.decision_evaluation.runtimes.decima_onnx as runtime
from muse.modalities.decision_evaluation.codec import encode_response, score_confidence
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionCancelled,
    DecisionInputError,
    DecisionModel,
)
from muse.modalities.decision_evaluation.runtimes.decima_onnx import DecimaOnnxModel
from muse.modalities.decision_evaluation.validation import parse_request


TEMPERATURE = 0.7269418400253115
CFG = {
    "max_state_tokens": 512,
    "max_choice_tokens": 64,
    "temperature": TEMPERATURE,
    "question_in_state": True,
    "question_in_choices": True,
    "state_prefix": "",
    "choice_prefix": "",
}
CLS, SEP, PAD = 1, 2, 0


def fake_normalize(text, lang="en"):
    assert lang == "en"
    return unicodedata.normalize("NFC", text).strip()


class FakeTokenizer:
    """CLS + one token per whitespace word + SEP. Records every call."""

    pad_token_id = PAD

    def __init__(self):
        self.calls = []
        self.vocab: dict[str, int] = {}

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        assert not kwargs.get("truncation"), "the adapter must never let the tokenizer truncate"
        assert isinstance(text, str)
        ids = [CLS] + [self.vocab.setdefault(w, 10 + len(self.vocab)) for w in text.split()] + [SEP]
        return {"input_ids": ids}


class FakeSession:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def run(self, outputs, feeds):
        assert outputs is None
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return self.fn(feeds)


class FakeDecimaOnnx:
    """Mimics decima.runtime.DecimaOnnx's attributes; its decide paths are off-limits."""

    instances: list["FakeDecimaOnnx"] = []
    cfg = CFG
    # scorer(texts of state, texts of options) -> raw logits per option
    raw_logits = None

    def __init__(self, path, threads=0):
        self.path = path
        self.threads = threads
        self.cfg = dict(type(self).cfg)
        self.tok = FakeTokenizer()
        self.enc = FakeSession(self._encode)
        self.sc = FakeSession(self._score)
        self.ordinal_calls = []
        FakeDecimaOnnx.instances.append(self)

    def _encode(self, feeds):
        ids = feeds["input_ids"]
        assert ids.dtype == np.int64 and feeds["attention_mask"].dtype == np.int64
        # hidden state = token id broadcast over a 2-wide feature axis
        return [np.stack([ids, ids], axis=-1).astype(np.float32)]

    def _score(self, feeds):
        n = feeds["choice_h"].shape[0]
        fn = type(self).raw_logits or (lambda n: [0.0] * n)
        s = np.array(fn(n), dtype=np.float32)
        z = np.zeros((n, 2), dtype=np.float32)
        return [s, z]

    def ordinal(self, s, z):
        # Recognizable stand-in: reverses the scores (the real head is a
        # cumulative-logit model; the adapter must route score through it).
        self.ordinal_calls.append(np.array(s))
        x = -np.asarray(s, dtype=np.float64)
        return x - np.log(np.exp(x - x.max()).sum()) - x.max()

    @classmethod
    def from_pretrained(cls, *a, **k):  # pragma: no cover - must not be called
        raise AssertionError("from_pretrained may download")

    def decide(self, *a, **k):  # pragma: no cover
        raise AssertionError("decide truncates; use the raw heads")

    decide_many = decide_batch = decide_logits = logits = encode = decide


@pytest.fixture
def export(tmp_path):
    d = tmp_path / "onnx" / "int8"
    (d / "tokenizer").mkdir(parents=True)
    for name in ("decima.json", "encoder.onnx", "scorer.onnx"):
        (d / name).write_text("x")
    FakeDecimaOnnx.instances = []
    FakeDecimaOnnx.cfg = CFG
    FakeDecimaOnnx.raw_logits = None
    with patch.object(runtime, "DecimaOnnx", FakeDecimaOnnx), \
         patch.object(runtime, "normalize", fake_normalize), \
         patch.object(runtime, "np", np):
        yield tmp_path


def _model(local_dir, **kw) -> DecimaOnnxModel:
    kw.setdefault("temperature", TEMPERATURE)
    return DecimaOnnxModel(
        model_id="decima-base", hf_repo="amyrmahdy/decima-base",
        local_dir=str(local_dir), **kw,
    )


def _req(questions, state="the customer was charged twice", model="decima-base"):
    return parse_request(json.dumps(
        {"model": model, "state": state, "questions": questions}, ensure_ascii=False,
    ).encode("utf-8"))


def _choice(criteria, instructions="Which team?"):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def _rendered(m):
    """Texts the fake tokenizer saw, in call order."""
    return [text for text, _ in m._rt.tok.calls]


# --- load -----------------------------------------------------------------


def test_loads_the_local_int8_export_with_two_threads(export):
    m = _model(export)
    rt = FakeDecimaOnnx.instances[-1]
    assert rt.path == export / "onnx" / "int8"
    assert rt.threads == 2
    assert m.device == "cpu"
    assert isinstance(m, DecisionModel)


@pytest.mark.parametrize("device", ["cuda", "mps", "cuda:0"])
def test_refuses_non_cpu_devices(device, export):
    with pytest.raises(RuntimeError):
        _model(export, device=device)
    assert FakeDecimaOnnx.instances == []


def test_auto_device_resolves_to_cpu(export):
    assert _model(export, device="auto").device == "cpu"


def test_incomplete_export_is_refused_without_leaking_the_path(export):
    (export / "onnx" / "int8" / "scorer.onnx").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        _model(export)
    assert "scorer.onnx" in str(exc.value)
    assert str(export) not in str(exc.value)


def test_temperature_mismatch_is_refused(export):
    FakeDecimaOnnx.cfg = {**CFG, "temperature": 1.0}
    with pytest.raises(ValueError):
        _model(export)


def test_temperature_is_the_pinned_fitted_value(export):
    assert _model(export).temperature == 0.7269418400253115


# --- head mapping ---------------------------------------------------------


def test_choice_uses_choose_head_with_temperature(export):
    FakeDecimaOnnx.raw_logits = lambda n: [1.0, 0.0, -1.0][:n]
    m = _model(export)
    req = _req({"q": _choice({"billing": "charges", "tech": None, "sales": ""})})
    probs = m.evaluate(m.prepare(req)).answers[0].probabilities
    expected = _softmax([x / TEMPERATURE for x in (1.0, 0.0, -1.0)])
    assert list(probs) == ["billing", "tech", "sales"]
    assert list(probs.values()) == pytest.approx(expected, rel=1e-6)
    assert m._rt.ordinal_calls == []


def test_choice_options_render_as_key_colon_description(export):
    m = _model(export)
    m.prepare(_req({"q": _choice({"billing": "charges", "tech": None, "sales": " "})}))
    texts = _rendered(m)
    assert texts[1:] == ["Which team? billing: charges", "Which team? tech", "Which team? sales"]


def test_state_text_carries_the_question_then_the_state(export):
    m = _model(export)
    m.prepare(_req({"q": _choice({"a": None, "b": None})}, state={"x": "é"}))
    assert _rendered(m)[0] == 'Which team?\n{"x": "é"}'


def test_choice_permutation_permutes_keys_not_meaning(export):
    # The raw scorer gives option-text-dependent logits; permuting the
    # criteria must move each key's probability with it.
    by_text = {"Which team? billing": 2.0, "Which team? tech": 0.0}

    def raw(n):
        rows = FakeDecimaOnnx.instances[-1].enc.calls[-1]["input_ids"]
        vocab = {v: k for k, v in FakeDecimaOnnx.instances[-1].tok.vocab.items()}
        texts = [" ".join(vocab[t] for t in row if t >= 10) for row in rows]
        return [by_text[t] for t in texts]

    FakeDecimaOnnx.raw_logits = raw
    m = _model(export)
    a = m.evaluate(m.prepare(_req({"q": _choice({"billing": None, "tech": None})}))).answers[0]
    b = m.evaluate(m.prepare(_req({"q": _choice({"tech": None, "billing": None})}))).answers[0]
    assert a.probabilities["billing"] == pytest.approx(b.probabilities["billing"])
    assert a.probabilities["billing"] > 0.9


@pytest.mark.parametrize("raw, true_wins", [([3.0, 0.0], True), ([0.0, 3.0], False)])
def test_noul_verify_polarity(raw, true_wins, export):
    """Raw index 0 is the verify head's "yes": P(true) follows it, whatever order."""
    FakeDecimaOnnx.raw_logits = lambda n: raw
    m = _model(export)
    req = _req({"n": {"type": "noul", "instructions": "The charge is a duplicate."}})
    out = encode_response(req, m.evaluate(m.prepare(req)), model_id=m.model_id)
    assert (out["answers"]["n"]["noul"] > 0.5) is true_wins
    texts = _rendered(m)
    assert texts[1:] == ["The charge is a duplicate. yes", "The charge is a duplicate. no"]


def test_noul_criteria_render_true_if_false_if_lines(export):
    m = _model(export)
    m.prepare(_req({"n": {"type": "noul", "instructions": "Refund?",
                          "criteria": {"false": "they want an exchange"}}}))
    assert _rendered(m)[0].startswith("Refund?\nTrue if: —\nFalse if: they want an exchange\n")


def test_noul_default_descriptions_are_not_rendered(export):
    m = _model(export)
    m.prepare(_req({"n": {"type": "noul", "instructions": "Refund?"}}))
    assert "True if" not in _rendered(m)[0]


def test_score_routes_through_the_ordinal_head_in_level_order(export):
    FakeDecimaOnnx.raw_logits = lambda n: [0.0, 1.0, 2.0][:n]
    m = _model(export)
    req = _req({"s": {"type": "score", "instructions": "How urgent?",
                      "criteria": ["routine", "prompt", "outage"]}})
    probs = m.evaluate(m.prepare(req)).answers[0].probabilities
    assert list(probs) == ["0", "1", "2"]
    # The fake ordinal reverses, so level 0 wins, proving its output is used.
    assert max(probs, key=probs.get) == "0"
    assert len(m._rt.ordinal_calls) == 1
    assert m._rt.ordinal_calls[0] == pytest.approx([0.0, 1 / TEMPERATURE, 2 / TEMPERATURE])
    assert _rendered(m)[1:] == ["How urgent? routine", "How urgent? prompt", "How urgent? outage"]


def test_score_confidence_comes_from_muse_not_the_sdk(export):
    # Bimodal ordinal output: decima.systemone would report
    # (K * p_max - 1) / (K - 1) = 0.25; Muse's ordered statistic is 0.
    m = _model(export)
    m._rt.ordinal = lambda s, z: np.log(np.array([0.5, 1e-12, 0.5]))
    req = _req({"s": {"type": "score", "instructions": "How?", "criteria": ["a", "b", "c"]}})
    out = encode_response(req, m.evaluate(m.prepare(req)), model_id=m.model_id)
    p = list(out["answers"]["s"]["probabilities"].values())
    sdk_confidence = (3 * max(p) - 1) / 2
    assert sdk_confidence == pytest.approx(0.25)
    assert out["answers"]["s"]["confidence"] == pytest.approx(score_confidence(tuple(p)))
    assert out["answers"]["s"]["confidence"] == 0.0
    # Precision is kept: no four-decimal rounding.
    assert p[1] != round(p[1], 4) or p[1] == 0.0


# --- token limits ---------------------------------------------------------


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_state_limit_512_passes_513_fails_before_any_inference(export):
    m = _model(export)
    # "Which team?" = 2 words, CLS + SEP = 2, so 508 state words = 512 tokens.
    ok = m.prepare(_req({"q": _choice({"a": None, "b": None})}, state=_words(508)))
    assert ok.questions[0].payload.state_ids.__len__() == 512
    with pytest.raises(DecisionInputError) as exc:
        m.prepare(_req({"q": _choice({"a": None, "b": None})}, state=_words(509)))
    assert (exc.value.code, exc.value.status) == ("input_too_long", 400)
    assert m._rt.enc.calls == [] and m._rt.sc.calls == []


def test_option_limit_64_passes_65_fails_before_any_inference(export):
    m = _model(export)
    # question 2 words + CLS + SEP = 4; a 60-word option renders to 64.
    ok = m.prepare(_req({"q": _choice({"a": _words(59), "b": None})}))
    assert max(len(r) for r in ok.questions[0].payload.choice_rows) == 64  # "a:" + 59 words
    with pytest.raises(DecisionInputError) as exc:
        m.prepare(_req({"q": _choice({"a": _words(60), "b": None})}))
    assert exc.value.code == "input_too_long"
    assert m._rt.enc.calls == []


def test_long_question_is_rejected_not_shortened(export):
    """The SDK's fit rendering would cut the question copy inside each
    option to make it fit; Muse refuses instead."""
    m = _model(export)
    with pytest.raises(DecisionInputError):
        m.prepare(_req({"q": _choice({"a": None, "b": None}, instructions=_words(70))}))
    assert not any("…" in text for text, _ in m._rt.tok.calls)
    assert m._rt.enc.calls == []


def test_too_long_last_question_means_zero_inference(export):
    m = _model(export)
    with pytest.raises(DecisionInputError):
        m.prepare(_req({
            "first": _choice({"a": None, "b": None}),
            "last": _choice({"a": _words(80), "b": None}),
        }))
    assert m._rt.enc.calls == [] and m._rt.sc.calls == []


def test_encoder_sees_exactly_the_counted_tokens(export):
    m = _model(export)
    batch = m.prepare(_req({"q": _choice({"a": None, "bb": "x y"})}))
    m.evaluate(batch)
    p = batch.questions[0].payload
    state_feed, choice_feed = m._rt.enc.calls
    assert state_feed["input_ids"].tolist() == [list(p.state_ids)]
    width = max(len(r) for r in p.choice_rows)
    assert choice_feed["input_ids"].tolist() == [
        list(r) + [PAD] * (width - len(r)) for r in p.choice_rows
    ]
    assert choice_feed["attention_mask"].tolist() == [
        [1] * len(r) + [0] * (width - len(r)) for r in p.choice_rows
    ]


# --- cache, usage, isolation ----------------------------------------------


def test_option_cache_is_bounded_and_does_not_change_usage(export):
    m = _model(export, option_cache_size=2)
    req = _req({"q": _choice({"a": None, "b": None})})
    first = m.prepare(req)
    m.evaluate(first)
    encodes = len(m._rt.enc.calls)
    again = m.prepare(req)
    m.evaluate(again)
    assert len(m._rt.enc.calls) == encodes + 1  # state only; options from cache
    assert again.input_tokens == first.input_tokens

    for keys in (("c", "d"), ("e", "f"), ("g", "h")):
        m.evaluate(m.prepare(_req({"q": _choice(dict.fromkeys(keys))})))
    assert len(m._cache) == 2


def test_usage_counts_state_and_every_option_per_question(export):
    m = _model(export)
    req = _req({"a": _choice({"x": None, "y": None}), "b": _choice({"x": None, "y": None})})
    batch = m.prepare(req)
    expected = sum(
        len(q.payload.state_ids) + sum(len(r) for r in q.payload.choice_rows)
        for q in batch.questions
    )
    assert batch.input_tokens == expected
    assert m.evaluate(batch).input_tokens == expected


def test_requests_do_not_share_state(export):
    m = _model(export)
    m.evaluate(m.prepare(_req({"q": _choice({"a": None, "b": None})}, state="FIRST")))
    m._rt.enc.calls.clear()
    m.evaluate(m.prepare(_req({"q": _choice({"a": None, "b": None})}, state="SECOND")))
    vocab = {v: k for k, v in m._rt.tok.vocab.items()}
    words = {vocab.get(t) for call in m._rt.enc.calls for row in call["input_ids"] for t in row}
    assert "FIRST" not in words and "SECOND" in words


# --- failures and lifecycle -----------------------------------------------


def test_non_finite_scores_rejected_without_poisoning(export):
    FakeDecimaOnnx.raw_logits = lambda n: [float("nan")] * n
    m = _model(export)
    req = _req({"q": _choice({"a": None, "b": None})})
    with pytest.raises(DecisionBackendError) as exc:
        m.evaluate(m.prepare(req))
    assert exc.value.code == "invalid_distribution"
    FakeDecimaOnnx.raw_logits = None
    assert m.evaluate(m.prepare(req)).answers[0].probabilities == {"a": 0.5, "b": 0.5}


def test_wrong_logit_count_rejected(export):
    FakeDecimaOnnx.raw_logits = lambda n: [0.0] * (n + 1)
    m = _model(export)
    with pytest.raises(DecisionBackendError):
        m.evaluate(m.prepare(_req({"q": _choice({"a": None, "b": None})})))


def test_session_failure_poisons_without_leaking_text(export):
    m = _model(export)
    batch = m.prepare(_req({"q": _choice({"a": None, "b": None})}))

    def boom(*a, **k):
        raise RuntimeError("onnxruntime failed reading /secret/encoder.onnx")

    m._rt.enc.run = boom
    with pytest.raises(DecisionBackendError) as exc:
        m.evaluate(batch)
    assert exc.value.code == "inference_failed" and "/secret" not in str(exc.value)
    with pytest.raises(DecisionBackendError) as again:
        m.evaluate(batch)
    assert again.value.code == "backend_unusable"


def test_batches_bind_to_their_instance(export):
    a, b = _model(export), _model(export)
    with pytest.raises(DecisionBackendError) as exc:
        b.evaluate(a.prepare(_req({"q": _choice({"x": None, "y": None})})))
    assert exc.value.code == "foreign_batch"


def test_request_for_another_model_is_refused(export):
    with pytest.raises(DecisionBackendError):
        _model(export).prepare(_req({"q": _choice({"x": None, "y": None})}, model="other"))


def test_close_clears_cache_and_refuses_further_work(export):
    m = _model(export)
    req = _req({"q": _choice({"x": None, "y": None})})
    m.evaluate(m.prepare(req))
    m.close()
    assert len(m._cache) == 0
    with pytest.raises(DecisionBackendError):
        m.prepare(req)


def test_should_stop_ends_the_batch_between_questions(export):
    m = _model(export)
    req = _req({
        "a": _choice({"a": None, "b": None}),
        "b": _choice({"a": None, "b": None}),
    })
    batch = m.prepare(req)
    m._rt.enc.calls.clear()
    with pytest.raises(DecisionCancelled):
        m.evaluate(batch, should_stop=lambda: True)
    assert m._rt.enc.calls == []
    assert m.usable
    assert len(m.evaluate(batch).answers) == 2
