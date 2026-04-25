"""HFTextClassifier runtime: mocked-dep tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern)."""
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    orig = (mod.torch, mod.AutoTokenizer, mod.AutoModelForSequenceClassification)
    yield
    mod.torch, mod.AutoTokenizer, mod.AutoModelForSequenceClassification = orig


def _make_logits_tensor(values_2d):
    """Build a fake logits tensor that has .detach().cpu().numpy() chain."""
    arr = np.array(values_2d, dtype="float32")
    t = MagicMock()
    t.detach.return_value.cpu.return_value.numpy.return_value = arr
    return t


def _wire_torch_with_softmax_and_sigmoid(mod):
    """Install MagicMock torch + numpy-backed softmax/sigmoid."""
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)

    def _fake_softmax(t, dim):
        arr = t.detach.return_value.cpu.return_value.numpy.return_value
        e = np.exp(arr - arr.max(axis=-1, keepdims=True))
        out_arr = e / e.sum(axis=-1, keepdims=True)
        out = MagicMock()
        out.detach.return_value.cpu.return_value.numpy.return_value = out_arr
        return out

    def _fake_sigmoid(t):
        arr = t.detach.return_value.cpu.return_value.numpy.return_value
        out_arr = 1.0 / (1.0 + np.exp(-arr))
        out = MagicMock()
        out.detach.return_value.cpu.return_value.numpy.return_value = out_arr
        return out

    mod.torch.softmax = _fake_softmax
    mod.torch.sigmoid = _fake_sigmoid


def _install_fake_model(mod, *, id2label, problem_type, logits):
    """Install fake tokenizer and AutoModelForSequenceClassification.

    Returns the fake_model so the caller can introspect call args.
    """
    fake_tok = MagicMock()
    fake_tok.return_value = MagicMock(to=MagicMock(return_value={}))
    mod.AutoTokenizer = MagicMock()
    mod.AutoTokenizer.from_pretrained.return_value = fake_tok

    fake_model = MagicMock()
    fake_model.config = SimpleNamespace(
        id2label=id2label, problem_type=problem_type,
    )
    # call() returns the SimpleNamespace with .logits
    fake_model.return_value = SimpleNamespace(logits=_make_logits_tensor(logits))
    # .to() and the inference-mode toggle both return the same fake_model
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    mod.AutoModelForSequenceClassification = MagicMock()
    mod.AutoModelForSequenceClassification.from_pretrained.return_value = fake_model
    return fake_model


def test_classify_single_label_uses_softmax():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    _wire_torch_with_softmax_and_sigmoid(mod)
    _install_fake_model(
        mod,
        id2label={0: "OK", 1: "H", 2: "V"},
        problem_type="single_label_classification",
        logits=[[0.1, 5.0, 0.5]],
    )

    m = mod.HFTextClassifier(
        model_id="text-moderation", hf_repo="x", local_dir="/fake", device="cpu",
    )
    results = m.classify("test input")
    assert len(results) == 1
    r = results[0]
    assert not r.multi_label
    # H should be the highest score (logit 5.0 dominates after softmax)
    assert max(r.scores, key=r.scores.get) == "H"
    assert abs(sum(r.scores.values()) - 1.0) < 1e-3


def test_classify_multi_label_uses_sigmoid():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    _wire_torch_with_softmax_and_sigmoid(mod)
    _install_fake_model(
        mod,
        id2label={0: "toxic", 1: "obscene"},
        problem_type="multi_label_classification",
        # logit 5.0 -> sigmoid ~0.99; logit -2.0 -> sigmoid ~0.12
        logits=[[5.0, -2.0]],
    )

    m = mod.HFTextClassifier(
        model_id="toxic-bert", hf_repo="x", local_dir="/fake", device="cpu",
    )
    results = m.classify("text")
    r = results[0]
    assert r.multi_label
    assert r.scores["toxic"] > 0.9
    assert r.scores["obscene"] < 0.2


def test_classify_batch_preserves_order():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    _wire_torch_with_softmax_and_sigmoid(mod)
    _install_fake_model(
        mod,
        id2label={0: "OK", 1: "H"},
        problem_type="single_label_classification",
        # Row 0: OK dominant; Row 1: H dominant
        logits=[[5.0, 0.1], [0.1, 5.0]],
    )

    m = mod.HFTextClassifier(
        model_id="text-moderation", hf_repo="x", local_dir="/fake", device="cpu",
    )
    results = m.classify(["safe text", "harmful text"])
    assert len(results) == 2
    assert max(results[0].scores, key=results[0].scores.get) == "OK"
    assert max(results[1].scores, key=results[1].scores.get) == "H"


def test_device_auto_selects_cuda_when_available():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = True
    mod.torch.backends = MagicMock(mps=None)
    _install_fake_model(
        mod, id2label={0: "OK"}, problem_type=None, logits=[[1.0]],
    )

    m = mod.HFTextClassifier(
        model_id="m", hf_repo="x", local_dir="/fake", device="auto",
    )
    assert m._device == "cuda"


def test_raises_when_transformers_not_installed(monkeypatch):
    """If transformers import fails, constructor raises a clear error."""
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)
    mod.AutoTokenizer = None
    mod.AutoModelForSequenceClassification = None
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFTextClassifier(
            model_id="m", hf_repo="x", local_dir="/fake", device="cpu",
        )
