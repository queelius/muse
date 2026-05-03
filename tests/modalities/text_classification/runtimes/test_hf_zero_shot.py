"""HFZeroShotPipeline runtime: mocked-dep tests.

Same fixture pattern as test_hf_text_classifier.py: module-level
sentinels (torch, pipeline) get patched per-test, autouse fixture
restores them on teardown.

The pipeline is mocked at the factory layer (transformers.pipeline);
the returned pipeline object is a callable returning {labels, scores}
dicts. The runtime's job is to dedupe candidate_labels, iterate
inputs, and rebuild a {label: score} dict per ClassificationResult.
"""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern)."""
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    orig = (mod.torch, mod.pipeline)
    yield
    mod.torch, mod.pipeline = orig


def _wire_pipeline(mod, pipeline_fn):
    """Install fake torch + pipeline factory; the factory returns
    `pipeline_fn` (a callable mock that itself returns the per-input dict)."""
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)

    pipe_obj = MagicMock()
    pipe_obj.side_effect = pipeline_fn
    pipe_obj.model = MagicMock()  # set_inference_mode is called on this
    factory = MagicMock(return_value=pipe_obj)
    mod.pipeline = factory
    return factory, pipe_obj


def test_classify_zero_shot_returns_dict_per_input():
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod

    def _pipe_fn(text, *, candidate_labels, multi_label):
        # Mimic the real pipeline shape: labels sorted by score desc.
        return {
            "sequence": text,
            "labels": ["positive", "neutral", "negative"],
            "scores": [0.7, 0.2, 0.1],
        }

    _wire_pipeline(mod, _pipe_fn)

    runtime = mod.HFZeroShotPipeline(
        model_id="zs-test",
        hf_repo="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device="cpu",
    )
    results = runtime.classify_zero_shot(
        "I love this", candidate_labels=["positive", "neutral", "negative"],
    )
    assert len(results) == 1
    r = results[0]
    assert r.scores == {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
    assert r.multi_label is False  # default


def test_classify_zero_shot_list_input_returns_one_result_per_input():
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod

    canned = [
        {"labels": ["a"], "scores": [0.9], "sequence": "x1"},
        {"labels": ["a"], "scores": [0.4], "sequence": "x2"},
    ]
    iter_canned = iter(canned)

    def _pipe_fn(text, *, candidate_labels, multi_label):
        return next(iter_canned)

    _wire_pipeline(mod, _pipe_fn)
    runtime = mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="x", device="cpu",
    )
    results = runtime.classify_zero_shot(
        ["x1", "x2"], candidate_labels=["a"],
    )
    assert len(results) == 2
    assert results[0].scores == {"a": 0.9}
    assert results[1].scores == {"a": 0.4}


def test_multi_label_flag_forwards_to_pipeline():
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    captured: dict = {}

    def _pipe_fn(text, *, candidate_labels, multi_label):
        captured["multi_label"] = multi_label
        return {"labels": ["a"], "scores": [0.5], "sequence": "x"}

    _wire_pipeline(mod, _pipe_fn)
    runtime = mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="x", device="cpu",
    )
    results = runtime.classify_zero_shot(
        "x", candidate_labels=["a"], multi_label=True,
    )
    assert captured["multi_label"] is True
    assert results[0].multi_label is True


def test_candidate_labels_stripped_and_deduped():
    """Whitespace + duplicate labels collapse to a clean list before
    the pipeline is called. Otherwise 'happy' and 'happy ' would yield
    two distinct logits for the same concept."""
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    captured: dict = {}

    def _pipe_fn(text, *, candidate_labels, multi_label):
        captured["labels"] = list(candidate_labels)
        return {"labels": candidate_labels, "scores": [0.5] * len(candidate_labels), "sequence": "x"}

    _wire_pipeline(mod, _pipe_fn)
    runtime = mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="x", device="cpu",
    )
    runtime.classify_zero_shot(
        "x", candidate_labels=["happy", " happy ", "sad", "", "  ", "sad"],
    )
    assert captured["labels"] == ["happy", "sad"]


def test_empty_candidate_labels_rejected():
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    _wire_pipeline(mod, lambda *a, **k: {})
    runtime = mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="x", device="cpu",
    )
    with pytest.raises(ValueError, match="non-empty"):
        runtime.classify_zero_shot("x", candidate_labels=[])


def test_all_whitespace_candidate_labels_rejected():
    """Even with non-empty list input, if every label strips to empty
    string the runtime must reject (otherwise it'd hand the pipeline
    an empty list)."""
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    _wire_pipeline(mod, lambda *a, **k: {})
    runtime = mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="x", device="cpu",
    )
    with pytest.raises(ValueError, match="reduced to empty"):
        runtime.classify_zero_shot("x", candidate_labels=["", "  ", "\t"])


def test_empty_input_returns_empty_list():
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    _wire_pipeline(mod, lambda *a, **k: {})
    runtime = mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="x", device="cpu",
    )
    out = runtime.classify_zero_shot([], candidate_labels=["a"])
    assert out == []


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.pipeline = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFZeroShotPipeline(model_id="zs", hf_repo="x", device="cpu")


def test_raises_when_transformers_not_installed(monkeypatch):
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.pipeline = None
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFZeroShotPipeline(model_id="zs", hf_repo="x", device="cpu")


def test_local_dir_preferred_over_hf_repo():
    import muse.modalities.text_classification.runtimes.hf_zero_shot as mod
    captured: dict = {}

    def _factory(*, task, model, device):
        captured["model"] = model
        pipe = MagicMock()
        pipe.return_value = {"labels": ["a"], "scores": [0.5], "sequence": "x"}
        pipe.model = MagicMock()
        return pipe

    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)
    mod.pipeline = _factory

    mod.HFZeroShotPipeline(
        model_id="zs", hf_repo="hub-id",
        local_dir="/tmp/local-snapshot", device="cpu",
    )
    assert captured["model"] == "/tmp/local-snapshot"
