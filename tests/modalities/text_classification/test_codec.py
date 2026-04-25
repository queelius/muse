"""Codec: ClassificationResult + threshold to OpenAI moderations envelope."""
import pytest

from muse.modalities.text_classification import ClassificationResult
from muse.modalities.text_classification.codec import (
    encode_moderations,
    _resolve_threshold,
    _flagged_categories,
)


# --- _resolve_threshold ---

def test_resolve_threshold_request_overrides_manifest():
    assert _resolve_threshold(0.7, {"flag_threshold": 0.3}) == 0.7


def test_resolve_threshold_falls_back_to_manifest():
    assert _resolve_threshold(None, {"flag_threshold": 0.6}) == 0.6


def test_resolve_threshold_default_is_half():
    assert _resolve_threshold(None, {}) == 0.5


def test_resolve_threshold_ignores_non_numeric_manifest():
    assert _resolve_threshold(None, {"flag_threshold": "high"}) == 0.5


# --- _flagged_categories ---

def test_flagged_multi_label_per_category():
    """Multi-label: each category True iff its score >= threshold."""
    cats, flagged = _flagged_categories(
        {"toxic": 0.9, "obscene": 0.4, "threat": 0.55}, 0.5, multi_label=True,
    )
    assert cats == {"toxic": True, "obscene": False, "threat": True}
    assert flagged is True


def test_flagged_multi_label_nothing_above_threshold():
    cats, flagged = _flagged_categories(
        {"toxic": 0.1, "obscene": 0.2}, 0.5, multi_label=True,
    )
    assert cats == {"toxic": False, "obscene": False}
    assert flagged is False


def test_flagged_single_label_argmax_above_threshold():
    """Single-label: only the argmax can be flagged, and only if >= threshold."""
    cats, flagged = _flagged_categories(
        {"H": 0.7, "V": 0.2, "OK": 0.1}, 0.5, multi_label=False,
    )
    assert cats == {"H": True, "V": False, "OK": False}
    assert flagged is True


def test_flagged_single_label_argmax_below_threshold():
    cats, flagged = _flagged_categories(
        {"H": 0.4, "V": 0.35, "OK": 0.25}, 0.5, multi_label=False,
    )
    assert cats == {"H": False, "V": False, "OK": False}
    assert flagged is False


# --- encode_moderations envelope ---

def test_encode_envelope_shape_single_input():
    results = [ClassificationResult(
        scores={"H": 0.7, "V": 0.2, "OK": 0.1}, multi_label=False,
    )]
    body = encode_moderations(results, model_id="text-moderation", threshold=0.5)
    assert body["model"] == "text-moderation"
    assert body["id"].startswith("modr-")
    assert len(body["results"]) == 1
    r0 = body["results"][0]
    assert r0["flagged"] is True
    assert r0["categories"] == {"H": True, "V": False, "OK": False}
    assert r0["category_scores"] == {"H": 0.7, "V": 0.2, "OK": 0.1}


def test_encode_envelope_batch_preserves_order():
    results = [
        ClassificationResult(scores={"toxic": 0.1}, multi_label=True),
        ClassificationResult(scores={"toxic": 0.9}, multi_label=True),
    ]
    body = encode_moderations(results, model_id="toxic-bert", threshold=0.5)
    assert len(body["results"]) == 2
    assert body["results"][0]["flagged"] is False
    assert body["results"][1]["flagged"] is True


def test_encode_envelope_id_unique_per_call():
    results = [ClassificationResult(scores={"OK": 1.0}, multi_label=False)]
    a = encode_moderations(results, model_id="m", threshold=0.5)
    b = encode_moderations(results, model_id="m", threshold=0.5)
    assert a["id"] != b["id"]


# --- safe_labels (v0.14.1 fix) ---

def test_flagged_single_label_safe_label_demoted():
    """Argmax that's in safe_labels never flags True, even if >= threshold.

    Regression: KoalaAI/Text-Moderation has an "OK" label meaning "safe".
    Confidence in OK shouldn't trigger the flagged boolean. Without
    safe_labels, the argmax-above-threshold rule would produce
    `flagged=True` on every benign input where the model is confident.
    """
    cats, flagged = _flagged_categories(
        {"OK": 0.99, "H": 0.005, "V": 0.005}, 0.5,
        multi_label=False, safe_labels=("OK",),
    )
    assert cats == {"OK": False, "H": False, "V": False}
    assert flagged is False


def test_flagged_single_label_safe_label_does_not_protect_other_categories():
    """If argmax is harmful, safe_labels doesn't suppress flagging."""
    cats, flagged = _flagged_categories(
        {"OK": 0.1, "H": 0.7, "V": 0.2}, 0.5,
        multi_label=False, safe_labels=("OK",),
    )
    assert cats == {"OK": False, "H": True, "V": False}
    assert flagged is True


def test_flagged_multi_label_safe_label_demoted():
    """Multi-label: a category in safe_labels never flags True
    even if its score >= threshold."""
    cats, flagged = _flagged_categories(
        {"toxic": 0.9, "safe": 0.95}, 0.5,
        multi_label=True, safe_labels=("safe",),
    )
    assert cats == {"toxic": True, "safe": False}
    assert flagged is True


def test_flagged_safe_labels_default_empty_preserves_old_behavior():
    """Without safe_labels (default ()), behavior matches v0.14.0."""
    cats, flagged = _flagged_categories(
        {"OK": 0.99, "H": 0.005}, 0.5, multi_label=False,
    )
    # OK is argmax with score >= threshold so flagged True (the v0.14.0 bug)
    assert cats == {"OK": True, "H": False}
    assert flagged is True


def test_resolve_safe_labels_from_manifest():
    """capabilities.safe_labels parses from manifest."""
    from muse.modalities.text_classification.codec import _resolve_safe_labels
    assert _resolve_safe_labels({"safe_labels": ["OK"]}) == ("OK",)
    assert _resolve_safe_labels({"safe_labels": ("OK", "neutral")}) == ("OK", "neutral")
    assert _resolve_safe_labels({}) == ()
    assert _resolve_safe_labels({"safe_labels": "not-a-list"}) == ()
    assert _resolve_safe_labels({"safe_labels": [1, 2]}) == ()


def test_encode_envelope_threads_safe_labels():
    """encode_moderations passes safe_labels through to flagging logic."""
    results = [ClassificationResult(
        scores={"OK": 0.99, "H": 0.005}, multi_label=False,
    )]
    body = encode_moderations(
        results, model_id="text-moderation", threshold=0.5,
        safe_labels=("OK",),
    )
    res0 = body["results"][0]
    assert res0["flagged"] is False
    assert res0["categories"] == {"OK": False, "H": False}
