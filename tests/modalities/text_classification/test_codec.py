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


# ----- v0.35.0: encode_classifications for /v1/text/classifications -----


def test_encode_classifications_sorts_by_score_desc():
    """Each per-input list of {label, score} pairs is sorted descending."""
    from muse.modalities.text_classification.codec import encode_classifications
    results = [ClassificationResult(
        scores={"a": 0.1, "b": 0.6, "c": 0.3}, multi_label=False,
    )]
    body = encode_classifications(results, model_id="x")
    assert body["model"] == "x"
    assert body["id"].startswith("classify-")
    pairs = body["results"][0]
    assert [p["label"] for p in pairs] == ["b", "c", "a"]
    assert pairs[0]["score"] == 0.6


def test_encode_classifications_top_k_truncates_per_input():
    from muse.modalities.text_classification.codec import encode_classifications
    results = [ClassificationResult(
        scores={f"l{i}": 0.1 * i for i in range(5)}, multi_label=False,
    )]
    body = encode_classifications(results, model_id="x", top_k=2)
    pairs = body["results"][0]
    assert len(pairs) == 2
    # Top two by score (l4=0.4, l3=0.3)
    assert [p["label"] for p in pairs] == ["l4", "l3"]


def test_encode_classifications_top_k_none_keeps_all():
    from muse.modalities.text_classification.codec import encode_classifications
    results = [ClassificationResult(
        scores={f"l{i}": 0.1 for i in range(7)}, multi_label=True,
    )]
    body = encode_classifications(results, model_id="x", top_k=None)
    assert len(body["results"][0]) == 7


def test_encode_classifications_zero_top_k_treated_as_no_truncation():
    """top_k=0 is degenerate; codec defaults to keeping all rather than
    returning empty lists. Route layer should reject 0 via Pydantic."""
    from muse.modalities.text_classification.codec import encode_classifications
    results = [ClassificationResult(
        scores={"a": 0.5, "b": 0.4}, multi_label=False,
    )]
    body = encode_classifications(results, model_id="x", top_k=0)
    assert len(body["results"][0]) == 2


def test_encode_classifications_list_inputs_become_list_of_lists():
    """One input per ClassificationResult -> one inner list per input."""
    from muse.modalities.text_classification.codec import encode_classifications
    results = [
        ClassificationResult(scores={"a": 0.9}, multi_label=False),
        ClassificationResult(scores={"a": 0.4, "b": 0.6}, multi_label=False),
    ]
    body = encode_classifications(results, model_id="x")
    assert len(body["results"]) == 2
    assert body["results"][0][0]["label"] == "a"
    assert body["results"][1][0]["label"] == "b"


def test_encode_classifications_float_coercion():
    """Scores from numpy / torch tensors get coerced to plain Python float
    so json.dumps doesn't choke."""
    from muse.modalities.text_classification.codec import encode_classifications
    results = [ClassificationResult(
        scores={"a": 0.9, "b": 0.1}, multi_label=False,
    )]
    body = encode_classifications(results, model_id="x")
    for pair in body["results"][0]:
        assert isinstance(pair["score"], float)
