"""Curated decision models and the documented TypeSafe text subset."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from muse.core.curated import _reset_curated_cache_for_tests, find_curated
from muse.core.discovery import _default_hf_plugin_dirs, discover_hf_plugins
from muse.core.resolvers_hf import HFResolver
from muse.modalities.decision_evaluation.codec import (
    choice_confidence,
    encode_response,
    score_confidence,
)
from muse.modalities.decision_evaluation.hf import SOURCES
from muse.modalities.decision_evaluation.protocol import DecisionBatch, QuestionDistribution
from muse.modalities.decision_evaluation.validation import parse_request


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "typesafe_text_subset.json").read_text()
)
EXPECTED = {
    "decision-0.8b-q8": ("hf://evalengine/decision-0.8b-gguf",
                         "ce993792ad283053004bf0230b1709ff7351bb24"),
    "decima-base": ("hf://amyrmahdy/decima-base",
                    "e5d84ab3a8436cedc45b0daefa91979f68319455"),
}


@pytest.fixture(autouse=True)
def _fresh_curated():
    _reset_curated_cache_for_tests()
    yield
    _reset_curated_cache_for_tests()


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_curated_rows_pin_the_exact_sources(model_id):
    entry = find_curated(model_id)
    uri, revision = EXPECTED[model_id]
    assert entry is not None and not entry.bundled
    assert (entry.uri, entry.revision) == (uri, revision)
    assert entry.modality == "decision/evaluation"
    assert "Apache 2.0" in entry.description
    repo = uri.removeprefix("hf://")
    assert SOURCES[repo]["revision"] == revision
    assert SOURCES[repo]["model_id"] == model_id  # URI and curated pulls share the id


def test_sources_carry_dependencies_limits_and_calibration_metadata():
    decision = SOURCES["evalengine/decision-0.8b-gguf"]
    decima = SOURCES["amyrmahdy/decima-base"]
    assert decision["pip_extras"] == ("llama-cpp-python==0.3.33",)
    assert decima["pip_extras"] == ("decima-ai==2.2.0", "onnxruntime==1.24.4")
    for source in (decision, decima):
        caps = source["capabilities"]
        assert source["license"] == "apache-2.0"
        assert caps["question_types"] == ["choice", "score", "noul"]
        assert caps["input_modalities"] == ["text"]
        assert caps["max_concurrency"] == 1
        assert caps["max_questions"] == 16 and caps["max_choice_options"] == 24
    assert decision["capabilities"]["precision"] == "Q8_0"
    assert decision["capabilities"]["calibrated"] is False
    assert decima["capabilities"]["precision"] == "int8"
    assert decima["capabilities"]["device"] == "cpu"
    assert decima["capabilities"]["calibrated"] is True
    assert "LICENSING.md" in decima["description"]


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_curated_modality_resolves_through_the_decision_plugin(model_id):
    uri, revision = EXPECTED[model_id]
    repo = uri.removeprefix("hf://")
    info = SimpleNamespace(
        id=repo, sha=revision, tags=["zero-shot-classification", "gguf"],
        siblings=[SimpleNamespace(rfilename=f) for f in SOURCES[repo]["files"]],
        card_data=SimpleNamespace(license="apache-2.0"),
    )
    resolver = HFResolver(plugins=discover_hf_plugins(_default_hf_plugin_dirs()))
    resolver._api = MagicMock()
    resolver._api.repo_info.return_value = info
    resolved = resolver.resolve_via_modality(uri, "decision/evaluation", revision=revision)
    assert resolved.manifest["model_id"] == model_id
    resolver._api.repo_info.assert_called_with(repo, revision=revision)


# --- documented TypeSafe text subset ---------------------------------------


def _documented_batch(request, answers):
    dists = []
    for qid, question in request.questions:
        answer = answers[qid]
        if question.kind == "noul":
            p = answer["noul"]
            probs = {"true": p, "false": 1 - p}
        else:
            probs = answer["probabilities"]
        dists.append(QuestionDistribution(qid, probs))
    return DecisionBatch(tuple(dists), 1)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_documented_requests_parse_and_answers_reencode(case):
    body = {"model": "decision-0.8b-q8", **case["request"]}
    request = parse_request(json.dumps(body).encode())
    assert [qid for qid, _ in request.questions] == list(case["request"]["questions"])

    out = encode_response(request, _documented_batch(request, case["answers"]),
                          model_id="decision-0.8b-q8")
    for qid, documented in case["answers"].items():
        ours = out["answers"][qid]
        assert ours["type"] == documented["type"]
        assert set(ours) == set(documented) | ({"legend"} if documented["type"] == "score" else set())
        if documented["type"] == "choice":
            assert ours["choice"] == documented["choice"]
            # Muse keeps request order for probability keys.
            assert list(ours["probabilities"]) == list(case["request"]["questions"][qid]["criteria"])
        if documented["type"] == "score":
            assert ours["score"] == pytest.approx(documented["score"], abs=0.006)
            assert list(ours["legend"].values()) == case["request"]["questions"][qid]["criteria"]
        if documented["type"] == "noul":
            assert ours["noul"] == pytest.approx(documented["noul"])
            assert "confidence" not in ours
        if "confidence" in documented:
            # The documents round to 2 decimals.
            assert ours["confidence"] == pytest.approx(documented["confidence"], abs=0.006)
    assert out["usage"]["output_tokens"] == 0


def test_documented_confidence_worked_examples():
    for probs, expected in FIXTURE["confidence_examples"]["choice"]:
        assert choice_confidence(tuple(probs)) == pytest.approx(expected, abs=0.005)
    for probs, expected in FIXTURE["confidence_examples"]["score"]:
        assert score_confidence(tuple(probs)) == pytest.approx(expected, abs=0.005)


def test_documented_requests_without_model_are_rejected():
    # TypeSafe can default its model; Muse requires an explicit catalog id.
    with pytest.raises(Exception) as exc:
        parse_request(json.dumps(FIXTURE["cases"][0]["request"]).encode())
    assert getattr(exc.value, "code", None) == "missing_field"


def test_structured_criteria_are_outside_the_supported_subset():
    body = {"model": "m", "state": "s", "questions": {"q": {
        "type": "score", "instructions": "How?",
        "criteria": [{"what": "low", "examples": ["a"]}, {"what": "high", "examples": ["b"]}],
    }}}
    with pytest.raises(Exception) as exc:
        parse_request(json.dumps(body).encode())
    assert exc.value.code == "invalid_criteria"
