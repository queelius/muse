"""The three-type decision probe run by `muse models probe` and `muse pull`."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from muse.cli_impl.probe_worker import _read_probe_defaults, _run_inference
from muse.modalities.decision_evaluation import PROBE_DEFAULTS
from muse.modalities.decision_evaluation.probe import probe_backend
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    PreparedBatch,
    PreparedQuestion,
    QuestionDistribution,
)


class FakeBackend:
    """Uniform distributions unless told to break one question."""

    model_id = "decision-0.8b-q8"

    def __init__(self, mangle=None):
        self.mangle = mangle or (lambda answers: answers)
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return PreparedBatch(
            model_id=self.model_id,
            questions=tuple(
                PreparedQuestion(qid, q.keys, None, 10) for qid, q in request.questions
            ),
            input_tokens=10 * len(request.questions),
        )

    def evaluate(self, batch):
        answers = [
            QuestionDistribution(q.question_id, {k: 1 / len(q.keys) for k in q.keys})
            for q in batch.questions
        ]
        return DecisionBatch(answers=tuple(self.mangle(answers)), input_tokens=batch.input_tokens)

    def close(self):
        pass


def test_probe_asks_one_question_of_each_type():
    backend = FakeBackend()
    out = probe_backend(backend)
    (request,) = backend.requests
    assert sorted(q.kind for _, q in request.questions) == ["choice", "noul", "score"]
    assert request.model == backend.model_id
    assert {a["type"] for a in out["answers"].values()} == {"choice", "score", "noul"}
    assert out["usage"] == {"input_tokens": 30, "output_tokens": 0}


@pytest.mark.parametrize("mangle", [
    lambda a: a[:-1],                                                      # missing answer
    lambda a: [QuestionDistribution(a[0].question_id, {"billing": float("nan"),
                                                       "technical": 0.5})] + a[1:],
    lambda a: [QuestionDistribution(a[0].question_id, {"billing": 1.0})] + a[1:],
    lambda a: [QuestionDistribution(a[0].question_id, {"billing": 0.2,
                                                       "technical": 0.2})] + a[1:],
])
def test_any_invalid_distribution_fails_the_probe(mangle):
    with pytest.raises(DecisionBackendError):
        probe_backend(FakeBackend(mangle))


def test_backend_errors_propagate():
    backend = FakeBackend()
    backend.evaluate = lambda batch: (_ for _ in ()).throw(DecisionBackendError("inference_failed"))
    with pytest.raises(DecisionBackendError):
        probe_backend(backend)


def test_probe_defaults_are_found_by_the_probe_worker():
    defaults = _read_probe_defaults("decision/evaluation")
    assert defaults is PROBE_DEFAULTS
    assert defaults["call"] is probe_backend
    assert "choice" in defaults["shape"] and "noul" in defaults["shape"]


def test_probe_worker_inference_runs_the_decision_probe_and_raises_on_failure():
    entry = SimpleNamespace(modality="decision/evaluation")
    shape, _ = _run_inference(FakeBackend(), entry, "cpu", None)
    assert shape == PROBE_DEFAULTS["shape"]
    with pytest.raises(DecisionBackendError):
        _run_inference(FakeBackend(lambda a: a[:-1]), entry, "cpu", None)
