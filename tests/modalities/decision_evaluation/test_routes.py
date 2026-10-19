"""POST /v1/systemone on a worker app (direct-worker entry)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.decision_evaluation import MODALITY, build_router
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionInputError,
    PreparedBatch,
    PreparedQuestion,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.validation import MAX_BODY_BYTES


class FakeBackend:
    def __init__(self, model_id="decision-0.8b-q8", *, dist=None, prepare_exc=None,
                 evaluate_exc=None):
        self.model_id = model_id
        self.dist = dist
        self.prepare_exc = prepare_exc
        self.evaluate_exc = evaluate_exc
        self.usable = True
        self.calls = 0

    def prepare(self, request):
        if self.prepare_exc:
            raise self.prepare_exc
        return PreparedBatch(self.model_id, tuple(
            PreparedQuestion(qid, q.keys, None, 7) for qid, q in request.questions
        ), 7 * len(request.questions))

    def evaluate(self, batch, *, should_stop=None):
        self.calls += 1
        if self.evaluate_exc:
            raise self.evaluate_exc
        answers = []
        for q in batch.questions:
            probs = (self.dist or {}).get(q.question_id)
            if probs is None:
                probs = {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(q.keys)}
            answers.append(QuestionDistribution(q.question_id, probs))
        return DecisionBatch(tuple(answers), batch.input_tokens)

    def close(self):
        pass


def _client(*backends, other=None):
    registry = ModalityRegistry()
    for b in backends:
        registry.register(MODALITY, b, manifest={"model_id": b.model_id, "modality": MODALITY})
    if other is not None:
        registry.register("chat/completion", other,
                          manifest={"model_id": other.model_id, "modality": "chat/completion"})
    app = create_app(registry=registry, routers={MODALITY: build_router(registry)})
    return TestClient(app)


def _post(client, payload=None, *, raw=None, headers=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    return client.post("/v1/systemone", content=data,
                       headers={"content-type": "application/json", **(headers or {})})


def _err(r):
    return r.json()["error"]


def test_mixed_request_returns_the_full_envelope(mixed_payload):
    backend = FakeBackend()
    r = _post(_client(backend), mixed_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "decision-0.8b-q8"
    assert list(body["answers"]) == ["team", "duplicate", "urgency"]
    assert body["answers"]["team"]["choice"] == "billing"
    assert body["answers"]["duplicate"] == {"type": "noul", "noul": 1.0}
    assert body["usage"] == {"input_tokens": 21, "output_tokens": 0}


def test_unknown_model_is_404(mixed_payload):
    mixed_payload["model"] = "nope"
    r = _post(_client(FakeBackend()), mixed_payload)
    assert r.status_code == 404 and _err(r)["code"] == "model_not_found"


def test_model_of_another_modality_is_404(mixed_payload):
    chat = FakeBackend("chatty")
    mixed_payload["model"] = "chatty"
    r = _post(_client(FakeBackend(), other=chat), mixed_payload)
    assert r.status_code == 404
    assert chat.calls == 0


def test_model_is_required_even_with_one_model_loaded(mixed_payload):
    del mixed_payload["model"]
    r = _post(_client(FakeBackend()), mixed_payload)
    assert r.status_code == 400 and _err(r)["code"] == "missing_field"


def test_duplicate_model_key_is_rejected():
    backend = FakeBackend()
    raw = (b'{"model":"other","model":"decision-0.8b-q8","state":"s",'
           b'"questions":{"q":{"type":"noul","instructions":"i"}}}')
    r = _post(_client(backend), raw=raw)
    assert r.status_code == 400 and _err(r)["code"] == "duplicate_key"
    assert backend.calls == 0


def test_body_over_limit_with_content_length_is_413(mixed_payload):
    raw = json.dumps(mixed_payload).encode()
    raw = raw[:-1] + b" " * (MAX_BODY_BYTES + 1 - len(raw)) + b"}"
    r = _post(_client(FakeBackend()), raw=raw)
    assert r.status_code == 413 and _err(r)["code"] == "body_too_large"


def test_streamed_body_over_limit_without_content_length_is_413():
    backend = FakeBackend()

    def chunks():
        yield b'{"model":"decision-0.8b-q8","state":"'
        for _ in range(MAX_BODY_BYTES // 4096 + 2):
            yield b"x" * 4096
        yield b'","questions":{}}'

    client = _client(backend)
    r = client.post("/v1/systemone", content=chunks(),
                    headers={"content-type": "application/json"})
    assert r.request.headers.get("content-length") is None
    assert r.status_code == 413
    assert backend.calls == 0


def test_schema_errors_are_400(mixed_payload):
    mixed_payload["questions"]["team"]["type"] = "rank"
    r = _post(_client(FakeBackend()), mixed_payload)
    assert r.status_code == 400 and _err(r)["code"] == "unsupported_question_type"


def test_state_over_limit_is_413():
    raw = json.dumps({"model": "decision-0.8b-q8", "state": "x" * (64 * 1024 + 1),
                      "questions": {"q": {"type": "noul", "instructions": "i"}}}).encode()
    r = _post(_client(FakeBackend()), raw=raw)
    assert r.status_code == 413 and _err(r)["code"] == "state_too_large"


def test_token_limit_from_prepare_is_400(mixed_payload):
    backend = FakeBackend(prepare_exc=DecisionInputError("input_too_long", "questions[0] is long"))
    r = _post(_client(backend), mixed_payload)
    assert r.status_code == 400 and _err(r)["code"] == "input_too_long"
    assert backend.calls == 0


def test_invalid_numerics_are_a_sanitized_500(mixed_payload):
    backend = FakeBackend(dist={"team": {"billing": float("nan"), "technical": 0.31337}})
    r = _post(_client(backend), mixed_payload)
    assert r.status_code == 500
    assert _err(r)["code"] == "decision_backend_error"
    assert "31337" not in r.text and "billing" not in r.text


def test_native_exception_text_never_reaches_the_client(mixed_payload):
    backend = FakeBackend(evaluate_exc=RuntimeError("failed at /home/secret/model.gguf: I was charged"))
    r = _post(_client(backend), mixed_payload)
    assert r.status_code == 500
    assert "/home/secret" not in r.text and "charged" not in r.text


def test_unusable_backend_is_503_and_fails_health(mixed_payload):
    backend = FakeBackend()
    backend.usable = False
    client = _client(backend)
    r = _post(client, mixed_payload)
    assert r.status_code == 503 and _err(r)["code"] == "model_unusable"
    assert backend.calls == 0
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["unusable_models"] == ["decision-0.8b-q8"]


def test_poisoning_failure_turns_health_red(mixed_payload):
    class Poisoning(FakeBackend):
        def evaluate(self, batch, *, should_stop=None):
            self.usable = False
            raise DecisionBackendError("inference_failed")

    backend = Poisoning()
    client = _client(backend)
    assert client.get("/health").status_code == 200
    assert _post(client, mixed_payload).status_code == 500
    assert client.get("/health").status_code == 503
    assert _post(client, mixed_payload).status_code == 503


def test_health_is_unaffected_for_backends_without_the_attribute():
    class Plain:
        model_id = "plain"

    registry = ModalityRegistry()
    registry.register("audio/speech", Plain(), manifest={"model_id": "plain", "modality": "audio/speech"})
    client = TestClient(create_app(registry=registry, routers={}))
    assert client.get("/health").status_code == 200


async def test_client_disconnect_stops_the_batch_between_questions():
    import asyncio
    import threading

    from muse.modalities.decision_evaluation.protocol import DecisionCancelled

    class Slow(FakeBackend):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.evaluated = 0

        def evaluate(self, batch, *, should_stop=None):
            for _ in batch.questions:
                if should_stop():
                    raise DecisionCancelled()
                self.started.set()
                self.evaluated += 1
                threading.Event().wait(0.05)
            return super().evaluate(batch)

    backend = Slow()
    registry = ModalityRegistry()
    registry.register(MODALITY, backend, manifest={"model_id": backend.model_id, "modality": MODALITY})
    app = create_app(registry=registry, routers={MODALITY: build_router(registry)})
    body = json.dumps({"model": backend.model_id, "state": "s", "questions": {
        f"q{i}": {"type": "noul", "instructions": "i"} for i in range(16)
    }}).encode()
    gone = asyncio.Event()
    sent = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if sent:
            return sent.pop()
        await gone.wait()
        return {"type": "http.disconnect"}

    responses = []

    async def send(message):
        responses.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/systemone", "raw_path": b"/v1/systemone",
        "root_path": "", "query_string": b"", "server": ("test", 80), "client": ("test", 1),
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    }
    call = asyncio.ensure_future(app(scope, receive, send))
    while not backend.started.is_set():
        await asyncio.sleep(0.005)
    gone.set()
    await asyncio.wait_for(call, 10)
    assert responses[0]["status"] == 499
    assert 1 <= backend.evaluated < 16
