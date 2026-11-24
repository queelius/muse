"""Tests for batched decision evaluation (POST /v1/systemone/batch)."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from muse.cli_impl.queueing import QueueFull, QueueTimeout
from muse.core.discovery import (
    disconnect_cancellable_paths,
    modality_discovery,
    strict_request_parsers,
)
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.decision_evaluation import (
    CANCEL_ON_DISCONNECT_PATHS,
    DISCOVERY,
    MODALITY,
    STRICT_REQUEST_PARSERS,
    DecisionBatchItem,
    DecisionBatchRequest,
    DecisionClient,
    build_router,
)
from muse.modalities.decision_evaluation.client import encode_batch_request
from muse.modalities.decision_evaluation.codec import encode_batch_response
from muse.modalities.decision_evaluation.execution import DecisionExecutor
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionCancelled,
    DecisionInputError,
    PreparedBatch,
    PreparedQuestion,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.validation import (
    MAX_BATCH_BODY_BYTES,
    MAX_BATCH_ITEMS,
    MAX_KEY_BYTES,
    MAX_STATE_BYTES,
    MAX_STATE_DEPTH,
    parse_batch_request,
)


def _noul(instructions: str = "Is it urgent?") -> dict:
    return {"type": "noul", "instructions": instructions}


def _choice(n: int = 2) -> dict:
    return {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {f"opt{i}": f"Option {i}" for i in range(n)},
    }


def _item(
    *,
    item_id: object = ...,
    state: object = "obs",
    questions: object = None,
) -> dict:
    out: dict = {
        "state": state,
        "questions": {"q": _noul()} if questions is None else questions,
    }
    if item_id is not ...:
        out["id"] = item_id
    return out


def _batch_payload(items: list | object, *, model: object = "decision-0.8b-q8") -> dict:
    return {"model": model, "items": items}


def _err(body: bytes) -> DecisionInputError:
    with pytest.raises(DecisionInputError) as exc:
        parse_batch_request(body)
    return exc.value


def _pad_to(body: bytes, size: int) -> bytes:
    assert body.endswith(b"}")
    return body[:-1] + b" " * (size - len(body)) + b"}"


def _nest(depth: int):
    value = "leaf"
    for i in range(depth):
        value = [value] if i % 2 else {"k": value}
    return value


class FakeBatchBackend:
    def __init__(
        self,
        model_id: str = "decision-0.8b-q8",
        *,
        dist: dict | None = None,
        prepare_exc: Exception | None = None,
        fail_prepare_index: int | None = None,
        evaluate_exc: Exception | None = None,
    ) -> None:
        self.model_id = model_id
        self.dist = dist
        self.prepare_exc = prepare_exc
        self.fail_prepare_index = fail_prepare_index
        self.evaluate_exc = evaluate_exc
        self.usable = True
        self.prepare_calls = 0
        self.evaluate_calls = 0
        self.seen_states: list = []

    def prepare(self, request):
        idx = self.prepare_calls
        self.prepare_calls += 1
        if self.prepare_exc is not None and (
            self.fail_prepare_index is None or idx == self.fail_prepare_index
        ):
            raise self.prepare_exc
        self.seen_states.append(request.state)
        return PreparedBatch(
            self.model_id,
            tuple(PreparedQuestion(qid, q.keys, request.state, 5) for qid, q in request.questions),
            5 * len(request.questions),
        )

    def evaluate(self, batch, *, should_stop=None):
        self.evaluate_calls += 1
        if self.evaluate_exc is not None:
            raise self.evaluate_exc
        answers = []
        for q in batch.questions:
            if should_stop is not None and should_stop():
                raise DecisionCancelled()
            probs = (self.dist or {}).get(q.question_id)
            if probs is None:
                probs = {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(q.keys)}
            answers.append(QuestionDistribution(q.question_id, probs))
        return DecisionBatch(tuple(answers), batch.input_tokens)

    def close(self) -> None:
        pass


def _client(*backends, other=None) -> TestClient:
    registry = ModalityRegistry()
    for b in backends:
        registry.register(MODALITY, b, manifest={"model_id": b.model_id, "modality": MODALITY})
    if other is not None:
        registry.register(
            "chat/completion",
            other,
            manifest={"model_id": other.model_id, "modality": "chat/completion"},
        )
    app = create_app(registry=registry, routers={MODALITY: build_router(registry)})
    return TestClient(app)


def _post_batch(client: TestClient, payload=None, *, raw: bytes | None = None, headers=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return client.post(
        "/v1/systemone/batch",
        content=data,
        headers={"content-type": "application/json", **(headers or {})},
    )


# --- discovery & registration ----------------------------------------------


def test_discovery_declares_batch_route_parser_and_disconnect():
    assert STRICT_REQUEST_PARSERS["/v1/systemone/batch"] == (
        "muse.modalities.decision_evaluation.validation:parse_batch_request"
    )
    assert strict_request_parsers()["/v1/systemone/batch"] == (
        "muse.modalities.decision_evaluation.validation:parse_batch_request"
    )
    assert isinstance(CANCEL_ON_DISCONNECT_PATHS, tuple)
    assert "/v1/systemone/batch" in CANCEL_ON_DISCONNECT_PATHS
    assert "/v1/systemone/batch" in disconnect_cancellable_paths()

    routes = {r["path"]: r for r in DISCOVERY["routes"]}
    assert "/v1/systemone/batch" in routes
    batch_decl = routes["/v1/systemone/batch"]
    assert batch_decl["method"] == "POST"
    assert batch_decl["request_content_type"] == "application/json"
    assert batch_decl["response_content_type"] == "application/json"
    assert batch_decl["compat"] is None
    assert batch_decl["mcp_tool"] == "muse_decide_batch"
    assert batch_decl["file_fields"] == []
    assert batch_decl["media_fields"] == []

    # Verify AST discovery and example validity.
    ast_routes = {r["path"]: r for r in modality_discovery()[MODALITY]["routes"]}
    example = {
        k: ("m" if v == "<model>" else v)
        for k, v in ast_routes["/v1/systemone/batch"]["example"].items()
    }
    parsed = parse_batch_request(json.dumps(example).encode("utf-8"))
    assert isinstance(parsed, DecisionBatchRequest)
    assert parsed.model == "m"
    assert len(parsed.items) == 1


# --- strict validation -----------------------------------------------------


def test_parse_batch_happy_path_preserves_order_ids_and_unicode(encode_body):
    payload = _batch_payload([
        _item(item_id="npc-é", state={"hp": 20}, questions={"alert": _choice(2)}),
        _item(state="patrol quiet", questions={"safe": _noul()}),
        _item(item_id=None, state=["event", 1], questions={
            "severity": {
                "type": "score",
                "instructions": "Severity?",
                "criteria": ["low", "high"],
            },
        }),
    ])
    req = parse_batch_request(encode_body(payload))
    assert isinstance(req, DecisionBatchRequest)
    assert req.model == "decision-0.8b-q8"
    assert len(req.items) == 3
    assert all(isinstance(it, DecisionBatchItem) for it in req.items)
    assert [it.id for it in req.items] == ["npc-é", None, None]
    assert req.items[0].id.encode("utf-8") == "npc-é".encode("utf-8")
    assert req.items[0].state == {"hp": 20}
    assert req.items[1].state == "patrol quiet"
    assert req.items[2].state == ["event", 1]


def test_batch_item_count_bounds(encode_body):
    assert _err(encode_body(_batch_payload([]))).code == "invalid_request"

    ok_max = _batch_payload([_item(item_id=f"i{i}") for i in range(MAX_BATCH_ITEMS)])
    assert len(parse_batch_request(encode_body(ok_max)).items) == MAX_BATCH_ITEMS

    over_max = _batch_payload([_item(item_id=f"i{i}") for i in range(MAX_BATCH_ITEMS + 1)])
    err = _err(encode_body(over_max))
    assert (err.code, err.status) == ("invalid_request", 400)


def test_batch_body_size_limit(encode_body):
    base = encode_body(_batch_payload([_item(item_id="a")]))
    assert len(parse_batch_request(_pad_to(base, MAX_BATCH_BODY_BYTES)).items) == 1
    err = _err(_pad_to(base, MAX_BATCH_BODY_BYTES + 1))
    assert (err.code, err.status) == ("body_too_large", 413)


def test_batch_per_item_state_limits_independent_of_outer_nesting(encode_body):
    # Depth MAX_STATE_DEPTH inside items[0].state passes; +1 fails.
    assert parse_batch_request(encode_body(_batch_payload([_item(state=_nest(MAX_STATE_DEPTH))])))
    err = _err(encode_body(_batch_payload([_item(state=_nest(MAX_STATE_DEPTH + 1))])))
    assert (err.code, err.status) == ("state_too_deep", 400)

    # State bytes > MAX_STATE_BYTES fails with 413 even inside 512 KiB batch budget.
    err_size = _err(encode_body(_batch_payload([_item(state="x" * (MAX_STATE_BYTES + 1))])))
    assert (err_size.code, err_size.status) == ("state_too_large", 413)


def test_batch_item_id_rules_and_duplicates(encode_body):
    # Boundary length in UTF-8 bytes passes; +1 fails.
    ok_id = "é" * (MAX_KEY_BYTES // 2)
    assert parse_batch_request(encode_body(_batch_payload([_item(item_id=ok_id)]))).items[0].id == ok_id
    assert _err(encode_body(_batch_payload([_item(item_id=ok_id + "a")]))).status == 400

    # Empty or non-string ids fail with 400.
    for bad in ("", 123, True, False, [], {}):
        err = _err(encode_body(_batch_payload([_item(item_id=bad)])))
        assert err.status == 400

    # Duplicate non-null ids fail with 400.
    dup = _batch_payload([_item(item_id="npc-1"), _item(item_id="npc-1")])
    err = _err(encode_body(dup))
    assert err.status == 400

    # Multiple null / omitted ids are allowed.
    nulls = _batch_payload([_item(), _item(item_id=None), _item(item_id=None)])
    assert [it.id for it in parse_batch_request(encode_body(nulls)).items] == [None, None, None]


@pytest.mark.parametrize("payload, code", [
    ({"items": [_item()]}, "missing_field"),
    ({"model": "m"}, "missing_field"),
    ({"model": "", "items": [_item()]}, "invalid_model"),
    ({"model": "m", "items": "not-a-list"}, "invalid_request"),
    ({"model": "m", "items": {}}, "invalid_request"),
    ({"model": "m", "items": ["not-an-object"]}, "invalid_request"),
    ({"model": "m", "items": [_item()], "extra": 1}, "unsupported_field"),
    ({"model": "m", "items": [_item()], "images": []}, "unsupported_media"),
    ({"model": "m", "items": [{**_item(), "extra": 1}]}, "unsupported_field"),
    ({"model": "m", "items": [{**_item(), "image": "x"}]}, "unsupported_media"),
    ({"model": "m", "items": [{"state": "s"}]}, "missing_field"),
    ({"model": "m", "items": [{"questions": {"q": _noul()}}]}, "missing_field"),
    ({"model": "m", "items": [_item(state="")]}, "invalid_state"),
    ({"model": "m", "items": [_item(questions={})]}, "invalid_questions"),
])
def test_batch_schema_rejections(payload, code, encode_body):
    assert _err(encode_body(payload)).code == code


def test_batch_strict_json_duplicates_nan_and_surrogates():
    dup_top = b'{"model":"a","model":"b","items":[{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}]}'
    assert _err(dup_top).code == "duplicate_key"

    dup_item = b'{"model":"m","items":[{"id":"a","id":"b","state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}]}'
    assert _err(dup_item).code == "duplicate_key"

    nan_body = b'{"model":"m","items":[{"state":{"v":NaN},"questions":{"q":{"type":"noul","instructions":"i"}}}]}'
    assert _err(nan_body).code == "non_finite_number"

    surr_id = b'{"model":"m","items":[{"id":"\\ud800","state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}]}'
    assert _err(surr_id).code == "invalid_unicode"

    bom = b'\xef\xbb\xbf{"model":"m","items":[{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}]}'
    assert _err(bom).status == 400


@pytest.mark.parametrize("payload", [
    _batch_payload([_item(item_id="SECRET" * 30)]),
    _batch_payload([_item(item_id="SECRET"), _item(item_id="SECRET")]),
    {"model": "m", "items": [_item()], "SECRETFIELD": 1},
    {"model": "m", "items": [{**_item(), "SECRETFIELD": 1}]},
    _batch_payload([_item(questions={"SECRET" * 30: _noul()})]),
])
def test_batch_validation_errors_never_echo_user_content(payload, encode_body):
    err = _err(encode_body(payload))
    assert "SECRET" not in err.message
    assert "SECRET" not in str(err)


# --- codec -----------------------------------------------------------------


def test_encode_batch_response_shape_and_counts(encode_body):
    req = parse_batch_request(encode_body(_batch_payload([
        _item(item_id="npc-1", questions={"q1": _choice(2), "q2": _noul()}),
        _item(questions={"q3": _noul()}),
    ])))
    results = [
        DecisionBatch(
            answers=(
                QuestionDistribution("q1", {"opt0": 0.75, "opt1": 0.25}),
                QuestionDistribution("q2", {"true": 0.6, "false": 0.4}),
            ),
            input_tokens=15,
        ),
        DecisionBatch(
            answers=(QuestionDistribution("q3", {"true": 0.2, "false": 0.8}),),
            input_tokens=8,
        ),
    ]
    out = encode_batch_response(req, results, model_id="decision-0.8b-q8", elapsed_ms=12.5)
    assert out["object"] == "decision.batch"
    assert out["model"] == "decision-0.8b-q8"
    assert out["usage"] == {"items": 2, "questions": 3, "elapsed_ms": 12.5}
    assert len(out["results"]) == 2

    r0, r1 = out["results"]
    assert r0["index"] == 0 and r0["id"] == "npc-1"
    assert list(r0["answers"]) == ["q1", "q2"]
    assert r0["answers"]["q1"]["choice"] == "opt0"
    assert r0["answers"]["q2"] == {"type": "noul", "noul": 0.6}
    assert r0["usage"] == {"input_tokens": 15, "output_tokens": 0}

    assert r1["index"] == 1 and r1["id"] is None
    assert r1["answers"]["q3"] == {"type": "noul", "noul": 0.2}
    assert r1["usage"] == {"input_tokens": 8, "output_tokens": 0}


def test_encode_batch_response_rejects_mismatched_or_invalid_results(encode_body):
    req = parse_batch_request(encode_body(_batch_payload([_item(item_id="a"), _item(item_id="b")])))
    good = DecisionBatch((QuestionDistribution("q", {"true": 0.5, "false": 0.5}),), 4)

    with pytest.raises(DecisionBackendError) as exc:
        encode_batch_response(req, [good], model_id="m")
    assert exc.value.code == "incomplete_results"

    bad = DecisionBatch((QuestionDistribution("q", {"true": float("nan"), "false": 0.5}),), 4)
    with pytest.raises(DecisionBackendError) as exc:
        encode_batch_response(req, [good, bad], model_id="m")
    assert exc.value.code == "invalid_distribution"

    with pytest.raises(DecisionBackendError) as exc:
        encode_batch_response(req, [good, good], model_id="m", elapsed_ms=-1.0)
    assert exc.value.code == "invalid_usage"


# --- execution -------------------------------------------------------------


def _deadline(s: float = 10.0) -> float:
    return time.monotonic() + s


async def _until(predicate, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "condition never became true"
        await asyncio.sleep(0.005)


async def test_execute_batch_single_slot_no_interleaving(encode_body):
    ex = DecisionExecutor("m", max_concurrency=1, max_queue_depth=32)
    try:
        order: list[str] = []
        active = 0
        max_active = 0
        lock = threading.Lock()
        first_started = threading.Event()
        gate = threading.Event()

        class TrackingBackend(FakeBatchBackend):
            def evaluate(self, batch, *, should_stop=None):
                nonlocal active, max_active
                state = batch.questions[0].payload
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    order.append(state)
                if state == "A0":
                    first_started.set()
                    assert gate.wait(5)
                try:
                    return super().evaluate(batch, should_stop=should_stop)
                finally:
                    with lock:
                        active -= 1

        backend = TrackingBackend("m")
        req_a = parse_batch_request(encode_body(_batch_payload(
            [_item(state="A0"), _item(state="A1"), _item(state="A2")], model="m",
        )))
        req_b = parse_batch_request(encode_body(_batch_payload(
            [_item(state="B0"), _item(state="B1")], model="m",
        )))

        t_a = asyncio.ensure_future(ex.execute_batch(backend, req_a, deadline=_deadline()))
        await _until(first_started.is_set)
        t_b = asyncio.ensure_future(ex.execute_batch(backend, req_b, deadline=_deadline()))
        await _until(lambda: ex.queue_depth == 1)
        gate.set()
        res_a, res_b = await asyncio.wait_for(asyncio.gather(t_a, t_b), 5)

        assert len(res_a) == 3 and len(res_b) == 2
        assert max_active == 1
        # All items of batch A ran contiguously before any item of batch B.
        assert order == ["A0", "A1", "A2", "B0", "B1"]
    finally:
        await ex.close(timeout=1)


async def test_execute_batch_prepare_failure_runs_zero_inference(encode_body):
    ex = DecisionExecutor("m", max_concurrency=1, max_queue_depth=32)
    try:
        backend = FakeBatchBackend(
            "m",
            prepare_exc=DecisionInputError("input_too_long", "questions[0] too long"),
            fail_prepare_index=1,
        )
        req = parse_batch_request(encode_body(_batch_payload(
            [_item(state="ok"), _item(state="too-long")], model="m",
        )))
        with pytest.raises(DecisionInputError) as exc:
            await ex.execute_batch(backend, req, deadline=_deadline())
        assert exc.value.code == "input_too_long"
        assert "items[1]:" in exc.value.message
        assert backend.evaluate_calls == 0

        # Slot was released cleanly.
        await _until(lambda: ex.running == 0)
        backend.prepare_exc = None
        res = await ex.execute_batch(backend, req, deadline=_deadline(0))
        assert len(res) == 2
    finally:
        await ex.close(timeout=1)


async def test_execute_batch_cancels_between_items_and_questions(encode_body):
    ex = DecisionExecutor("m", max_concurrency=1, max_queue_depth=32)
    try:
        started = threading.Event()
        release_first = threading.Event()

        class SlowBackend(FakeBatchBackend):
            def evaluate(self, batch, *, should_stop=None):
                self.evaluate_calls += 1
                started.set()
                assert release_first.wait(5)
                if should_stop is not None and should_stop():
                    raise DecisionCancelled()
                return DecisionBatch(
                    tuple(
                        QuestionDistribution(q.question_id, {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(q.keys)})
                        for q in batch.questions
                    ),
                    batch.input_tokens,
                )

        backend = SlowBackend("m")
        req = parse_batch_request(encode_body(_batch_payload(
            [_item(state=f"s{i}") for i in range(5)], model="m",
        )))
        task = asyncio.ensure_future(ex.execute_batch(backend, req, deadline=_deadline()))
        await _until(started.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Slot still held until in-flight native item finishes.
        assert ex.running == 1
        release_first.set()
        await _until(lambda: ex.running == 0)
        # Never proceeded to items 1..4.
        assert backend.evaluate_calls == 1
    finally:
        await ex.close(timeout=1)


async def test_execute_batch_queue_timeout_and_full(encode_body, monkeypatch):
    monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
    ex = DecisionExecutor("m", max_concurrency=1, max_queue_depth=32)
    try:
        started = threading.Event()
        gate = threading.Event()

        class Blocking(FakeBatchBackend):
            def evaluate(self, batch, *, should_stop=None):
                started.set()
                assert gate.wait(5)
                return super().evaluate(batch, should_stop=should_stop)

        backend = Blocking("m")
        req = parse_batch_request(encode_body(_batch_payload([_item()], model="m")))
        first = asyncio.ensure_future(ex.execute_batch(backend, req, deadline=_deadline()))
        await _until(started.is_set)

        waiter = asyncio.ensure_future(ex.execute_batch(backend, req, deadline=_deadline()))
        await _until(lambda: ex.queue_depth == 1)
        with pytest.raises(QueueFull):
            await ex.execute_batch(backend, req, deadline=_deadline())

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(QueueTimeout):
            await ex.execute_batch(backend, req, deadline=_deadline(0.03))

        gate.set()
        await asyncio.wait_for(first, 5)
    finally:
        await ex.close(timeout=1)


# --- HTTP routes -----------------------------------------------------------


def test_batch_route_happy_path():
    backend = FakeBatchBackend()
    payload = _batch_payload([
        _item(item_id="npc-1", state="sees intruder", questions={"alert": _choice(2), "urgent": _noul()}),
        _item(state="merchant resting", questions={"open": _noul()}),
    ])
    r = _post_batch(_client(backend), payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "decision.batch"
    assert body["model"] == "decision-0.8b-q8"
    assert len(body["results"]) == 2
    assert body["results"][0]["index"] == 0
    assert body["results"][0]["id"] == "npc-1"
    assert body["results"][0]["answers"]["alert"]["choice"] == "opt0"
    assert body["results"][0]["usage"] == {"input_tokens": 10, "output_tokens": 0}
    assert body["results"][1]["index"] == 1
    assert body["results"][1]["id"] is None
    assert body["results"][1]["usage"] == {"input_tokens": 5, "output_tokens": 0}
    assert body["usage"]["items"] == 2
    assert body["usage"]["questions"] == 3
    assert body["usage"]["elapsed_ms"] >= 0.0


def test_batch_route_error_statuses():
    backend = FakeBatchBackend()
    client = _client(backend)

    # 404 unknown model
    r = _post_batch(client, _batch_payload([_item()], model="missing"))
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"

    # 400 empty items
    r = _post_batch(client, _batch_payload([]))
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"

    # 400 duplicate item id
    r = _post_batch(client, _batch_payload([_item(item_id="x"), _item(item_id="x")]))
    assert r.status_code == 400

    # 413 over 512 KiB with Content-Length
    raw = json.dumps(_batch_payload([_item()])).encode("utf-8")
    over = _pad_to(raw, MAX_BATCH_BODY_BYTES + 1)
    r = _post_batch(client, raw=over)
    assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"
    assert backend.evaluate_calls == 0


def test_batch_route_streamed_body_over_limit_is_413():
    backend = FakeBatchBackend()

    def chunks():
        yield b'{"model":"decision-0.8b-q8","items":[{"state":"'
        for _ in range(MAX_BATCH_BODY_BYTES // 4096 + 2):
            yield b"x" * 4096
        yield b'","questions":{}}]}'

    r = _client(backend).post(
        "/v1/systemone/batch",
        content=chunks(),
        headers={"content-type": "application/json"},
    )
    assert r.request.headers.get("content-length") is None
    assert r.status_code == 413
    assert backend.evaluate_calls == 0


def test_batch_route_sanitizes_backend_errors_and_respects_unusable():
    bad_backend = FakeBatchBackend(
        evaluate_exc=RuntimeError("crash at /secret/path with user text SECRET"),
    )
    r = _post_batch(_client(bad_backend), _batch_payload([_item()]))
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "decision_backend_error"
    assert "/secret/path" not in r.text and "SECRET" not in r.text

    unusable = FakeBatchBackend()
    unusable.usable = False
    r2 = _post_batch(_client(unusable), _batch_payload([_item()]))
    assert r2.status_code == 503 and r2.json()["error"]["code"] == "model_unusable"
    assert unusable.evaluate_calls == 0


async def test_batch_route_client_disconnect_stops_between_items():
    class SlowItemBackend(FakeBatchBackend):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.evaluated_items = 0

        def evaluate(self, batch, *, should_stop=None):
            if should_stop is not None and should_stop():
                raise DecisionCancelled()
            self.started.set()
            self.evaluated_items += 1
            threading.Event().wait(0.05)
            return super().evaluate(batch, should_stop=should_stop)

    backend = SlowItemBackend()
    registry = ModalityRegistry()
    registry.register(MODALITY, backend, manifest={"model_id": backend.model_id, "modality": MODALITY})
    app = create_app(registry=registry, routers={MODALITY: build_router(registry)})
    body = json.dumps(_batch_payload([_item(item_id=f"i{i}") for i in range(16)])).encode("utf-8")

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
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/systemone/batch",
        "raw_path": b"/v1/systemone/batch",
        "root_path": "",
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 1),
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    call = asyncio.ensure_future(app(scope, receive, send))
    while not backend.started.is_set():
        await asyncio.sleep(0.005)
    gone.set()
    await asyncio.wait_for(call, 10)
    assert responses[0]["status"] == 499
    assert 1 <= backend.evaluated_items < 16


# --- client ----------------------------------------------------------------


def test_decision_client_evaluate_batch_posts_validated_bytes():
    envelope = {
        "object": "decision.batch",
        "model": "decima-base",
        "results": [
            {
                "index": 0,
                "id": "npc-é",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        ],
        "usage": {"items": 1, "questions": 1, "elapsed_ms": 4.2},
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = envelope
    resp.raise_for_status.return_value = None

    items = [_item(item_id="npc-é", state={"msg": "café"})]
    with patch("muse.modalities.decision_evaluation.client.requests.post", return_value=resp) as post:
        out = DecisionClient("http://muse/").evaluate_batch(model="decima-base", items=items)

    assert out == envelope
    (url,), kwargs = post.call_args
    assert url == "http://muse/v1/systemone/batch"
    sent = kwargs["data"]
    assert sent == encode_batch_request(model="decima-base", items=items)
    assert "npc-é".encode("utf-8") in sent


@pytest.mark.parametrize("items", [
    [],
    [_item(item_id="dup"), _item(item_id="dup")],
    [_item(state={"x": float("nan")})],
    [{**_item(), "image": "x"}],
])
def test_client_evaluate_batch_fails_locally_before_http(items):
    with patch("muse.modalities.decision_evaluation.client.requests.post") as post:
        with pytest.raises(DecisionInputError):
            DecisionClient("http://muse").evaluate_batch(model="m", items=items)
    post.assert_not_called()


def test_client_evaluate_batch_keyword_only_and_http_errors():
    with pytest.raises(TypeError):
        DecisionClient("http://muse").evaluate_batch("m", [_item()])

    resp = MagicMock()
    resp.status_code = 503
    resp.raise_for_status.side_effect = requests.HTTPError("503")
    with patch("muse.modalities.decision_evaluation.client.requests.post", return_value=resp):
        with pytest.raises(requests.HTTPError):
            DecisionClient("http://muse").evaluate_batch(model="m", items=[_item()])
