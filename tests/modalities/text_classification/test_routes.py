"""Route tests for POST /v1/moderations."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_classification import (
    MODALITY,
    ClassificationResult,
    build_router,
)


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "text-moderation"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def _fake_classify_single(scores, multi_label=False):
    """Build a fake backend whose classify returns one ClassificationResult."""
    backend = MagicMock()
    backend.model_id = "text-moderation"
    backend.classify.return_value = [
        ClassificationResult(scores=scores, multi_label=multi_label),
    ]
    return backend


def test_returns_openai_envelope_for_scalar_input():
    backend = _fake_classify_single({"H": 0.7, "OK": 0.3}, multi_label=False)
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={
        "input": "I hate everything",
        "model": "text-moderation",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "text-moderation"
    assert body["id"].startswith("modr-")
    assert len(body["results"]) == 1
    res0 = body["results"][0]
    assert res0["flagged"] is True
    assert res0["categories"]["H"] is True
    assert res0["category_scores"]["H"] == 0.7

    args, _ = backend.classify.call_args
    assert args[0] == "I hate everything"


def test_returns_envelope_for_batch_input():
    # Single-label: argmax wins. "OK" at 0.9 is the argmax of result[0],
    # so flagged=True (0.9 >= 0.5 default threshold). Labels are opaque;
    # no special "safe" label treatment. "H" at 0.9 is argmax of result[1],
    # also flagged=True. This test verifies ordering and count, not the
    # flagging semantics (those live in test_codec.py).
    backend = MagicMock()
    backend.model_id = "text-moderation"
    backend.classify.return_value = [
        ClassificationResult(scores={"OK": 0.4, "H": 0.1}, multi_label=False),
        ClassificationResult(scores={"OK": 0.1, "H": 0.9}, multi_label=False),
    ]
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={
        "input": ["hello world", "I hate everything"],
        "model": "text-moderation",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["flagged"] is False
    assert body["results"][1]["flagged"] is True


def test_threshold_request_field_overrides_default():
    """A request with threshold=0.9 demotes a 0.7 score to not-flagged."""
    backend = _fake_classify_single({"H": 0.7, "OK": 0.3}, multi_label=False)
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={
        "input": "borderline",
        "model": "text-moderation",
        "threshold": 0.9,
    })
    assert r.status_code == 200
    res0 = r.json()["results"][0]
    assert res0["flagged"] is False
    assert res0["categories"]["H"] is False


def test_manifest_threshold_used_when_request_omits():
    backend = _fake_classify_single({"H": 0.7}, multi_label=True)
    client = _make_client(backend, manifest={
        "model_id": "text-moderation",
        "capabilities": {"flag_threshold": 0.9},
    })

    r = client.post("/v1/moderations", json={
        "input": "borderline",
        "model": "text-moderation",
    })
    res0 = r.json()["results"][0]
    assert res0["flagged"] is False  # 0.7 < manifest threshold 0.9


def test_threshold_out_of_range_returns_400():
    backend = _fake_classify_single({"OK": 1.0})
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={
        "input": "x",
        "model": "text-moderation",
        "threshold": 1.5,
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "invalid_parameter"


def test_empty_input_returns_400():
    backend = _fake_classify_single({"OK": 1.0})
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={"input": "", "model": "text-moderation"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_unknown_model_returns_404_envelope():
    backend = _fake_classify_single({"OK": 1.0})
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={
        "input": "x",
        "model": "no-such-model",
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_default_model_used_when_field_omitted():
    """Requests without `model` use the registry's default for this modality."""
    backend = _fake_classify_single({"OK": 1.0})
    client = _make_client(backend)

    r = client.post("/v1/moderations", json={"input": "hi"})
    assert r.status_code == 200
    assert r.json()["model"] == "text-moderation"
