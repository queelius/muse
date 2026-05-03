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


def test_oversized_batch_returns_400(monkeypatch):
    """A list bigger than MUSE_MODERATIONS_MAX_BATCH is rejected up front."""
    monkeypatch.setenv("MUSE_MODERATIONS_MAX_BATCH", "3")
    # Reimport so the new env var is reflected (module reads at import time).
    import importlib
    from muse.modalities.text_classification import routes as routes_mod
    importlib.reload(routes_mod)

    backend = _fake_classify_single({"OK": 1.0})
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "text-moderation"})
    app = create_app(registry=reg, routers={MODALITY: routes_mod.build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/moderations", json={
        "input": ["a", "b", "c", "d"],
    })
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "invalid_parameter"
    assert "MUSE_MODERATIONS_MAX_BATCH=3" in body["message"]


def test_oversized_item_returns_400(monkeypatch):
    """A single string bigger than the per-item cap is rejected."""
    monkeypatch.setenv("MUSE_MODERATIONS_MAX_CHARS_PER_ITEM", "10")
    import importlib
    from muse.modalities.text_classification import routes as routes_mod
    importlib.reload(routes_mod)

    backend = _fake_classify_single({"OK": 1.0})
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "text-moderation"})
    app = create_app(registry=reg, routers={MODALITY: routes_mod.build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/moderations", json={
        "input": "a" * 100,
    })
    assert r.status_code == 400
    assert "MUSE_MODERATIONS_MAX_CHARS_PER_ITEM=10" in r.json()["error"]["message"]


# ---------- v0.35.0: /v1/text/classifications ----------


def _fake_zs_backend(model_id="zs-fake"):
    """Build a fake backend that satisfies ZeroShotClassifierModel."""
    backend = MagicMock()
    backend.model_id = model_id

    def _classify_zs(input, *, candidate_labels, multi_label=False):
        # Mimic NLI scores: first label wins.
        items = [input] if isinstance(input, str) else list(input)
        out = []
        for _ in items:
            scores = {label: max(0.0, 1.0 - 0.1 * i)
                      for i, label in enumerate(candidate_labels)}
            out.append(ClassificationResult(scores=scores, multi_label=multi_label))
        return out

    backend.classify_zero_shot = MagicMock(side_effect=_classify_zs)
    # Don't set classify; we want hasattr(backend, 'classify_zero_shot') to be
    # True and the route to dispatch to it. Backend may or may not have
    # classify; the capability flag decides.
    return backend


def _fake_classifier_backend(model_id="cls-fake"):
    """Build a fake backend that satisfies TextClassifierModel only."""
    backend = MagicMock()
    backend.model_id = model_id
    backend.classify.return_value = [
        ClassificationResult(
            scores={"positive": 0.7, "neutral": 0.2, "negative": 0.1},
            multi_label=False,
        ),
    ]
    # Critical: del classify_zero_shot so hasattr returns False; the
    # MagicMock auto-attributes it otherwise.
    del backend.classify_zero_shot
    return backend


def test_classifications_with_fine_tuned_classifier():
    """A fine-tuned classifier (supports_classification=True) returns
    the per-input list of {label, score} pairs sorted by score desc."""
    backend = _fake_classifier_backend("sentiment-fake")
    manifest = {
        "model_id": "sentiment-fake",
        "capabilities": {"supports_classification": True},
    }
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "sentiment-fake",
        "input": "I love this!",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "sentiment-fake"
    assert body["id"].startswith("classify-")
    assert len(body["results"]) == 1
    pairs = body["results"][0]
    assert pairs[0]["label"] == "positive"
    assert pairs[0]["score"] == 0.7


def test_classifications_with_zero_shot_dispatch():
    """A zero-shot model (supports_zero_shot=True) accepts candidate_labels
    and routes through classify_zero_shot."""
    backend = _fake_zs_backend("zs-fake")
    manifest = {
        "model_id": "zs-fake",
        "capabilities": {"supports_zero_shot": True},
    }
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "zs-fake",
        "input": "I love this!",
        "candidate_labels": ["happy", "sad", "angry"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0][0]["label"] == "happy"


def test_classifications_zero_shot_with_multi_label_forwarded():
    """multi_label=True is passed through to classify_zero_shot."""
    backend = _fake_zs_backend("zs-fake")
    manifest = {"model_id": "zs-fake",
                "capabilities": {"supports_zero_shot": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "zs-fake",
        "input": "x",
        "candidate_labels": ["a", "b"],
        "multi_label": True,
    })
    assert r.status_code == 200
    call_kwargs = backend.classify_zero_shot.call_args.kwargs
    assert call_kwargs["multi_label"] is True


def test_classifications_capability_gate_zero_shot_required():
    """A request with candidate_labels against a model that does NOT
    advertise supports_zero_shot returns 400 zero_shot_not_supported."""
    backend = _fake_classifier_backend("sentiment-fake")
    manifest = {"model_id": "sentiment-fake",
                "capabilities": {"supports_classification": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "sentiment-fake",
        "input": "x",
        "candidate_labels": ["a", "b"],
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "zero_shot_not_supported"
    backend.classify.assert_not_called()


def test_classifications_capability_gate_classification_required():
    """A request without candidate_labels against a zero-shot-only
    model returns 400 candidate_labels_required."""
    backend = _fake_zs_backend("zs-fake")
    manifest = {"model_id": "zs-fake",
                "capabilities": {"supports_zero_shot": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "zs-fake",
        "input": "x",
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "candidate_labels_required"
    backend.classify_zero_shot.assert_not_called()


def test_classifications_top_k_truncates():
    """top_k=2 returns at most 2 labels per input."""
    backend = _fake_classifier_backend("sentiment-fake")
    manifest = {"model_id": "sentiment-fake",
                "capabilities": {"supports_classification": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "sentiment-fake",
        "input": "x",
        "top_k": 2,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"][0]) == 2
    # Top two: positive (0.7) and neutral (0.2)
    assert [p["label"] for p in body["results"][0]] == ["positive", "neutral"]


def test_classifications_list_input_yields_list_of_lists():
    """input=[s1, s2] returns results=[[...], [...]]."""
    backend = MagicMock()
    backend.model_id = "sentiment-fake"
    del backend.classify_zero_shot
    backend.classify.return_value = [
        ClassificationResult(scores={"a": 0.9}, multi_label=False),
        ClassificationResult(scores={"b": 0.8}, multi_label=False),
    ]
    manifest = {"model_id": "sentiment-fake",
                "capabilities": {"supports_classification": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "sentiment-fake",
        "input": ["s1", "s2"],
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["results"][0][0]["label"] == "a"
    assert body["results"][1][0]["label"] == "b"


def test_classifications_empty_input_returns_400():
    backend = _fake_classifier_backend("sentiment-fake")
    manifest = {"model_id": "sentiment-fake",
                "capabilities": {"supports_classification": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "sentiment-fake",
        "input": "",
    })
    assert r.status_code == 400


def test_classifications_unknown_model_returns_404():
    backend = _fake_classifier_backend("real-model")
    manifest = {"model_id": "real-model",
                "capabilities": {"supports_classification": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "ghost",
        "input": "x",
    })
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_classifications_invalid_top_k_returns_422():
    """Pydantic validates top_k: must be >= 1."""
    backend = _fake_classifier_backend("sentiment-fake")
    manifest = {"model_id": "sentiment-fake",
                "capabilities": {"supports_classification": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "sentiment-fake",
        "input": "x",
        "top_k": 0,
    })
    assert r.status_code == 422


def test_classifications_empty_candidate_labels_returns_400():
    """Empty candidate_labels (not None) returns 400."""
    backend = _fake_zs_backend("zs-fake")
    manifest = {"model_id": "zs-fake",
                "capabilities": {"supports_zero_shot": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "zs-fake",
        "input": "x",
        "candidate_labels": [],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_classifications_misconfigured_model_returns_500():
    """A model that claims supports_zero_shot=True but doesn't have a
    classify_zero_shot method is a misconfigured curated entry."""
    backend = _fake_classifier_backend("misconfigured")
    manifest = {"model_id": "misconfigured",
                "capabilities": {"supports_zero_shot": True}}
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/text/classifications", json={
        "model": "misconfigured",
        "input": "x",
        "candidate_labels": ["a"],
    })
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
