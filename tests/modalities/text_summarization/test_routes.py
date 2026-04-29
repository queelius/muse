"""Route tests for POST /v1/summarize."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_summarization import (
    MODALITY,
    SummarizationResult,
    build_router,
)


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "bart-large-cnn"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def _fake_backend(result):
    backend = MagicMock()
    backend.model_id = "bart-large-cnn"
    backend.summarize.return_value = result
    return backend


def _fake_result(**overrides):
    base = dict(
        summary="muse is a multi-modality server.",
        length="medium",
        format="paragraph",
        model_id="bart-large-cnn",
        prompt_tokens=412,
        completion_tokens=67,
    )
    base.update(overrides)
    return SummarizationResult(**base)


def test_summarize_returns_envelope_for_minimal_request():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "bart-large-cnn"
    assert body["id"].startswith("sum-")
    assert body["summary"] == "muse is a multi-modality server."
    assert body["usage"]["prompt_tokens"] == 412
    assert body["usage"]["completion_tokens"] == 67
    assert body["usage"]["total_tokens"] == 479
    assert body["meta"]["length"] == "medium"
    assert body["meta"]["format"] == "paragraph"


def test_summarize_passes_length_to_backend():
    backend = _fake_backend(_fake_result(length="short"))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={
        "text": "hello", "length": "short",
    })
    assert r.status_code == 200
    args, _ = backend.summarize.call_args
    # backend.summarize(text, length, format)
    assert args[1] == "short"


def test_summarize_passes_format_to_backend():
    backend = _fake_backend(_fake_result(format="bullets"))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={
        "text": "hello", "format": "bullets",
    })
    assert r.status_code == 200
    args, _ = backend.summarize.call_args
    assert args[2] == "bullets"


def test_summarize_default_length_medium():
    backend = _fake_backend(_fake_result(length="medium"))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "hello"})
    assert r.status_code == 200
    args, _ = backend.summarize.call_args
    assert args[1] == "medium"


def test_summarize_default_format_paragraph():
    backend = _fake_backend(_fake_result(format="paragraph"))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "hello"})
    assert r.status_code == 200
    args, _ = backend.summarize.call_args
    assert args[2] == "paragraph"


def test_summarize_400_on_empty_text():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": ""})
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "invalid_parameter"
    assert "text" in body["error"]["message"].lower()


def test_summarize_400_on_invalid_length():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={
        "text": "hello", "length": "WRONG",
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "length" in body["error"]["message"].lower()


def test_summarize_400_on_invalid_format():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={
        "text": "hello", "format": "WRONG",
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "format" in body["error"]["message"].lower()


def test_summarize_400_on_text_too_long(monkeypatch):
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    huge_text = "x" * 100_001  # default cap is 100000
    r = client.post("/v1/summarize", json={"text": huge_text})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_summarize_404_on_unknown_model():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={
        "text": "hello", "model": "nonexistent-summarizer",
    })
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_summarize_default_model_resolves_first_registered():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "hello"})
    assert r.status_code == 200
    assert r.json()["model"] == "bart-large-cnn"


def test_summarize_passes_through_runtime_metadata():
    """Backend metadata flows into response.meta unchanged."""
    backend = _fake_backend(SummarizationResult(
        summary="x",
        length="medium",
        format="paragraph",
        model_id="bart-large-cnn",
        prompt_tokens=2000,
        completion_tokens=10,
        metadata={"truncation_warning": True, "truncated_from_tokens": 2000},
    ))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "long input"})
    body = r.json()
    assert body["meta"]["truncation_warning"] is True
    assert body["meta"]["truncated_from_tokens"] == 2000


def test_summarize_envelope_total_tokens_is_sum():
    backend = _fake_backend(_fake_result(
        prompt_tokens=100, completion_tokens=23,
    ))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "x"})
    body = r.json()
    assert body["usage"]["total_tokens"] == 123


def test_summarize_envelope_id_unique_per_request():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    a = client.post("/v1/summarize", json={"text": "x"}).json()
    b = client.post("/v1/summarize", json={"text": "y"}).json()
    assert a["id"] != b["id"]


def test_summarize_400_uses_error_envelope_not_detail():
    """muse error envelope is {"error": {...}}, not FastAPI's {"detail": ...}."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": ""})
    body = r.json()
    assert "detail" not in body
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert "type" in body["error"]


def test_summarize_404_uses_error_envelope_not_detail():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={
        "text": "hello", "model": "nonexistent",
    })
    body = r.json()
    assert "detail" not in body
    assert "error" in body


def test_summarize_passes_text_argument_to_backend():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)

    client.post("/v1/summarize", json={"text": "specific input string"})
    args, _ = backend.summarize.call_args
    assert args[0] == "specific input string"


def test_summarize_returns_model_id_from_backend_not_request():
    """The response 'model' field should come from backend.model_id,
    not the request body. Important for canonical id reporting."""
    backend = _fake_backend(_fake_result(model_id="bart-large-cnn"))
    client = _make_client(backend)

    r = client.post("/v1/summarize", json={"text": "hello"})
    body = r.json()
    assert body["model"] == "bart-large-cnn"
