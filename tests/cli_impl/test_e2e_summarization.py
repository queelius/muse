"""End-to-end: /v1/summarize through FastAPI + codec correctly.

Uses a fake SummarizationModel backend; no real weights.
"""
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_summarization import (
    MODALITY,
    SummarizationResult,
    build_router,
)


pytestmark = pytest.mark.slow


class _FakeSummarizer:
    """Deterministic fake: returns the first N words of input, padded
    with token counts derived from input/output length. Honors length
    via a frozen mapping so the e2e can assert max_new_tokens flowed
    through correctly."""

    LENGTH_TO_WORDS = {"short": 5, "medium": 10, "long": 20}

    def __init__(self):
        self.calls = []
        self.model_id = "fake-summarizer"

    def summarize(self, text, length="medium", format="paragraph"):
        self.calls.append((text, length, format))
        words = text.split()[: self.LENGTH_TO_WORDS[length]]
        summary = " ".join(words) + "."
        return SummarizationResult(
            summary=summary,
            length=length,
            format=format,
            model_id=self.model_id,
            prompt_tokens=len(text.split()),
            completion_tokens=len(words),
        )


@pytest.mark.timeout(10)
def test_summarize_full_request_response_cycle():
    fake = _FakeSummarizer()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-summarizer"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/summarize", json={
        "text": "the quick brown fox jumps over the lazy dog and runs away into the forest",
        "length": "short",
        "format": "paragraph",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-summarizer"
    assert body["id"].startswith("sum-")
    assert body["meta"]["length"] == "short"
    assert body["meta"]["format"] == "paragraph"
    # short=5 words -> "the quick brown fox jumps."
    assert body["summary"] == "the quick brown fox jumps."
    # usage rolls up correctly
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


@pytest.mark.timeout(10)
def test_summarize_e2e_length_changes_output_words():
    fake = _FakeSummarizer()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-summarizer"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    text = " ".join(f"w{i}" for i in range(50))

    short = client.post("/v1/summarize", json={
        "text": text, "length": "short",
    }).json()
    medium = client.post("/v1/summarize", json={
        "text": text, "length": "medium",
    }).json()
    long_ = client.post("/v1/summarize", json={
        "text": text, "length": "long",
    }).json()

    # Word counts in the summary correspond to the length.
    assert len(short["summary"].split()) == 5
    assert len(medium["summary"].split()) == 10
    assert len(long_["summary"].split()) == 20


@pytest.mark.timeout(10)
def test_summarize_e2e_default_length_is_medium():
    fake = _FakeSummarizer()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-summarizer"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/summarize", json={
        "text": " ".join(f"w{i}" for i in range(50)),
    })
    body = r.json()
    assert body["meta"]["length"] == "medium"


@pytest.mark.timeout(10)
def test_summarize_e2e_400_on_invalid_length():
    fake = _FakeSummarizer()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-summarizer"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/summarize", json={
        "text": "hello", "length": "WRONG",
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.timeout(10)
def test_summarize_e2e_empty_text_400():
    fake = _FakeSummarizer()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "fake-summarizer"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/summarize", json={"text": ""})
    assert r.status_code == 400
