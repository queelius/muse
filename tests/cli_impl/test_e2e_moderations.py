"""End-to-end: /v1/moderations through FastAPI + codec correctly.

Uses a fake TextClassifierModel backend; no real weights.
"""
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_classification import (
    MODALITY,
    ClassificationResult,
    build_router,
)


pytestmark = pytest.mark.slow


class _FakeClassifier:
    def __init__(self):
        self.called_with = None
        self.model_id = "text-moderation"

    def classify(self, input):
        self.called_with = input
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = list(input)
        # Simple rule: if "hate" in text, score H high; else score H low
        return [
            ClassificationResult(
                scores={
                    "H": 0.95 if "hate" in t.lower() else 0.1,
                },
                multi_label=False,
            )
            for t in inputs
        ]


@pytest.mark.timeout(10)
def test_moderations_full_request_response_cycle():
    fake = _FakeClassifier()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "text-moderation"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    client = TestClient(app)
    r = client.post("/v1/moderations", json={
        "input": ["hello world", "I hate you"],
        "model": "text-moderation",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "text-moderation"
    assert body["id"].startswith("modr-")
    assert len(body["results"]) == 2
    assert body["results"][0]["flagged"] is False
    assert body["results"][1]["flagged"] is True
    assert body["results"][1]["categories"]["H"] is True
