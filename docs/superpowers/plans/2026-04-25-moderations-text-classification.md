# /v1/moderations (text/classification) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship muse's 6th modality, `text/classification`, mounted at OpenAI-compat `/v1/moderations`. Generic `HFTextClassifier` runtime over any HuggingFace text-classification model. One curated default (KoalaAI/Text-Moderation). HF resolver sniffs the `text-classification` tag.

**Architecture:** Modality subpackage at `src/muse/modalities/text_classification/` mirrors the five existing modalities. Runtime wraps `transformers.AutoModelForSequenceClassification` with deferred imports. Routes layer mounts `POST /v1/moderations` (OpenAI URL) on a `text/classification` modality (broad MIME tag) so future routes (e.g., `/v1/text/classifications` for sentiment-only) can share the runtime.

**Tech Stack:** transformers + torch, FastAPI JSON, pytest, requests (ModerationsClient).

**Spec:** `docs/superpowers/specs/2026-04-25-moderations-text-classification-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/text_classification/__init__.py` | create | exports `MODALITY` + `build_router`; re-exports Protocol + dataclasses + client |
| `src/muse/modalities/text_classification/protocol.py` | create | `ClassificationResult` dataclass; `TextClassifierModel` Protocol |
| `src/muse/modalities/text_classification/codec.py` | create | `encode_moderations(results, model_id, threshold)` build OpenAI envelope |
| `src/muse/modalities/text_classification/routes.py` | create | `build_router(registry)` with `POST /v1/moderations` |
| `src/muse/modalities/text_classification/client.py` | create | `ModerationsClient` |
| `src/muse/modalities/text_classification/runtimes/__init__.py` | create | empty package marker |
| `src/muse/modalities/text_classification/runtimes/hf_text_classifier.py` | create | `HFTextClassifier` generic runtime |
| `src/muse/core/resolvers_hf.py` | modify | 4th sniff branch + resolve + search |
| `src/muse/curated.yaml` | modify | +1 entry: `text-moderation` |
| `tests/modalities/text_classification/` | create | full unit suite (5 files + 2 inits) |
| `tests/core/test_resolvers_hf.py` | modify | +5 tests for the new branch |
| `tests/core/test_curated.py` | modify | +1 test for new entry |
| `tests/cli_impl/test_e2e_moderations.py` | create | one slow-marked end-to-end test |
| `tests/integration/conftest.py` | modify | `text_moderation_model` fixture |
| `tests/integration/test_remote_moderations.py` | create | opt-in end-to-end probe |
| `CLAUDE.md` | modify | modality list + tag-vs-URL note |
| `README.md` | modify | modality list + endpoint |
| `pyproject.toml` | modify | version 0.13.x to 0.14.0 |

---

### Task 1: Protocol + dataclasses + package skeleton + stub routes

**Files:**
- Create: `src/muse/modalities/text_classification/__init__.py`
- Create: `src/muse/modalities/text_classification/protocol.py`
- Create: `src/muse/modalities/text_classification/runtimes/__init__.py` (one-line docstring)
- Create: `src/muse/modalities/text_classification/routes.py` (stub returning empty APIRouter; replaced in Task 3)
- Create: `tests/modalities/text_classification/__init__.py` (empty)
- Test: `tests/modalities/text_classification/test_protocol.py`

Lesson learned from the ASR migration: ship a working stub `routes.py` in Task 1 so `discover_modalities` then `build_router(registry)` doesn't crash workers. Task 3 replaces the stub with the real implementation.

- [ ] **Step 1: Write the failing protocol tests**

Create `tests/modalities/text_classification/__init__.py` (empty) and `tests/modalities/text_classification/test_protocol.py`:

```python
"""Protocol + dataclass shape tests for text/classification."""
from muse.modalities.text_classification import (
    MODALITY,
    ClassificationResult,
    TextClassifierModel,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "text/classification"


def test_classification_result_minimal():
    r = ClassificationResult(
        scores={"H": 0.8, "V": 0.1, "OK": 0.1},
        multi_label=False,
    )
    assert r.scores["H"] == 0.8
    assert r.multi_label is False


def test_classification_result_multi_label():
    r = ClassificationResult(
        scores={"toxic": 0.9, "obscene": 0.7, "threat": 0.05},
        multi_label=True,
    )
    assert r.multi_label is True
    assert len(r.scores) == 3


def test_text_classifier_protocol_accepts_structural_impl():
    """A class that implements `classify(...)` satisfies the protocol."""
    class Fake:
        def classify(self, input):
            return [ClassificationResult(scores={"OK": 1.0}, multi_label=False)]
    assert isinstance(Fake(), TextClassifierModel)


def test_text_classifier_protocol_rejects_missing_method():
    class Missing:
        pass
    assert not isinstance(Missing(), TextClassifierModel)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/text_classification/test_protocol.py -v`
Expected: ModuleNotFoundError on `muse.modalities.text_classification`.

- [ ] **Step 3: Create the modality package**

Create `src/muse/modalities/text_classification/protocol.py`:

```python
"""Protocol + dataclasses for text/classification."""
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ClassificationResult:
    """One input's classification output.

    scores: {label: confidence in [0, 1]} from the model's id2label space.
    multi_label: True if scores are independent (sigmoid head). False if
    mutually exclusive (softmax; sums to ~1.0).

    The codec uses multi_label to decide whether `flagged` derives from
    "any score >= threshold" (multi-label) or "argmax score >= threshold"
    (single-label).
    """
    scores: dict[str, float]
    multi_label: bool


@runtime_checkable
class TextClassifierModel(Protocol):
    """Structural protocol any text-classifier backend satisfies.

    HFTextClassifier (the generic runtime) satisfies this without
    inheriting. Tests use fakes that match the signature structurally.
    """

    def classify(self, input: str | list[str]) -> list[ClassificationResult]:
        """Return one ClassificationResult per input.

        Even when `input` is a scalar str, returns a list of length 1.
        Order matches input order in batch mode.
        """
        ...
```

Create `src/muse/modalities/text_classification/__init__.py`:

```python
"""text/classification modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - ClassificationResult dataclass
  - TextClassifierModel Protocol
  - ModerationsClient

Wire contract (OpenAI-compat):
  - POST /v1/moderations

Modality vs URL: this is muse's first modality whose MIME tag
(text/classification) is broader than its primary URL (/v1/moderations).
Future routes (e.g., /v1/text/classifications for sentiment) can share
the same runtime + dataclasses without a new modality package.
"""
from muse.modalities.text_classification.protocol import (
    ClassificationResult,
    TextClassifierModel,
)
from muse.modalities.text_classification.routes import build_router


MODALITY = "text/classification"


__all__ = [
    "MODALITY",
    "build_router",
    "ClassificationResult",
    "TextClassifierModel",
]
```

Create `src/muse/modalities/text_classification/runtimes/__init__.py`:

```python
"""Generic runtimes for text/classification."""
```

Create `src/muse/modalities/text_classification/routes.py` (stub for Task 1; replaced in Task 3):

```python
"""Stub routes for text/classification.

Replaced in Task 3 with the real /v1/moderations endpoint. For now
this exists only so the modality's build_router() in __init__.py has
something to import, which keeps run_worker's "mount all discovered
modality routers" path working between Task 1 (protocol) and Task 3
(routes).
"""
from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Empty router placeholder; Task 3 adds POST /v1/moderations."""
    return APIRouter()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/text_classification/test_protocol.py -v`
Expected: 5 passed.

Then: `pytest -m "not slow" -q 2>&1 | tail -3`
Expected: full fast lane stays green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_classification/__init__.py \
        src/muse/modalities/text_classification/protocol.py \
        src/muse/modalities/text_classification/runtimes/__init__.py \
        src/muse/modalities/text_classification/routes.py \
        tests/modalities/text_classification/__init__.py \
        tests/modalities/text_classification/test_protocol.py
git commit -m "$(cat <<'EOF'
feat(moderations): text/classification modality skeleton + stub routes

MODALITY tag, ClassificationResult dataclass, TextClassifierModel
structural protocol. routes.py is a stub that returns an empty
APIRouter so discover_modalities + build_router(registry) work
across worker startup; Task 3 replaces it with the real
POST /v1/moderations endpoint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Codec + threshold resolution

**Files:**
- Create: `src/muse/modalities/text_classification/codec.py`
- Test: `tests/modalities/text_classification/test_codec.py`

- [ ] **Step 1: Write the failing codec tests**

Create `tests/modalities/text_classification/test_codec.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/text_classification/test_codec.py -v`
Expected: ModuleNotFoundError on `muse.modalities.text_classification.codec`.

- [ ] **Step 3: Implement codec**

Create `src/muse/modalities/text_classification/codec.py`:

```python
"""Encoding for /v1/moderations responses.

Pure functions: ClassificationResult + threshold to OpenAI envelope dict.
Tested without FastAPI.
"""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.text_classification.protocol import ClassificationResult


def encode_moderations(
    results: list[ClassificationResult],
    *,
    model_id: str,
    threshold: float,
) -> dict[str, Any]:
    """Build the OpenAI-shape moderations response.

    Returns a dict; the route layer wraps it in a JSONResponse.
    `id` is a fresh modr-<uuid4> per call so logs and traces can
    correlate request to response.
    """
    out_results: list[dict[str, Any]] = []
    for r in results:
        cats, flagged = _flagged_categories(
            r.scores, threshold, multi_label=r.multi_label,
        )
        out_results.append({
            "flagged": flagged,
            "categories": cats,
            "category_scores": dict(r.scores),
        })
    return {
        "id": f"modr-{uuid.uuid4().hex[:24]}",
        "model": model_id,
        "results": out_results,
    }


def _resolve_threshold(
    request_threshold: float | None,
    manifest_capabilities: dict,
) -> float:
    """Pick the effective threshold for this request.

    Precedence: request > MANIFEST.capabilities.flag_threshold > 0.5.
    Non-numeric manifest values are silently ignored (default applies).
    """
    if request_threshold is not None:
        return float(request_threshold)
    cap = manifest_capabilities.get("flag_threshold")
    if isinstance(cap, (int, float)):
        return float(cap)
    return 0.5


def _flagged_categories(
    scores: dict[str, float],
    threshold: float,
    *,
    multi_label: bool,
) -> tuple[dict[str, bool], bool]:
    """Convert scores to (per-category booleans, overall flagged).

    Multi-label: each category True iff its score >= threshold.
    Single-label: only the argmax can be True, and only if >= threshold.

    Returns (categories_dict, any_flagged_bool). The overall `flagged`
    is just `any(categories.values())`.
    """
    if multi_label:
        cats = {k: v >= threshold for k, v in scores.items()}
    else:
        if not scores:
            cats = {}
        else:
            top = max(scores, key=scores.get)
            cats = {k: (k == top and v >= threshold) for k, v in scores.items()}
    flagged = any(cats.values())
    return cats, flagged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/text_classification/test_codec.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_classification/codec.py \
        tests/modalities/text_classification/test_codec.py
git commit -m "$(cat <<'EOF'
feat(moderations): codec + threshold resolution

Pure functions: encode_moderations builds the OpenAI envelope from
ClassificationResult; _resolve_threshold layers request > manifest
> 0.5 default; _flagged_categories handles multi-label (any-above)
vs single-label (argmax-above) flagging. id is modr-<uuid4hex24> per
call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Routes (POST /v1/moderations)

**Files:**
- Modify (replace stub): `src/muse/modalities/text_classification/routes.py`
- Test: `tests/modalities/text_classification/test_routes.py`

- [ ] **Step 1: Write the failing route tests**

Create `tests/modalities/text_classification/test_routes.py`:

```python
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
    backend = MagicMock()
    backend.model_id = "text-moderation"
    backend.classify.return_value = [
        ClassificationResult(scores={"OK": 0.9, "H": 0.1}, multi_label=False),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/text_classification/test_routes.py -v`
Expected: most fail (the stub router has no routes).

- [ ] **Step 3: Implement routes**

Replace `src/muse/modalities/text_classification/routes.py` entirely:

```python
"""FastAPI routes for /v1/moderations.

OpenAI-compat shape:
  request:  {"input": str | list[str], "model"?: str, "threshold"?: float}
  response: {"id", "model", "results": [{"flagged", "categories", "category_scores"}]}

Error envelopes follow muse's OpenAI-compat convention (see
audio_transcription/routes.py for full discussion). 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry

# MODALITY defined locally to avoid the __init__ circular import that
# bit ASR T1; sibling modalities all do this.
MODALITY = "text/classification"


logger = logging.getLogger(__name__)


class _ModerationsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    threshold: float | None = None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/moderations")
    async def moderations(req: _ModerationsRequest):
        # Lazy import to avoid a circular import with __init__.
        from muse.modalities.text_classification.codec import (
            encode_moderations, _resolve_threshold,
        )

        if req.threshold is not None and not (0.0 <= req.threshold <= 1.0):
            return error_response(
                400, "invalid_parameter",
                f"threshold must be in [0, 1]; got {req.threshold}",
            )

        if isinstance(req.input, str) and not req.input:
            return error_response(
                400, "invalid_parameter", "input must not be empty",
            )
        if isinstance(req.input, list) and (
            len(req.input) == 0 or any(not s for s in req.input)
        ):
            return error_response(
                400, "invalid_parameter",
                "input must be a non-empty string or list of non-empty strings",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "(default)", modality=MODALITY,
            )

        # Resolve effective model_id for the response envelope. Prefer
        # the backend's own model_id (set on instantiation) so that the
        # response reflects what actually answered, not the request's
        # model field which may have been None.
        effective_id = getattr(backend, "model_id", req.model or "unknown")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}
        threshold = _resolve_threshold(req.threshold, capabilities)

        results = backend.classify(req.input)
        body = encode_moderations(
            results, model_id=effective_id, threshold=threshold,
        )
        return JSONResponse(content=body)

    return router
```

If `ModalityRegistry` doesn't have a `manifest(modality, model_id)` accessor, check `src/muse/core/registry.py`. The existing `register()` likely stores the manifest in an internal dict. If no public accessor exists, add a tiny one.

- [ ] **Step 4: Verify the registry exposes manifests**

Run:
```bash
grep -n "manifest" src/muse/core/registry.py | head -20
```

If `ModalityRegistry` already has a `manifest(modality, model_id)` method, use it. If only `register(..., manifest=...)` stores the manifest internally as `self._manifests[(modality, model_id)] = manifest`, then add this accessor on the registry:

```python
def manifest(self, modality: str, model_id: str) -> dict | None:
    return self._manifests.get((modality, model_id))
```

If you must add this accessor, also add a one-line test in `tests/core/test_registry.py`:

```python
def test_registry_exposes_manifest_after_register():
    reg = ModalityRegistry()
    fake = MagicMock(model_id="m1")
    reg.register("text/classification", fake, manifest={
        "model_id": "m1", "capabilities": {"flag_threshold": 0.7},
    })
    m = reg.manifest("text/classification", "m1")
    assert m["capabilities"]["flag_threshold"] == 0.7
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/modalities/text_classification/test_routes.py -v`
Expected: 8 passed.

Then: `pytest -m "not slow" -q 2>&1 | tail -3`
Expected: full fast lane green.

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/text_classification/routes.py \
        tests/modalities/text_classification/test_routes.py \
        $(git diff --name-only -- src/muse/core/registry.py tests/core/test_registry.py 2>/dev/null)
git commit -m "$(cat <<'EOF'
feat(moderations): POST /v1/moderations route

OpenAI-shape JSON request and response. Pydantic-validated body with
input (str | list[str]), optional model, optional threshold.
Threshold resolution layers request > MANIFEST.capabilities
.flag_threshold > 0.5. Empty input is 400; out-of-range threshold
is 400; unknown model is 404. Replaces the Task 1 scaffolding stub.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Client (ModerationsClient)

**Files:**
- Create: `src/muse/modalities/text_classification/client.py`
- Test: `tests/modalities/text_classification/test_client.py`

- [ ] **Step 1: Write the failing client tests**

Create `tests/modalities/text_classification/test_client.py`:

```python
"""Tests for ModerationsClient HTTP client."""
import json
from unittest.mock import MagicMock, patch


def _make_response(body: dict, status: int = 200):
    mock = MagicMock(
        status_code=status,
        headers={"content-type": "application/json"},
    )
    mock.json = MagicMock(return_value=body)
    mock.text = json.dumps(body)
    mock.raise_for_status = MagicMock()
    return mock


def test_default_server_url():
    from muse.modalities.text_classification import ModerationsClient
    c = ModerationsClient()
    assert c.server_url == "http://localhost:8000"


def test_trailing_slash_stripped():
    from muse.modalities.text_classification import ModerationsClient
    c = ModerationsClient(server_url="http://lan:8000/")
    assert c.server_url == "http://lan:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    from muse.modalities.text_classification import ModerationsClient
    c = ModerationsClient()
    assert c.server_url == "http://custom:9999"


def test_classify_scalar_returns_first_result():
    """Scalar input returns dict (the single results[0])."""
    body = {
        "id": "modr-1", "model": "text-moderation",
        "results": [{
            "flagged": True,
            "categories": {"H": True}, "category_scores": {"H": 0.9},
        }],
    }
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        out = c.classify("I hate everything", model="text-moderation")
    assert isinstance(out, dict)
    assert out["flagged"] is True
    assert out["categories"]["H"] is True


def test_classify_list_returns_list_of_results():
    body = {
        "id": "modr-2", "model": "text-moderation",
        "results": [
            {"flagged": False, "categories": {}, "category_scores": {}},
            {"flagged": True, "categories": {}, "category_scores": {}},
        ],
    }
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        out = c.classify(["a", "b"], model="text-moderation")
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[1]["flagged"] is True


def test_classify_threshold_forwarded():
    body = {"id": "x", "model": "m", "results": [{"flagged": False, "categories": {}, "category_scores": {}}]}
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        c.classify("x", model="m", threshold=0.7)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["threshold"] == 0.7
        assert sent["input"] == "x"
        assert sent["model"] == "m"


def test_raise_for_status_invoked():
    """4xx propagates as requests.HTTPError."""
    import requests
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        resp = _make_response({"error": {"code": "model_not_found"}}, status=404)
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError("404 model_not_found"),
        )
        mock_post.return_value = resp
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        import pytest
        with pytest.raises(requests.HTTPError):
            c.classify("x", model="no-such-model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/text_classification/test_client.py -v`
Expected: ImportError on `ModerationsClient`.

- [ ] **Step 3: Implement client**

Create `src/muse/modalities/text_classification/client.py`:

```python
"""HTTP client for /v1/moderations.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class ModerationsClient:
    """Minimal HTTP client for the text/classification modality."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        url = (
            server_url
            or os.environ.get("MUSE_SERVER")
            or _DEFAULT_SERVER
        )
        self.server_url = url.rstrip("/")
        self._timeout = timeout

    def classify(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Send a moderation request.

        Returns:
          - dict (the single results[0]) when input is a scalar str
          - list[dict] (the full results array) when input is a list
        """
        body: dict[str, Any] = {"input": input}
        if model is not None:
            body["model"] = model
        if threshold is not None:
            body["threshold"] = threshold

        r = requests.post(
            f"{self.server_url}/v1/moderations",
            json=body, timeout=self._timeout,
        )
        r.raise_for_status()
        envelope = r.json()
        if isinstance(input, str):
            return envelope["results"][0]
        return envelope["results"]
```

Update `src/muse/modalities/text_classification/__init__.py` to re-export the client. Add the import:

```python
from muse.modalities.text_classification.client import ModerationsClient
```

and add `"ModerationsClient"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/text_classification/test_client.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_classification/client.py \
        src/muse/modalities/text_classification/__init__.py \
        tests/modalities/text_classification/test_client.py
git commit -m "$(cat <<'EOF'
feat(moderations): ModerationsClient for /v1/moderations

Parallel API to the other muse clients. Server URL precedence:
explicit arg > MUSE_SERVER env > http://localhost:8000. Trailing
slashes stripped. Scalar input returns dict; list input returns
list. raise_for_status before parsing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: HFTextClassifier runtime

**Files:**
- Create: `src/muse/modalities/text_classification/runtimes/hf_text_classifier.py`
- Test: `tests/modalities/text_classification/runtimes/__init__.py` (empty)
- Test: `tests/modalities/text_classification/runtimes/test_hf_text_classifier.py`

- [ ] **Step 1: Write the failing runtime tests**

Create `tests/modalities/text_classification/runtimes/__init__.py` (empty).

Create `tests/modalities/text_classification/runtimes/test_hf_text_classifier.py`:

```python
"""HFTextClassifier runtime: mocked-dep tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern)."""
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    orig = (mod.torch, mod.AutoTokenizer, mod.AutoModelForSequenceClassification)
    yield
    mod.torch, mod.AutoTokenizer, mod.AutoModelForSequenceClassification = orig


def _make_logits_tensor(values_2d):
    """Build a fake logits tensor that has .detach().cpu().numpy() chain."""
    arr = np.array(values_2d, dtype="float32")
    t = MagicMock()
    t.detach.return_value.cpu.return_value.numpy.return_value = arr
    return t


def _wire_torch_with_softmax_and_sigmoid(mod):
    """Install MagicMock torch + numpy-backed softmax/sigmoid."""
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)

    def _fake_softmax(t, dim):
        arr = t.detach.return_value.cpu.return_value.numpy.return_value
        e = np.exp(arr - arr.max(axis=-1, keepdims=True))
        out_arr = e / e.sum(axis=-1, keepdims=True)
        out = MagicMock()
        out.detach.return_value.cpu.return_value.numpy.return_value = out_arr
        return out

    def _fake_sigmoid(t):
        arr = t.detach.return_value.cpu.return_value.numpy.return_value
        out_arr = 1.0 / (1.0 + np.exp(-arr))
        out = MagicMock()
        out.detach.return_value.cpu.return_value.numpy.return_value = out_arr
        return out

    mod.torch.softmax = _fake_softmax
    mod.torch.sigmoid = _fake_sigmoid


def _install_fake_model(mod, *, id2label, problem_type, logits):
    """Install fake tokenizer and AutoModelForSequenceClassification.

    Returns the fake_model so the caller can introspect call args.
    """
    fake_tok = MagicMock()
    fake_tok.return_value = MagicMock(to=MagicMock(return_value={}))
    mod.AutoTokenizer = MagicMock()
    mod.AutoTokenizer.from_pretrained.return_value = fake_tok

    fake_model = MagicMock()
    fake_model.config = SimpleNamespace(
        id2label=id2label, problem_type=problem_type,
    )
    # call() returns the SimpleNamespace with .logits
    fake_model.return_value = SimpleNamespace(logits=_make_logits_tensor(logits))
    # .to() and the inference-mode toggle both return the same fake_model
    fake_model.to.return_value = fake_model
    # PyTorch's inference-mode toggle method (not run as a script;
    # we just need the chained call to return fake_model).
    fake_model.eval.return_value = fake_model

    mod.AutoModelForSequenceClassification = MagicMock()
    mod.AutoModelForSequenceClassification.from_pretrained.return_value = fake_model
    return fake_model


def test_classify_single_label_uses_softmax():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    _wire_torch_with_softmax_and_sigmoid(mod)
    _install_fake_model(
        mod,
        id2label={0: "OK", 1: "H", 2: "V"},
        problem_type="single_label_classification",
        logits=[[0.1, 5.0, 0.5]],
    )

    m = mod.HFTextClassifier(
        model_id="text-moderation", hf_repo="x", local_dir="/fake", device="cpu",
    )
    results = m.classify("test input")
    assert len(results) == 1
    r = results[0]
    assert not r.multi_label
    # H should be the highest score (logit 5.0 dominates after softmax)
    assert max(r.scores, key=r.scores.get) == "H"
    assert abs(sum(r.scores.values()) - 1.0) < 1e-3


def test_classify_multi_label_uses_sigmoid():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    _wire_torch_with_softmax_and_sigmoid(mod)
    _install_fake_model(
        mod,
        id2label={0: "toxic", 1: "obscene"},
        problem_type="multi_label_classification",
        # logit 5.0 -> sigmoid ~0.99; logit -2.0 -> sigmoid ~0.12
        logits=[[5.0, -2.0]],
    )

    m = mod.HFTextClassifier(
        model_id="toxic-bert", hf_repo="x", local_dir="/fake", device="cpu",
    )
    results = m.classify("text")
    r = results[0]
    assert r.multi_label
    assert r.scores["toxic"] > 0.9
    assert r.scores["obscene"] < 0.2


def test_classify_batch_preserves_order():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    _wire_torch_with_softmax_and_sigmoid(mod)
    _install_fake_model(
        mod,
        id2label={0: "OK", 1: "H"},
        problem_type="single_label_classification",
        # Row 0: OK dominant; Row 1: H dominant
        logits=[[5.0, 0.1], [0.1, 5.0]],
    )

    m = mod.HFTextClassifier(
        model_id="text-moderation", hf_repo="x", local_dir="/fake", device="cpu",
    )
    results = m.classify(["safe text", "harmful text"])
    assert len(results) == 2
    assert max(results[0].scores, key=results[0].scores.get) == "OK"
    assert max(results[1].scores, key=results[1].scores.get) == "H"


def test_device_auto_selects_cuda_when_available():
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = True
    mod.torch.backends = MagicMock(mps=None)
    _install_fake_model(
        mod, id2label={0: "OK"}, problem_type=None, logits=[[1.0]],
    )

    m = mod.HFTextClassifier(
        model_id="m", hf_repo="x", local_dir="/fake", device="auto",
    )
    assert m._device == "cuda"


def test_raises_when_transformers_not_installed(monkeypatch):
    """If transformers import fails, constructor raises a clear error."""
    import muse.modalities.text_classification.runtimes.hf_text_classifier as mod
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)
    mod.AutoTokenizer = None
    mod.AutoModelForSequenceClassification = None
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    with pytest.raises(RuntimeError, match="transformers is not installed"):
        mod.HFTextClassifier(
            model_id="m", hf_repo="x", local_dir="/fake", device="cpu",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/modalities/text_classification/runtimes/ -v`
Expected: ImportError on the module.

- [ ] **Step 3: Implement runtime**

Create `src/muse/modalities/text_classification/runtimes/hf_text_classifier.py`:

```python
"""HFTextClassifier: generic runtime over any HF text-classification model.

One class wraps `AutoModelForSequenceClassification` + `AutoTokenizer`
for any HuggingFace text-classifier. Pulled via the HF resolver:
`muse pull hf://KoalaAI/Text-Moderation` synthesizes a manifest
pointing at this class.

Deferred imports follow the muse pattern: torch, AutoTokenizer, and
AutoModelForSequenceClassification stay as module-top sentinels (None)
until _ensure_deps() lazy-imports them. Tests patch the sentinels
directly; _ensure_deps short-circuits on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.text_classification.protocol import ClassificationResult


logger = logging.getLogger(__name__)

torch: Any = None
AutoTokenizer: Any = None
AutoModelForSequenceClassification: Any = None


def _ensure_deps() -> None:
    global torch, AutoTokenizer, AutoModelForSequenceClassification
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFTextClassifier torch unavailable: %s", e)
    if AutoTokenizer is None:
        try:
            from transformers import AutoTokenizer as _tk
            AutoTokenizer = _tk
        except Exception as e:  # noqa: BLE001
            logger.debug("HFTextClassifier AutoTokenizer unavailable: %s", e)
    if AutoModelForSequenceClassification is None:
        try:
            from transformers import AutoModelForSequenceClassification as _m
            AutoModelForSequenceClassification = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("HFTextClassifier AutoModel unavailable: %s", e)


class HFTextClassifier:
    """Generic HuggingFace text-classification runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` or "
                "install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._max_length = max_length

        src = local_dir or hf_repo
        logger.info(
            "loading text classifier from %s (device=%s)",
            src, self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForSequenceClassification.from_pretrained(src)
        self._model = self._model.to(self._device)
        # Switch to inference mode (no autograd) without using a context
        # manager so the deferred-import discipline stays clean.
        self._model = self._model.eval()

        cfg = self._model.config
        self._id2label: dict[int, str] = dict(getattr(cfg, "id2label", {}))
        self._multi_label = (
            getattr(cfg, "problem_type", None) == "multi_label_classification"
        )

    def classify(self, input: str | list[str]) -> list[ClassificationResult]:
        texts = [input] if isinstance(input, str) else list(input)
        if not texts:
            return []

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = encoded.to(self._device)

        outputs = self._model(**encoded)
        logits = outputs.logits  # shape: (batch, num_labels)

        if self._multi_label:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.detach().cpu().numpy()  # (batch, num_labels)

        results: list[ClassificationResult] = []
        for row in probs_np:
            scores = {
                self._id2label.get(i, str(i)): float(row[i])
                for i in range(row.shape[-1])
            }
            results.append(ClassificationResult(
                scores=scores,
                multi_label=self._multi_label,
            ))
        return results


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/modalities/text_classification/runtimes/ -v`
Expected: 5 passed.

Then: `pytest -m "not slow" -q 2>&1 | tail -3`
Expected: full fast lane green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_classification/runtimes/hf_text_classifier.py \
        tests/modalities/text_classification/runtimes/__init__.py \
        tests/modalities/text_classification/runtimes/test_hf_text_classifier.py
git commit -m "$(cat <<'EOF'
feat(moderations): HFTextClassifier generic runtime

Wraps any HuggingFace text-classification model via
AutoModelForSequenceClassification + AutoTokenizer. Deferred imports
(torch + transformers stay None until instantiation). Detects
multi-label vs single-label via model.config.problem_type and uses
sigmoid or softmax accordingly. Returns ClassificationResult per
input with model-native id2label keys.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: HF resolver text-classification sniff

**Files:**
- Modify: `src/muse/core/resolvers_hf.py`
- Test: `tests/core/test_resolvers_hf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_resolvers_hf.py`:

```python
# --- text-classification branch ---

def test_sniff_detects_text_classification_tag():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["text-classification", "transformers"],
    )
    assert _sniff_repo_shape(info) == "text-classification"


def test_sniff_text_classification_does_not_override_gguf():
    """If a repo has both .gguf siblings AND text-classification tag,
    gguf wins (it's the more specific format)."""
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="model.q4_k_m.gguf"),
            SimpleNamespace(rfilename="config.json"),
        ],
        tags=["text-classification", "transformers"],
    )
    assert _sniff_repo_shape(info) == "gguf"


def test_resolve_text_classification_synthesizes_manifest():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["text-classification"],
        card_data=SimpleNamespace(license="apache-2.0"),
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://KoalaAI/Text-Moderation")
    assert resolved.manifest["modality"] == "text/classification"
    assert resolved.manifest["hf_repo"] == "KoalaAI/Text-Moderation"
    assert resolved.manifest["model_id"] == "text-moderation"
    assert "transformers>=4.36.0" in resolved.manifest["pip_extras"]
    assert "torch>=2.1.0" in resolved.manifest["pip_extras"]
    assert resolved.backend_path == (
        "muse.modalities.text_classification.runtimes.hf_text_classifier"
        ":HFTextClassifier"
    )


def test_search_text_classification_yields_results():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    fake_repos = [
        SimpleNamespace(id="KoalaAI/Text-Moderation", downloads=5000, siblings=[]),
        SimpleNamespace(id="unitary/toxic-bert", downloads=12000, siblings=[]),
    ]
    with patch.object(resolver._api, "list_models", return_value=fake_repos):
        results = list(resolver.search("toxic", modality="text/classification"))
    assert len(results) == 2
    assert all(r.modality == "text/classification" for r in results)
    assert results[0].model_id == "text-moderation"


def test_resolve_unknown_error_message_lists_text_classification():
    """Sanity: the unknown-shape error mentions all 4 supported branches."""
    from muse.core.resolvers_hf import HFResolver, ResolverError
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename="random.bin")],
        tags=["something-unknown"],
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        try:
            resolver.resolve("hf://x/y")
        except ResolverError as e:
            msg = str(e)
            assert "text-classification" in msg
        else:
            raise AssertionError("expected ResolverError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_resolvers_hf.py -v -k "text_class"`
Expected: AssertionError (sniff returns "unknown") and import errors on the resolve path.

- [ ] **Step 3: Implement the sniff**

In `src/muse/core/resolvers_hf.py`, add module-level helpers and constants:

```python
TEXT_CLASSIFIER_RUNTIME_PATH = (
    "muse.modalities.text_classification.runtimes.hf_text_classifier"
    ":HFTextClassifier"
)
TEXT_CLASSIFIER_PIP_EXTRAS = ("transformers>=4.36.0", "torch>=2.1.0")
TEXT_CLASSIFIER_SYSTEM_PACKAGES = ()


def _looks_like_text_classifier(siblings: list[str], tags: list[str]) -> bool:
    """HF text-classification repos carry the `text-classification` tag.
    Sibling shape varies (PyTorch / safetensors / older bin formats); we
    don't gate on file presence, only on the tag, since transformers
    handles the loading ambiguity for us at AutoModelForSequenceClassification
    time.
    """
    return "text-classification" in tags
```

Update `_sniff_repo_shape` to add a fourth branch BEFORE the final `return "unknown"`:

```python
    if _looks_like_text_classifier(siblings, tags):
        return "text-classification"
```

(Keep this branch AFTER the existing gguf, sentence-transformers, and faster-whisper branches so more-specific signals win.)

Add the dispatch in `HFResolver.resolve`:

```python
        if shape == "text-classification":
            return self._resolve_text_classifier(repo_id, info)
```

Update the unknown-shape error message to include the new branch:

```python
        raise ResolverError(
            f"cannot infer modality for {repo_id!r} "
            f"(no .gguf siblings, no sentence-transformers tag, "
            f"no CT2 shape with ASR tag, no text-classification tag; "
            f"tags={tags})"
        )
```

Add the resolve method on `HFResolver`:

```python
    def _resolve_text_classifier(self, repo_id: str, info) -> ResolvedModel:
        manifest = {
            "model_id": repo_id.split("/", 1)[-1].lower(),
            "modality": "text/classification",
            "hf_repo": repo_id,
            "description": f"Text classifier: {repo_id}",
            "license": _repo_license(info),
            "pip_extras": list(TEXT_CLASSIFIER_PIP_EXTRAS),
            "system_packages": list(TEXT_CLASSIFIER_SYSTEM_PACKAGES),
            "capabilities": {},
        }

        def _download(cache_root: Path) -> Path:
            return Path(snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=TEXT_CLASSIFIER_RUNTIME_PATH,
            download=_download,
        )
```

Update `HFResolver.search`'s modality dispatch. Add an elif before the else:

```python
        elif modality == "text/classification":
            yield from self._search_text_classifier(query, sort=sort, limit=limit)
```

And update the `else` branch's error message to include `text/classification`:

```python
        else:
            raise ResolverError(
                f"HFResolver.search does not support modality {modality!r}; "
                f"supported: chat/completion, embedding/text, "
                f"audio/transcription, text/classification"
            )
```

Add the search method:

```python
    def _search_text_classifier(self, query: str, *, sort: str, limit: int):
        repos = self._api.list_models(
            search=query, filter="text-classification",
            sort=sort, limit=limit,
        )
        for repo in repos:
            yield SearchResult(
                uri=f"hf://{repo.id}",
                model_id=repo.id.split("/", 1)[-1].lower(),
                modality="text/classification",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo.id,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_resolvers_hf.py -v`
Expected: all prior tests still pass + 5 new ones.

Then: `pytest -m "not slow" -q 2>&1 | tail -3`
Expected: full fast lane green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/resolvers_hf.py tests/core/test_resolvers_hf.py
git commit -m "$(cat <<'EOF'
feat(resolver): HF resolver sniffs text-classification repos

Adds a fourth _sniff_repo_shape branch after gguf + sentence-transformers
+ faster-whisper: text-classification, gated by the HF tag.
resolve() synthesizes a text/classification manifest pointing at the
HFTextClassifier runtime; search() routes --modality text/classification
to list_models(filter=text-classification). Unknown-shape error message
updated to list all four supported branches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Curated entry

**Files:**
- Modify: `src/muse/curated.yaml`
- Test: `tests/core/test_curated.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_curated.py`:

```python
def test_load_curated_includes_text_moderation_entry():
    """Curated text-moderation alias exists and points at KoalaAI."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}
    assert "text-moderation" in by_id
    e = by_id["text-moderation"]
    assert e.modality == "text/classification"
    assert e.uri == "hf://KoalaAI/Text-Moderation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_curated.py -v -k text_moderation`
Expected: AssertionError (entry missing).

- [ ] **Step 3: Add the curated entry**

Append to `src/muse/curated.yaml`:

```yaml
# ---------- text/classification (moderation) ----------

- id: text-moderation
  uri: hf://KoalaAI/Text-Moderation
  modality: text/classification
  size_gb: 0.14
  description: "9-category text moderation (S/H/V/HR/SH/S3/H2/V2/OK), CPU-friendly"
```

- [ ] **Step 4: Verify**

Run: `pytest tests/core/test_curated.py -v`
Expected: all curated tests green.

Run: `python -c "from muse.core.curated import load_curated, _reset_curated_cache_for_tests; _reset_curated_cache_for_tests(); ents = [e for e in load_curated() if e.modality == 'text/classification']; print(ents)"`
Expected: prints the new entry.

- [ ] **Step 5: Commit**

```bash
git add src/muse/curated.yaml tests/core/test_curated.py
git commit -m "$(cat <<'EOF'
feat(curated): text-moderation shortcut for KoalaAI/Text-Moderation

Newbie-friendly alias for the curated default moderation model.
Users can pull other classifiers via `muse pull hf://...` or find
via `muse search <query> --modality text/classification`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Slow e2e + integration test

**Files:**
- Create: `tests/cli_impl/test_e2e_moderations.py`
- Modify: `tests/integration/conftest.py`
- Create: `tests/integration/test_remote_moderations.py`

- [ ] **Step 1: Slow e2e test**

Create `tests/cli_impl/test_e2e_moderations.py`:

```python
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
        # Simple rule: if "hate" in text, score H high
        return [
            ClassificationResult(
                scores={
                    "H": 0.95 if "hate" in t.lower() else 0.05,
                    "OK": 0.05 if "hate" in t.lower() else 0.95,
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
```

- [ ] **Step 2: Add the integration fixture**

Append to `tests/integration/conftest.py`:

```python


@pytest.fixture(scope="session")
def text_moderation_model(remote_health) -> str:
    """The text/classification model id integration tests should target.

    Defaults to text-moderation. Override via MUSE_MODERATION_MODEL_ID.
    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_MODERATION_MODEL_ID", "text-moderation")
    _require_model(remote_health, model_id)
    return model_id
```

- [ ] **Step 3: Integration test**

Create `tests/integration/test_remote_moderations.py`:

```python
"""End-to-end /v1/moderations against a running muse server. Opt-in.

Requires MUSE_REMOTE_SERVER set + the target server has the
text_moderation_model loaded (default text-moderation).
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.slow


def test_protocol_classifies_safe_text(openai_client, text_moderation_model):
    """A neutral input should return a non-flagged result."""
    r = openai_client.moderations.create(
        model=text_moderation_model,
        input="hello world, this is a friendly message",
    )
    assert len(r.results) == 1
    res = r.results[0]
    assert hasattr(res, "flagged")
    assert hasattr(res, "category_scores")


def test_protocol_classifies_batch_returns_ordered_results(
    openai_client, text_moderation_model,
):
    r = openai_client.moderations.create(
        model=text_moderation_model,
        input=["hello world", "another safe text"],
    )
    assert len(r.results) == 2
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/cli_impl/test_e2e_moderations.py -v`
Expected: 1 pass.

Run (offline): `pytest tests/integration/test_remote_moderations.py -v`
Expected: 2 skipped (no MUSE_REMOTE_SERVER).

Run full: `pytest -q 2>&1 | tail -3`
Expected: prior count + 1 e2e + 2 skipped integration.

- [ ] **Step 5: Commit**

```bash
git add tests/cli_impl/test_e2e_moderations.py \
        tests/integration/conftest.py \
        tests/integration/test_remote_moderations.py
git commit -m "$(cat <<'EOF'
test(moderations): e2e + opt-in integration coverage

One slow e2e test exercises the full /v1/moderations request/response
cycle with a fake backend. Two opt-in integration tests hit a live
muse server with a real classifier (auto-skipped without
MUSE_REMOTE_SERVER). text_moderation_model fixture defaults to
text-moderation; override via MUSE_MODERATION_MODEL_ID.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

In the project-overview list of modalities, add a 6th bullet:

```
- **text/classification**: text moderation/classification via `/v1/moderations` (any HuggingFace text-classification model)
```

In the Modality conventions section, add a new bullet about the new pattern this introduces:

```
- `text_classification/` is muse's first modality whose internal MIME tag (`text/classification`) is broader than its primary URL route (`/v1/moderations`). The wire path is OpenAI-specific; the modality tag is broad enough to host future routes (`/v1/text/classifications` for sentiment/intent) sharing the same runtime + dataclasses without a new modality package.
```

- [ ] **Step 2: Update README.md**

Add `text/classification` to the modality summary list. Add `POST /v1/moderations` to the endpoints block. Match existing style.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "$(cat <<'EOF'
docs(moderations): CLAUDE.md and README.md note the new modality

Modality list gains text/classification at /v1/moderations.
CLAUDE.md notes the new MIME-tag-vs-URL-route convention this
modality introduces.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: v0.14.0 release

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change `version = "0.13.x"` to `version = "0.14.0"`.

- [ ] **Step 2: Final test sweep**

```bash
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: ~626 passed (from ~612 baseline + ~14 new tests across this plan).

```bash
pytest -q 2>&1 | tail -3
```

Expected: ~628 passed + ~14 skipped.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
chore(release): v0.14.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Tag**

```bash
git tag -a v0.14.0 -m "$(cat <<'EOF'
v0.14.0: /v1/moderations modality (text/classification)

New 6th modality: POST /v1/moderations, OpenAI-wire-compat.
Generic HFTextClassifier runtime over any HuggingFace
text-classification model. Single curated alias: text-moderation
(KoalaAI/Text-Moderation, 9-category, CPU-friendly).

Threshold precedence: request.threshold > MANIFEST.capabilities
.flag_threshold > 0.5. Multi-label (sigmoid) and single-label
(softmax) handling auto-detected from model.config.problem_type.
Category labels passthrough from model's id2label space (no lossy
remap to OpenAI's fixed schema).

This is muse's first modality whose internal MIME tag
(text/classification) is broader than its primary URL
(/v1/moderations); future text-classification routes can share
the same runtime + dataclasses.

Pull a moderation model:

  muse pull text-moderation
  muse pull hf://unitary/toxic-bert
  muse search toxic --modality text/classification

Use via OpenAI SDK:

  from openai import OpenAI
  c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
  r = c.moderations.create(model="text-moderation", input="some text")
  print(r.results[0].flagged, r.results[0].categories)
EOF
)"
```

- [ ] **Step 5: Push**

```bash
git push origin main
git push origin v0.14.0
```

---

## Success criteria

- `muse pull text-moderation` creates a venv with transformers and downloads KoalaAI/Text-Moderation.
- `muse serve` loads the moderation worker alongside existing models.
- `curl -X POST -H 'Content-Type: application/json' -d '{"input":"some text","model":"text-moderation"}' http://localhost:8000/v1/moderations` returns the OpenAI envelope.
- OpenAI Python SDK `client.moderations.create(model="text-moderation", input="...")` works.
- Multi-label and single-label models both produce correct envelopes.
- `muse search toxic --modality text/classification` lists HF text-classification repos.
- Full fast-lane suite green; full suite including slow green.
- Tag `v0.14.0` on origin.

## Out of scope (confirmed)

- Image moderation / OpenAI omni-shape input arrays.
- Streaming.
- Per-category request thresholds.
- id2label remapping to OpenAI's fixed schema.
- Other text-classification routes (sentiment-only `/v1/text/classifications`).
