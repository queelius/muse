# `text/rerank` Modality Implementation Plan (#98)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `text/rerank` modality with `POST /v1/rerank` (Cohere-shape). Bundled `bge-reranker-v2-m3` script (BAAI/bge-reranker-v2-m3) and HF plugin sniffing cross-encoder reranker repos. Generic `CrossEncoderRuntime` over `sentence_transformers.CrossEncoder`.

**Architecture:** Mirror existing modalities. New `text_rerank/` package with protocol, codec, routes, client, `hf.py`, and `runtimes/cross_encoder.py`. New bundled `bge_reranker_v2_m3.py`. Plugin priority 115 (specific to cross-encoder rerankers; wins over text-classification's broad 200 catch-all).

**Spec:** `docs/superpowers/specs/2026-04-28-text-rerank-modality-design.md`

**Target version:** v0.19.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/text_rerank/__init__.py` | create | exports `MODALITY`, `build_router`, Protocol, Result, Client, PROBE_DEFAULTS |
| `src/muse/modalities/text_rerank/protocol.py` | create | `RerankResult` dataclass, `RerankerModel` Protocol |
| `src/muse/modalities/text_rerank/codec.py` | create | `encode_rerank_response` |
| `src/muse/modalities/text_rerank/routes.py` | create | `POST /v1/rerank`, request validation, threshold-of-validity |
| `src/muse/modalities/text_rerank/client.py` | create | `RerankClient` (HTTP) |
| `src/muse/modalities/text_rerank/runtimes/__init__.py` | create | empty marker |
| `src/muse/modalities/text_rerank/runtimes/cross_encoder.py` | create | `CrossEncoderRuntime` generic runtime |
| `src/muse/modalities/text_rerank/hf.py` | create | HF plugin for cross-encoder rerankers (priority 115) |
| `src/muse/models/bge_reranker_v2_m3.py` | create | bundled script (BAAI/bge-reranker-v2-m3) |
| `src/muse/curated.yaml` | modify | +1 entry: `bge-reranker-v2-m3` (bundled) |
| `pyproject.toml` | modify | bump 0.18.3 to 0.19.0 |
| `src/muse/__init__.py` | modify | docstring v0.19.0; add `text/rerank` to bundled modalities list |
| `CLAUDE.md` | modify | document new modality (Cohere-compat note) |
| `README.md` | modify | add `text/rerank` to route list + curl example |
| `tests/modalities/text_rerank/` (full tree) | create | protocol, codec, routes, client, hf_plugin, runtime |
| `tests/models/test_bge_reranker_v2_m3.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_supervisor.py` | modify | (optional) extend slow e2e to cover the new modality |
| `tests/integration/test_remote_rerank.py` | create | opt-in integration tests |
| `tests/integration/conftest.py` | modify | `rerank_model` fixture |

---

## Task A: Protocol + Codec

Smallest, most isolated. No callers. Foundation for everything else.

**Files:**
- Create: `src/muse/modalities/text_rerank/__init__.py` (skeleton with re-exports; build_router stubbed in routes)
- Create: `src/muse/modalities/text_rerank/protocol.py`
- Create: `src/muse/modalities/text_rerank/codec.py`
- Create: `src/muse/modalities/text_rerank/routes.py` (stub returning empty APIRouter; replaced in Task C)
- Create: `tests/modalities/text_rerank/__init__.py` (empty)
- Create: `tests/modalities/text_rerank/test_protocol.py`
- Create: `tests/modalities/text_rerank/test_codec.py`

- [ ] **Step 1: Write the failing protocol test**

Create `tests/modalities/text_rerank/__init__.py` (empty) and `tests/modalities/text_rerank/test_protocol.py`:

```python
"""Protocol + dataclass shape tests for text/rerank."""
from muse.modalities.text_rerank import (
    MODALITY,
    RerankResult,
    RerankerModel,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "text/rerank"


def test_rerank_result_minimal():
    r = RerankResult(index=0, relevance_score=0.9, document_text="hello")
    assert r.index == 0
    assert r.relevance_score == 0.9
    assert r.document_text == "hello"


def test_rerank_result_supports_negative_scores():
    """Cross-encoder logits can be negative pre-sigmoid; runtime may pass through."""
    r = RerankResult(index=2, relevance_score=-3.4, document_text="x")
    assert r.relevance_score == -3.4


def test_reranker_protocol_accepts_structural_impl():
    class Fake:
        def rerank(self, query, documents, top_n=None):
            return [RerankResult(index=0, relevance_score=1.0, document_text="x")]
    assert isinstance(Fake(), RerankerModel)


def test_reranker_protocol_rejects_missing_method():
    class Missing:
        pass
    assert not isinstance(Missing(), RerankerModel)
```

- [ ] **Step 2: Write the failing codec test**

Create `tests/modalities/text_rerank/test_codec.py`:

```python
"""Codec: list[RerankResult] + return_documents -> Cohere envelope."""
from muse.modalities.text_rerank import RerankResult
from muse.modalities.text_rerank.codec import encode_rerank_response


def _sample():
    return [
        RerankResult(index=3, relevance_score=0.97, document_text="alpha"),
        RerankResult(index=0, relevance_score=0.81, document_text="beta"),
    ]


def test_encode_envelope_minimum_shape():
    body = encode_rerank_response(
        _sample(), model_id="bge-reranker-v2-m3", return_documents=False,
    )
    assert body["model"] == "bge-reranker-v2-m3"
    assert body["id"].startswith("rrk-")
    assert body["meta"] == {"billed_units": {"search_units": 1}}
    assert len(body["results"]) == 2


def test_encode_envelope_omits_document_when_flag_false():
    body = encode_rerank_response(
        _sample(), model_id="m", return_documents=False,
    )
    for row in body["results"]:
        assert "document" not in row
        assert "index" in row
        assert "relevance_score" in row


def test_encode_envelope_includes_document_text_when_flag_true():
    body = encode_rerank_response(
        _sample(), model_id="m", return_documents=True,
    )
    assert body["results"][0]["document"] == {"text": "alpha"}
    assert body["results"][1]["document"] == {"text": "beta"}


def test_encode_preserves_input_order():
    """Codec is dumb: it preserves caller order. Sorting + truncation
    happen in the runtime."""
    rows = [
        RerankResult(index=2, relevance_score=0.1, document_text="c"),
        RerankResult(index=0, relevance_score=0.9, document_text="a"),
        RerankResult(index=1, relevance_score=0.5, document_text="b"),
    ]
    body = encode_rerank_response(rows, model_id="m", return_documents=False)
    indices = [r["index"] for r in body["results"]]
    assert indices == [2, 0, 1]


def test_encode_id_unique_per_call():
    a = encode_rerank_response(_sample(), model_id="m", return_documents=False)
    b = encode_rerank_response(_sample(), model_id="m", return_documents=False)
    assert a["id"] != b["id"]


def test_encode_empty_results_is_valid():
    body = encode_rerank_response([], model_id="m", return_documents=False)
    assert body["results"] == []
    assert body["model"] == "m"
    assert body["meta"] == {"billed_units": {"search_units": 1}}
```

- [ ] **Step 3: Run, expect ImportError**

```bash
pytest tests/modalities/text_rerank/ -v
```

- [ ] **Step 4: Implement protocol**

Create `src/muse/modalities/text_rerank/protocol.py`:

```python
"""Protocol + dataclasses for text/rerank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RerankResult:
    """One scored (query, document) pair from a rerank call.

    index: position of this document in the request's `documents` array.
           Stable so a client can map back from result row to input.
    relevance_score: float; higher means more relevant. Cross-encoders
           often emit raw logits or sigmoid-normalized scores in [0, 1].
           muse passes through whatever the runtime returns.
    document_text: original document. The codec uses this when the
           request asks `return_documents=True` and drops it otherwise.
           Holding it on the dataclass keeps the codec pure (no need
           to re-index the request).
    """
    index: int
    relevance_score: float
    document_text: str


@runtime_checkable
class RerankerModel(Protocol):
    """Structural protocol any reranker backend satisfies.

    CrossEncoderRuntime (the generic runtime) and the bundled
    bge-reranker-v2-m3 Model satisfy this without inheritance.
    """

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score query against each document; return sorted descending.

        When `top_n is None`, returns all documents in score-descending
        order. When set, returns the top-N.
        """
        ...
```

- [ ] **Step 5: Implement codec**

Create `src/muse/modalities/text_rerank/codec.py`:

```python
"""Encoding for /v1/rerank responses (Cohere-shape).

Pure functions: list[RerankResult] + return_documents -> envelope dict.
Tested without FastAPI.
"""
from __future__ import annotations

import uuid
from typing import Any

from muse.modalities.text_rerank.protocol import RerankResult


def encode_rerank_response(
    results: list[RerankResult],
    *,
    model_id: str,
    return_documents: bool,
) -> dict[str, Any]:
    """Build the Cohere-shape rerank response.

    Returns a dict; the route layer wraps it in JSONResponse.
    `id` is a fresh rrk-<24hex> per call so logs and traces can
    correlate request to response.

    `meta.billed_units.search_units` is a Cohere artifact (their pricing
    unit). muse always reports `1`; the field exists for SDK
    compatibility, not for billing.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "index": r.index,
            "relevance_score": r.relevance_score,
        }
        if return_documents:
            row["document"] = {"text": r.document_text}
        rows.append(row)
    return {
        "id": f"rrk-{uuid.uuid4().hex[:24]}",
        "model": model_id,
        "results": rows,
        "meta": {"billed_units": {"search_units": 1}},
    }
```

- [ ] **Step 6: Stub runtimes package and routes**

Create `src/muse/modalities/text_rerank/runtimes/__init__.py`:

```python
"""Generic runtimes for text/rerank."""
```

Create `src/muse/modalities/text_rerank/routes.py` (stub for Task A; replaced in Task C):

```python
"""Stub routes for text/rerank.

Replaced in Task C with the real /v1/rerank endpoint. For now this
exists so build_router(registry) is importable, keeping discovery
clean between Task A (protocol + codec) and Task C (full route).
"""
from __future__ import annotations

from fastapi import APIRouter

from muse.core.registry import ModalityRegistry


def build_router(registry: ModalityRegistry) -> APIRouter:
    """Empty placeholder. Task C adds POST /v1/rerank."""
    return APIRouter()
```

Create `src/muse/modalities/text_rerank/__init__.py`:

```python
"""text/rerank modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - RerankResult dataclass
  - RerankerModel Protocol
  - RerankClient
  - PROBE_DEFAULTS

Wire contract (Cohere-compat):
  - POST /v1/rerank

This is muse's first modality with a Cohere-shape wire envelope rather
than OpenAI-compat. OpenAI has no rerank API; Cohere's /v1/rerank is
the de-facto standard, and downstream tooling (LangChain, LlamaIndex,
Haystack) expects it.
"""
from muse.modalities.text_rerank.protocol import (
    RerankResult,
    RerankerModel,
)
from muse.modalities.text_rerank.routes import build_router


MODALITY = "text/rerank"


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": "1 query, 4 documents",
    "call": lambda m: m.rerank(
        "what is muse?",
        [
            "muse is an audio server",
            "muse is a server",
            "purple cats",
            "model serving",
        ],
        None,
    ),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "RerankResult",
    "RerankerModel",
]
```

(`RerankClient` re-export added in Task D.)

- [ ] **Step 7: Run tests; verify they pass**

```bash
pytest tests/modalities/text_rerank/ -v
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: protocol + codec tests pass; full fast lane stays green.

- [ ] **Step 8: Commit**

```bash
git add src/muse/modalities/text_rerank/__init__.py \
        src/muse/modalities/text_rerank/protocol.py \
        src/muse/modalities/text_rerank/codec.py \
        src/muse/modalities/text_rerank/runtimes/__init__.py \
        src/muse/modalities/text_rerank/routes.py \
        tests/modalities/text_rerank/__init__.py \
        tests/modalities/text_rerank/test_protocol.py \
        tests/modalities/text_rerank/test_codec.py
git commit -m "$(cat <<'EOF'
feat(rerank): text/rerank modality skeleton + codec

MODALITY tag, RerankResult dataclass, RerankerModel structural protocol,
encode_rerank_response codec building Cohere-shape envelopes.
routes.py is a stub returning an empty APIRouter so build_router is
importable; Task C replaces it with the real POST /v1/rerank endpoint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task B: CrossEncoderRuntime generic runtime

Wraps `sentence_transformers.CrossEncoder`. Lazy imports. Honors
`device="auto"` and `max_length` from manifest capabilities. The
runtime owns the sort + slice; the route layer doesn't touch scores.

**Files:**
- Create: `src/muse/modalities/text_rerank/runtimes/cross_encoder.py`
- Create: `tests/modalities/text_rerank/runtimes/__init__.py` (empty)
- Create: `tests/modalities/text_rerank/runtimes/test_cross_encoder.py`

- [ ] **Step 1: Write the failing runtime test**

Create `tests/modalities/text_rerank/runtimes/__init__.py` (empty) and `tests/modalities/text_rerank/runtimes/test_cross_encoder.py`:

```python
"""Tests for CrossEncoderRuntime (sentence-transformers CrossEncoder wrapper)."""
from unittest.mock import MagicMock, patch

import pytest

import muse.modalities.text_rerank.runtimes.cross_encoder as ce_mod
from muse.modalities.text_rerank import RerankResult
from muse.modalities.text_rerank.runtimes.cross_encoder import (
    CrossEncoderRuntime,
)


def _patched_runtime(predict_return):
    """Return a CrossEncoderRuntime with sentence_transformers stubbed."""
    fake_ce = MagicMock()
    fake_ce.predict.return_value = predict_return
    fake_ce_class = MagicMock(return_value=fake_ce)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", fake_torch):
        rt = CrossEncoderRuntime(
            model_id="test", hf_repo="org/repo",
            local_dir=None, device="cpu", max_length=512,
        )
    return rt, fake_ce, fake_ce_class


def test_runtime_constructs_with_local_dir_preference():
    """Runtime prefers local_dir over hf_repo as the source path."""
    fake_ce = MagicMock()
    fake_ce_class = MagicMock(return_value=fake_ce)
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", MagicMock()):
        CrossEncoderRuntime(
            model_id="m", hf_repo="org/repo",
            local_dir="/tmp/cache/abc", device="cpu",
        )
    args, kwargs = fake_ce_class.call_args
    assert args[0] == "/tmp/cache/abc"


def test_runtime_falls_back_to_hf_repo_when_no_local_dir():
    fake_ce = MagicMock()
    fake_ce_class = MagicMock(return_value=fake_ce)
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", MagicMock()):
        CrossEncoderRuntime(
            model_id="m", hf_repo="org/repo",
            local_dir=None, device="cpu",
        )
    args, _ = fake_ce_class.call_args
    assert args[0] == "org/repo"


def test_rerank_returns_descending_score_order():
    rt, fake_ce, _ = _patched_runtime([0.1, 0.9, 0.4])
    out = rt.rerank("q", ["a", "b", "c"])
    assert isinstance(out, list)
    assert all(isinstance(r, RerankResult) for r in out)
    indices = [r.index for r in out]
    assert indices == [1, 2, 0]
    scores = [r.relevance_score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_rerank_passes_pairs_to_predict():
    rt, fake_ce, _ = _patched_runtime([0.5, 0.5])
    rt.rerank("hello", ["doc1", "doc2"])
    args, _ = fake_ce.predict.call_args
    pairs = args[0]
    assert pairs == [("hello", "doc1"), ("hello", "doc2")]


def test_rerank_top_n_truncates():
    rt, fake_ce, _ = _patched_runtime([0.1, 0.9, 0.5, 0.7])
    out = rt.rerank("q", ["a", "b", "c", "d"], top_n=2)
    assert len(out) == 2
    assert [r.index for r in out] == [1, 3]


def test_rerank_top_n_none_returns_all():
    rt, fake_ce, _ = _patched_runtime([0.1, 0.9, 0.5])
    out = rt.rerank("q", ["a", "b", "c"], top_n=None)
    assert len(out) == 3


def test_rerank_empty_documents_returns_empty():
    rt, fake_ce, _ = _patched_runtime([])
    out = rt.rerank("q", [])
    assert out == []
    fake_ce.predict.assert_not_called()


def test_rerank_preserves_document_text():
    rt, fake_ce, _ = _patched_runtime([0.2, 0.8])
    out = rt.rerank("q", ["alpha", "beta"])
    text_by_index = {r.index: r.document_text for r in out}
    assert text_by_index == {0: "alpha", 1: "beta"}


def test_runtime_raises_when_sentence_transformers_missing():
    with patch.object(ce_mod, "CrossEncoder", None), \
            patch.object(ce_mod, "torch", MagicMock()):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            CrossEncoderRuntime(
                model_id="m", hf_repo="org/repo",
                local_dir=None, device="cpu",
            )


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    with patch.object(ce_mod, "torch", None):
        from muse.modalities.text_rerank.runtimes.cross_encoder import _select_device
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(ce_mod, "torch", fake_torch):
        from muse.modalities.text_rerank.runtimes.cross_encoder import _select_device
        assert _select_device("auto") == "cuda"
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/text_rerank/runtimes/ -v
```

- [ ] **Step 3: Implement runtime**

Create `src/muse/modalities/text_rerank/runtimes/cross_encoder.py`:

```python
"""CrossEncoderRuntime: generic runtime over any HF cross-encoder reranker.

One class wraps `sentence_transformers.CrossEncoder` for any repo on
HuggingFace that ships a cross-encoder. Pulled via the HF resolver:
`muse pull hf://BAAI/bge-reranker-v2-m3` synthesizes a manifest
pointing at this class.

Deferred imports follow the muse pattern: torch + CrossEncoder stay
as module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.text_rerank.protocol import RerankResult


logger = logging.getLogger(__name__)


torch: Any = None
CrossEncoder: Any = None


def _ensure_deps() -> None:
    global torch, CrossEncoder
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("CrossEncoderRuntime: torch unavailable: %s", e)
    if CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as _ce
            CrossEncoder = _ce
        except Exception as e:  # noqa: BLE001
            logger.debug("CrossEncoderRuntime: sentence_transformers unavailable: %s", e)


class CrossEncoderRuntime:
    """Generic cross-encoder reranker runtime."""

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
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed; run `muse pull` "
                "or install `sentence-transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._max_length = max_length

        src = local_dir or hf_repo
        logger.info(
            "loading cross-encoder reranker from %s (device=%s, max_length=%d)",
            src, self._device, max_length,
        )
        self._model = CrossEncoder(
            src, max_length=max_length, device=self._device,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score query against each document; return sorted descending.

        Empty `documents` returns []. Otherwise builds [(query, doc), ...]
        pairs, runs `predict`, sorts by score descending, slices to
        `top_n` if set, and emits one RerankResult per surviving row.
        """
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        scored = sorted(
            [(i, float(s)) for i, s in enumerate(scores)],
            key=lambda kv: kv[1], reverse=True,
        )
        if top_n is not None:
            scored = scored[:top_n]
        return [
            RerankResult(
                index=i,
                relevance_score=s,
                document_text=documents[i],
            )
            for i, s in scored
        ]


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

- [ ] **Step 4: Run tests; verify they pass**

```bash
pytest tests/modalities/text_rerank/runtimes/ -v
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: ~11 passed; full fast lane stays green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_rerank/runtimes/cross_encoder.py \
        tests/modalities/text_rerank/runtimes/__init__.py \
        tests/modalities/text_rerank/runtimes/test_cross_encoder.py
git commit -m "$(cat <<'EOF'
feat(rerank): CrossEncoderRuntime generic runtime

Wraps sentence_transformers.CrossEncoder. Lazy-imports torch +
CrossEncoder; honors device='auto' and max_length from manifest
capabilities. rerank() builds (query, doc) pairs, runs predict, sorts
descending, slices to top_n.

No callers yet; routes wire it in next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task C: Routes (POST /v1/rerank)

Replaces the Task A stub with the real endpoint. Validates request,
resolves the registered backend, calls `backend.rerank(...)` (offloaded
to a thread; cross-encoder predict is sync), encodes the response.

**Files:**
- Modify (replace stub): `src/muse/modalities/text_rerank/routes.py`
- Create: `tests/modalities/text_rerank/test_routes.py`

- [ ] **Step 1: Write the failing route test**

Create `tests/modalities/text_rerank/test_routes.py`:

```python
"""Route tests for POST /v1/rerank."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_rerank import (
    MODALITY,
    RerankResult,
    build_router,
)


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "bge-reranker-v2-m3"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def _fake_backend(results):
    backend = MagicMock()
    backend.model_id = "bge-reranker-v2-m3"
    backend.rerank.return_value = results
    return backend


def test_rerank_returns_envelope_for_minimal_request():
    backend = _fake_backend([
        RerankResult(index=1, relevance_score=0.9, document_text="b"),
        RerankResult(index=0, relevance_score=0.1, document_text="a"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a", "b"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "bge-reranker-v2-m3"
    assert body["id"].startswith("rrk-")
    assert body["meta"] == {"billed_units": {"search_units": 1}}
    assert len(body["results"]) == 2
    assert body["results"][0]["index"] == 1
    assert body["results"][0]["relevance_score"] == 0.9
    assert "document" not in body["results"][0]


def test_rerank_includes_documents_when_flag_true():
    backend = _fake_backend([
        RerankResult(index=0, relevance_score=0.7, document_text="alpha"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["alpha"],
        "return_documents": True,
    })
    body = r.json()
    assert body["results"][0]["document"] == {"text": "alpha"}


def test_rerank_top_n_passed_to_backend():
    backend = _fake_backend([
        RerankResult(index=0, relevance_score=0.7, document_text="x"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["x", "y", "z"],
        "top_n": 2,
    })
    assert r.status_code == 200
    args, kwargs = backend.rerank.call_args
    # rerank(query, documents, top_n) signature
    if "top_n" in kwargs:
        assert kwargs["top_n"] == 2
    else:
        assert args[2] == 2


def test_rerank_400_on_empty_query():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "",
        "documents": ["a"],
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_parameter"


def test_rerank_400_on_empty_documents():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": [],
    })
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_parameter"


def test_rerank_400_on_empty_document_string():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["good", ""],
    })
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_parameter"


def test_rerank_400_on_top_n_zero():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a"],
        "top_n": 0,
    })
    assert r.status_code in (400, 422)


def test_rerank_404_on_unknown_model():
    backend = _fake_backend([])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["a"],
        "model": "nonexistent",
    })
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "model_not_found"


def test_rerank_default_model_resolves_first_registered():
    backend = _fake_backend([
        RerankResult(index=0, relevance_score=0.5, document_text="x"),
    ])
    client = _make_client(backend)

    r = client.post("/v1/rerank", json={
        "query": "q",
        "documents": ["x"],
    })
    assert r.status_code == 200
    assert r.json()["model"] == "bge-reranker-v2-m3"
```

- [ ] **Step 2: Run, expect failures (stub returns nothing)**

```bash
pytest tests/modalities/text_rerank/test_routes.py -v
```

- [ ] **Step 3: Implement routes**

Replace `src/muse/modalities/text_rerank/routes.py` with:

```python
"""FastAPI routes for /v1/rerank.

Cohere-compat shape:
  request:  {"query": str, "documents": list[str],
             "top_n"?: int, "return_documents"?: bool, "model"?: str}
  response: {"id", "model", "results": [{"index", "relevance_score",
             "document"?: {"text": str}}], "meta": {"billed_units": ...}}

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.text_rerank.codec import encode_rerank_response


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/rerank"


logger = logging.getLogger(__name__)


# Defaults are conservative. A request with documents=10000 can OOM the
# worker by trying to materialize a giant batch of pairs. Caps are
# tunable via env so power users with big GPUs can lift them.
_MAX_DOCUMENTS = int(os.environ.get("MUSE_RERANK_MAX_DOCUMENTS", "1000"))
_MAX_QUERY_CHARS = int(os.environ.get("MUSE_RERANK_MAX_QUERY_CHARS", "4000"))
_MAX_DOC_CHARS = int(os.environ.get("MUSE_RERANK_MAX_DOC_CHARS", "100000"))


class _RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1, le=1000)
    model: str | None = None
    return_documents: bool = False


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/rerank")
    async def rerank(req: _RerankRequest):
        if not req.query:
            return error_response(
                400, "invalid_parameter", "query must not be empty",
            )
        if len(req.query) > _MAX_QUERY_CHARS:
            return error_response(
                400, "invalid_parameter",
                f"query exceeds MUSE_RERANK_MAX_QUERY_CHARS={_MAX_QUERY_CHARS}",
            )
        if not req.documents:
            return error_response(
                400, "invalid_parameter", "documents must be a non-empty list",
            )
        if len(req.documents) > _MAX_DOCUMENTS:
            return error_response(
                400, "invalid_parameter",
                f"documents batch size {len(req.documents)} exceeds "
                f"MUSE_RERANK_MAX_DOCUMENTS={_MAX_DOCUMENTS}",
            )
        empty_idx = next(
            (i for i, s in enumerate(req.documents) if not s), None,
        )
        if empty_idx is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{empty_idx}] must not be empty",
            )
        too_long = next(
            (i for i, s in enumerate(req.documents) if len(s) > _MAX_DOC_CHARS),
            None,
        )
        if too_long is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{too_long}] exceeds "
                f"MUSE_RERANK_MAX_DOC_CHARS={_MAX_DOC_CHARS}",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        effective_id = backend.model_id

        # backend.rerank is sync (cross-encoder predict); offload so a
        # slow inference doesn't block sibling /health, /v1/models, or
        # other in-flight requests on the same worker.
        results = await asyncio.to_thread(
            backend.rerank, req.query, req.documents, req.top_n,
        )
        body = encode_rerank_response(
            results,
            model_id=effective_id,
            return_documents=req.return_documents,
        )
        return JSONResponse(content=body)

    return router
```

- [ ] **Step 4: Run tests; verify they pass**

```bash
pytest tests/modalities/text_rerank/test_routes.py -v
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: 9 passed; full fast lane stays green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_rerank/routes.py \
        tests/modalities/text_rerank/test_routes.py
git commit -m "$(cat <<'EOF'
feat(rerank): POST /v1/rerank route (Cohere-compat)

Validates query + documents (non-empty, size caps via env), resolves
the backend, offloads sync rerank to a thread, encodes the
Cohere-shape envelope. Errors use muse's OpenAI-shape error envelopes
(400 invalid_parameter, 404 model_not_found).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task D: RerankClient

Mirror `ModerationsClient`. Server URL public attribute, MUSE_SERVER
env fallback, requests + raise_for_status.

**Files:**
- Create: `src/muse/modalities/text_rerank/client.py`
- Modify: `src/muse/modalities/text_rerank/__init__.py` (re-export client)
- Create: `tests/modalities/text_rerank/test_client.py`

- [ ] **Step 1: Write the failing client test**

Create `tests/modalities/text_rerank/test_client.py`:

```python
"""Tests for RerankClient (HTTP wrapper)."""
from unittest.mock import MagicMock, patch

from muse.modalities.text_rerank import RerankClient


def _fake_response(json_body):
    r = MagicMock()
    r.json.return_value = json_body
    r.raise_for_status.return_value = None
    return r


def test_default_server_url_uses_localhost():
    c = RerankClient()
    assert c.server_url == "http://localhost:8000"


def test_server_url_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://gpu-host:8000")
    c = RerankClient()
    assert c.server_url == "http://gpu-host:8000"


def test_server_url_arg_beats_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env:8000")
    c = RerankClient("http://arg:8000")
    assert c.server_url == "http://arg:8000"


def test_rerank_post_minimal_body():
    c = RerankClient("http://localhost:8000")
    body = {
        "id": "rrk-x", "model": "m", "results": [], "meta": {},
    }
    with patch("muse.modalities.text_rerank.client.requests.post",
               return_value=_fake_response(body)) as mock_post:
        out = c.rerank(query="q", documents=["a", "b"])
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/rerank"
    assert kwargs["json"] == {"query": "q", "documents": ["a", "b"]}
    assert out == body


def test_rerank_includes_optional_fields():
    c = RerankClient("http://x:8000")
    with patch("muse.modalities.text_rerank.client.requests.post",
               return_value=_fake_response({"results": []})) as mock_post:
        c.rerank(
            query="q",
            documents=["a"],
            top_n=5,
            model="bge-reranker-v2-m3",
            return_documents=True,
        )
    _, kwargs = mock_post.call_args
    sent = kwargs["json"]
    assert sent["top_n"] == 5
    assert sent["model"] == "bge-reranker-v2-m3"
    assert sent["return_documents"] is True


def test_rerank_raises_on_http_error():
    c = RerankClient()
    failing = MagicMock()
    failing.raise_for_status.side_effect = RuntimeError("503")
    with patch("muse.modalities.text_rerank.client.requests.post",
               return_value=failing):
        try:
            c.rerank(query="q", documents=["a"])
        except RuntimeError as e:
            assert "503" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/text_rerank/test_client.py -v
```

- [ ] **Step 3: Implement client**

Create `src/muse/modalities/text_rerank/client.py`:

```python
"""HTTP client for /v1/rerank.

Parallel to other muse clients: server_url public attribute, MUSE_SERVER
env fallback, requests under the hood, raise_for_status before parsing.
"""
from __future__ import annotations

import os
from typing import Any

import requests


_DEFAULT_SERVER = "http://localhost:8000"


class RerankClient:
    """Minimal HTTP client for the text/rerank modality.

    Cohere-compat: returns the full response envelope unchanged so the
    caller sees `id`, `model`, `results`, `meta` exactly as Cohere SDKs
    expect.
    """

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

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        model: str | None = None,
        return_documents: bool = False,
    ) -> dict[str, Any]:
        """Send a rerank request; return the full Cohere-shape envelope."""
        body: dict[str, Any] = {"query": query, "documents": documents}
        if top_n is not None:
            body["top_n"] = top_n
        if model is not None:
            body["model"] = model
        if return_documents:
            body["return_documents"] = True

        r = requests.post(
            f"{self.server_url}/v1/rerank",
            json=body, timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 4: Re-export in __init__.py**

Update `src/muse/modalities/text_rerank/__init__.py`:

```python
from muse.modalities.text_rerank.client import RerankClient
from muse.modalities.text_rerank.protocol import (
    RerankResult,
    RerankerModel,
)
from muse.modalities.text_rerank.routes import build_router


MODALITY = "text/rerank"


PROBE_DEFAULTS = {
    "shape": "1 query, 4 documents",
    "call": lambda m: m.rerank(
        "what is muse?",
        [
            "muse is an audio server",
            "muse is a server",
            "purple cats",
            "model serving",
        ],
        None,
    ),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "RerankResult",
    "RerankerModel",
    "RerankClient",
]
```

(Keep the docstring at the top.)

- [ ] **Step 5: Run tests; verify they pass**

```bash
pytest tests/modalities/text_rerank/test_client.py -v
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: 6 passed; full fast lane stays green.

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/text_rerank/client.py \
        src/muse/modalities/text_rerank/__init__.py \
        tests/modalities/text_rerank/test_client.py
git commit -m "$(cat <<'EOF'
feat(rerank): RerankClient HTTP wrapper

Mirrors other muse clients (server_url + MUSE_SERVER env fallback +
requests + raise_for_status). Returns the full Cohere-shape envelope
unchanged so downstream tooling sees id/model/results/meta exactly
as Cohere SDKs expect.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task E: Bundled bge_reranker_v2_m3 script

Hand-written script for `BAAI/bge-reranker-v2-m3`. Wraps `CrossEncoder`
directly (not via the generic runtime) so the script demonstrates the
pattern muse uses for other bundled models. Lazy imports.

**Files:**
- Create: `src/muse/models/bge_reranker_v2_m3.py`
- Create: `tests/models/test_bge_reranker_v2_m3.py`

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_bge_reranker_v2_m3.py`:

```python
"""Tests for the bundled bge_reranker_v2_m3 script."""
from unittest.mock import MagicMock, patch

import pytest

import muse.models.bge_reranker_v2_m3 as bge_mod
from muse.modalities.text_rerank import RerankResult


def test_manifest_required_fields():
    m = bge_mod.MANIFEST
    assert m["model_id"] == "bge-reranker-v2-m3"
    assert m["modality"] == "text/rerank"
    assert m["hf_repo"] == "BAAI/bge-reranker-v2-m3"
    assert "pip_extras" in m
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("sentence-transformers" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = bge_mod.MANIFEST["capabilities"]
    assert caps["max_length"] == 8192
    assert caps["device"] == "auto"
    assert "memory_gb" in caps


def test_model_class_exists():
    assert hasattr(bge_mod, "Model")
    assert bge_mod.Model.model_id == "bge-reranker-v2-m3"


def test_model_construction_lazy_imports():
    """Model.__init__ should call _ensure_deps and read CrossEncoder
    via the module's sentinel so tests can patch it."""
    fake_ce_inst = MagicMock()
    fake_ce_class = MagicMock(return_value=fake_ce_inst)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        m = bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
    assert m._device == "cpu"
    fake_ce_class.assert_called_once()


def test_model_prefers_local_dir():
    fake_ce_class = MagicMock(return_value=MagicMock())
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir="/tmp/bge")
    args, _ = fake_ce_class.call_args
    assert args[0] == "/tmp/bge"


def test_rerank_returns_sorted_results():
    fake_ce = MagicMock()
    fake_ce.predict.return_value = [0.1, 0.9, 0.5]
    fake_ce_class = MagicMock(return_value=fake_ce)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        m = bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
        out = m.rerank("q", ["a", "b", "c"], top_n=2)
    assert all(isinstance(r, RerankResult) for r in out)
    assert [r.index for r in out] == [1, 2]


def test_rerank_empty_documents():
    fake_ce_class = MagicMock(return_value=MagicMock())
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        m = bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
        out = m.rerank("q", [])
    assert out == []


def test_model_raises_when_sentence_transformers_missing():
    fake_torch = MagicMock()
    with patch.object(bge_mod, "CrossEncoder", None), \
            patch.object(bge_mod, "torch", fake_torch):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/models/test_bge_reranker_v2_m3.py -v
```

- [ ] **Step 3: Implement bundled script**

Create `src/muse/models/bge_reranker_v2_m3.py`:

```python
"""BAAI bge-reranker-v2-m3 (multilingual cross-encoder reranker).

Curated default for `text/rerank`. ~568MB on disk; works on CPU.
8192-token context window (the m3 lineage). Multilingual.

License: Apache 2.0.

Wraps `sentence_transformers.CrossEncoder`; lazy imports so muse pull
+ muse --help work without ML deps installed.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.text_rerank.protocol import RerankResult


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches sd_turbo, soprano_80m, etc.).
torch: Any = None
CrossEncoder: Any = None


def _ensure_deps() -> None:
    global torch, CrossEncoder
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("bge_reranker_v2_m3: torch unavailable: %s", e)
    if CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as _ce
            CrossEncoder = _ce
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "bge_reranker_v2_m3: sentence_transformers unavailable: %s", e,
            )


MANIFEST = {
    "model_id": "bge-reranker-v2-m3",
    "modality": "text/rerank",
    "hf_repo": "BAAI/bge-reranker-v2-m3",
    "description": (
        "BAAI bge-reranker-v2-m3: multilingual cross-encoder reranker, "
        "8192-token context, ~568MB"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "sentence-transformers>=2.2.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "max_length": 8192,
        # Measured peak fp16 inference, query + 32 documents @ 8192 ctx.
        "memory_gb": 1.2,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "tokenizer*", "spiece.model",
    ],
}


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


class Model:
    """bge-reranker-v2-m3 backend (cross-encoder)."""

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        max_length: int = 8192,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed; run "
                "`muse pull bge-reranker-v2-m3` or install "
                "`sentence-transformers` into this venv"
            )
        self._device = _select_device(device)
        self._max_length = max_length

        src = local_dir or hf_repo
        logger.info(
            "loading bge-reranker-v2-m3 from %s (device=%s, max_length=%d)",
            src, self._device, max_length,
        )
        self._model = CrossEncoder(
            src, max_length=max_length, device=self._device,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        scored = sorted(
            [(i, float(s)) for i, s in enumerate(scores)],
            key=lambda kv: kv[1], reverse=True,
        )
        if top_n is not None:
            scored = scored[:top_n]
        return [
            RerankResult(
                index=i,
                relevance_score=s,
                document_text=documents[i],
            )
            for i, s in scored
        ]
```

- [ ] **Step 4: Run tests; verify they pass**

```bash
pytest tests/models/test_bge_reranker_v2_m3.py -v
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: 8 passed; fast lane stays green.

- [ ] **Step 5: Commit**

```bash
git add src/muse/models/bge_reranker_v2_m3.py \
        tests/models/test_bge_reranker_v2_m3.py
git commit -m "$(cat <<'EOF'
feat(rerank): bundled bge-reranker-v2-m3 script

Wraps sentence_transformers.CrossEncoder for BAAI/bge-reranker-v2-m3.
Lazy imports torch + CrossEncoder. MANIFEST: ~568MB, Apache 2.0,
8192-token max_length, multilingual. Honors device='auto' and
prefers local_dir over hf_repo.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task F: HF plugin for cross-encoder rerankers

`src/muse/modalities/text_rerank/hf.py`. Sniff: `cross-encoder` tag OR
(`text-classification` tag AND `rerank` in repo name). Priority **115**
so it beats `text/classification`'s 200 catch-all but loses to
file-pattern plugins at 100.

**Files:**
- Create: `src/muse/modalities/text_rerank/hf.py`
- Create: `tests/modalities/text_rerank/test_hf_plugin.py`

- [ ] **Step 1: Write the failing plugin test**

Create `tests/modalities/text_rerank/test_hf_plugin.py`:

```python
"""Tests for the text/rerank HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.text_rerank.hf import HF_PLUGIN


def _fake_info(repo_id, tags=(), siblings=(), card=None):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "text/rerank"
    # 115: between embedding/text (110) and text/classification (200).
    # Wins over text-classification's catch-all because reranker repos
    # commonly carry the text-classification tag too.
    assert HF_PLUGIN["priority"] == 115


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.text_rerank.runtimes.cross_encoder"
        ":CrossEncoderRuntime"
    )


def test_sniff_true_for_cross_encoder_tag():
    info = _fake_info("any/repo", tags=["cross-encoder"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_text_classification_with_rerank_in_name():
    info = _fake_info("BAAI/bge-reranker-v2-m3",
                     tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_for_text_classification_without_rerank_name():
    info = _fake_info("KoalaAI/Text-Moderation",
                     tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_unrelated_repo():
    info = _fake_info("Qwen/Qwen3-8B-GGUF",
                     tags=["chat", "text-generation"])
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_synthesizes_manifest():
    info = _fake_info(
        "BAAI/bge-reranker-v2-m3",
        tags=["cross-encoder", "text-classification"],
        card=SimpleNamespace(license="apache-2.0"),
    )
    resolved = HF_PLUGIN["resolve"]("BAAI/bge-reranker-v2-m3", None, info)
    m = resolved.manifest
    assert m["model_id"] == "bge-reranker-v2-m3"
    assert m["modality"] == "text/rerank"
    assert m["hf_repo"] == "BAAI/bge-reranker-v2-m3"
    assert m["license"] == "apache-2.0"
    # max_length heuristic: m3 lineage ships 8K context.
    assert m["capabilities"]["max_length"] == 8192


def test_resolve_max_length_heuristic_default_for_unknown_repo():
    info = _fake_info("acme/some-cross-encoder", tags=["cross-encoder"])
    resolved = HF_PLUGIN["resolve"]("acme/some-cross-encoder", None, info)
    assert resolved.manifest["capabilities"]["max_length"] == 512


def test_resolve_max_length_heuristic_jina_v2():
    info = _fake_info("jinaai/jina-reranker-v2-base-multilingual",
                     tags=["cross-encoder"])
    resolved = HF_PLUGIN["resolve"](
        "jinaai/jina-reranker-v2-base-multilingual", None, info,
    )
    assert resolved.manifest["capabilities"]["max_length"] == 1024


def test_resolve_backend_path_points_to_cross_encoder_runtime():
    info = _fake_info("acme/x", tags=["cross-encoder"])
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_search_yields_results():
    api = MagicMock()
    repo1 = SimpleNamespace(id="BAAI/bge-reranker-v2-m3", downloads=1000)
    repo2 = SimpleNamespace(id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                            downloads=500)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](api, "rerank", sort="downloads", limit=10))
    assert len(out) == 2
    assert out[0].uri == "hf://BAAI/bge-reranker-v2-m3"
    assert out[0].modality == "text/rerank"
    assert out[0].downloads == 1000
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/text_rerank/test_hf_plugin.py -v
```

- [ ] **Step 3: Implement plugin**

Create `src/muse/modalities/text_rerank/hf.py`:

```python
"""HF resolver plugin for cross-encoder rerankers.

Sniffs HF repos with the `cross-encoder` tag, OR repos with both the
`text-classification` tag and `rerank` in the repo name (case-insensitive).
This second branch catches BAAI/bge-reranker-v2-m3 and similar repos
that ship under text-classification without the dedicated cross-encoder
tag.

Priority 115: more specific than embedding/text (110); wins over
text/classification (200, catch-all).

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.text_rerank.runtimes.cross_encoder:CrossEncoderRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "sentence-transformers>=2.2.0")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "cross-encoder" in tags:
        return True
    if "text-classification" in tags:
        repo_id = (getattr(info, "id", "") or "").lower()
        return "rerank" in repo_id
    return False


def _max_length_for(repo_id: str) -> int:
    """Heuristic max_length per known reranker lineage.

    Most rerankers default to 512. The bge-m3 lineage ships an 8K
    context. Jina v2 defaults to 1024. Override via curated capabilities
    when needed.
    """
    name = repo_id.lower()
    if "bge-reranker-v2-m3" in name or "bge-m3" in name:
        return 8192
    if "jina-reranker-v2" in name:
        return 1024
    return 512


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "text/rerank",
        "hf_repo": repo_id,
        "description": f"Cross-encoder reranker: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {
            "device": "auto",
            "max_length": _max_length_for(repo_id),
        },
    }

    def _download(cache_root: Path) -> Path:
        # Keep weights light: prefer safetensors, drop tf/flax/onnx.
        siblings = [s.rfilename for s in getattr(info, "siblings", [])]
        has_fp16 = any(".fp16." in f for f in siblings)
        if has_fp16:
            allow_patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
        else:
            allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
        # Some rerankers still ship pytorch_model.bin (no safetensors).
        allow_patterns.append("pytorch_model.bin")
        # spiece.model + tokenizer files for tokenization.
        allow_patterns.extend(["tokenizer*", "spiece.model"])
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HuggingFace for cross-encoder rerankers.

    Filter: cross-encoder tag (when available). Some reranker repos
    ship under text-classification only and won't show up via the tag
    filter; we surface those via the `query` term `rerank` when the
    user passes one.
    """
    repos = api.list_models(
        search=query, filter="cross-encoder",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="text/rerank",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "text/rerank",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 115: more specific than embedding/text (110); wins over
    # text/classification (200, catch-all). Loses to file-pattern
    # plugins at 100 (GGUF, faster-whisper, diffusers).
    "priority": 115,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

- [ ] **Step 4: Run tests; verify they pass**

```bash
pytest tests/modalities/text_rerank/test_hf_plugin.py -v
pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: 11 passed; fast lane stays green. Also re-run the
text-classification tests to make sure no regression on its plugin:

```bash
pytest tests/modalities/text_classification/ -v
pytest tests/core/test_resolvers_hf.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/text_rerank/hf.py \
        tests/modalities/text_rerank/test_hf_plugin.py
git commit -m "$(cat <<'EOF'
feat(rerank): HF plugin for cross-encoder rerankers (priority 115)

Sniffs HF repos by cross-encoder tag, OR by text-classification tag
combined with 'rerank' in the repo name. Priority 115 so the plugin
wins over text/classification's catch-all 200 (since reranker repos
commonly carry the text-classification tag too) but loses to
file-pattern plugins at 100. max_length heuristic per lineage:
bge-m3=8192, jina-v2=1024, fallback=512.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task G: Curated entry

Add `bge-reranker-v2-m3` to curated.yaml as a bundled-script alias.
The catalog now surfaces `bge-reranker-v2-m3` in `muse models list
--available`.

**Files:**
- Modify: `src/muse/curated.yaml`
- Modify: `tests/core/test_curated.py` (or wherever the curated parse
  test lives)

- [ ] **Step 1: Locate and inspect the curated test**

```bash
grep -rn "bge-reranker\|text-moderation\|kokoro-82m" tests/core/
```

Pick the file that asserts on curated entries (commonly
`tests/core/test_curated.py`); add a test for the new entry.

- [ ] **Step 2: Write the failing test**

Add to `tests/core/test_curated.py`:

```python
def test_curated_includes_bge_reranker_v2_m3(curated):
    """bge-reranker-v2-m3 is curated as a bundled alias."""
    entry = next((c for c in curated if c["id"] == "bge-reranker-v2-m3"), None)
    assert entry is not None
    assert entry.get("bundled") is True
```

(If the existing test file uses a different fixture shape, mirror it.)

- [ ] **Step 3: Run, expect failure**

```bash
pytest tests/core/test_curated.py -v -k bge_reranker
```

- [ ] **Step 4: Add the curated entry**

In `src/muse/curated.yaml`, after the `text/classification` block,
add a new `text/rerank` section:

```yaml
# ---------- text/rerank (cross-encoder rerankers) ----------

- id: bge-reranker-v2-m3
  bundled: true
```

- [ ] **Step 5: Run tests; verify they pass**

```bash
pytest tests/core/test_curated.py -v
pytest -m "not slow" -q 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/curated.yaml tests/core/test_curated.py
git commit -m "$(cat <<'EOF'
feat(rerank): curated entry for bge-reranker-v2-m3

Adds bge-reranker-v2-m3 (BAAI/bge-reranker-v2-m3) as a curated
bundled-script alias. Now appears in `muse models list --available
--modality text/rerank`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task H: Slow e2e + integration tests

Two test layers: a `@pytest.mark.slow` in-process supervisor smoke
test (no network, mocked runtime), and an opt-in integration test
that hits a real muse server with a real reranker loaded.

**Files:**
- Modify (or create as new file): `tests/cli_impl/test_e2e_supervisor.py`
- Modify: `tests/integration/conftest.py`
- Create: `tests/integration/test_remote_rerank.py`

- [ ] **Step 1: Add the slow e2e test**

Either extend the existing supervisor e2e or add a new file. Look at
`tests/cli_impl/test_e2e_supervisor.py` for the pattern: spawn the
supervisor with a fake registry, hit `/v1/rerank`, assert envelope.

A minimal new file `tests/cli_impl/test_e2e_rerank.py` (if the
existing supervisor e2e is hard to extend):

```python
"""Slow e2e smoke for /v1/rerank.

Spawns the supervisor in-process; registers a fake reranker that
returns deterministic scores; hits POST /v1/rerank; asserts the
Cohere envelope shape.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_rerank import (
    MODALITY,
    RerankResult,
    build_router,
)


@pytest.mark.slow
def test_rerank_e2e_through_app_factory():
    reg = ModalityRegistry()
    backend = MagicMock()
    backend.model_id = "fake-rerank"
    backend.rerank.return_value = [
        RerankResult(index=2, relevance_score=0.95, document_text="winner"),
        RerankResult(index=0, relevance_score=0.10, document_text="loser"),
    ]
    reg.register(MODALITY, backend, manifest={"model_id": "fake-rerank"})

    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post("/v1/rerank", json={
        "query": "what is muse?",
        "documents": ["a", "b", "c"],
        "top_n": 2,
        "return_documents": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-rerank"
    assert body["meta"]["billed_units"]["search_units"] == 1
    assert len(body["results"]) == 2
    assert body["results"][0]["index"] == 2
    assert body["results"][0]["document"] == {"text": "winner"}
```

- [ ] **Step 2: Add integration fixture**

In `tests/integration/conftest.py`, add a `rerank_model_id` fixture
mirroring the existing pattern (whisper_model_id, chat_model_id,
text_moderation_model_id). The default model id should be
`bge-reranker-v2-m3`; allow `MUSE_RERANK_MODEL_ID` env override.

- [ ] **Step 3: Add the integration test**

Create `tests/integration/test_remote_rerank.py`:

```python
"""Integration tests for /v1/rerank against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable
or no rerank model is loaded.

Targets the configurable rerank model id (default bge-reranker-v2-m3,
override via MUSE_RERANK_MODEL_ID).
"""
from __future__ import annotations

import os

import pytest
import requests

from muse.modalities.text_rerank import RerankClient


REMOTE = os.environ.get("MUSE_REMOTE_SERVER")
RERANK_MODEL = os.environ.get("MUSE_RERANK_MODEL_ID", "bge-reranker-v2-m3")


pytestmark = pytest.mark.skipif(
    REMOTE is None,
    reason="MUSE_REMOTE_SERVER not set; integration tests opt-in",
)


def _model_loaded(server: str, model_id: str) -> bool:
    try:
        r = requests.get(f"{server}/v1/models", timeout=5)
        if r.status_code != 200:
            return False
        return any(m["id"] == model_id for m in r.json().get("data", []))
    except Exception:
        return False


@pytest.fixture(scope="module")
def rerank_server():
    if not _model_loaded(REMOTE, RERANK_MODEL):
        pytest.skip(f"{RERANK_MODEL} not loaded on {REMOTE}")
    return REMOTE


def test_protocol_basic_rerank(rerank_server):
    """Hard claim: a rerank with 4 candidates returns 4 sorted results."""
    client = RerankClient(rerank_server)
    out = client.rerank(
        query="what is muse?",
        documents=[
            "muse is an audio server",
            "muse is a generation server with multiple modalities",
            "purple cats are illegal in some places",
            "model serving with HTTP APIs",
        ],
        model=RERANK_MODEL,
    )
    assert "results" in out
    assert len(out["results"]) == 4
    scores = [r["relevance_score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True)
    indices = {r["index"] for r in out["results"]}
    assert indices == {0, 1, 2, 3}


def test_protocol_top_n_truncates(rerank_server):
    client = RerankClient(rerank_server)
    out = client.rerank(
        query="q",
        documents=["a", "b", "c", "d", "e"],
        top_n=2,
        model=RERANK_MODEL,
    )
    assert len(out["results"]) == 2


def test_protocol_return_documents_includes_text(rerank_server):
    client = RerankClient(rerank_server)
    out = client.rerank(
        query="q",
        documents=["alpha", "beta"],
        return_documents=True,
        model=RERANK_MODEL,
    )
    for row in out["results"]:
        assert "document" in row
        assert "text" in row["document"]


def test_protocol_return_documents_default_false(rerank_server):
    client = RerankClient(rerank_server)
    out = client.rerank(
        query="q",
        documents=["alpha", "beta"],
        model=RERANK_MODEL,
    )
    for row in out["results"]:
        assert "document" not in row


def test_protocol_envelope_meta_present(rerank_server):
    """Cohere SDK compatibility: envelope must include meta.billed_units."""
    client = RerankClient(rerank_server)
    out = client.rerank(
        query="q", documents=["a"], model=RERANK_MODEL,
    )
    assert "meta" in out
    assert "billed_units" in out["meta"]
    assert out["meta"]["billed_units"]["search_units"] == 1


def test_observe_relevance_score_above_zero_for_relevant_query(rerank_server):
    """Spot-check that the model returns sensible relevance scores."""
    client = RerankClient(rerank_server)
    out = client.rerank(
        query="capital of France",
        documents=[
            "Paris is the capital of France.",
            "Bananas are yellow.",
        ],
        model=RERANK_MODEL,
    )
    rows = sorted(out["results"], key=lambda r: r["relevance_score"], reverse=True)
    # Top-scored row should be the Paris document (index 0).
    assert rows[0]["index"] == 0
```

- [ ] **Step 4: Run tests**

Slow e2e:

```bash
pytest tests/cli_impl/test_e2e_rerank.py -v -m slow
```

Integration (skipped without env var):

```bash
pytest tests/integration/test_remote_rerank.py -v  # should skip
```

If a remote muse server with `bge-reranker-v2-m3` is reachable, set
`MUSE_REMOTE_SERVER` to verify the suite runs.

Full fast-lane:

```bash
pytest -m "not slow" -q 2>&1 | tail -3
pytest -q 2>&1 | tail -3  # full lane (slow included)
```

- [ ] **Step 5: Commit**

```bash
git add tests/cli_impl/test_e2e_rerank.py \
        tests/integration/conftest.py \
        tests/integration/test_remote_rerank.py
git commit -m "$(cat <<'EOF'
test(rerank): slow e2e + opt-in integration

Slow lane: in-process supervisor smoke for POST /v1/rerank with a
fake backend, asserting Cohere envelope shape.

Integration lane (opt-in via MUSE_REMOTE_SERVER): protocol-level
asserts (sorted scores, top_n truncation, return_documents flag,
meta.billed_units presence) plus an observation-style sanity check
that 'capital of France' ranks the Paris document above bananas.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task I: v0.19.0 release

Final wrap-up: docs, version bump, GitHub release notes, tag.

**Files:**
- Modify: `pyproject.toml` (version 0.18.3 to 0.19.0)
- Modify: `src/muse/__init__.py` (docstring; bundled-modalities list)
- Modify: `CLAUDE.md` (modality list + Cohere-compat note)
- Modify: `README.md` (modality list + curl example)

- [ ] **Step 1: Run full test suite**

```bash
pytest -m "not slow" -q 2>&1 | tail -3
pytest -q 2>&1 | tail -3
```

Both green before bumping.

- [ ] **Step 2: Bump version**

In `pyproject.toml`:

```toml
version = "0.19.0"
```

- [ ] **Step 3: Update src/muse/__init__.py**

Add `text/rerank` to the bundled-modalities list in the docstring;
update version reference.

- [ ] **Step 4: Update CLAUDE.md**

Add `text/rerank` to the modality list with a one-line description
of what's bundled. Add a note: "muse's first Cohere-compat modality;
the wire shape mirrors Cohere `/v1/rerank` (results[] with index +
relevance_score, optional document.text, meta.billed_units) so
downstream tooling like LangChain reranker integrations work
unchanged. OpenAI has no rerank API."

- [ ] **Step 5: Update README.md**

Add `/v1/rerank` to the route list at the top. Add a curl example:

```bash
# Rerank
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is muse?",
    "documents": [
      "muse is an audio server",
      "muse is a multi-modality generation server",
      "muse is the goddess of inspiration"
    ],
    "model": "bge-reranker-v2-m3",
    "top_n": 2,
    "return_documents": true
  }'
```

Add a brief subsection (or row in a "modality table") naming the
default reranker.

- [ ] **Step 6: Commit docs + version**

```bash
git add pyproject.toml src/muse/__init__.py CLAUDE.md README.md
git commit -m "$(cat <<'EOF'
chore(release): v0.19.0

text/rerank modality (Cohere-compat /v1/rerank). Bundled
bge-reranker-v2-m3 (BAAI/bge-reranker-v2-m3, Apache 2.0). Generic
CrossEncoderRuntime over sentence_transformers.CrossEncoder. HF
plugin sniffs cross-encoder rerankers at priority 115.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Tag and push**

```bash
git tag -a v0.19.0 -m "v0.19.0: text/rerank modality"
git push origin main
git push origin v0.19.0
```

- [ ] **Step 8: Create GitHub release**

```bash
gh release create v0.19.0 --title "v0.19.0: text/rerank modality" --notes "$(cat <<'EOF'
## What's new

muse's 8th modality: **text/rerank** (cross-encoder rerankers).

- New endpoint: `POST /v1/rerank` with Cohere-compat shape (`query`,
  `documents`, `top_n`, `return_documents`, `model`).
- Response envelope mirrors Cohere's: `results[]` with `index` +
  `relevance_score`, optional `document.text`, plus
  `meta.billed_units.search_units` for SDK compatibility.
- Bundled default: **bge-reranker-v2-m3** (BAAI/bge-reranker-v2-m3,
  Apache 2.0, multilingual, 8192-token context, ~568MB).
- Generic `CrossEncoderRuntime` over `sentence_transformers.CrossEncoder`
  serves any cross-encoder reranker.
- HF resolver plugin (priority 115) sniffs cross-encoder rerankers:
  `cross-encoder` tag, OR `text-classification` tag combined with
  `rerank` in the repo name. `muse pull hf://BAAI/bge-reranker-v2-m3`,
  `muse pull hf://mixedbread-ai/mxbai-rerank-large-v1`, etc.
- `RerankClient` HTTP wrapper.
- `muse models probe` works with rerank models (4-document smoke).

## Why Cohere shape, not OpenAI shape

OpenAI has no rerank API; Cohere's `/v1/rerank` is the de-facto
standard. Downstream tooling (LangChain, LlamaIndex, Haystack)
expects the Cohere envelope. A user with
`cohere.Client(api_key="x", base_url="http://localhost:8000")` can
call `.rerank(...)` against muse and get the right shape back.

## Quick start

```bash
muse pull bge-reranker-v2-m3
muse serve

curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is muse?",
    "documents": [
      "muse is an audio server",
      "muse is a multi-modality generation server",
      "muse is the goddess of inspiration"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

## Out of scope (filed for later)

Multi-modal rerank, ColBERT-style late-interaction rerankers,
learned-to-rank fine-tuning hooks, per-document score thresholds.
EOF
)"
```

- [ ] **Step 9: Verify release**

```bash
gh release view v0.19.0
```

Check that the release is published and the tag points at the right commit.

---

## Self-review checklist (after Task I)

- [ ] 7+ modalities now have HF plugin coverage (audio/speech remains
  bundled-only).
- [ ] `text/rerank` discovered by `discover_modalities`; appears in
  `/v1/models` after a worker loads a reranker.
- [ ] `POST /v1/rerank` works end-to-end with `bge-reranker-v2-m3`.
- [ ] Curated entry parses correctly.
- [ ] Cohere SDK can call `.rerank()` against muse and get the right
  shape (verified manually if a Cohere-compatible client is handy).
- [ ] Plugin priority correctly set so text-classification doesn't
  shadow text-rerank for cross-encoder repos.
- [ ] All tests pass (fast lane and slow lane).
- [ ] v0.19.0 tagged and pushed; GitHub release published.
- [ ] No em-dashes anywhere in new files.
