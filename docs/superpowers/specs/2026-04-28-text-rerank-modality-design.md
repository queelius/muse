# `text/rerank` modality (cross-encoder rerankers)

**Date:** 2026-04-28
**Status:** approved
**Target release:** v0.19.0

## Goal

Add muse's 8th modality: `text/rerank`, mounted at the Cohere-compat URL
`POST /v1/rerank`. One generic `CrossEncoderRuntime` over the
`sentence_transformers.CrossEncoder` API serves any cross-encoder
reranker on HuggingFace. One curated default (BAAI/bge-reranker-v2-m3).
HF resolver sniffs cross-encoder reranker repos with priority 115 so
they win over the broad `text-classification` plugin (200) without
disturbing the more specific embedding plugin (110).

This is muse's first modality with a Cohere-compat wire shape rather
than OpenAI-compat. The Cohere `/v1/rerank` API is the de-facto
standard for the reranker space; the response envelope (`results[]`
with `index` + `relevance_score`, optional `document.text`,
`meta.billed_units.search_units`) is what every Cohere SDK and
LangChain/LlamaIndex reranker integration expects.

## Scope

**In v1:**
- `POST /v1/rerank` with Cohere-shape JSON request and response.
- Cross-encoder rerankers via `sentence_transformers.CrossEncoder`.
- `query: str`, `documents: list[str]`, optional `top_n`,
  `return_documents`, `model`.
- Sort: results sorted by `relevance_score` descending; up to `top_n`
  returned (or all if `top_n is None`).
- `return_documents=False` (default): result rows omit `document`.
- `return_documents=True`: result rows include `{"document": {"text": "..."}}`.
- Generic `CrossEncoderRuntime` over `CrossEncoder.predict(list[(query, doc)])`.
- HF resolver fifth sniff branch for cross-encoder rerankers
  (priority 115; specific to rerank repos).
- Search routes `--modality text/rerank` to a hybrid `list_models` query
  (cross-encoder filter combined with `rerank` term in repo name).
- One curated entry: `bge-reranker-v2-m3` (bundled-script alias to
  `BAAI/bge-reranker-v2-m3`).
- One bundled script: `src/muse/models/bge_reranker_v2_m3.py`.
- `RerankClient` parallel to other muse clients; minimal HTTP wrapper.
- `PROBE_DEFAULTS` so `muse models probe <id>` exercises a 4-document
  rerank with a sample query.

**Not in v1 (deferred):**
- Multi-modal rerank (image-text rerank). Different route, different
  modality. Future task.
- ColBERT-style late-interaction rerankers (per-token contextual
  scoring, multi-vector encoding). Different runtime shape; would need
  a `LateInteractionRuntime` or `ColBERTRuntime`. Future task.
- Learned-to-rank fine-tuning hooks (LoRA on rerankers, custom loss
  functions). Out of scope for serving.
- Per-document score thresholds in the request. Cohere's API only
  supports `top_n`; we honor that and return whatever the model gives.
- Streaming responses. Reranking is one forward pass; nothing to stream.
- Bi-encoder reranker fallback (using embedding cosine sim). Bi-encoders
  belong in `embedding/text`; users compose them client-side.

## Why generic runtime, not bundled-script-only

Matches muse's trajectory (sentence-transformers, llama-cpp,
faster-whisper, transformers AutoModelForSequenceClassification,
diffusers AutoPipeline). One runtime serves any cross-encoder; curated
entries pin the recommended specific. Adding `mxbai-rerank-large-v1`
or `jina-reranker-v2` later is a curated.yaml edit (or a `muse pull
hf://...`), not a new Python script.

## Why Cohere shape (and Cohere-only) at /v1/rerank

Cohere's `/v1/rerank` is the wire contract that downstream tools
(LangChain, LlamaIndex, Haystack) hit. OpenAI has no rerank API; there
is no competing standard to reconcile. A user with
`cohere.Client(api_key="x", base_url="http://localhost:8000")` should
be able to call `.rerank(...)` and get the right response shape. We
make Cohere SDK compatibility a first-class goal:

- Field names: `query`, `documents`, `top_n`, `return_documents`,
  `model`.
- Response: `id`, `model`, `results: [{index, relevance_score, document?}]`,
  `meta: {billed_units: {search_units: 1}}`.
- `meta.billed_units.search_units` is a Cohere artifact (price unit);
  muse always reports `1` for compatibility. No real billing.

The MIME-tag (`text/rerank`) is broad enough to host future routes
sharing the same runtime + dataclasses (e.g., a more muse-native
`/v1/text/rerank` that doesn't require `meta.billed_units`) without
needing a second modality package. Same precedent set by
`text/classification` at `/v1/moderations`.

## Package layout

```
src/muse/modalities/text_rerank/
|-- __init__.py          # MODALITY = "text/rerank" + build_router + exports + PROBE_DEFAULTS
|-- protocol.py          # RerankerModel Protocol + RerankResult dataclass
|-- routes.py            # build_router; mounts POST /v1/rerank
|-- codec.py             # RerankResult list + return_documents -> Cohere envelope
|-- client.py            # RerankClient
|-- hf.py                # HF_PLUGIN sniffing cross-encoder rerankers
`-- runtimes/
    |-- __init__.py
    `-- cross_encoder.py  # CrossEncoderRuntime generic runtime
```

Bundled script:

```
src/muse/models/
`-- bge_reranker_v2_m3.py   # BAAI/bge-reranker-v2-m3 curated default
```

## Protocol

```python
@dataclass
class RerankResult:
    """One scored (query, document) pair from a rerank call.

    index: position of this document in the request's `documents` array.
           Stable across the rerank call so a client can map back.
    relevance_score: float in [0, 1] (or roughly so; cross-encoders
           often emit logits that the runtime sigmoid-normalizes).
    document_text: the original document string. The codec uses this
           when `return_documents=True` and drops it otherwise. Held
           on the dataclass (rather than re-indexing the request) so
           the runtime is self-contained and the codec stays pure.
    """
    index: int
    relevance_score: float
    document_text: str


@runtime_checkable
class RerankerModel(Protocol):
    """Structural protocol any reranker backend satisfies."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score query against each document; return sorted descending.

        When `top_n` is None, returns all documents. When set, returns
        the top-N by score. Order of equal-score ties is unspecified
        (cross-encoder scores are continuous; ties are unlikely).
        """
        ...
```

## Wire contract

**Request** (`POST /v1/rerank`, `application/json`):

| Field | Type | Required | Validation | Notes |
|---|---|---|---|---|
| `query` | `str` | yes | `1 <= len <= 4000` | The search query |
| `documents` | `list[str]` | yes | `1 <= len <= 1000`; each non-empty | Pool of candidates |
| `top_n` | `int | None` | no | `1 <= top_n <= 1000` when set | Defaults to all |
| `model` | `str | None` | no | catalog id | Defaults to first registered under `text/rerank` |
| `return_documents` | `bool` | no | default `False` | Include doc text in response |

**Response** (`application/json`, Cohere shape):

```json
{
  "id": "rrk-<24-hex>",
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 3, "relevance_score": 0.97, "document": {"text": "..."}},
    {"index": 0, "relevance_score": 0.81, "document": {"text": "..."}}
  ],
  "meta": {
    "billed_units": {"search_units": 1}
  }
}
```

When `return_documents=False`: `document` field omitted from each row.
When `top_n=None`: all documents returned, sorted descending by score.

**Error envelopes** (OpenAI-shape, used by all muse modalities):

- 400 `invalid_parameter`: `query` empty; `documents` empty;
  any document empty; `top_n` out of [1, 1000]; `documents` too long.
- 404 `model_not_found`: `model` unknown (raised via `ModelNotFoundError`).

## Runtime: CrossEncoderRuntime

`src/muse/modalities/text_rerank/runtimes/cross_encoder.py:CrossEncoderRuntime`:

```python
class CrossEncoderRuntime:
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
                "sentence-transformers is not installed; run "
                "`muse pull` or install `sentence-transformers`"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        src = local_dir or hf_repo
        self._model = CrossEncoder(src, max_length=max_length, device=self._device)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []
        pairs = [(query, d) for d in documents]
        scores = self._model.predict(pairs)
        scored = sorted(
            [(i, float(s)) for i, s in enumerate(scores)],
            key=lambda kv: kv[1], reverse=True,
        )
        if top_n is not None:
            scored = scored[:top_n]
        return [
            RerankResult(index=i, relevance_score=s, document_text=documents[i])
            for i, s in scored
        ]
```

Key points:

- Lazy-import sentence_transformers + torch (sentinel pattern shared by
  embedding_text and text_classification runtimes).
- Honors `device="auto"` via the same `_select_device` shape as siblings.
- `max_length` is a manifest capability (default 512). Cross-encoders
  with 8K context (e.g., bge-reranker-v2-m3) can override per model.
- `CrossEncoder.predict` returns numpy array; runtime casts to float.
- The runtime never raises sentinel `RerankResult` rows; the route
  layer is responsible for validating the request body.

## RerankResult dataclass

```python
@dataclass
class RerankResult:
    index: int
    relevance_score: float
    document_text: str
```

`document_text` is held on the result so the codec is pure: it
receives a list of `RerankResult` and a `return_documents: bool` and
emits the Cohere envelope without consulting the original request.

## Codec

```python
def encode_rerank_response(
    results: list[RerankResult],
    *,
    model_id: str,
    return_documents: bool,
) -> dict[str, Any]:
    """Build the Cohere-shape rerank response.

    Returns:
      {
        "id": "rrk-<24-hex>",
        "model": model_id,
        "results": [
          {"index": int, "relevance_score": float, "document"?: {"text": str}}
        ],
        "meta": {"billed_units": {"search_units": 1}}
      }
    """
    rows = []
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

## HF resolver plugin

`src/muse/modalities/text_rerank/hf.py`:

Sniff: any HF repo with the `cross-encoder` tag, OR a repo whose
`text-classification` tag co-occurs with `rerank` in the repo name
(case-insensitive). The latter pattern catches `BAAI/bge-reranker-v2-m3`
and similar repos that ship under the text-classification tag without
the dedicated cross-encoder tag.

```python
def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "cross-encoder" in tags:
        return True
    if "text-classification" in tags:
        repo_id = (getattr(info, "id", "") or "").lower()
        return "rerank" in repo_id
    return False
```

Priority **115** (more specific than `embedding/text`'s 110, less
specific than file-pattern plugins at 100). Beats the
`text/classification` plugin's catch-all 200 because reranker repos
normally also carry `text-classification`.

Capability defaults: `max_length=512` for most models. Repo-name
heuristic for known long-context rerankers:
- `bge-reranker-v2-m3`: max_length=8192
- `mxbai-rerank-large-v1`: max_length=512
- `jina-reranker-v2`: max_length=1024
- fallback: max_length=512

`HFResolver.search` adds a branch for `modality == "text/rerank"`:
search HF for the query string, filter by `cross-encoder` tag (with
fallback to `rerank` substring in name when the tag filter is empty).

## Bundled script: bge_reranker_v2_m3

`src/muse/models/bge_reranker_v2_m3.py`:

```python
MANIFEST = {
    "model_id": "bge-reranker-v2-m3",
    "modality": "text/rerank",
    "hf_repo": "BAAI/bge-reranker-v2-m3",
    "description": "BAAI bge-reranker-v2-m3: multilingual cross-encoder, ~568MB",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "sentence-transformers>=2.2.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "max_length": 8192,
        "memory_gb": 1.2,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "tokenizer*", "spiece.model",
    ],
}
```

The `Model` class wraps `CrossEncoder` directly (rather than going
through `CrossEncoderRuntime`) so the script demonstrates the
same shape muse uses for other bundled models. Lazy imports.

## Curated entry

```yaml
- id: bge-reranker-v2-m3
  bundled: true
```

Curated entries that alias bundled scripts use `bundled: true` and
inherit MANIFEST data from the script.

## PROBE_DEFAULTS

```python
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
```

Used by `muse models probe <id>` so a power user can verify a fresh
pull works end-to-end without opening a Python REPL.

## Test strategy

Unit-heavy. Mocks for `sentence_transformers.CrossEncoder.predict`.
One slow e2e test exercises FastAPI + codec + mocked runtime. One
opt-in integration test against a live muse server with a real
reranker loaded.

Coverage targets:

- Protocol + dataclass shape (5 tests).
- Codec: envelope shape, `return_documents` toggle, `top_n` truncation,
  ordering.
- Routes: 200 happy path, 400 envelope for bad input
  (empty query/documents, out-of-range top_n), 404 for unknown model,
  `return_documents` flag plumbed.
- Runtime: deferred imports, predict path called with correct pairs,
  sort + slice correctness, device auto-select.
- HF plugin: positive sniffs (cross-encoder tag, text-classification +
  rerank in name), negative (text-classification only without rerank
  in name), priority correctness, max_length heuristic, search branch.
- Curated: bge-reranker-v2-m3 entry parses as bundled.
- Bundled script: MANIFEST shape, Model construction with lazy imports
  patched.
- E2E slow: full JSON-in / JSON-out round-trip through the supervisor.
- Integration opt-in: real server + real reranker. `MUSE_RERANK_MODEL_ID`
  env override (default `bge-reranker-v2-m3`).

## Documentation

- CLAUDE.md: add `text/rerank` to modality list; note that this
  modality is Cohere-compat rather than OpenAI-compat.
- README.md: modality list + `/v1/rerank` endpoint + curl example.

## Release

v0.19.0. Minor bump (new feature, no breaking changes). Tag message
calls out: new modality, new endpoint, new curated default, Cohere-SDK
compatibility (sets precedent for non-OpenAI wire shapes when an
established alternative exists).

## Out of scope

- Multi-modal rerank (image-text rerank).
- ColBERT-style late-interaction rerankers.
- Learned-to-rank fine-tuning hooks.
- Per-document score thresholds.
- Streaming.
- Bi-encoder reranker fallback (those belong in `embedding/text`).
