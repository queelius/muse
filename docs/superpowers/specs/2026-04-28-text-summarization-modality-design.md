# `text/summarization` modality (BART / PEGASUS / seq2seq summarizers)

**Date:** 2026-04-28
**Status:** approved
**Target release:** v0.22.0

## Goal

Add muse's 10th modality: `text/summarization`, mounted at the
Cohere-compat URL `POST /v1/summarize`. One generic
`BartSeq2SeqRuntime` over `transformers.AutoModelForSeq2SeqLM` serves
any BART/PEGASUS-shape summarizer on HuggingFace. One bundled default
(`facebook/bart-large-cnn`) plus one curated dialog-tuned addition
(`philschmid/bart-large-cnn-samsum`). HF resolver sniffs HF repos
tagged `summarization` with priority **110** so they resolve to the
summarization runtime ahead of any catch-all classification plugin.

This is muse's second modality with a Cohere-compat wire shape (after
`text/rerank` at `/v1/rerank`). Cohere's `/v1/summarize` API was the
de-facto reference for the summarization request/response shape until
its 2024 deprecation; the envelope (`{id, model, summary, usage}`) is
the closest thing the industry has to a stable summarization wire
contract, so we adopt it and let Cohere SDK clients (and any tooling
written against the Cohere shape) work against muse with no code
changes.

## Scope

**In v1:**
- `POST /v1/summarize` with Cohere-shape JSON request and response.
- BART/PEGASUS-shape summarizers via
  `transformers.AutoModelForSeq2SeqLM`.
- `text: str` (1 to 100,000 chars), `length: "short"|"medium"|"long"`,
  `format: "paragraph"|"bullets"`, optional `model`.
- Length maps deterministically to `max_new_tokens`:
  short=80, medium=180, long=400.
- Format is recorded in `meta.format` for client introspection. For
  BART (which doesn't take instructions), format does NOT affect
  generation; the modality documents this honestly. Future
  instruction-tuned summarizers can consult `format` in their runtime.
- Generic `BartSeq2SeqRuntime` over `AutoModelForSeq2SeqLM.generate()`.
- Usage stats (prompt_tokens, completion_tokens, total_tokens) computed
  from the tokenizer.
- HF resolver sixth sniff branch for `summarization`-tagged repos
  (priority 110).
- Search routes `--modality text/summarization` to a hybrid
  `list_models` query (summarization tag combined with the user's
  query).
- Two curated entries: `bart-large-cnn` (bundled-script alias to
  `facebook/bart-large-cnn`) and `bart-cnn-samsum`
  (`hf://philschmid/bart-large-cnn-samsum`, dialog-tuned).
- One bundled script: `src/muse/models/bart_large_cnn.py`.
- `SummarizationClient` parallel to other muse clients; minimal HTTP
  wrapper.
- `PROBE_DEFAULTS` so `muse models probe <id>` exercises a
  ~200-word-input, medium-length summarization.

**Not in v1 (deferred):**
- Streaming summarization. BART/PEGASUS are seq2seq; partial decoding
  is technically possible but the wire contract and most client
  expectations are non-streaming. Could land later if a streaming
  summarizer becomes worth supporting.
- Beam search hyperparameters in the request (num_beams, length_penalty,
  no_repeat_ngram_size). Curated capabilities can override defaults
  per-model; per-request control is out of scope for v1.
- Multi-document summarization (concatenated input is fine; explicit
  `documents: list[str]` API is not v1).
- Bullet generation that actually produces bullets. BART-CNN summaries
  are paragraphs; we record `format` in the response metadata but the
  caller gets paragraph-shaped output. A future instruction-tuned
  summarizer (FLAN-T5, Llama-summarize) could honor `format`.
- T5/PEGASUS as separate runtimes. T5 is `AutoModelForSeq2SeqLM`-shaped
  and works under this runtime; PEGASUS is too. We pick BART as the
  default because of its size/quality tradeoff for CPU users.
- Long-context summarization (Longformer, LED). BART caps inputs at
  ~1024 tokens; longer inputs get truncated honestly with a
  `truncation_warning` field set in `meta`.
- Chat-style summarizers (Qwen-summarize). Those route through
  `chat/completion` with a system prompt; not this modality's concern.

## Why generic runtime, not bundled-script-only

Matches muse's trajectory (sentence-transformers, llama-cpp,
faster-whisper, transformers AutoModelForSequenceClassification,
diffusers AutoPipeline, sentence-transformers CrossEncoder). One
runtime serves any seq2seq summarizer; curated entries pin the
recommended specific. Adding `philschmid/bart-large-xsum-samsum` or
`google/pegasus-xsum` later is a curated.yaml edit (or a `muse pull
hf://...`), not a new Python script.

## Why Cohere shape (and Cohere-only) at /v1/summarize

Cohere's `/v1/summarize` was the closest thing to a stable wire
contract for summarization. OpenAI has no summarization API. Cohere
deprecated theirs in 2024 but the shape is well-known and was widely
mimicked in client tooling. We make Cohere SDK compatibility a
first-class goal:

- Field names: `text`, `length`, `format`, `model`.
- Length values: `short`, `medium`, `long`.
- Format values: `paragraph`, `bullets`.
- Response: `id`, `model`, `summary`, `usage` (token counts), `meta`
  (echoed length/format).

The MIME-tag (`text/summarization`) is broad enough to host future
routes sharing the same runtime + dataclasses (e.g., a more muse-native
`/v1/text/summarize` that exposes beam search params) without needing
a second modality package. Same precedent set by `text/rerank` at
`/v1/rerank` and `text/classification` at `/v1/moderations`.

## Package layout

```
src/muse/modalities/text_summarization/
|-- __init__.py          # MODALITY = "text/summarization" + build_router + exports + PROBE_DEFAULTS
|-- protocol.py          # SummarizationModel Protocol + SummarizationResult dataclass
|-- routes.py            # build_router; mounts POST /v1/summarize
|-- codec.py             # SummarizationResult + length/format -> Cohere envelope
|-- client.py            # SummarizationClient
|-- hf.py                # HF_PLUGIN sniffing summarization-tagged repos
`-- runtimes/
    |-- __init__.py
    `-- bart_seq2seq.py  # BartSeq2SeqRuntime generic runtime
```

Bundled script:

```
src/muse/models/
`-- bart_large_cnn.py   # facebook/bart-large-cnn curated default
```

## Protocol

```python
@dataclass
class SummarizationResult:
    """One summary produced by a `text/summarization` model.

    summary: the produced summary text (paragraph or bullet shape;
             see format).
    length: the length budget used ("short"|"medium"|"long"). Echoed
            so a client can confirm the runtime honored its request.
    format: the format requested ("paragraph"|"bullets"). For BART-CNN
            and similar non-instruction models this is metadata only;
            the produced summary is whatever the model gave.
    model_id: catalog id of the model that produced this summary.
    prompt_tokens: input token count (post-truncation when applicable).
    completion_tokens: output token count.
    metadata: optional per-call extras the runtime wants surfaced
              (e.g., truncation_warning, language detected).
    """
    summary: str
    length: str
    format: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class SummarizationModel(Protocol):
    """Structural protocol any summarizer backend satisfies."""

    def summarize(
        self,
        text: str,
        length: str = "medium",
        format: str = "paragraph",
    ) -> SummarizationResult:
        """Produce a summary for `text`.

        length controls max_new_tokens; format is metadata for non-
        instruction models, instructional for instruction-tuned ones.
        """
        ...
```

## Wire contract

**Request** (`POST /v1/summarize`, `application/json`):

| Field | Type | Required | Validation | Notes |
|---|---|---|---|---|
| `text` | `str` | yes | `1 <= len <= 100000` | The input to summarize |
| `length` | `str` | no | one of "short", "medium", "long" | default "medium" |
| `format` | `str` | no | one of "paragraph", "bullets" | default "paragraph" |
| `model` | `str | None` | no | catalog id | Defaults to first registered under `text/summarization` |

**Response** (`application/json`, Cohere shape):

```json
{
  "id": "sum-<24-hex>",
  "model": "bart-large-cnn",
  "summary": "muse is a multi-modality generation server...",
  "usage": {
    "prompt_tokens": 412,
    "completion_tokens": 67,
    "total_tokens": 479
  },
  "meta": {
    "length": "medium",
    "format": "paragraph"
  }
}
```

**Error envelopes** (OpenAI-shape, used by all muse modalities):

- 400 `invalid_parameter`: `text` empty; `text` too long;
  `length` not in valid set; `format` not in valid set.
- 404 `model_not_found`: `model` unknown (raised via `ModelNotFoundError`).

## Runtime: BartSeq2SeqRuntime

`src/muse/modalities/text_summarization/runtimes/bart_seq2seq.py:BartSeq2SeqRuntime`:

```python
class BartSeq2SeqRuntime:
    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        default_length: str = "medium",
        default_format: str = "paragraph",
        max_input_tokens: int = 1024,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` "
                "or install `transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = dtype
        self._default_length = default_length
        self._default_format = default_format
        self._max_input_tokens = max_input_tokens

        src = local_dir or hf_repo
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            src, torch_dtype=_resolve_dtype(dtype),
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def summarize(self, text, length=None, format=None) -> SummarizationResult:
        length = length or self._default_length
        fmt = format or self._default_format
        max_new_tokens = _LENGTH_TO_MAX_TOKENS[length]
        # tokenize with truncation
        # call self._model.generate(input_ids, max_new_tokens=...)
        # decode
        # tokenize the output to count completion tokens
        # build SummarizationResult
```

Key points:

- Lazy-import transformers + torch (sentinel pattern shared by all
  other muse runtimes).
- Honors `device="auto"` via the same `_select_device` shape as
  siblings.
- `max_input_tokens` defaults to 1024 (BART's hard limit). Tokenized
  inputs longer than this get truncated; metadata records
  `truncation_warning: True`.
- `length` maps to `max_new_tokens` via a frozen lookup:
  `{"short": 80, "medium": 180, "long": 400}`. The codec does not
  re-derive these; the runtime is the single source of truth.
- `format` is recorded but does not affect BART generation. Documented
  in the README + CLAUDE.md.
- The runtime emits one `SummarizationResult` per call; the route layer
  wraps it in the Cohere envelope.

## SummarizationResult dataclass

```python
@dataclass
class SummarizationResult:
    summary: str
    length: str
    format: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    metadata: dict = field(default_factory=dict)
```

`prompt_tokens` and `completion_tokens` are computed by the runtime
using the model's own tokenizer; the codec just sums them for
`total_tokens` and projects onto the Cohere envelope.

## Codec

```python
def encode_summarization_response(
    result: SummarizationResult,
) -> dict[str, Any]:
    """Build the Cohere-shape summarize response.

    Returns:
      {
        "id": "sum-<24-hex>",
        "model": model_id,
        "summary": str,
        "usage": {prompt_tokens, completion_tokens, total_tokens},
        "meta": {length, format, **runtime metadata}
      }
    """
    meta = dict(result.metadata)
    meta.setdefault("length", result.length)
    meta.setdefault("format", result.format)
    return {
        "id": f"sum-{uuid.uuid4().hex[:24]}",
        "model": result.model_id,
        "summary": result.summary,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "meta": meta,
    }
```

## HF resolver plugin

`src/muse/modalities/text_summarization/hf.py`:

Sniff: any HF repo with the `summarization` task tag.

```python
def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "summarization" in tags
```

Priority **110** (matches embedding/text). The `summarization` tag is
specific enough that no other plugin should be picking it up; we don't
need to be more specific than the embedding tier. Higher than the
catch-all `text-classification` (200) but loses to file-pattern plugins
at 100 (GGUF, faster-whisper, diffusers).

The plugin only handles BART/PEGASUS-shape repos via
`AutoModelForSeq2SeqLM`. Other architectures (chat-completion-style
summarizers like Qwen/Llama-summarize) fall outside; they'd be routed
via the `chat/completion` modality.

Capability defaults: `default_length="medium"`,
`default_format="paragraph"`, `memory_gb=1.5` (BART-large class),
`device="auto"`. Repo-name heuristic for dialog-tuned models:
`supports_dialog_summarization=True` if the repo name contains
`samsum`, `dialog`, `chat`, or `meeting` (case-insensitive). Otherwise
`False`.

## Bundled script: bart_large_cnn

`src/muse/models/bart_large_cnn.py`:

```python
MANIFEST = {
    "model_id": "bart-large-cnn",
    "modality": "text/summarization",
    "hf_repo": "facebook/bart-large-cnn",
    "description": "BART large CNN: news summarization, ~400MB, CPU-friendly, Apache 2.0",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cpu",
        "default_length": "medium",
        "default_format": "paragraph",
        "supports_dialog_summarization": False,
        "memory_gb": 1.5,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "merges.txt", "vocab.json", "tokenizer*",
    ],
}
```

The `Model` class wraps `transformers.AutoModelForSeq2SeqLM` directly
(rather than going through `BartSeq2SeqRuntime`) so the script
demonstrates the same shape muse uses for other bundled models. Lazy
imports.

## Curated entries

```yaml
- id: bart-large-cnn
  bundled: true

- id: bart-cnn-samsum
  uri: hf://philschmid/bart-large-cnn-samsum
  modality: text/summarization
  size_gb: 0.4
  description: "BART CNN samsum: dialog summarization (meetings, chats), ~400MB"
  capabilities:
    supports_dialog_summarization: true
    device: cpu
    memory_gb: 1.5
```

## PROBE_DEFAULTS

```python
PROBE_DEFAULTS = {
    "shape": "200 word input, medium length output",
    "call": lambda m: m.summarize(
        "muse is a model-agnostic multi-modality generation server. It hosts text, "
        "image, audio, and video models behind a unified HTTP API that mirrors OpenAI "
        "where possible. Each modality is a self-contained plugin: it declares its MIME "
        "tag, contributes a build_router function, and the discovery layer wires it in. "
        "Models are pulled into per-model venvs so that conflicting dependencies between "
        "different model families never break each other.",
        length="medium",
    ),
}
```

Used by `muse models probe <id>` so a power user can verify a fresh
pull works end-to-end without opening a Python REPL.

## Test strategy

Unit-heavy. Mocks for `transformers.AutoModelForSeq2SeqLM` +
`AutoTokenizer`. One slow e2e test exercises FastAPI + codec + mocked
runtime. One opt-in integration test against a live muse server with a
real summarizer loaded.

Coverage targets:

- Protocol + dataclass shape (5 tests).
- Codec: envelope shape, usage rollup, meta echoes length/format,
  metadata pass-through, deterministic given fixed inputs.
- Routes: 200 happy path, 400 envelope for bad input
  (empty text, too-long text, invalid length/format), 404 for
  unknown model, default model resolution.
- Runtime: deferred imports, generate path called with the right
  `max_new_tokens`, sort + slice correctness for length mapping,
  device auto-select, truncation warning for inputs >max_input_tokens.
- HF plugin: positive sniff (summarization tag), negative (no tag),
  priority correctness, dialog heuristic for samsum/dialog/chat/meeting,
  search branch.
- Curated: `bart-large-cnn` parses as bundled; `bart-cnn-samsum`
  parses as URI with capabilities overlay.
- Bundled script: MANIFEST shape, Model construction with lazy imports
  patched, summarize() returns SummarizationResult with the right
  length/format echoed.
- E2E slow: full JSON-in / JSON-out round-trip through the supervisor.
- Integration opt-in: real server + real summarizer.
  `MUSE_SUMMARIZATION_MODEL_ID` env override (default
  `bart-large-cnn`).

## Documentation

- CLAUDE.md: add `text/summarization` to modality list; note that this
  modality is Cohere-compat and that `format` is metadata-only for
  BART. Bump count from 9 modalities to 10.
- README.md: modality list + `/v1/summarize` endpoint + curl example.
- src/muse/__init__.py docstring: bump version to 0.22.0; add
  `text/summarization` to bundled modalities list.

## Release

v0.22.0. Minor bump (new feature, no breaking changes). Tag message
calls out: new modality, new endpoint, new bundled default, Cohere-SDK
compatibility (consistent with v0.19.0's text/rerank precedent for
non-OpenAI wire shapes when an established alternative exists).

## Out of scope

- Streaming summarization.
- Beam search hyperparameters per request.
- Multi-document summarization with explicit `documents: list[str]`.
- Format-respecting bullet generation (needs an instruction-tuned
  summarizer; future task).
- Long-context summarization (Longformer, LED).
- Chat-style summarizers (route through `chat/completion`).
