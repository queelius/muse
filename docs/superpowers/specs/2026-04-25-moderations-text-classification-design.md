# /v1/moderations modality (text classification): design

**Date:** 2026-04-25
**Status:** approved
**Target release:** v0.14.0

## Goal

Add muse's 6th modality: `text/classification`, mounted at the
OpenAI-compat URL `/v1/moderations`. One generic `HFTextClassifier`
runtime serves any HuggingFace text-classification model; one curated
default (KoalaAI/Text-Moderation). HF resolver sniffs the
`text-classification` tag; search routes the modality to
`list_models(filter="text-classification")`.

This is muse's first modality whose modality tag (`text/classification`)
deliberately doesn't match its primary HTTP route (`/v1/moderations`),
so the package directory and the wire path can evolve independently.

## Scope

**In v1:**
- `POST /v1/moderations` with OpenAI-shape JSON request and response.
- `input: str | list[str]` (scalar or batch).
- Optional `threshold: float` request field (muse extension; overrides
  MANIFEST capability + 0.5 default; out-of-range returns 400).
- Response shape: `id`, `model`, `results[]` with `flagged`,
  `categories`, `category_scores`.
- Category labels come from the model's native `id2label` config
  (honest passthrough; no lossy remapping to OpenAI's fixed schema).
- Generic `HFTextClassifier` runtime over
  `transformers.AutoModelForSequenceClassification` + tokenizer.
- Multi-label (sigmoid) and single-label (softmax) handling, switched
  via `model.config.problem_type`.
- Threshold precedence: request > MANIFEST.capabilities.flag_threshold
  > 0.5 default.
- HF resolver fourth sniff branch for `text-classification` tag.
- Search routes `--modality text/classification` (the MIME tag) to
  HF `list_models(filter="text-classification")`.
- One curated entry: `text-moderation` -> `hf://KoalaAI/Text-Moderation`.
- `ModerationsClient` parallel to other muse clients.

**Not in v1 (deferred):**
- Image moderation / OpenAI omni-moderation shape.
- Per-category thresholds in the request body (manifest can declare
  per-category, but request is scalar only for v1).
- Stable id2label-to-OpenAI-keys remapping.
- Other text-classification use cases (sentiment, intent, language ID)
  ride the same runtime under `text/classification` but at different
  wire paths; future tasks (#101).

## Why generic runtime, not bundled script

Matches muse's trajectory (sentence-transformers, llama-cpp,
faster-whisper). One runtime serves any model in its class; curated
entries pin recommended specifics. Adding a second moderation model
(or a sentiment classifier) is a curated.yaml edit, not a Python
script.

## Why text/classification (broad MIME) at /v1/moderations (narrow URL)

OpenAI's `/v1/moderations` is the wire contract; that's what their
SDKs hit. Internally, moderation is a special case of text
classification: same model architecture, same runtime, same response
shape, only the labels differ. Naming the package and modality
`text/classification` (broad) lets us mount additional URL routes
later (e.g., `/v1/text/classifications` for non-OpenAI generic
sentiment / intent uses) without a second modality package or runtime
duplication. The route layer in `routes.py` is responsible for the
URL-to-modality mapping.

## Package layout

```
src/muse/modalities/text_classification/
|-- __init__.py          # MODALITY = "text/classification" + build_router
|-- protocol.py          # TextClassifierModel Protocol + ClassificationResult
|-- routes.py            # build_router; mounts POST /v1/moderations
|-- codec.py             # ClassificationResult + threshold -> OpenAI envelope
|-- client.py            # ModerationsClient
`-- runtimes/
    |-- __init__.py
    `-- hf_text_classifier.py   # HFTextClassifier generic runtime
```

## Protocol

```python
@dataclass
class ClassificationResult:
    """One input's classification.

    scores: dict from the model's id2label space to confidence in [0, 1].
    multi_label: True if scores are independent (sigmoid).
                 False if mutually exclusive (softmax sums to 1.0).
    """
    scores: dict[str, float]
    multi_label: bool


class TextClassifierModel(Protocol):
    def classify(self, input: str | list[str]) -> list[ClassificationResult]:
        """Return one ClassificationResult per input (always a list,
        even for a scalar input)."""
        ...
```

## Wire contract

**Request** (`POST /v1/moderations`, `application/json`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `input` | `str` or `list[str]` | yes | Single string or batch |
| `model` | `str` | no | Catalog id; defaults to first registered under `text/classification` |
| `threshold` | `float` | no | Muse extension. Overrides MANIFEST + 0.5 default. Must be in [0, 1]. |

**Response** (`application/json`):

```json
{
  "id": "modr-<uuid4>",
  "model": "text-moderation",
  "results": [
    {
      "flagged": true,
      "categories": {"H": false, "V": true, ...},
      "category_scores": {"H": 0.012, "V": 0.873, ...}
    }
  ]
}
```

`results` is always a list, ordered to match `input`. `categories`
booleans are derived by comparing `category_scores` against the
resolved threshold.

**Error envelopes** (OpenAI-shape):
- 400 `invalid_parameter`: `input` missing or empty; `threshold`
  out of [0, 1].
- 404 `model_not_found`: `model` unknown.

## Multi-label vs. single-label handling

The runtime detects via `model.config.problem_type`:
- `"multi_label_classification"`: sigmoid per logit. Each score
  independent in [0, 1]. `flagged` = `any(score >= threshold)`.
- `"single_label_classification"` (or unset): softmax. Scores sum to
  1.0. `flagged` = the argmax score >= threshold.

KoalaAI/Text-Moderation is single-label (the model picks one of nine
labels including OK). unitary/toxic-bert is multi-label (each toxicity
type evaluated independently). Same runtime serves both.

## Threshold resolution

```python
def _resolve_threshold(request_threshold, manifest_capabilities):
    if request_threshold is not None:
        return float(request_threshold)
    cap = manifest_capabilities.get("flag_threshold")
    if isinstance(cap, (int, float)):
        return float(cap)
    return 0.5
```

Layer 1 (per-request): scalar only in v1; applies to all categories.
Layer 2 (per-model): scalar in MANIFEST.capabilities.flag_threshold,
declared by curated entry or HF synthesizer. Per-category dict in
manifest is a future extension; v1 keeps it scalar.
Layer 3 (default): 0.5.

## HF resolver extension

```python
def _looks_like_text_classifier(siblings, tags):
    return "text-classification" in tags
```

`_sniff_repo_shape` gains a fourth branch (after gguf,
sentence-transformers, faster-whisper) returning
`"text-classification"`. `_resolve_text_classifier(repo_id, info)`
synthesizes a manifest with:
- `model_id`: repo-name slug, lowercase
- `modality`: `"text/classification"`
- `pip_extras`: `("transformers>=4.36.0", "torch>=2.1.0")`
- `system_packages`: `()`
- `capabilities`: `{}` (curated overlay or manifest defaults fill in
  flag_threshold)
- `backend_path`:
  `"muse.modalities.text_classification.runtimes.hf_text_classifier:HFTextClassifier"`
- `download`: `snapshot_download(repo_id, ...)`

`HFResolver.search` adds a branch for
`modality == "text/classification"` -> `list_models(filter="text-classification")`.

## Curated entry

```yaml
- id: text-moderation
  uri: hf://KoalaAI/Text-Moderation
  modality: text/classification
  size_gb: 0.14
  description: "9-category text moderation (S/H/V/HR/SH/S3/H2/V2/OK), CPU-friendly"
```

## Test strategy

Unit-heavy. Mocks for `transformers.AutoModelForSequenceClassification`
+ tokenizer. One slow e2e test exercises FastAPI + codec + mocked
runtime. One opt-in integration test against a live muse server with
a real text-moderation model loaded.

Coverage targets:
- Protocol + dataclass shape (5 tests).
- Codec: 5-format-equivalent here is just one OpenAI envelope; tests
  cover scalar and batch inputs, multi-label and single-label scoring,
  threshold resolution per layer, `flagged` derivation.
- Routes: 400 envelope for bad input, 404 envelope for unknown model,
  threshold honored when supplied, default applied otherwise.
- Runtime: deferred imports, multi-label sigmoid, single-label
  softmax, batching, device auto-select.
- Resolver: positive sniff, gguf-still-takes-priority negative,
  sentence-transformers-still-takes-priority negative, resolve shape,
  search routing.
- Curated: text-moderation entry parses with expected fields.
- E2E slow: full multipart-equivalent (JSON in, JSON out) round-trip.
- Integration opt-in: real server + real model.

## Documentation

- CLAUDE.md: add `text/classification` to modality list; new note
  about the modality-tag-vs-URL-route distinction this modality
  introduces.
- README.md: modality list + `/v1/moderations` endpoint.

## Release

v0.14.0. Minor bump (new feature, no breaking changes). Tag message
calls out: new endpoint, new curated entry, the modality-tag /
URL-route distinction (sets precedent for future modalities like
text/translation that may want non-OpenAI URLs).

## Out of scope

- Image moderation / OpenAI omni-moderation.
- Streaming.
- Per-category request thresholds.
- id2label remapping to OpenAI's fixed schema.
