# Decision API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local Choice, Score and Noul inference through one Muse API backed by Decision GGUF and Decima ONNX.

**Architecture:** Add the `decision_evaluation` modality with pure validation/codec contracts, two deferred-import adapters and a cancellation-safe execution boundary. Install both models through the existing pull path (curated entries plus an exact-repo HF plugin, verified by the post-pull probe), then reuse ordinary gateway admission, worker supervision, authorization, telemetry and storage. HTTP, Python and MCP return the same complete response envelope.

**Tech Stack:** Python 3.10+, existing FastAPI/httpx/requests clients, llama-cpp-python 0.3.33, decima-ai 2.2.0, onnxruntime 1.24.4, pytest.

**Spec:** [Decision API design](../specs/2026-10-18-decision-api-design.md). **Dependency:** complete the [pull path hardening plan](2026-10-18-pull-path-hardening.md) first (native placement, and a failed post-pull probe failing the pull). The [safe installation plan](2026-10-18-safe-model-installation.md) is deferred and is not a prerequisite.

**Revised 2026-10-18:** Task 4 originally depended on the safe-install foundation's `RuntimeRecipe`/private-verification records. It now uses curated entries and an exact-repo HF plugin on the existing pull path; Tasks 5 and 7 drop their generation/strict-install steps accordingly.

## Global Constraints

- Modality `decision/evaluation`; package `src/muse/modalities/decision_evaluation/`; route `POST /v1/systemone`; explicit Muse catalog `model` is required everywhere.
- Text primitives only: Choice, Score, Noul, string/JSON state, buffered whole-batch responses. No media, ranking, streaming, hosted service, automatic tools or model training.
- Body 128 KiB; serialized state 64 KiB; nesting depth 8; 1–16 questions; Choice 2–24 options; Score 2–10 levels; IDs/option keys 128 UTF-8 bytes; instructions 4 KiB; descriptions 2 KiB.
- Reject duplicate JSON keys, unknown fields, unsupported types/media and non-finite numbers. Preserve original keys and order; never coerce or silently truncate.
- Prevalidate every rendered question before any inference: Decision 2,048 tokens; Decima 512 state/question tokens and 64 tokens per rendered option.
- Choice confidence `(max(p) - 1/K) / (1 - 1/K)`; Score confidence `max(0, 1 - sum(p[i] * abs(i - m)) / U)`, with first-argmax `m` and `U = sum(abs(i - (K - 1)/2)) / K`.
- Normalize only roundoff within 1e-5; finite, complete distributions are mandatory. Do not independently round to four decimals; exact ties select the first supplied option/level.
- Logical `input_tokens` includes repeated context and option encodings before cache reuse; `output_tokens` is 0. Confidence is concentration, not correctness.
- Both backends have a hard concurrency ceiling of 1 and per-model queue ceiling of 32. Positive global limits may lower ceilings; global 0 does not remove them.
- Decision source `evalengine/decision-0.8b-gguf`, commit `ce993792ad283053004bf0230b1709ff7351bb24`, file `decision-0.8b-Q8_0.gguf`, 811,843,168 bytes; catalog ID `decision-0.8b-q8`.
- Decima source `amyrmahdy/decima-base`, commit `e5d84ab3a8436cedc45b0daefa91979f68319455`; catalog ID `decima-base`; local INT8 ONNX only, CPU pinned, temperature `0.7269418400253115`.
- Default telemetry/logging excludes state, questions, criteria and answers. No credentials, local paths or user text in backend errors.
- Use isolated catalogs/caches and owned workers for tests; no production service restart, driver changes, user-model replacement or VM operations.

## Review Focus

- A duplicate `model` key or a streamed oversized body must fail before model selection/auth-sensitive loading, including direct-worker requests — Tasks 1 and 5.
- Unicode byte limits, invalid Unicode and deeply nested finite JSON must not be confused with character counts or normalized user IDs — Task 1.
- A special-token-looking string in state/options must remain literal data, and a low-level tokenizer must never silently shorten a request — Tasks 2–3.
- Reversed true/false heads or reordered labels must not invert Noul, Score levels or the reported original Choice keys — Tasks 1–3.
- Cancelling a request while native work continues must not release its execution slot early, create unbounded waiting threads or corrupt later calls — Task 5.

---

## File Responsibilities and Shared Contracts

New package files under `src/muse/modalities/decision_evaluation/`:

- `protocol.py`: immutable requests/prepared inputs/distributions and the backend protocol, without FastAPI or ML imports.
- `validation.py`: strict raw-JSON parsing and byte/depth/schema validation; shared by gateway and direct worker.
- `codec.py`: complete numerical validation, answer shapes, concentration statistics and usage.
- `templates.py`: trusted Decision task framing, literal user segments and candidate-label token mapping.
- `runtimes/llama_cpp.py`, `runtimes/decima_onnx.py`: local-only prepared-input inference adapters; `runtimes/__init__.py` stays lightweight.
- `hf.py`, `probe.py`: exact-repo Hub dispatch with pinned revisions/files/extras, and the all-three-type probe run by `muse pull` and `muse models probe`.
- `execution.py`, `routes.py`: cancellation-safe async execution and bounded HTTP entry point.
- `client.py`, `__init__.py`: Python client and discoverable modality exports.

Other changes stay focused: extend `cli_impl/queueing.py` rather than introduce another gateway scheduler; add `mcp/tools/inference_decision.py` and register it; append curated entries only in the final delivery task. Mirror these boundaries under `tests/modalities/decision_evaluation/`.

Define the following in Task 1; `JsonValue` is recursive finite JSON, `QuestionKind = Literal["choice", "score", "noul"]`:

| Type | Contract |
| --- | --- |
| `DecisionQuestion` | `kind: QuestionKind`, `instructions: str`, `criteria: tuple[tuple[str, str \| None], ...]`; Score keys become decimal indices; Noul keys are `true`, `false` |
| `DecisionRequest` | `model: str`, `state: str \| dict[str, JsonValue] \| list[JsonValue]`, `questions: tuple[tuple[str, DecisionQuestion], ...]` |
| `PreparedQuestion` | `question_id: str`, `keys: tuple[str, ...]`, `payload: object`, `input_tokens: int`; payload is adapter-private token/array data, never wire JSON |
| `PreparedBatch` | `model_id: str`, `questions: tuple[PreparedQuestion, ...]`, `input_tokens: int` |
| `QuestionDistribution` | `question_id: str`, `probabilities: dict[str, float]` with every expected key |
| `DecisionBatch` | `answers: tuple[QuestionDistribution, ...]`, `input_tokens: int` |
| `DecisionInputError` | Exception with safe `code: str`, `message: str`; no user content in message |
| `DecisionBackendError` | Exception with safe `code: str`; output/cleanup errors do not expose native exception text |
| `DecisionModel` | Protocol: `model_id: str`; `prepare(request: DecisionRequest) -> PreparedBatch`; `evaluate(batch: PreparedBatch) -> DecisionBatch`; `close() -> None` |

Prepared batches validate model identity and originate from the same adapter instance. Preparing all questions precedes evaluation of any; tokenization is not inference. Test fixtures in `tests/modalities/decision_evaluation/conftest.py` supply ordered mixed-type requests, fake complete logits/heads and a blocking native-call fake controlled by thread events.

### Task 1: Define Strict Requests and Shared Answer Encoding

**Files:** Create `src/muse/modalities/decision_evaluation/{__init__,protocol,validation,codec}.py`, `tests/modalities/decision_evaluation/{conftest,test_validation,test_codec}.py`.

**Interfaces:** Produce the records above, `parse_request(body: bytes) -> DecisionRequest`, `encode_response(request: DecisionRequest, result: DecisionBatch, *, model_id: str) -> dict[str, JsonValue]`, `choice_confidence(probabilities: tuple[float, ...]) -> float`, `score_confidence(probabilities: tuple[float, ...]) -> float`.

- [ ] Write strict-parser tests for all exact limits at boundary and boundary + 1, duplicate keys at every object level, missing/empty model, unknown fields/media, booleans masquerading as numbers, NaN/Infinity, malformed UTF-8, unpaired surrogates, empty string state and duplicate Score descriptions. Empty object/array state remains valid. Preserve Unicode IDs byte-for-byte; count compact `ensure_ascii=False` serialized state bytes and state-container depth with the root at depth 1. Noul criteria may supply only `true`/`false` string descriptions, filling omitted descriptions with literal `True`/`False`. Add codec assertions:
  ```python
  assert choice_confidence((0.7, 0.2, 0.1)) == pytest.approx(0.55)
  assert score_confidence((0.7, 0.2, 0.1)) == pytest.approx(0.4)
  assert choice_confidence((0.5, 0.5)) == 0.0
  ```
  Assert Score expectation 0.4 for that distribution, decimal-index `legend` and `probabilities` objects, Noul equals `P(true)`, first-option ties, complete answer IDs, no confidence field for Noul, `output_tokens == 0`, tiny normalization accepted, and missing/extra/NaN/materially non-normalized probabilities rejected.
- [ ] Run `pytest tests/modalities/decision_evaluation/test_validation.py tests/modalities/decision_evaluation/test_codec.py -q`; expect RED for missing contracts, not dependency imports.
- [ ] Implement pure strict parsing with duplicate-preserving object-pair checks and finite JSON validation, and encode validated distributions centrally. Raise bounded safe errors with 400/413 distinction. Compute Score's ordered statistic independently of the upstream SDK; reject a batch with missing/repeated question results rather than returning partial success. Keep answer precision and key order intact.
- [ ] Run the Task 1 command; require GREEN without torch, llama_cpp, decima or onnxruntime installed/imported.
- [ ] Commit `feat(decision): define strict text primitives and numerical codec` with Task 1 files only.

### Task 2: Implement the Decision GGUF Candidate Scorer

**Files:** Create `src/muse/modalities/decision_evaluation/{templates.py,runtimes/__init__.py,runtimes/llama_cpp.py}`, `tests/modalities/decision_evaluation/runtimes/test_llama_cpp.py`, `tests/modalities/decision_evaluation/fixtures/decision_prompt.json`.

**Interfaces:** Produce `DecisionLlamaCppModel(*, model_id: str, hf_repo: str, local_dir: str, gguf_file: str, device: str = "auto", n_gpu_layers: int = -1, context_length: int = 2048, n_threads: int = 2, **kwargs)` implementing `DecisionModel`; `render_decision_tokens(request: DecisionRequest, question_id: str, backend: object) -> tuple[int, ...]`; `final_candidate_logits(backend: object, candidate_token_ids: tuple[int, ...]) -> tuple[float, ...]`. Consume `muse.core.placement` (cherry-picked in the hardening plan's Task 1). The latter helper reads logits after prompt evaluation; its token IDs identify candidates, not the input prompt.

- [ ] Add fake-native tests for all 24 labels A–X mapping to distinct single tokens; unsupported mappings fail. Pin a trusted no-thinking template fixture to the approved model format; assert question IDs are absent from instructions and special-token strings in state/options are tokenized with special parsing disabled. Assert fully rendered 2,048-token inputs pass and 2,049 fail before any `eval`; an invalid last question yields zero inference calls. Assert Choice key permutations, Score index order, Noul true/false polarity, stable full-candidate softmax, NaN rejection, reset before every question, and close on failed load/probe. Verify the final native accessor against a fake/full-logits reference; never read `.scores` with `logits_all=False`.
- [ ] Run `pytest tests/modalities/decision_evaluation/runtimes/test_llama_cpp.py -q`; expect RED for missing scorer and template boundary.
- [ ] Implement deferred llama_cpp loading and trusted-template/literal-data token segments. Validate label IDs and context capacity; prepare every question before evaluating. Read the final native logits through the pinned 0.3.33 `_ctx.get_logits_ith(-1)` boundary, copy all candidate values before reset, and softmax only those candidates. Return raw distributions; codec owns answers. Do not generate/parse letters, use top-logprobs, add cross-request caches or silently fall back devices.
- [ ] Run the Task 2 command plus `pytest tests/modalities/chat_completion/runtimes/test_llama_cpp.py -q`; require GREEN. Add an opt-in real-native comparison of final-accessor versus `logits_all=True` outputs to Task 7's live matrix.
- [ ] Commit `feat(decision): score complete GGUF candidate distributions` with Task 2 files only.

### Task 3: Implement the Local Decima ONNX Adapter

**Files:** Create `src/muse/modalities/decision_evaluation/runtimes/decima_onnx.py`, `tests/modalities/decision_evaluation/runtimes/test_decima_onnx.py`.

**Interfaces:** Produce `DecimaOnnxModel(*, model_id: str, hf_repo: str, local_dir: str, device: str = "cpu", n_threads: int = 2, option_cache_size: int = 256, **kwargs)` implementing `DecisionModel`. Use the pinned package's raw `Question`/`decide` heads, not its System One wire serializer. The pulled snapshot's `local_dir` contains the declared `onnx/int8/` subtree.

- [ ] Write fake-package tests for choose/verify/ordinal mapping, reversed raw verify-head polarity and permuted choices. Assert 512 state/question and 64 rendered-choice tokens pass; 513 or 65 fail before encoder/scorer execution. Cover the SDK's description-shortening path and a too-long last question; assert no silent truncation or partial inference. Assert CPU-only placement, local paths, no downloader/chunker calls, fitted temperature `0.7269418400253115`, bounded 256-entry cache, logical token usage unchanged on cache hits, inter-request input isolation and cleanup. Include a raw distribution where the SDK's Score confidence differs from Muse's codec.
- [ ] Run `pytest tests/modalities/decision_evaluation/runtimes/test_decima_onnx.py -q`; expect RED for missing adapter/prevalidation.
- [ ] Implement deferred package loading from complete local INT8 assets, explicit two-thread default and validated fitted temperature/head ordering. Prepare rendered state/question/options without SDK truncation or shortening; only then call raw heads. Bound option encodings, keep raw probability precision, and clear per-request mutable state. Unsupported device/configuration fails rather than selecting another export/runtime.
- [ ] Run the Task 3 command and Task 1 codec tests; require GREEN without a real ONNX download.
- [ ] Commit `feat(decision): add local bounded Decima ONNX inference` with Task 3 files only.

### Task 4: Add the Exact-Repo HF Plugin and the Three-Type Probe

**Files:** Create `src/muse/modalities/decision_evaluation/{hf,probe}.py`, `tests/modalities/decision_evaluation/{test_hf,test_probe}.py`; test `tests/core/test_hf_plugin_discovery.py`, `tests/modalities/text_classification/test_hf_plugin.py`. Defer public curated rows to Task 7.

**Interfaces:** Produce `HF_PLUGIN` (the existing contract in `docs/HF_PLUGINS.md`) at a priority that runs before `text/classification`'s 110 tier, and `probe_backend(backend: DecisionModel) -> dict[str, JsonValue]`. A module-level `SOURCES` table holds, per repo: pinned commit, exact allow-list, runtime class path, `pip_extras`, and capabilities (`device`, `memory_gb`, `max_concurrency: 1`, Decima's temperature).

- [ ] Write plugin tests: `sniff` claims exactly `evalengine/decision-0.8b-gguf` and `amyrmahdy/decima-base` (case-insensitive) and nothing else, including repos that merely carry a `decision` tag or a copied config. `hf://amyrmahdy/decima-base` resolves to `decision/evaluation`, not `text/classification` (regression for the audit's NLI misdispatch). `resolve` pins `revision` to the table's commit and refuses another revision unless it is the pinned one. Decision allow-lists only `decision-0.8b-Q8_0.gguf` (811,843,168 bytes, checked after download) plus license/attribution; Decima allow-lists exactly `config.json`, `onnx/int8/{decima.json,encoder.onnx,scorer.onnx,tokenizer/tokenizer.json,tokenizer/tokenizer_config.json}` and `LICENSE`, with no FP32/PyTorch files. `pip_extras` are `llama-cpp-python==0.3.33` and `decima-ai==2.2.0` + `onnxruntime==1.24.4`. Decima is pinned `device: cpu`. The plugin imports only stdlib, Hub and lightweight core modules.
- [ ] Write probe tests: `probe_backend` runs one bounded Choice, Score and Noul batch, validates complete outputs through the codec, and raises on any missing/invalid distribution, so the probe-on-pull (now fatal after the hardening plan) fails the pull.
- [ ] Run `pytest tests/modalities/decision_evaluation/test_hf.py tests/modalities/decision_evaluation/test_probe.py -q`; expect RED.
- [ ] Implement. Keep reservations conservative (`memory_gb` from the published file sizes) until Task 7's live probes measure them.
- [ ] Run the Task 4 command plus `pytest tests/core/test_hf_plugin_discovery.py tests/modalities/text_classification/ -q`; require GREEN and unchanged unrelated plugin dispatch.
- [ ] Commit `feat(decision): pin decision sources and probe all three question types` with Task 4 files only.

### Task 5: Serve With Bounded Queues and Cancellation-Safe Execution

**Files:** Create `src/muse/modalities/decision_evaluation/{execution,routes}.py`, `tests/modalities/decision_evaluation/{test_execution,test_routes}.py`, `tests/cli_impl/test_gateway_decision.py`; modify `src/muse/modalities/decision_evaluation/__init__.py`, `src/muse/cli_impl/{queueing,gateway,worker}.py`; test `tests/cli_impl/{test_queueing,test_queueing_gateway,test_gateway_telemetry}.py`.

**Interfaces:** Produce `build_router(registry: ModalityRegistry) -> APIRouter` and `DecisionExecutor(model_id: str)` with `async execute(backend: DecisionModel, request: DecisionRequest, *, deadline: float) -> DecisionBatch`, `async close() -> None`. Extend `ConcurrencyGate.slot` and `acquire_slot` with optional `max_queue_depth: int | None = None`; effective depth is the smallest positive model/server limit. Gateway `_effective_max_concurrency(manifest: dict) -> int | None` also clamps against a modality-declared ceiling: the package exports `CONCURRENCY_CEILING = {"max_concurrency": 1, "max_queue_depth": 32}`, aggregated by `discover_modalities` like `MODEL_OPTIONAL_PATHS`, so a catalog edit cannot raise it. Existing callers retain current semantics.

- [ ] Write direct-worker and gateway tests for explicit model, unknown/disabled/forbidden models, streaming body above 128 KiB without Content-Length, duplicate `model` keys, 400 token/schema errors, 413 body errors and sanitized 500 invalid numerics. Assert validation occurs before cold-load acquisition. With global unlimited, assert one executes, 32 wait and the next gets the existing queue-full 503; a positive global ceiling lowers depth. Test a surviving gate changing from cap 4 to hard cap 1 without restart: no new permit while active count is at least 1, then at most 1 thereafter. Use event-controlled native fakes to assert loop/health responsiveness, no waiter threads, one cold load, exact-once references/slots, cancelled waiter isolation, and an executing cancelled request retaining its backend slot until completion or owned shutdown. Assert subsequent calls cannot reuse a poisoned backend; shutdown drains or terminates only owned workers. Assert generic discovery reports `decision/evaluation` and path telemetry reports `systemone` without payload content.
- [ ] Run `pytest tests/modalities/decision_evaluation/test_execution.py tests/modalities/decision_evaluation/test_routes.py tests/cli_impl/test_gateway_decision.py -q`; expect RED for missing route/ceilings/cancellation boundary.
- [ ] Implement raw bounded parsing at both entries, before gateway model extraction; leave `/v1/systemone` out of optional-model routes. Register `MODALITY`, router and `PROBE_DEFAULTS`. Park waiters on existing asyncio gates; run a single admitted prepare/evaluate batch off-loop. The executor owns a retained task/future and its permit until native completion; caller cancellation must not cancel that owner or release its permit early. Reuse/extend worker cleanup for bounded drain then owned termination; uncertain runtime state is unservable, never silently reused. Preserve one gateway gate/director reference per batch and existing coalesced-load shielding. Clamp limits from the modality's `CONCURRENCY_CEILING`, not an overridable manifest capability; adjust cached gate capacity without restarting or over-issuing permits when a lower ceiling takes effect. Reuse current generic capability discovery and `systemone` path telemetry; neither needs a new hard-coded routing map.
- [ ] Run the Task 5 command plus `pytest tests/cli_impl/test_queueing.py tests/cli_impl/test_queueing_gateway.py tests/cli_impl/test_gateway_telemetry.py tests/cli_impl/test_signal_shutdown.py -q`; require GREEN. Run isolated marked slow cold-load/cancellation/shutdown cases separately.
- [ ] Commit `feat(decision): serve bounded cancellation-safe decision batches` with Task 5 paths only.

### Task 6: Expose Matching Python and MCP Clients

**Files:** Create `src/muse/modalities/decision_evaluation/client.py`, `src/muse/mcp/tools/inference_decision.py`, `tests/modalities/decision_evaluation/test_client.py`, `tests/mcp/test_tools_inference_decision.py`; modify `src/muse/modalities/decision_evaluation/__init__.py`, `src/muse/mcp/{client,server}.py`, `src/muse/mcp/tools/__init__.py`; test `tests/mcp/{test_model_contract,test_http_sse,test_no_token_leakage}.py`.

**Interfaces:** Produce `DecisionClient(server_url: str | None = None, *, timeout: float = 300.0)` with `decide(*, model: str, state: str | dict | list, questions: dict) -> dict`; `MuseClient.decide(*, model: str, state: str | dict | list, questions: dict) -> dict`; tool `muse_decide` with handler `_handle_decide(client: MuseClient, args: dict) -> list[dict]`. All return the full HTTP envelope, not a selected answer.

- [ ] Add mocked HTTP and MCP tests asserting the identical `/v1/systemone` body, explicit model in schema and handler, unchanged Unicode IDs and full answer/usage envelope. Assert missing model fails before HTTP, existing authorization is enforced, unsupported fields are rejected, and HTTP 400/403/404/413/503 are not converted into fabricated answers. Assert tool registration once, sync/async transports stay responsive and secret/path/input-bearing upstream exceptions are sanitized.
- [ ] Run `pytest tests/modalities/decision_evaluation/test_client.py tests/mcp/test_tools_inference_decision.py tests/mcp/test_model_contract.py -q`; expect RED for absent clients/tool.
- [ ] Implement the thin clients through existing URL/config conventions and gateway calls, with required keyword-only model and local strict validation where JSON is already available. Register the dedicated tool and include it in existing per-model authorization/required-model checks. Use safe error codes/envelopes, not raw exception strings. Do not bypass gateway admission or include state/questions/answers in operational logs.
- [ ] Run the Task 6 command plus `pytest tests/mcp/test_http_sse.py tests/mcp/test_no_token_leakage.py tests/mcp/test_client.py -q`; require GREEN.
- [ ] Commit `feat(mcp): expose explicit-model decision inference clients` with Task 6 files only.

### Task 7: Publish Both Curated Models After End-to-End Conformance

**Files:** Modify `src/muse/curated.yaml`, `README.md`, `pyproject.toml`; create `docs/DECISIONS.md`, `examples/decisions.py`, `tests/integration/test_remote_decisions.py`, `tests/modalities/decision_evaluation/test_curated.py`, `tests/modalities/decision_evaluation/fixtures/typesafe_text_subset.json`.

**Interfaces:** Add curated IDs `decision-0.8b-q8` and `decima-base` pointing at the Task 4 URIs with their pinned `revision`, plus probability/calibration provenance. Each live test consumes the ordinary public `muse pull` and gateway API, not hand-written catalog records. Compatibility fixtures include source SDK version/commit, serialization provenance and the tested contract date `2026-10-18`.

- [ ] Add curated tests for exact IDs, sources, revisions, dependencies, licenses, hard limits, Decision Q8 and Decima INT8/CPU, text-only question types and distinct calibration metadata. Pin official SDK request/response serialization fixtures for the supported three text primitives, preserving source version/commit and noting intentional usage/model-selection differences; do not advertise untested SDK features. Add opt-in live cases for a fresh pull (probe passes, model enabled), all-three-type batches, offline reload, unchanged re-pull, a forced probe failure leaving the model disabled, ordinary auth/warmup/idle unload and `muse models remove` cleanup. Verify real native final logits against the full-logits reference; inspect Decima token-boundary rejection without truncation.
- [ ] Run `pytest tests/modalities/decision_evaluation/test_curated.py -q`; expect RED before public catalog rows/metadata are added.
- [ ] Add the two rows only after Tasks 1–6 and the pull path hardening plan meet their gates. Document curl/Python/MCP examples, exact limits, answer/confidence/usage semantics, complete probability behavior, differing calibration and non-guaranteed accuracy. Preserve Apache-2.0 attribution and link Decima's upstream training-data licensing caveat without claiming commercial clearance. Update modality/capability counts and packaged assets only where existing checks require it; heavyweight runtimes stay in model environments. Document explicit opt-in/download/build settings and `MUSE_REMOTE_SERVER` for live tests.
- [ ] Run `pytest tests/modalities/decision_evaluation/ tests/mcp/test_tools_inference_decision.py tests/cli_impl/test_gateway_decision.py -q`, `pytest -m "not slow" -q`, `pytest tests/ -q`, and `python scripts/preflight.py`; require zero unexpected failures. Separately run both pinned pull-to-inference smoke tests with isolated storage and a gateway running this code. Record actual placement, load/inference memory and latency, probability checks and limitations; do not turn the previous 24-case feasibility result into a calibration claim. Do not mark either model ready if its live lifecycle cannot complete.
- [ ] Commit `feat(catalog): ship verified Decision and Decima text primitives` with Task 7 files only after final review.

## Delivery Boundary

The pull path hardening plan ships first. This plan then delivers the common protocol/codec, both adapters, the pinned plugin and probe, bounded serving and clients, and finally both curated entries. Release is gated on both models completing pull → probe → serve on the ordinary pull path; no manually registered model, silent CPU/GPU fallback or generated-prose decision substitute counts as completion.

Implement on a feature branch from `main` after the hardening plan lands. No service restart, commit/push of pre-existing unrelated edits, release tag or external publication is implied by plan approval.
