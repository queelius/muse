# Decision API and curated models design

**Date:** 2026-10-18
**Status:** Approved by user on 2026-10-18
**Dependency:** [Safe model installation](2026-10-18-safe-model-installation-design.md)

## Purpose and scope

Add local Jev-style decision inference to Muse using existing open models, not a
new training project or a wrapper around generated prose. Users supply shared
state and typed questions and receive structured judgments and distributions.

Ship both `decision-0.8b-q8` and `decima-base` in the curated catalog. They use one
`decision/evaluation` modality and `POST /v1/systemone`, with an explicitly
documented text-only subset of TypeSafe's request/response contract. Model
quality, probabilities and calibration remain backend-dependent.

Initial scope is Choice, Score and Noul; strings and JSON state; buffered JSON
responses; ordinary HTTP/Python/MCP clients; and Muse's existing worker lifecycle.
Images, audio, video, Box, ranking, streaming, model training, automatic tool
execution, automatic escalation between models and hosted TypeSafe access are
out of scope. Retain a modality boundary that can add media later without
misclassifying decisions as chat or ordinary NLI.

## Curated models and artifacts

| Catalog ID | Source and pinned commit | Runtime and downloaded subset |
| --- | --- | --- |
| `decision-0.8b-q8` | `evalengine/decision-0.8b-gguf` at `ce993792ad283053004bf0230b1709ff7351bb24` | llama.cpp candidate scorer; `decision-0.8b-Q8_0.gguf`, license and attribution files |
| `decima-base` | `amyrmahdy/decima-base` at `e5d84ab3a8436cedc45b0daefa91979f68319455` | Decima ONNX CPU scorer; `config.json`, `onnx/int8/decima.json`, encoder, scorer, tokenizer files and license |

The Decision GGUF is 811,843,168 bytes. Decima's selected INT8 model and tokenizer
assets are about 371 MB; do not download its FP32 or PyTorch exports. File sizes
are not runtime RAM requirements. The initial environment recipes pin
llama-cpp-python 0.3.33 for Decision and decima-ai 2.2.0 with onnxruntime 1.24.4
for Decima, plus a resolved platform-specific dependency receipt. These are the
versions exercised in the local feasibility check; any later upgrade requires
recipe revalidation. Check Python/platform compatibility before installation.

Curated metadata declares supported question types, text-only inputs, token
limits, `max_concurrency: 1`, runtime/profile version, numerical precision and
probability semantics. Use conservative memory reservations and replace estimates
with verified observations under the shared placement policy. Decima is CPU
pinned. Decision permits CPU or a validated native GPU build; `auto` reports its
actual placement. Never claim GPU use because PyTorch or NVML sees a GPU.

Decima ships fitted temperature scaling; Decision's candidate probabilities are
uncalibrated. Neither metadata entry claims validated calibration for arbitrary
user tasks. Record this distinction without labeling one backend's confidence
as a probability of being correct.

Both publishers declare Apache-2.0 releases. Decima also discloses training-data
licensing caveats. Preserve attribution and link that disclosure in model info;
do not describe the entry as commercially cleared. This release does not bundle
the upstream training datasets.

## Request contract

Require `model`, `state` and `questions`. The model is an explicit Muse catalog
ID, not an implicit `jev-latest` alias. Question IDs are user keys returned
unchanged and are not instructions to the model.

```json
{
  "model": "decision-0.8b-q8",
  "state": {"message": "I was charged twice and need a refund."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Charges and refunds", "technical": "Software faults"}
    },
    "duplicate": {"type": "noul", "instructions": "The customer reports a duplicate charge."},
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request?",
      "criteria": ["Routine", "Needs prompt attention", "Immediate outage response"]
    }
  }
}
```

State accepts a nonempty string, JSON object or JSON array. Questions use string
instructions. Choice criteria are a map of distinct option keys to string or
null descriptions. Score criteria are an ordered array of distinct nonempty
level descriptions. Noul optionally accepts string descriptions under `true`
and `false` criteria. Reject unsupported question types, media/assets, unknown
parameters and duplicate JSON keys rather than silently ignoring them.

Initial server limits are a 128 KiB body, 64 KiB serialized state, nesting depth
8, 1–16 questions, Choice 2–24 options, Score 2–10 levels, IDs/option keys up to
128 UTF-8 bytes, instructions up to 4 KiB and descriptions up to 2 KiB. Request
limits also apply at direct worker entry points, not only the gateway. Count
UTF-8 bytes where specified; reject non-finite JSON numbers. Individual maxima
do not override the total body cap.

Before inference, validate each backend's fully rendered inputs, including the
question, candidate descriptions and template overhead. Decision supports up to
2,048 input tokens per question; Decima supports 512 state/question tokens and
64 tokens for each rendered option. Reject excess rather than truncating or
silently shortening descriptions. Validate all questions before executing any;
return no partial batch when one question is invalid.

## Answer contract

Return `model`, an `answers` map keyed by every request question ID, and `usage`.
The following is an illustrative Choice answer, not a model measurement:

```json
{
  "model": "decision-0.8b-q8",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.9, "technical": 0.1},
      "confidence": 0.8
    }
  },
  "usage": {"input_tokens": 128, "output_tokens": 0}
}
```

Choice returns the winning original key and a probability for every option.
Score returns the probability-weighted level index, a legend and the complete
level distribution keyed by decimal index strings. Noul returns `noul`, the
probability of true; it has no separate confidence field. Never infer missing
candidate probabilities from a truncated top-k list.

The shared codec, not an upstream SDK, computes wire confidence. With K outcomes,
Choice uses `(max(p) - 1/K) / (1 - 1/K)`. Score uses the documented ordered-level
spread statistic: let m be the first most-probable level and
`U = sum(abs(i - (K - 1)/2) for i in range(K)) / K`; confidence is
`max(0, 1 - sum(p[i] * abs(i - m) for i in range(K)) / U)`. Clamp only tiny
floating-point excursions to [0, 1]. These summarize concentration, not correctness.

Validate finite values, complete keys and normalization before encoding. Permit
normalization of numerical roundoff within 1e-5; materially invalid distributions
are backend errors, not successful answers. Preserve numerical precision rather
than independently rounding probabilities to four decimals. Exact ties select
the first supplied Choice option or first Score level, deterministically.

`input_tokens` counts logical tokenized model inputs across questions, including
repeated context and option encodings, before cache reuse. It is not billing or
an estimate of native compute. The adapters do not generate answer text, so
`output_tokens` is zero. Document this accounting and report the actual responding
catalog ID. Generation and calibration provenance live in model/admin metadata.

## Runtime adapters

### Decision GGUF scorer

Load the pinned GGUF in the model's own environment. Match its published
no-thinking task format, map options to validated single-token labels, evaluate
the prompt and read all candidate logits at the answer position. Apply a stable
softmax over exactly those labels. Choice keys/descriptions remain data; special
token strings in untrusted text must not become template control tokens.

Do not generate a letter and parse it, use chat top-logprobs, or read an
unpopulated Python logits buffer. llama-cpp-python 0.3.33 does not populate
`scores` when `logits_all=False`; verify the final native output accessor against
the full-logits path. Reject unsupported token-label mappings during verification.

Noul uses explicit true/false candidates with correct polarity and optional
criteria. Score scores the ordered level descriptions and the codec computes
the expected index. Each question sees only the shared state and its own
instructions/options. Reset native state between independent questions and
requests; cross-request prefix caching is not part of this release.

### Decima ONNX scorer

Use the pinned audited runtime package against local INT8 exports, with explicit
CPU thread bounds. Map Choice to its choose head, Noul to verify with true/false
alignment, and Score to the ordinal head. Obtain raw result distributions and
let Muse's codec construct the wire answers, rather than forwarding the SDK's
already-rounded System One response.

Check token lengths before invoking the SDK: its low-level runtime can silently
truncate. Do not use its automatic remote download fallback or long-document
chunking. Bound its option-encoding cache; isolate state and mutable runtime
operations so one request cannot change another request's interpretation.

## Installation and discovery

Add a focused `decision_evaluation` package following existing protocol, codec,
routes, client, runtime and HF-plugin conventions. Keep ML imports deferred.
The two curated entries specify the decision recipe rather than relying on
generic chat/NLI tag matching. Direct `hf://` references to those same sources
receive the same recipe selection and immutable pins through curated resolution.

The decision plugin may automatically claim only declared, validated recipe
formats. A generic `decision-model` tag is insufficient to infer a prompt or
head architecture. Unsupported adapters fail with a custom-recipe hint. A repo
that validly supports multiple tasks requires a selection under the installer
contract, not another higher-priority heuristic.

Both entries must pass private load plus typed-inference probes before active
publication. Probes exercise all three question types, complete distributions
and cleanup, using a bounded short input. They do not assert broad model quality.
Runtime/code/precision changes invalidate the relevant verification evidence.

## Serving and integration

Register the route in the existing modality discovery and proxy path. Use
ordinary model authorization, enable/readiness checks, LoadDirector admission,
coalesced cold loads, idle eviction, admin warmup, telemetry and storage cleanup.
No separate decision server or port. Installing a model does not change another
default; this endpoint always requires an explicit model.

Each HTTP request consumes one existing per-model concurrency slot and one
director reference through all its questions. Declare concurrency 1 for both
backends as a hard runtime ceiling, not an operator-overridable suggestion.
Park queued callers on the gateway event loop and use the existing queue
depth/deadline mechanisms. Give decision recipes a per-model queue ceiling of
32; a positive server-wide limit may lower it, while a global unlimited setting
does not remove the recipe ceiling. Extend the existing gate to honor this
ceiling rather than adding a second scheduler. No thread per waiting request.
Direct worker calls also need a bounded async single-execution gate so native
state cannot race outside the gateway. Inference runs off the ASGI event loop.

Cancellation releases queue slots/references exactly once. A cancelled waiter
must not cancel another caller's coalesced load. A native call already running
must finish or be terminated through owned-worker shutdown before its execution
slot is reusable; dropping an await does not make native execution stop.
Never reuse a backend whose cleanup or cancellation leaves its state uncertain.

Add a Python decision client and an MCP `muse_decide` tool using the same body and
full response envelope. MCP requires a model in both schema and handler, uses
existing per-model authorization, and must not bypass the gateway. Add the new
route to any modality-aware capability/telemetry mappings; path-derived telemetry
labels may remain `systemone` where that is the existing convention.

Return ordinary Muse errors: 400 for invalid/unsupported requests or input token
limits, 413 for body limits, 404 for unknown models, 403 for forbidden models,
503 for unverified/unservable models or existing queue/capacity failures, and a
sanitized 500 backend error for invalid numerical output. Disabled models follow
Muse's existing disabled-model contract. A client never receives a partial
success or raw local paths from a backend exception.

## Compatibility and safety boundaries

Compatibility targets the text primitive request/answer shapes, explicit Muse
model selection, and the confidence statistics documented by TypeSafe on
2026-10-18. It does not claim TypeSafe model equivalence, identical probabilities,
their full input-size limits, media support, billing semantics or hosted service
features. Pin SDK serialization/response fixtures during implementation; only
advertise SDK compatibility for the tested subset.

State and criteria are untrusted data, never executable commands or URLs to
fetch. Model instructions are not a security boundary: prompt injection, numeric
rules, out-of-domain judgments and option wording/order can change answers.
Keep deterministic policy enforcement outside these models. Do not automatically
approve tools, remove files, route Muse itself or change scheduling based on a
decision response.

Default telemetry records model, timing, queue/capacity and error metadata, not
state, question text, criteria or answer contents. Diagnostic exceptions must
not leak those inputs, credentials or local artifact paths.

The local 24-case CPU smoke test found 23/24 canonical answers for Decision and
17/24 for Decima, with respectively 3/48 and 0/48 answer changes after reordering.
This small hand-authored set does not establish calibration or general accuracy.
In particular, Decision's fraud-veto answer changed after reordering; neither
model is approved as an autonomous safety gate.

## Acceptance and delivery

Unit tests cover strict request validation, byte/token boundaries, duplicate keys,
Unicode key preservation, each head mapping, true/false polarity, Score order,
confidence formulas, ties, missing/NaN logits, numeric normalization, template
control-token handling and input isolation. Codec tests include distributions
where Choice and Score confidence differ. Validate mapping under permutations;
do not require a model's probabilities to be order-invariant unless its recipe
specifically guarantees that property.

Serving tests cover explicit-model selection, authorization, staging visibility,
resource-pool selection, same-model cold-load coalescing, bounded queueing,
cancellation during load/inference, shutdown, idle unload and storage leases.
MCP/client tests assert the same model and error contracts as HTTP.

Pinned opt-in live tests pull each curated entry through the strict installer,
reload offline, send mixed typed questions, validate response distributions, and
exercise a running gateway without manual model registration. Re-pull and failed
update tests must preserve the previous generation. Run preflight and the full
suite before release, not only new modality tests.

Deliver after the installation foundation, then add the common protocol/codec,
both runtime recipes, curated entries, serving/client/MCP wiring and the live
conformance checks. An entry that cannot complete this lifecycle is not shipped
as a ready curated decision model.

## References

- [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice),
  [Score](https://docs.typesafe.ai/primitives/score),
  [Noul](https://docs.typesafe.ai/primitives/noul) and
  [confidence definitions](https://docs.typesafe.ai/confidence).
- [Decision model card](https://huggingface.co/evalengine/decision-0.8b) and
  [GGUF artifacts](https://huggingface.co/evalengine/decision-0.8b-gguf).
- [Decima model card](https://huggingface.co/amyrmahdy/decima-base),
  [runtime source](https://github.com/amyrmahdy/decima) and publisher
  [licensing disclosure](https://github.com/amyrmahdy/decima/blob/main/release/LICENSING.md).
