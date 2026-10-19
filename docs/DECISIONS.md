# Decisions (`decision/evaluation`)

`POST /v1/systemone` answers typed questions about one shared `state` with
local decision models. Instead of generating prose, each answer comes from
a complete probability distribution over the options you list. The wire
shape is a text-only subset of TypeSafe's System One API (Choice, Score,
Noul), with one difference: you must always name a Muse catalog `model`.

```bash
muse pull decima-base            # CPU, ~0.37 GB
muse pull decision-0.8b-q8       # llama.cpp, ~0.81 GB
muse serve
```

## Models

| Catalog id | Source (pinned commit) | Runtime | Probabilities |
|---|---|---|---|
| `decision-0.8b-q8` | `evalengine/decision-0.8b-gguf` @ `ce993792ad283053004bf0230b1709ff7351bb24`, `decision-0.8b-Q8_0.gguf` (811,843,168 bytes) | llama.cpp (llama-cpp-python 0.3.33): softmax over the final logits of the option letters A–X | **uncalibrated** |
| `decima-base` | `amyrmahdy/decima-base` @ `e5d84ab3a8436cedc45b0daefa91979f68319455`, INT8 ONNX export only | decima-ai 2.2.0 sessions + onnxruntime 1.24.4, CPU only, 2 threads | one fitted temperature (0.7269418400253115) from the publisher, not validated for arbitrary tasks |

Both are Apache-2.0. Decima's publisher also discloses training-data
licensing caveats; read
[LICENSING.md](https://github.com/amyrmahdy/decima/blob/main/release/LICENSING.md)
before commercial use. Muse does not describe either model as cleared for
any use.

Only these two repositories are accepted, at their pinned commits. A bare
`muse pull hf://amyrmahdy/decima-base` is refused when the repository has
moved past the pin; the curated id supplies it. Every downloaded file is
checked against its pinned size, and large files against their SHA-256.
Pulling runs one Choice, Score and Noul question as a probe. If the probe
fails, the pull exits 1 and the model is left disabled.

## Request

```json
{
  "model": "decima-base",
  "state": {"message": "I was charged twice and need a refund."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Charges and refunds", "technical": null}
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

- `state`: a nonempty string, a JSON object or a JSON array. It is
  treated as data, never as instructions or URLs.
- Question ids are your own keys and are returned unchanged, byte for
  byte. They are never shown to the model.
- `choice`: `criteria` maps 2–24 distinct option keys to a description or
  `null`.
- `score`: `criteria` is an ordered array of 2–10 distinct, nonempty level
  descriptions, lowest first.
- `noul`: optional `criteria` with `true` and/or `false` descriptions.

Muse rejects the following rather than ignoring them:

- unknown fields;
- media inputs;
- structured (object) instructions or criteria;
- duplicate JSON keys at any depth;
- `NaN` or `Infinity`;
- invalid UTF-8 or unpaired surrogates.

### Limits

| Limit | Value | Error |
|---|---|---|
| Body | 128 KiB, enforced as the body streams in at both the gateway and the worker | 413 `body_too_large` |
| State | 64 KiB, measured as UTF-8 bytes of the string or of compact JSON | 413 `state_too_large` |
| State nesting | 8 levels | 400 `state_too_deep` |
| Questions | 1–16 | 400 `invalid_questions` |
| Ids and option keys | 128 UTF-8 bytes | 400 |
| Instructions | 4 KiB | 400 |
| Descriptions | 2 KiB | 400 |
| `decision-0.8b-q8` rendered question | 2,048 tokens | 400 `input_too_long` |
| `decima-base` state plus question | 512 tokens | 400 `input_too_long` |
| `decima-base` each option, rendered with its question | 64 tokens | 400 `input_too_long` |

Every question is rendered and checked before any is evaluated. If one
question is too long, the whole batch is rejected; nothing is truncated
or shortened. Decima puts the question text into each option, so a long
question can exceed its 64-token option budget.

## Answers

```json
{
  "model": "decima-base",
  "answers": {
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.93, "technical": 0.07}, "confidence": 0.86},
    "duplicate": {"type": "noul", "noul": 0.97},
    "urgency": {"type": "score", "score": 0.61,
                "legend": {"0": "Routine", "1": "Needs prompt attention", "2": "Immediate outage response"},
                "probabilities": {"0": 0.45, "1": 0.49, "2": 0.06}, "confidence": 0.21}
  },
  "usage": {"input_tokens": 190, "output_tokens": 0}
}
```

The numbers above are illustrative, not a measurement.

- **Choice**: the winning option key and a probability for every option,
  in your order.
  - A tie goes to the first option you listed.
  - `confidence = (max(p) − 1/K) / (1 − 1/K)`.
- **Score**: `score` is the probability-weighted level index. It comes
  with a `legend` and the full distribution, keyed `"0"`, `"1"`, and so on.
  - `confidence = max(0, 1 − Σ pᵢ·|i − m| / U)`, where `m` is the first
    most probable level and `U = Σ|i − (K−1)/2| / K`.
- **Noul**: `noul` is the probability that the statement is true. There
  is no confidence field.
- Probabilities are complete, finite and keep their full precision. Sums
  within 1e-5 of 1 are renormalized; anything else is an error, never a
  partial answer.
- `input_tokens` counts every token fed to the model across all questions,
  including the state repeated per question and option encodings even
  when they are served from cache. `output_tokens` is always 0.

**Decision-0.8B's Score answers are often bimodal.** It can put most of
the probability on the two end levels and little in the middle. Then
`score` (the weighted mean) lands between them on a level the model
barely chose, and `confidence` is near 0 even though no level looked
uncertain. In the 2026-10-19 live run it scored a checkout outage 1.29
(confidence 0.00) on a 0-2 scale, although its most probable level was 2.
Read the `probabilities` for Score questions; do not threshold `score`
alone. Decima-base's ordinal head did not show this pattern in the same
run.

**Confidence measures concentration, not correctness.** Neither model is a
safety gate. Their answers can change with option wording, option order,
prompt injection in `state`, rules involving numbers, or out-of-domain
questions. Enforce policy deterministically, outside these models.

## Errors

| Status | When |
|---|---|
| 400 | invalid or unsupported request, or an input-token limit |
| 413 | body or state too large |
| 404 | unknown model, or one that is not a decision model |
| 409 | the model is disabled |
| 503 | queue full or timed out, the model cannot be served, or the backend is unusable until its worker restarts |
| 500 | the backend returned an invalid distribution, or failed |
| 499 | the client disconnected; nobody receives this, it appears only in telemetry |

Responses never include your text, local paths or native exception text.

## Serving

Each model runs **one batch at a time**, with at most **32 requests
queued**. These are hard ceilings declared by the modality
(`CONCURRENCY_CEILING`). A manifest edit or `server.default_max_concurrency`
cannot raise them; a positive `server.max_queue_depth` can lower the
queue limit.

The gateway parses the body strictly before choosing a model, so a
malformed request never triggers a cold load.

**A client that gives up stops its batch.** The modality lists
`/v1/systemone` in `CANCEL_ON_DISCONNECT_PATHS`, so when a client
disconnects, the gateway closes its worker connection at once. The
worker sees the disconnect and stops the batch before its next question.
The question already running always finishes, and the model takes no
other request until it does. Both sides record the request as status 499.

If a native failure leaves a model's state uncertain, the worker stops
serving it and its `/health` returns 503, so the supervisor replaces the
worker. Telemetry records the model, timings and status, never the state,
questions or answers.

## Clients

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "decima-base", "state": "Our checkout is down",
  "questions": {"urgent": {"type": "noul", "instructions": "This needs action now."}}}'
```

```python
from muse.modalities.decision_evaluation import DecisionClient

out = DecisionClient().decide(model="decima-base", state="Our checkout is down",
                              questions={"urgent": {"type": "noul",
                                                    "instructions": "This needs action now."}})
print(out["answers"]["urgent"]["noul"])
```

Both clients check the request with the server's parser before sending it.
MCP's `muse_decide` tool requires `model`, `state` and `questions`, and
returns the full envelope. See also `examples/decisions.py`.

## Compatibility

Muse matches the documented TypeSafe request and answer shapes and
confidence formulas for the text subset above, as of 2026-10-18. The
documented examples are pinned in
`tests/modalities/decision_evaluation/fixtures/typesafe_text_subset.json`.

Muse does not claim:

- TypeSafe model equivalence or identical probabilities;
- TypeSafe's input sizes, media, Box or ranking support;
- billing semantics;
- an implicit default model.

## Live verification

Live tests are opt-in: `MUSE_REMOTE_SERVER=http://host:8000 pytest
tests/integration/test_remote_decisions.py`. Set
`MUSE_DECISION_MODEL_IDS` to choose which models to test.

### Live run, 2026-10-18

Run with `muse pull` against an isolated `MUSE_CATALOG_DIR`, then
`muse serve` from this tree.

Host:

- 4-core CPU, 31 GB RAM, RTX 3060 12 GB.
- The default llama-cpp-python 0.3.33 build has no GPU offload, so
  Decision's `auto` placement resolved to **CPU**. A CUDA build of
  llama-cpp-python is needed to use the GPU.

| | `decima-base` | `decision-0.8b-q8` |
|---|---|---|
| Pull (download, venv, probe) | 99 s | 434 s (most of it building llama-cpp-python) |
| Probe | passed: load 7.1 s, peak 0.95 GB (CPU) | passed: load 0.8 s, peak 1.00 GB (CPU) |
| First 3-question request, cold load included | 0.50 s | 5.41 s |
| Logical input tokens for that request | 139 | 374 |

What the run established:

- **Native logits.** Reading the final row through `get_logits_ith(-1)`
  matched llama.cpp's `logits_all=True` values exactly on the real
  GGUF (max absolute difference 0.0).
- **Contract tests.** `tests/integration/test_remote_decisions.py`
  passed 8 of 8:
  - complete, normalized distributions;
  - option reordering moves the keys with their options;
  - over-budget input is rejected rather than truncated;
  - a duplicate `model` key is refused.
- **Forced probe failure.** A same-size corrupted `decima.json` made
  `muse pull` exit 1 and leave the model disabled. Restoring the file,
  then running `muse models probe` and `muse models enable`, recovered it.
- **Lifecycle.**
  - Probing with `HF_HUB_OFFLINE=1` worked; the model reloads from local
    files only.
  - An unchanged re-pull exited 0.
  - `muse models remove --purge` removed the model's catalog entry, its
    venv and its unshared weights.

Both models chose `billing` and judged the duplicate charge true, but they
put the same message's urgency at different levels. This was a
handful of hand-written questions, so it says nothing about accuracy or
calibration.

### GPU run, 2026-10-19

The same host, with `install.llama_cpp_wheel: cu130` (see
[GPU builds of llama-cpp-python](CHAT_COMPLETION.md#gpu-builds-of-llama-cpp-python))
and `muse models refresh decision-0.8b-q8`, then `muse-local` restarted.
Decision's `auto` placement now resolves to **CUDA**. Decima-base is
ONNX and stays on CPU.

| | `decima-base` (CPU) | `decision-0.8b-q8` (GPU) |
|---|---|---|
| Probe | (unchanged) | load 1.1 s, peak 1.40 GB VRAM |
| Hot 3-question request through the gateway, median of 5 | 219 ms | 81 ms (CPU build: about 7.5 s) |
| 50 simultaneous requests | 33 served, 17 `queue_full`, 5.7 s | 33 served, 17 `queue_full`, 2.5 s |
| 16-question batch, run to completion | 35.4 s | 3.2 s |
| Next request after abandoning that batch at 0.3 s | 1.98 s | 0.23 s |

- **Same logits.** A scratch benchmark of the cu130 wheel produced the
  same option logits on GPU as on CPU (max difference 0.0).
- **Panel unchanged.** On the 12 hand-labelled questions, Decision matched
  12 and Decima 11, as on CPU. Decision changed one of five Choice answers
  when the options were reversed; Decima changed none.
- **Contract unchanged.** The error contract, MCP `muse_decide` and
  content-free telemetry all checked out again.
- **Abandoned batches stop.** Before 0.62.1, an abandoned 16-question
  request kept the model busy for its full run (189 s on the CPU build).

