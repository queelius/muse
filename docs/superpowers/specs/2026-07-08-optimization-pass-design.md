# Optimization Pass Design (benchmark-first; muse-vs-ollama + modality sweep)

- **Date:** 2026-07-08
- **Status:** approved design, pre-implementation
- **Ships as:** v0.57.0 (fixes) + committed benchmark harness + before/after report

## Problem

Perceived and measured slowness relative to the runtime floor:

- LLM generation through muse is "much slower than ollama" (user report).
  Confirmed structural cause-candidates: `LlamaCppModel` constructs
  `Llama(...)` with only 5 kwargs (model_path, n_ctx, n_gpu_layers,
  chat_format, verbose) -- `n_threads` (llama-cpp-python default:
  cpu_count/2), `flash_attn` (default off), `n_batch`, `use_mlock`,
  KV-cache dtype are all unset defaults, while ollama tunes these per
  box. Multi-turn prompt-prefix reuse behavior is unverified.
- The other modalities (especially the anima-relevant ones: image gen,
  SFX/music, TTS) have never been latency-baselined; per-runtime knobs
  (whisper compute_type, diffusers steps/scheduler, batch sizes) are
  set once and never measured.

User-set success bar: **measure all, fix top wins** -- a rigorous
baseline report first, then ship only the fixes the numbers indict,
then re-measure and publish before/after.

## Non-goals

- Instrument-first tracing spans through gateway/worker (the v0.53-0.55
  telemetry already attributes queued_ms + latency_ms; enough for round
  one).
- Architectural changes (worker model, gateway design). This pass tunes
  within the existing architecture.
- Quality-affecting silent changes: any knob that changes OUTPUT quality
  (e.g. diffusion steps) is a per-curated-entry documented change, never
  a silent global default flip (no-surprises rule).

## Phase 1: benchmark harness (committed, reusable)

New `scripts/bench/` (stdlib + httpx only, no new deps):

### `scripts/bench/bench_llm.py`

Head-to-head on ONE box with the SAME model weights: ollama's
`qwen2.5:3b-instruct` (Q4_K_M) vs muse serving
`hf://bartowski/Qwen2.5-3B-Instruct-GGUF@q4_k_m` (identical quant/arch).
Primary target box: the 64GB/12-core CPU box (192.168.0.102, where
ollama 0.14.3 is installed). Scenarios, each 3 hot reps after 1 discarded
warmup, medians reported:

1. **short-gen**: ~20-token prompt, max_tokens 128, non-streaming ->
   generation tok/s (completion_tokens / elapsed).
2. **stream-ttft**: same prompt, streaming -> time-to-first-token and
   inter-token gap median.
3. **long-prompt**: ~1000-token prompt, max_tokens 64 -> prompt-eval
   throughput isolated (elapsed minus generation share).
4. **multi-turn**: 4-turn conversation with cumulative history ->
   detects prompt-prefix KV reuse (turn N prompt-eval time flat vs
   growing).
5. **muse-internal split**: scenario 1 against the WORKER port directly
   vs via the gateway -> isolates proxy/relay overhead.

Adapters: `OllamaClient` (POST /api/chat, stream and non-stream;
`eval_count`/`eval_duration`/`prompt_eval_duration` from the response
give ollama's own numbers), `MuseClient` (OpenAI-shape; usage block +
wall-clock). Output: `docs/benchmarks/<date>-llm.json` + a markdown
table.

### `scripts/bench/bench_modalities.py`

Per-modality representative request against a muse server (default
frodo, 192.168.0.204), hot (1 warmup + 3 reps, median wall-clock),
skipping models that are not enabled on the target:

| Modality | Model | Request |
|---|---|---|
| image/generation | sd-turbo | 512x512, default steps |
| image/generation (LoRA) | pixelartredmond-1-5v... | 512x512 sprite |
| audio/speech | kokoro-82m | ~12-word sentence |
| audio/speech (CPU) | supertonic-3 | same sentence |
| audio/transcription | whisper (smallest enabled) | ~10s wav the script generates at runtime via the target server's own TTS (no binary fixture in the repo) |
| embedding/text | minilm (or enabled default) | batch of 16 short strings |
| audio/generation | stable-audio-open-1.0 | 3s SFX prompt |
| chat/completion | smallest enabled GGUF | 64-token completion |

Output: `docs/benchmarks/<date>-modalities.json` + markdown table with
per-modality median latency + notes column (device, steps, etc.).

Both scripts: `--server URL`, `--json PATH`, `--md PATH`, `--reps N`;
resilient (a failing modality records `error`, never aborts the run).

## Phase 2: triage + fixes (evidence-gated)

Triage bar: ship a fix only if the harness shows **>= 20% improvement**
on the relevant scenario AND output quality is unchanged (or the change
is a documented curated-entry edit). Candidate buckets, pre-identified
but NOT pre-committed (the numbers decide):

1. **llama.cpp construction kwargs** (highest prior): forward
   `n_threads`, `n_threads_batch`, `n_batch`, `flash_attn`, `use_mlock`,
   `type_k`, `type_v` from manifest capabilities through the EXISTING
   load_backend kwargs merge into `Llama(...)`. Zero new plumbing: add
   explicit constructor params (or a vetted passthrough) in
   `LlamaCppModel`, defaults = today's behavior; tuned values land as
   catalog/curated capabilities per box (e.g. `n_threads: 12` for the
   32B on the CPU box). Explicit knobs, no auto-detection magic beyond
   llama.cpp's own defaults.
2. **Prompt-prefix reuse**: if multi-turn shows full re-eval per turn,
   evaluate llama-cpp-python's prefix-matching / LlamaCache options;
   scope any fix to a capability flag.
3. **Gateway overhead**: only if scenario 5 shows > ~5% delta; then
   profile the relay (httpx client reuse, chunk size).
4. **Whisper**: `compute_type=int8` on CPU targets if transcription
   baseline is slow (faster-whisper supports it natively; documented
   capability, default unchanged).
5. **Diffusers**: verify sd-turbo runs 1-4 steps as intended (it is a
   1-step model; if the default steps knob inflates it, fix the curated
   entry); LoRA fuse/unfuse overhead check.
6. **Embeddings/TTS**: batch-size + chunking checks only if baselines
   look off.

Each shipped fix: TDD where testable (kwargs forwarding gets unit tests
mirroring the n_gpu_layers precedence tests), plus a harness re-run.

## Phase 3: after-report

Re-run both scripts, produce `docs/benchmarks/<date>-after.md` with a
before/after table per scenario, commit alongside the fixes. Release
v0.57.0 gated on user go, deploy both boxes, spot-validate live.

## Deliverables

1. `scripts/bench/bench_llm.py` + `scripts/bench/bench_modalities.py`
   (committed, reusable for future regression checks).
2. `docs/benchmarks/` baseline + after reports (md + json).
3. The evidence-gated fixes (expected center of mass: llama.cpp kwargs
   as capabilities + tuned catalog values for the 32B/CPU box).
4. CLAUDE.md note pointing at the harness for future perf work.

## Testing

- Harness itself: unit-testable pure functions for stats (median,
  tok/s derivation) + JSON/markdown rendering; network paths exercised
  live (the harness IS the test).
- Kwarg forwarding fixes: unit tests in
  tests/modalities/chat_completion/ mirroring existing constructor
  tests (fake Llama capturing kwargs; capability -> constructor 
  passthrough; defaults preserve today's construction exactly).

## Rollout

Benchmarks run against live boxes (read-only + generation load;
schedule-friendly, no restarts needed). Fixes ship as v0.57.0 after the
before/after table exists. Both boxes redeployed; tuned capabilities
applied via catalog (32B on .102) or curated entries as appropriate.
