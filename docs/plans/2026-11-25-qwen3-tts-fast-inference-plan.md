# Plan: Make `qwen3-tts-0.6b-base` Fast (Sub-Real-Time Latency + Multi-Voice Micro-Batching)

**Date:** 2026-11-25 (revised after code review the same day)  
**Target Host:** RTX 3060 12 GB (`192.168.0.204:8000`, `torch 2.15.0+cu132`, `qwen-tts==0.1.1`, no `flash-attn` installed)  
**Goal:** Run fast, real-time NPC dialogue (`borrowed-stars-npc-qwen3.5-0.8b`) **and** expressive multi-character Qwen3-TTS voices (`qwen3-tts-0.6b-base`, 60 saved character voices) concurrently on a single 12 GB GPU with sub-real-time TTS latency (`RTF < 1.0x`) and high multi-NPC throughput.

---

## 1. Current Measured Baseline

Already verified live on this host (`main @ 239ebce`):

- **Co-residency fits comfortably in 12 GB VRAM**:
  - `borrowed-stars-npc-qwen3.5-0.8b` (`llama-server b12728` CUDA): **`690 MiB` VRAM**, **`0.20 s`** warm `/v1/completions` (GBNF + `cache_prompt: true`), 4 parallel slots.
  - `qwen3-tts-0.6b-base` (one loaded model serving all **60 saved character voices** under `~/.muse/voices/qwen3-tts-0.6b-base/`): **`2,882 MiB` VRAM**.
  - Combined: **`3,600 MiB` used / `8,311 MiB` free**.
- **Current Qwen3-TTS bottleneck** (via `POST /v1/audio/speech`):
  - Cold first call (`wispwood_nettle`): **`23.76 s`** wall for `2.00 s` audio (`RTF 11.88x`).
  - Warm cached prompt (`wispwood_nettle`): **`7.75 s`** wall for `2.32 s` audio (`RTF 3.34x`).
  - Warm longer ref voice (`wispwood_tallow`, 7.5 s ref clip): **`15.01 s`** wall for `3.36 s` audio (`RTF 4.47x`).
  - 2 concurrent TTS requests: **`6.87 s`** and **`12.19 s`** (strictly serialized by `model._inference_lock`).

### Exact Time Breakdown (Measured Inside `qwen3-tts-0.6b-base` Venv)

Profiled with `attn_implementation="sdpa"` on `wispwood_nettle` (`"The lanterns stay lit until dawn."`, `2.24 s` audio, 28 codec frames @ 12 Hz):

| Stage | Wall Time | Share | Root Cause |
| --- | ---: | ---: | --- |
| **`talker.code_predictor.generate()`** | **4.35 s** | **71.5%** | Calls HuggingFace `transformers`'s full Python `.generate()` loop (validation, logits processors, dynamic KV cache, plus **unused `output_hidden_states=True`**) for 15 fixed steps on *every single 12 Hz frame* (`~155 ms/frame`). |
| **`talker` backbone** | **1.64 s** | **27.0%** | Eager PyTorch Python dispatch (28 decode steps). |
| **`speech_tokenizer.decode()` (vocoder)** | **0.09 s** | **1.5%** | Already negligible. |
| **Total** | **6.08 s** | **100%** | `RTF = 2.71x` (vs `3.34x` without explicit `sdpa`). |

### Proven Multi-Voice Batching Win (Before Any Compile Work)

`qwen_tts.Qwen3TTSModel.generate_voice_clone` **natively accepts heterogeneous batches** (`text: list[str]`, `voice_clone_prompt: list[VoiceClonePromptItem]` — different character voices in the same GPU batch):

| Test (`sdpa`, ICL mode) | Wall Time | Total Audio | Effective RTF |
| --- | ---: | ---: | ---: |
| Batch = 1 (`wispwood_nettle`) | `5.96 s` | `2.08 s` | `2.87x` |
| **Batch = 3 mixed voices** (`wispwood_nettle` + `archivist_linnea` + `wispwood_tallow`) | **`12.06 s`** | **`8.64 s`** | **`1.40x`** (~3× faster than `36.3 s` sequential) |

Note on `x_vector_only_mode=True`: measured `6.51 s` for `2.08 s` audio (`RTF 3.13x`) — **not** faster than ICL (`5.96 s`) because the bottleneck is the per-frame `code_predictor.generate()` loop, not prefill length. Keep full ICL quality as the default.

---

## 2. Key Files & Constraints

- Runtime: `src/muse/modalities/audio_speech/runtimes/qwen3_tts.py`
- Model manifests:
  - `src/muse/models/qwen3_tts_0_6b_base.py`
  - `src/muse/models/qwen3_tts_1_7b_voicedesign.py`
- Speech routes (lock handling!): `src/muse/modalities/audio_speech/routes.py` (lines 364–387, 457–480)
- Unit tests: `tests/models/test_qwen3_tts.py`, `tests/modalities/audio_speech/test_routes_saved_voices.py`
- Upstream `qwen-tts` reference (in venv):
  - `~/.muse/venvs/qwen3-tts-0.6b-base/lib/python3.12/site-packages/qwen_tts/inference/qwen3_tts_model.py` (lines 470–625)
  - `~/.muse/venvs/qwen3-tts-0.6b-base/lib/python3.12/site-packages/qwen_tts/core/models/modeling_qwen3_tts.py` (lines 1665–1695)
- **Important repo invariants**:
  - Keep heavy imports (`torch`, `qwen_tts`) deferred inside `_ensure_deps()` in `qwen3_tts.py` (unit tests replace module-level `torch` / `QwenTTS` sentinels).
  - Any new capability keys added to `MANIFEST["capabilities"]` must be classified in `src/muse/core/capability_keys.py` (`tests/core/test_capability_keys.py`).
  - `tests/core/test_runtime_helpers_meta.py` checks runtime helper reuse (`select_device`, `dtype_for_name`, `set_inference_mode`, `LoadTimer`).

---

## 3. Implementation Workstreams

### Phase 1: Single-Request Latency — SDPA + Fast Compiled `code_predictor` (Biggest Win)

**Target:** Cut `code_predictor` from `~155 ms/frame` down to `< 15 ms/frame`, bringing total single-line latency from `6–8 s` down to **`~1.5–2.2 s` (`RTF <= 1.0x`)**.

1. **Enable SDPA explicitly on CUDA**:
   - In `Qwen3TTSRuntime.__init__`, pass `attn_implementation="sdpa"` in `load_kwargs` when `resolved.startswith("cuda")` (with a clean fallback if `from_pretrained` raises `TypeError` in tests/fakes, or check capability/config).
2. **Replace per-frame `code_predictor.generate()` with a fast 15-step static-KV-cache loop**:
   - Inspect `modeling_qwen3_tts.py` around line 1671 (`self.code_predictor.generate(...)`) and `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward` (around line 1250–1320).
   - Notice `code_predictor` **always** generates exactly `num_code_groups - 1` (15) tokens from a 2-token prefill `[past_hidden, last_id_hidden]`, with fixed max sequence length `2 + 15 = 17`! And `output_hidden_states=True` is completely unused by the caller.
   - **Each of the 15 steps uses its own weights**: step `k` reads input embedding table `get_input_embeddings()[k - 1]` and output head `lm_head[k]` (`modeling_qwen3_tts.py:1279`, `:1298`, tracked via `generation_steps`). A static / CUDA-graph loop must therefore be **unrolled over 15 fixed steps**, not a single generic step function.
   - Implement a small, self-contained fast path (monkey-patched onto `self._tts.model.talker.code_predictor.generate` or a wrapper installed in `Qwen3TTSRuntime.__init__` when running on CUDA, opt-outable via capability/env for debugging):
     - Pre-allocate or reuse a tiny fixed-shape KV cache `(batch, num_kv_heads, 17, head_dim)` (or static cache) so there is zero `torch.cat` allocation and zero HF `.generate()` Python overhead per step.
     - Preserve exact sampling semantics (`subtalker_dosample`, `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p`) and return a lightweight namespace with `.sequences` matching `predictor_result.sequences`.
     - **Sampling equivalence**: HF `.generate()` applies temperature -> top-k -> top-p (defaults `do_sample=True, top_k=50, top_p=1.0, temperature=0.9`). Greedy must match upstream token-for-token; sampled output is checked distributionally / with a fixed seed.
     - **Version guard**: `qwen-tts==0.1.1` is pinned, but the patch checks the installed version and the shape of the patched module, and falls back to upstream `.generate()` (with a warning) on mismatch so a bump never silently breaks synthesis.
     - Compile the 15-step predictor (or its inner step/forward) with `torch.compile(..., mode="reduce-overhead")` **on static buffers** (note: compiling `code_predictor.forward` with dynamic `past_key_values` failed with `CUDAGraphs tensor output overwritten`; using a pre-allocated static KV buffer + `torch.compiler.cudagraph_mark_step_begin()` avoids that completely — or fall back gracefully to `mode="default"` / eager if compile fails).
   - **Also profile `talker` backbone after `code_predictor` is fast**: once `code_predictor` drops from `4.35 s` → `~0.3 s`, the `talker` backbone (`1.64 s`) becomes the new #1 item. Check if `torch.compile` on `talker.model` (with `mode="default"` or static KV cache via `cache_implementation="static"`) drops `talker` from `1.64 s` → `< 0.6 s`!
   - **CUDA graphs are batch-size specific**: once Phase 2 batches requests, every new batch size recompiles / re-captures. Pad batches to a small set of bucket sizes (1/2/4/8) or warm each bucket at load.
   - **Warmup compile at load time** (optional short dummy forward or lazy on first request — document cold-start tradeoff, keeping worker startup well under `server.worker_ready_timeout_seconds` (300 s) and the probe timeout (600 s)).
   - **VRAM accounting**: CUDA-graph memory pools and static KV buffers add VRAM. Re-run `muse models probe qwen3-tts-0.6b-base` after Phase 1 and update the hand-set `memory_gb: 3.0`.
   - **Manifest text**: update the `limits` line ("Slow: ~10 s per short line ...") to the measured numbers.
3. **Size the prompt cache from `limits.max_saved_voices`** (instead of the hard-coded `PROMPT_CACHE_SIZE = 32`):
   - Cache capacity = `max(PROMPT_CACHE_SIZE, limits.max_saved_voices + bundled voices)`, so every saved voice (60+ Borrowed Stars characters) stays cached once touched (each `VoiceClonePromptItem` is tiny — speaker embedding + 12 Hz ref codes). A hard-coded 128 would silently fall behind if the voice limit is raised.

---

### Phase 1 status (2026-11-25): done, talker backbone included (option b)

Implemented in `src/muse/modalities/audio_speech/runtimes/qwen3_tts_fast.py` (new) and `qwen3_tts.py`:
SDPA on CUDA; `FastCodePredictor` (unrolled 15-step static-KV decoder reusing the predictor's own modules, HF-order sampling via Gumbel-max, one CUDA graph per batch shape / sampling params, capped at 16); `install()` gated on qwen-tts 0.1.1 plus a layout check, upstream fallback for any other call shape, and a process-wide switch-off on unexpected failure (OOM re-raised); constructor kwargs `fast_code_predictor` / `cuda_graphs` (default on, graphs CUDA only); prompt cache sized `limits.max_saved_voices + bundled`.

Measured on this host (RTX 3060, i5-4460):

| | Before | After |
| --- | ---: | ---: |
| code predictor per frame | 152.9 ms | 15.9 ms (eager static path alone: 124 ms; the host is CPU-launch-bound) |
| greedy token match vs upstream on real inputs | n/a | 480/480 (0.6B), 450/450 (1.7B) |
| 0.6B short line, in process | 6.64 s / 2.48 s audio (RTF 2.68) | 2.46 s / 3.04 s audio (RTF 0.81) |
| 0.6B short line, HTTP warm | 7.75 s | ~2.3 s / ~2.4 s audio |
| 0.6B 3-sentence line, HTTP warm | n/a | 5.04 s / 5.36 s audio (RTF 0.94) |
| 0.6B cold first request (spawn + load) | 23.76 s | 15.4 s |
| 1.7B VoiceDesign short line | ~11 s / ~2.8 s audio | ~3.6-4.8 s / ~2.6-3.7 s audio (RTF ~1.3) |
| peak VRAM (0.6B, torch max allocated) | 2.5 GB | 2.29 GiB |

ASR round-trip (`qwen3-asr-1.7b`) transcribes every fast-path output exactly. `muse models probe` was not re-run (it refuses while the live server has the model loaded); `memory_gb: 3.0` still covers the measured peak.

**Talker backbone: chose (b), a hand-written CUDA-graphed talker.** With the predictor fast, `talker.model` was ~70% of wall time (~50 ms per 12 Hz step). Options considered:
- Whole-module `torch.compile`: short line 2.33 s -> 1.22 s, but ~290 s first compile on a cold inductor cache and ~100 s warm, paid on every cold load (the compiled code dies with the worker on eviction); too close to the 300 s `worker_ready_timeout_seconds`. Regional per-layer compile gave no gain (recompiles per `layer_idx`). Not shipped.
- (b) `FastTalker` (same module), patched over `talker.generate` by `install_talker()` (after `install()`; constructor kwarg `fast_talker`, default on). Batch 1 only (anything else, a padded mask, an unknown kwarg or a non-default talker `generation_config` uses upstream). Eager prefill, then one CUDA graph per 12 Hz frame: predictor decode, codec-embedding sum + trailing text, one talker token over a static KV cache (512 tokens, doubling on demand; graphs recaptured after growth in a fresh memory pool), transformers' processors in order (repetition penalty, min new tokens, suppress tokens, then temperature / top-k / top-p) and sampling. It returns the `hidden_states` tuple qwen-tts reads.

Results (RTX 3060, nothing else running; Phase 1 = fast predictor only):

| | Phase 1 | + fast talker |
| --- | ---: | ---: |
| exact (eager) greedy vs upstream, real inputs | n/a | 149/149 + 149/149 frames, hidden states bit-identical (0.6B); 35/35 (1.7B) |
| graphed frame vs the same frame run eagerly | n/a | 59/59 identical (capture is faithful) |
| graphed vs exact, teacher-forced 60 steps | n/a | max logit diff 0.375 (3 bf16 steps), mean KL 7e-4, 1 argmax flip at a 0.125 tie |
| talker per frame | 70-110 ms | 25-30 ms |
| 0.6B short line, in process warm | 2.33 s / 2.56 s audio | 0.95 s / 2.16 s audio (RTF 0.44) |
| 0.6B paragraph, in process warm | 10.48 s / 10.3 s audio | 3.44 s / 11.0 s audio (RTF 0.31) |
| 1.7B short line, in process warm | 3.95 s / 3.0 s audio | 0.95 s / 3.1 s audio (RTF 0.31) |
| 1.7B paragraph, in process warm | 16.8 s / 12.1 s audio | 5.0 s / 14.2 s audio (RTF 0.35) |
| HTTP warm, 0.6B saved voice / 1.7B design | n/a | ~1.9 s / 6.1 s audio; ~2.0 s / 5.3 s audio |
| first request after load (graph capture) | n/a | ~+1 s |
| peak VRAM (torch max allocated) | 2.29 GiB (0.6B) | 2.57 GiB (0.6B), 4.42 GiB (1.7B) |

ASR (`qwen3-asr-1.7b`) transcribed every HTTP output exactly (8 short lines, a 34 s and a 68 s paragraph). The 68 s paragraph found a bug, now fixed: after cache growth the recapture reused the dropped graphs' pool handle and failed an allocator assert; the fallback served the line through upstream (slowly, correctly). The remaining per-frame cost is mostly the predictor's ~1500 tiny kernels (16 ms); fusing them would need a compiler.

### Kernel hill-climb (2026-11-26): frame 22.3 -> 11.5 ms, decoder -20%

Karpathy-style autoresearch on branch `autoresearch/qwen3-tts-speed`: one change per experiment, kept only if the metric improved beyond noise (~0.3%) and every gate passed. Metric: CUDA-event GPU time of one graphed 12 Hz frame (immune to host load), timing the default-sampling graph (the first eleven rows timed the greedy one; the harness was fixed and the baseline re-measured). Gates: exact eager talker == upstream token for token; graphed == masked-eager; talker teacher-forced KL <= 1.1e-3; code predictor KL against an fp32 reference <= 1.5x upstream bf16's own KL (its floor) under teacher forcing; graphed predictor == eager; decoded waveform == upstream's on the same codes; unit tests. A harness run is marked contaminated if any other GPU process appears during it.

| experiment | frame GPU ms | kept |
| --- | ---: | --- |
| baseline (option b, main) | 22.28 | |
| fuse q/k/v and gate/up GEMMs (parameters become views of the fused weight, no extra memory) | 20.73 | yes |
| SDPA `enable_gqa` instead of repeated K/V | 19.64 | yes |
| Triton RMSNorm (strided per-head rows) | 16.42 | yes |
| fused per-head RMSNorm + RoPE | 14.70 | yes |
| Triton decode attention over the live keys only (length read on device) | 13.70 | yes |
| fused SiLU * up | 13.48 | yes |
| fused residual add + next RMSNorm | 13.31 | yes |
| Triton GEMV for one-row projections (cuBLAS's M=1 GEMV ran 66-89% of bandwidth) | 12.11 | yes |
| up-to-4-row GEMV in the same kernel | 19.95 (contended) | no: M=2 no faster, M=1 numerics moved |
| decode attention for the predictor's one-token steps | 11.88 | yes |
| re-measured on the sampling graph | 12.70 | |
| one-kernel temperature + top-k + Gumbel-max sampler (k-th largest by binary search on order-preserving int keys; torch's Exp(1) noise, so RNG stays graph-safe) | 11.96 | yes |
| separate few-row GEMV for the predictor's two-token step | 11.91 | yes |
| K/V written into the caches by the norm + RoPE kernel (~200 fewer kernels) | 11.57 | yes |
| SiLU fused into the down-projection GEMV input | 11.54 | no: -0.25%, noise level |
| one-kernel SnakeBeta in the speech decoder (bit-identical) | 11.55; decode 115.0 -> 92.1 ms | yes |

Final state (RTX 3060, in process, warm): 0.6B short line ~0.5 s, RTF ~0.17-0.18; 1.7B frame 17.6 ms, short line ~0.8 s, RTF ~0.24, decode 88.6 -> 72.7 ms; both pass every gate (1.7B predictor KL 8.4e-4 vs floor 8.5e-4, talker KL 4.6e-4). Peak VRAM (torch max allocated): 0.6B 2.33 GiB short line / 2.80 GiB 25 s paragraph, 1.7B 4.03 / 4.70 GiB. ASR (`qwen3-asr-1.7b`, six lines per model and path): fast-path WER equals upstream's (1/81 on both models; the one miss everywhere is "traveller" spelled "traveler"), except one 0.6B fast draw heard "roots" as "routes" (2/81), within sampling variation.

Where the frame goes now: GEMV ~10.3 of 11.5 ms at ~92% of the 360 GB/s floor (the predictor re-reads its 154 MB of weights 15 times per frame). Kernel-level work is near its end; the next large lever is fewer bytes (weight-only int8 would roughly halve GEMV time), which changes numerics and needs a quality decision. Lessons: `tl.sin` / `tl.exp` are fast approximations (use libdevice for bit-exactness), and Triton's default FP fusion contracts a multiply-add across explicit bf16 round trips, so a kernel that must match upstream's rounding needs `enable_fp_fusion=False`.

### int8 layer weights (2026-11-26): frame 11.6 -> 7.2 ms, opt-in

Branch `autoresearch/qwen3-tts-int8`. `int8_weights=True` quantizes every talker and code-predictor layer projection (fused q/k/v, o, fused gate/up, down) per output row (symmetric, row absmax -> 127) at the first graphed frame, from the loaded bf16 checkpoint; the GEMV kernels gained a `SCALED` variant that reads int8 tiles and multiplies each output by its row scale. Output heads, the eager prefill and every fallback keep the float weights. Off, the kernels compile to the same code as before (both KL gates reproduced to the last digit).

| | bf16 | int8 layers |
| --- | ---: | ---: |
| 0.6B frame GPU ms / RTF | 11.57 / 0.176 | 7.23 / 0.120 |
| 1.7B frame GPU ms / RTF | 17.61 / 0.243 | 10.19 / 0.148 |
| 0.6B talker KL vs exact / predictor KL vs fp32 (bf16 floor 1.52e-3) | 7.5e-4 / 1.44e-3 | 9.4e-4 / 4.45e-3 |
| 1.7B talker KL / predictor KL (floor 8.5e-4) | 4.6e-4 / 8.4e-4 | 8.9e-4 / 2.97e-3 |
| 0.6B ASR errors / UTMOS, 48 takes | 2/628 / 4.465 | 2/628 / 4.450 |
| 1.7B ASR errors / UTMOS, 48 takes | 2/628 / 4.396 | 2/628 / 4.390 |
| 0.6B peak GiB, short line / 25 s paragraph | 2.33 / 2.80 | 2.83 / 3.31 |
| 1.7B peak GiB | 4.03 / 4.70 | 5.59 / 6.09 |

Gates for int8 runs: talker KL <= 1.1e-3 as before; predictor KL <= 4x its bf16 floor; the rest unchanged. Perceptual check: 24 lines x 2 seeds per arm and model, `qwen3-asr-1.7b` word errors (all four are "traveller" heard as "traveler") and UTMOS (paired differences -0.014 +/- 0.011 and -0.006 +/- 0.015).

What was tried: int8 heads too (talker KL 1.25e-3, over its gate, for 0.08 ms: no). Offline on the real weights, per-row int8 sits at ~41 dB SNR; finer scale groups (256 / 128 / 64 columns) gain 1.2 / 1.9 / 2.7 dB and row-and-column equilibration nothing, because the weights are close to Gaussian. Getting meaningfully closer needs calibration data (GPTQ-style rounding), i.e. a published quantized checkpoint.

The table above is the first form, with the int8 copies beside the float weights (+0.5 GiB on the 0.6B, +1.4 GiB on the 1.7B, float prefill). A blind listening test on it (owner, 14 pairs, 8 mixed pairs judged): bf16 preferred in 3, int8 in 3, no difference in 2; one take preferred in 3 of the 4 bf16-only control pairs.

### int8-only storage (2026-11-26): memory falls instead of rising

`_FusedLayer.quantize` now stores the int8 and drops the float weights: each bias-free projection module gets its rows of the fused int8 matrix, `weight = None`, and a `forward` that computes from them, so upstream's own calls of the modules (prefill, larger batches, fallbacks) keep working. Few-row products use the int8 GEMV; anything else dequantizes per call with a one-pass Triton kernel into `F.linear`. Decoders quantize at construction, so every request computes from the same weights. Quantization runs in 1024-row chunks to keep the load-time temporaries small.

| | bf16 | int8 only |
| --- | ---: | ---: |
| 0.6B frame GPU ms / RTF, 48 takes | 11.56 / 0.174 | 7.22 / 0.123 |
| 1.7B frame GPU ms / RTF, 48 takes | 17.61 / 0.244 | 10.20 / 0.154 |
| 0.6B prefill, 72 tokens | 41.0 ms | 57.6 ms |
| 0.6B talker KL vs a float copy / predictor KL vs fp32 | 7.5e-4 / 1.44e-3 | 1.87e-3 / 4.45e-3 |
| 1.7B talker KL / predictor KL | 4.6e-4 / 8.4e-4 | 1.37e-3 / 2.97e-3 |
| 0.6B ASR errors / UTMOS, 48 takes | 2/628 / 4.465 | 2/628 / 4.460 |
| 1.7B ASR errors / UTMOS, 48 takes | 2/628 / 4.396 | 2/628 / 4.406 |
| 0.6B peak GiB: load / short line / 25 s paragraph | - / 2.33 / 2.80 | 2.0 / 1.85 / 2.34 |
| 1.7B peak GiB: load / short line / 25 s paragraph | - / 4.03 / 4.70 | 3.94 / 2.64 / 3.32 |

Cost of dropping the floats: the prefill is int8 too, so the talker's KL against a float copy rises from 0.9e-3 (float prefill) to 1.4-1.9e-3, above the 1.1e-3 gate used for the float path; the int8 gate is 2.5e-3, set after ASR and UTMOS came back unchanged (paired UTMOS differences -0.005 +/- 0.013 and +0.009 +/- 0.017). The exact path still equals upstream token for token (both now read the int8 layers), and graphed still equals masked-eager. The process holds no float reference any more: the harness deep-copies the talker and predictor before install for its KL gates. The prefill's extra 17 ms is mostly the launch overhead of ~200 dequantize kernels; dequantizing the fused q/k/v and gate/up once each would roughly halve it.

---

### Phase 2: Dynamic Multi-Voice Micro-Batching (Concurrency Win)

**Target:** Allow 2–8 simultaneous `/v1/audio/speech` requests (for the *same* or *different* saved character voices) to execute in **one** GPU `generate_voice_clone` batch instead of serializing one by one.

1. **Critical lock detail in `routes.py`**:
   - Currently `src/muse/modalities/audio_speech/routes.py:370` wraps `model.synthesize(...)` in `with model._inference_lock:` (and `routes.py:466` passes `lock=model._inference_lock` to `threaded_sse_response`).
   - If `model._inference_lock` is a real mutex, concurrent HTTP threads block *outside* `synthesize()` and can **never** be coalesced into a batch!
   - Clean solution (zero impact on Kokoro or other backends):
     - **Not** `contextlib.nullcontext()`: `threaded_sse_response` calls `lock.acquire(timeout=...)` / `lock.release()` (`_common/streaming.py:136`), which a null context lacks, so streaming would break.
     - Follow the existing precedent instead: `LlamaServerModel` sets `self._inference_lock = threading.Semaphore(parallel)` (`chat_completion/runtimes/llama_server.py:105`), and `ModalityRegistry.register` preserves a backend-declared lock (`core/registry.py:63`). Set `self._inference_lock = threading.Semaphore(max_batch_size)`: routes stay unchanged, the semaphore bounds the admission queue, and an internal `_gpu_lock` (held only by the batch runner) owns the GPU.
     - Keep `save_voice` / `delete_voice` / `_clone_prompt` thread-safe under `_prompts_lock` (already locked).
2. **Micro-batcher design in `Qwen3TTSRuntime`** (clone mode, and optionally design mode):
   - Maintain a small queue + condition variable + single GPU worker thread (or leader-follower batching inside `synthesize()`):
     - First arriving request takes the GPU immediately (or waits a tiny coalesce window, e.g., `10–25 ms`, configurable via capability `batch_window_ms` / `max_batch_size`, default `max_batch_size=4` or `8`).
     - Better yet: **zero-latency-penalty leader/queue pattern** — when the GPU is idle, start immediately (batch=1, zero added wait!); any requests that arrive *while* the current batch is running accumulate in the queue and all launch together as the next single GPU batch (up to `max_batch_size`)! Optionally add a tiny `10 ms` grace window when batching is active if bursts arrive slightly staggered.
     - For `mode == "clone"`: resolve each queued item's `_clone_prompt(voice)` (each returns a 1-element `list[VoiceClonePromptItem]`), unpack `[p[0] for p in prompts]`, and call:
       ```python
       wavs, sr = self._tts.generate_voice_clone(
           text=texts,
           language=[None] * len(texts),  # or "Auto"
           voice_clone_prompt=prompt_items,
       )
       ```
     - Verify edge cases:
       - If one item in a batch has an invalid voice name, validate `self._store.get(voice)` **before** enqueuing so bad requests fail fast without poisoning valid peers in the batch.
       - If `generate_voice_clone` fails mid-batch (e.g. OOM on a huge batch), fall back to single-item execution or propagate cleanly to the affected future(s).
       - **Batch latency coupling**: the talker loop runs until **every** item emits EOS (`modeling_qwen3_tts.py:2284`), so a short line (`"Hello!"`) batched with a three-sentence line waits for the long one. Throughput improves but per-request latency of short lines regresses — which is what an NPC bark feels. Bucket by input length (or cap the length ratio within a batch), and add a per-request acceptance check that a short line's wall time is not much worse batched than solo.
       - Check variable-length output trimming (upstream trims per item via `effective_lengths` and cuts the reference prefix per item, `qwen3_tts_model.py:626`; still verify live): verify `qwen_tts` properly trims padding/EOS per item in `wavs[i]` when batching a short line (`"Hello!"`) with a long line (3 sentences) in the same batch! (Write a test/live check specifically for this!)
       - Clean shutdown (`close()` / daemon worker thread) so tests and worker teardowns never hang.
   - **`FastTalker` is batch 1 only** (Phase 1): a batched `generate_voice_clone` call falls back to the upstream talker loop (~70-110 ms per frame instead of ~25-30 ms), so batching only wins once `FastTalker` handles batch > 1: left-padded prefill with a key-padding mask (and unmasked fully-padded query rows, or pad K/V becomes NaN), per-row RoPE offsets (`rope_deltas`), per-row finished flags padding with EOS, and graphs keyed by batch size. Measure batch-4 throughput against four batch-1 runs at ~1 s each before building the batcher; at RTF ~0.3 serial may already suffice.
   - Check gateway/worker concurrency limits:
     - Ensure `qwen3-tts-0.6b-base` does **not** cap `max_concurrency: 1` (currently unset -> defaults to `0` = unlimited in `LoadDirector`, which allows concurrent requests to reach the worker's micro-batcher).

---

### Phase 3 (Optional Follow-Up if Needed): Chunked / Sentence Streaming or `flash-attn`

Only pursue if Phase 1 + Phase 2 don't already hit the desired latency budget:
1. **Sentence-level streaming in `synthesize_stream`**: Split multi-sentence text into sentences and yield each sentence's audio chunk as soon as it finishes (or batch the sentences on GPU) so time-to-first-audio (TTFA) for multi-sentence lines drops sharply.
2. **Prebuilt `flash-attn` wheel** (for `torch 2.15 + cu132 + sm_86` RTX 3060) if SDPA leaves any attention gap, though on short sequence lengths (`< 200` tokens) PyTorch SDPA + `torch.compile` usually matches or beats `flash-attn`.
3. **Optional `qwen3-tts-1.7b-voicedesign` parity**: Both `0.6b-base` and `1.7b-voicedesign` share `Qwen3TTSRuntime`, so Phase 1 (`sdpa` + fast `code_predictor`) automatically speeds up VoiceDesign too!

---

## 4. Verification & Acceptance Criteria

### Unit Tests
```bash
pytest tests/models/test_qwen3_tts.py tests/modalities/audio_speech/ tests/core/test_capability_keys.py tests/core/test_runtime_helpers_meta.py -q
```
Add regression tests for:
- `attn_implementation="sdpa"` forwarded on CUDA (not on CPU),
- fast `code_predictor` numerical/shape compatibility (greedy & sampled),
- multi-voice micro-batching coalescing concurrent `synthesize()` calls across different voices into a single `generate_voice_clone(text=[...], voice_clone_prompt=[...])` call,
- bad voice rejection before batch enqueue (doesn't poison sibling requests),
- prompt cache holding every saved voice (capacity follows `limits.max_saved_voices`),
- `_inference_lock` is a `threading.Semaphore` (acquire/release with timeout works for the streaming path),
- fast path falls back to upstream `.generate()` on a version / shape mismatch.

### Live Benchmark on RTX 3060 (`192.168.0.204:8000`)
After reloading `muse-local.service`:
1. **Single warm voice latency**: `POST /v1/audio/speech` (`qwen3-tts-0.6b-base`, `wispwood_nettle`, `"The lanterns stay lit until dawn."`) — target **`< 2.0 s` wall (`RTF < 1.0x`)** (down from `7.75 s`).
2. **Audio quality sanity check**: Save WAVs before/after optimization and verify clean speech, proper EOS trimming (no trailing garbage/silence on short lines batched with long lines), and faithful voice cloning across `wispwood_nettle`, `wispwood_tallow`, and `archivist_linnea`.
3. **Concurrent NPC SLM + multi-voice TTS stress test**:
   - Fire **4 concurrent `borrowed-stars-npc-qwen3.5-0.8b` `/completion` calls (GBNF + `cache_prompt: true`) + 4 concurrent `qwen3-tts-0.6b-base` calls across 4 distinct character voices** simultaneously.
   - Verify all 4 NPC turns finish in `< 0.5 s` warm, all 4 TTS lines complete in a single batched pass (much faster than `4 × single`), and total GPU VRAM stays well under `10 GiB`.
4. **Short-line latency under batching**: a short line batched with a long line must not take much longer than the short line solo (see "Batch latency coupling").
