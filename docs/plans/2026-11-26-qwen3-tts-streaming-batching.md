# Qwen3-TTS: streaming and batching (design, 2026-11-26)

Follows `2026-11-25-qwen3-tts-fast-inference-plan.md`, whose Phase 2 (micro-batching) and Phase 3 (sentence streaming) this replaces. State of `main` when written: `b839050`, one 12 Hz frame per CUDA graph, int8 layer weights, RTF ~0.12 (0.6B) / ~0.15 (1.7B) on an RTX 3060, one request at a time, audio returned as one chunk.

## Goal

Two independent switches, so four configurations can be compared on the same code:

| | one request at a time | up to N requests per frame |
|---|---|---|
| **audio at the end** | A: today | C: batching |
| **audio as it is generated** | B: streaming | D: both |

- Streaming is per request (`stream: true`, the SSE route that already exists).
- Batching is per model (`max_batch_size`, a hidden capability; 1 is today's behaviour).
- What each is for: streaming cuts the time to first audio for one listener. Batching raises throughput when requests overlap. Only D gives every one of several simultaneous listeners a fast start, because in B a stream holds the model until it ends.

## Measurements that shaped the design

Scripts: `ar/design_probe.py`, `ar/design_probe2.py` in the session scratchpad; 0.6B, int8, RTX 3060.

**A frame at 2-4 rows.** In-graph time of the int8 GEMVs, summed over a frame:

| rows | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 0.6B talker, 28 layers | 1.44 ms | 1.80 | 3.08 | 3.10 |
| code predictor, 5 layers x 16 positions | 4.13 ms | 5.14 | 8.81 | 8.87 |
| 1.7B talker, 28 layers | 4.28 ms | 5.21 | 7.45 | 7.55 |

Today's 0.6B frame is 7.2 ms, of which 5.6 ms is these GEMVs. So a frame for 2 lines should cost about 8.7 ms and for 4 lines about 14.5 ms: roughly 1.6x and 2x the throughput, with each line still 4x faster than playback. Three rows cost the same as four (the kernel rounds rows up to a power of two), and only the two-row tile configuration was ever tuned.

**The speech decoder.** It is causal throughout: a causal conv, an 8-layer causal transformer (sliding window 72) at the frame rate, then a conv stack that upsamples 1920x.

- Upstream-style chunking (decode `[start - L, end)` and drop the context) is unusable for small chunks: against the one-pass audio it reaches 13 dB SNR at L = 25 and 28 dB at L = 72. The transformer's memory is long.
- Split the decoder instead. *Front* = codes -> quantizer -> `pre_conv` -> transformer. *Conv stack* = everything after. `conv_stack(front(codes))` is bit-identical to `decoder(codes)`.
- The front is causal, so its output for frame t never changes once computed. The conv stack has a short receptive field: one frame of left context already reaches the noise floor.
- Streaming with the front over the frames so far and the conv stack over `[start - 4, end)` gives 36-37 dB against the one-pass audio. That is the decoder's own floor: decoding a prefix of the same codes in one pass differs from the full decode by the same 37 dB (bf16 rounding that depends on the length).
- Costs, eager: the front takes 25 ms whatever the length (launch overhead, not GPU time). The conv stack takes 7.6 ms for 8 frames, 10.7 for 16, 17.8 for 32, 80.6 for 181. Captured as a CUDA graph for a fixed shape it is 6.0 / 10.0 / 13.6 ms for 8 / 16 / 24 frames, all GPU time.
- Clone mode decodes the reference clip's codes in front of the generated ones and cuts that audio off. The narrator's reference is 49 frames; a 30 s reference would be 375, about 165 ms of conv stack thrown away per request. Only the front needs the reference.

## Shared pieces (both features use them)

1. **Prepare.** Upstream's prompt building (text tokenizing, embeddings, speaker and reference handling) stays upstream's. What the talker needs for one request is exactly the keyword arguments upstream passes to `talker.generate` (`inputs_embeds`, `trailing_text_hidden`, `tts_pad_embed`, sampling parameters, `max_new_tokens`), plus the reference codes for clone mode. `prepare(text, voice | instructions)` returns those as plain data. Suggested mechanism: call the upstream entry point with `FastTalker` in a capture mode that records the arguments and unwinds; a helper thread blocked inside upstream is the fallback mechanism. Either way the request becomes data that a loop we own can run.
2. **Stepper.** `FastTalker` already has prefill and a per-frame `_step`; expose them as start / step so a caller can interleave other work between frames. Upstream's post-processing is small and is reproduced: the codes are the frames before the first EOS.
3. **Split decoder.** One object per request: it takes frames as they arrive, keeps the front's hidden states (clone mode seeds it with the reference codes), and turns `[start, end)` into audio with a few frames of conv context. It also decodes a finished line in one call, skipping the reference's conv stack.
4. **Fallback.** Wherever the fast talker is not installed (CPU, unsupported qwen-tts, disabled after a failure), requests take today's path: one chunk, one at a time.

## Streaming

`synthesize_stream` becomes: prepare; prefill; then step frames, and whenever the chunk schedule says so, decode the new frames and yield an `AudioChunk`. One thread, under the lock the route already takes. The wire format does not change (SSE events of base64 PCM16, then `done`).

- **Chunk schedule.** Small first chunk, then larger: for example 4 frames (0.32 s of audio), then 8, then 16-24. Generation runs about 8x faster than playback, so after the first chunk the listener is always ahead and larger chunks only save decode overhead. Two hidden capabilities set the first and the steady size.
- **Expected time to first audio, 0.6B:** prefill ~58 ms + 4 frames ~29 ms + first decode ~31 ms eager, so about 120 ms in process against ~650 ms today. Untested.
- **Levers once it works, each a measured experiment:** capture the conv stack as a CUDA graph per chunk shape (6 ms instead of 7.6, and no launch jitter); make the front incremental with a KV cache so it stops costing 25 ms per chunk; skip the reference's conv stack in the non-streaming path too.
- **What cannot stream:** `timestamps`, `align`, non-WAV formats (the route already refuses these).

## Batching

One engine thread owns the GPU for a model. Request threads prepare, hand the engine a request, and wait on a queue.

- **Slots, not batches.** The talker state gets a leading row dimension. Each row has its own position, text cursor, repetition set, stop flag and KV length. A request joins at a frame boundary (its prefill runs alone, as today, into its row) and leaves at its own EOS, so a short line never waits for a long one and a new request never waits for the current ones to finish.
- **Frames.** One CUDA graph per (rows, KV capacity, sampling parameters), rows in {1, 2, 4}. With one active request the one-row graph runs, so a lone request costs what it costs today. Per-row KV writes, attention masks and RoPE positions replace the single shared position. The code predictor runs the same rows; its first step has two positions per row, which at 3-4 rows is past the kernels' four-row limit and has to be split or the kernel extended.
- **Only like requests share a frame.** Requests with different sampling parameters queue for the next compatible frame (muse always sends the same ones).
- **Decoding** runs on the engine thread between frames as jobs, a first chunk ahead of everything else. Voice saves and anything else that touches the GPU go through the same thread.
- **The route's lock.** `_inference_lock` becomes a `threading.Semaphore(max_batch_size)`, as `LlamaServerModel` does; the registry keeps a backend's own lock and the stream helper only needs `acquire(timeout=)` / `release()`.
- **Failure and cancellation.** A disconnected client frees its row at the next frame. An error in a frame fails the requests in it, disables the engine and returns the model to the one-at-a-time path.
- **Memory.** Each extra row is one KV cache (57 MB per 512 tokens on the 0.6B) plus a graph per row count; `memory_gb` is re-measured at the largest batch.

## Gates

Kept from the speed work: every existing gate in `ar/run.py`, bit-identical numerics with both switches off, unit tests.

Streaming:
- Same seed, same request: the streamed codes equal the non-streamed codes token for token.
- The concatenated chunks have exactly the non-streamed length and are within 3 dB of the measured floor (so >= 33 dB) of the one-pass decode of the same codes.
- Seams: the error in a 10 ms window around each chunk boundary is no worse than twice the median window.
- ASR errors and UTMOS over the 24-line set match the non-streamed arm within noise.

Batching:
- Rows are independent: under greedy decoding, a line's tokens are the same whatever shares its frame, and a row count of one reproduces today's tokens.
- Graphed equals masked-eager for each row count, as today.
- Teacher-forced along its solo trajectory, each row's KL against the one-row path is at the level of today's graph-versus-eager difference.
- A short line batched with long ones finishes within 1.5x of its solo time.
- ASR errors and UTMOS for batched takes match solo takes within noise.

## The comparison

In process, both models, warm, GPU otherwise idle, median of at least 5 runs:

- Workloads: one client, lines in sequence; 2, 4 and 8 simultaneous requests; a second request arriving one second into a first; one short line among long ones.
- Per configuration A-D: time to first audio (median, worst), time to last audio, audio seconds per wall second, the smallest margin by which a chunk beat its playback time, peak VRAM.
- Then one end-to-end run over HTTP against a throwaway server, to confirm the numbers survive the gateway.

## Not in this work

Sentence splitting, batching the conv stack across requests, 4-bit or calibrated quantization, a streaming decoder with carried conv state (the split already reaches the floor), opus/mp3 streaming.
