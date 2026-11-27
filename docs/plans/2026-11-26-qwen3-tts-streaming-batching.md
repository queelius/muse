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

## Addendum, 2026-11-26: the target workload (changes the batching stage)

The owner's use: an RPG where as many NPCs as possible speak at once in real time (a crowded city), on one 12 GB GPU that also runs the small language model writing their lines, speech-to-text for the player, and other real-time work. The clone model (0.6B, saved character voices) is the one that matters; the 1.7B designs voices offline.

What that changes:

- **The metric is capacity, not speed.** A line only has to stay ahead of playback (1x). Report the largest number of simultaneous streams that all keep a positive playback margin, the GPU milliseconds spent per second of audio, and VRAM per added stream. Single-stream RTF is secondary.
- **Aim for 16-32 rows, not 4.** A first, contaminated probe (`ar/rows_probe.py`; another process was on the GPU, so re-measure) suggests the int8 GEMV kernel is the right path up to about 4 rows and stops scaling after (about 2 ms of layer products per line at 8-16 rows), while float weights through cuBLAS cost about the same for 1 row as for 16 (roughly 17-20 ms per frame), so about 1 ms per line at 16 rows and under 1 ms at 32. Dequantizing int8 per call is far slower than either. So the weight path should depend on the row count, which means keeping float layer weights (+0.5 GiB on the 0.6B) or writing a real int8 GEMM. Decide by measurement on an idle GPU.
- **Pace to playback.** Generating at 8x real time takes GPU time from the language model for no benefit. The engine should keep each stream a bounded lead ahead of its playback (a second or two), serve the stream nearest to running dry first, and otherwise leave the GPU idle.
- **Decode across streams together.** At 16 streams the decoder's per-call overhead (25 ms of launches for the front) dominates on one engine thread; the front and conv stack need to run batched over the streams that are due, and the front incrementally.
- **Measure under contention.** muse runs each model in its own process, and separate processes on one GPU take turns unless NVIDIA MPS is on. Capacity must be measured with a language model generating in another process, with and without MPS.
- **Text can arrive while speaking.** In clone mode the talker reads one text token per frame (12.5 per second), so a line can start once its first few tokens exist. Keep each row's text buffer appendable so a later stage can feed it from the language model's token stream; no API for that in this stage.

## Stage 1 results: streaming (2026-11-26)

Branch `worktree-agent-a72e3ef4085d52583`. RTX 3060, in process unless said otherwise, both manifests as shipped (`int8_weights`). Every run below was checked for other processes on the GPU; the ones that were contaminated are marked or were re-run.

### What was built

- **Prepare** (`FastTalker.capture` / `request`): the qwen-tts entry point (`generate_voice_clone` / `generate_voice_design`) runs as for the one-call path until it reaches `talker.generate`, which raises a private `BaseException` carrying its kwargs; `request` turns them into a `TalkerRequest` (plain data: `inputs_embeds`, `trailing_text_hidden`, `tts_pad_embed`, `_TalkerParams`, `max_new_tokens`, `ref_codes`). qwen-tts 0.1.1 has no `try` between its entry points and the talker call; the unwind is a `BaseException` so a broad `except Exception` added later cannot swallow it. Capture is per thread. The suggested helper-thread fallback was not needed.
- **Stepper** (`FastTalker.start` / `step`, `TalkerRun`): the same prefill and per-frame graphs, driven from a caller's loop. `generate` itself now runs on the same `_start` / `_advance` pair, with the same tensor operations in the same order. One decode state per talker: a later start retires earlier runs (a stale run raises).
- **Split decoder** (`runtimes/qwen3_tts_stream.py`): `SplitDecoder.front` / `conv_stack` (both take several streams along the leading axis), `StreamDecoder` per request (front over every frame so far, reference included; conv stack over the new frames plus 6 frames of context), `SplitDecoder.decode(codes, skip=)` for a whole line without the reference's conv stack. On CUDA each half runs as a CUDA graph per padded shape: codes padded at the end to 64, 128, 256... frames for the front, widths rounded up to 4 frames for the conv stack (both causal, so padding at the end leaves the real frames' outputs alone).
- **Streaming** in `Qwen3TTSRuntime.synthesize_stream`, one thread, under the route's lock. Chunk schedule: `stream_first_frames` (2) then each chunk twice the last up to `stream_chunk_frames` (16); both hidden capabilities. CPU, another qwen-tts and a fast path disabled after a failure keep the one-chunk stream; a failure before the first chunk falls back to it for that request (talker failure disables the fast talker, decoder failure disables streaming), after a chunk it ends the stream with the route's error event.

### Gates

| gate | 0.6B (clone, `wispwood_nettle`) | 1.7B (design) | pass |
| --- | --- | --- | --- |
| `run.py` / `run17.py`, `INT8=1` (all existing gates) | pass; frame 7.231 ms, talker KL 1.87e-3, cp KL 4.45e-3 (as before) | pass; frame 10.196 ms, KL 1.37e-3 / 2.97e-3 | yes |
| one-call path vs `main`, 16 takes, same seeds | codes identical in all 16; audio bit-identical in 14 | codes identical; audio bit-identical in 2 | yes (see below) |
| streamed codes == one-call codes, same seed, 48 takes | 48 / 48 | 48 / 48 | yes |
| stream length == one-call length | 46 / 48; 2 are 1 sample shorter | 48 / 48 | no, by one sample (see below) |
| SNR vs one-pass decode of the same codes, min / median | 33.8 / 40.6 dB | 34.4 / 38.8 dB | yes (>= 33) |
| seams as worded: boundary 10 ms window / the take's median window | max 10.7 (27% of all windows exceed 2x) | max 9.6 (19%) | no, see below |
| seams, local: boundary window / its 8 neighbours | median 1.01; 3.2% over 2x vs 4.6% of non-boundary positions | median 0.99; 3.7% vs 4.8% | yes (indistinguishable) |
| ASR word errors, 48 takes, one-call / stream | 2 / 2 of 628 | 4 / 6 of 628 | within noise |
| UTMOS, one-call / stream | 4.473 / 4.473 | 4.418 / 4.419 | yes |

- **Main vs branch.** The one-call audio is not run-to-run deterministic even on `main`: two runs of `main` differ from each other exactly as `main` and the branch do (same 14 / 16 and 2 / 16 takes bit-equal, the rest ~50 dB apart), while the codes are identical in every run. The difference is the speech decoder's own nondeterminism, not this work.
- **Length.** qwen-tts cuts the reference's audio at `int(ref / total * samples)`, a float product that lands one sample short of `ref * 1920` for about 5% of lengths, so the one-call audio then starts with one sample of the reference clip. The stream emits exactly the generated frames' samples. It cannot match without knowing the line's final length before its first chunk. Gate not met as worded; I did not change either side.
- **Seams.** The decoder floor's error follows the signal, so as worded the gate fails for most positions, not only boundaries: a quarter of all 10 ms windows exceed twice their take's median. Against the error of the surrounding 80 ms, boundary windows are like any other position (the boundary rate over 2x is below the non-boundary rate). Reported both ways; the gate as worded is not met.
- **ASR.** All 0.6B errors are "traveller" heard as "traveler". The 1.7B's extra two are one take where the ASR wrote "94" for "ninety four" (scored as two words); the same codes, ~38 dB apart.

### Comparison, configurations A and B

Median of 5 runs per workload, warm. TTFA = time to first audio from submit; TTLA = to last audio; rate = all audio seconds / wall seconds; margin = the smallest amount by which any chunk after the first arrived before its playback time; GPU ms/s = summed CUDA kernel time per second of audio over the six `seq` lines (torch.profiler; host gaps excluded). One process per configuration.

0.6B (narrator), GPU kernel time per second of audio: A 104.5 ms, B 110.1 ms.

| workload | A TTFA med / worst | B TTFA med / worst | A TTLA med / worst | B TTLA med / worst | A rate | B rate | B margin | peak GiB A / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seq (6 lines, one client) | 561 / 717 | 87 / 94 | 561 / 717 | 562 / 718 | 8.27 | 8.33 | 115 ms | 1.94 / 1.67 |
| burst2 | 945 / 1249 | 387 / 810 | 945 / 1249 | 928 / 1275 | 8.13 | 8.18 | 121 ms | 1.93 / 1.67 |
| burst4 | 1494 / 2410 | 966 / 1924 | 1494 / 2410 | 1438 / 2410 | 8.13 | 8.29 | 121 ms | 1.94 / 1.67 |
| burst8 | 2497 / 4642 | 2059 / 4141 | 2497 / 4642 | 2519 / 4543 | 7.94 | 8.06 | 121 ms | 1.94 / 1.67 |
| late (20 s line, a short one 1 s in) | 2558 / 3394 | 609 / 1322 | | 1874 / 2202 | 6.64* | 8.83 | 121 ms | 2.34 / 1.67 |
| short among 3 long | 5567 / 9803 | 3265 / 4811 | | 4436 / 6700 | 7.36* | 9.01 | 121 ms | 2.34 / 1.67 |

\* The A run of `late` and `short_among_long` is slower than an earlier A run of the same code (1747 ms and 4211 ms median TTFA, rates 9.24 and 9.46) with no other GPU process seen; host load varies on this machine (two VirtualBox VMs). Read A's long-line rows with that in mind; the GPU kernel figure is the robust one.

1.7B (designed narrator), GPU kernel time per second of audio: A 140.0 ms, B 148.8 ms.

| workload | A TTFA med / worst | B TTFA med / worst | B TTLA med / worst | A rate | B rate | B margin | peak GiB A / B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seq | 799 / 1137 | 91 / 94 | 816 / 1175 | 6.66 | 6.49 | 111 ms | 2.80 / 2.60 |
| burst2 | 1323 / 1836 | 441 / 1023 | 1230 / 1753 | 6.61 | 6.43 | 111 ms | 2.77 / 2.60 |
| burst4 | 2135 / 3691 | 1556 / 3060 | 2353 / 3832 | 6.58 | 6.44 | 111 ms | 2.80 / 2.60 |
| burst8 | 3708 / 6686 | 3052 / 6114 | 3813 / 6704 | 6.50 | 6.42 | 111 ms | 2.80 / 2.60 |
| late | 3343 / 3817 | 1288 / 3203 | 3655 / 4105 | 7.02 | 6.59 | 106 ms | 3.26 / 2.61 |
| short among 3 long | 7472 / 11743 | 5808 / 8688 | 7873 / 12635 | 7.07 | 6.66 | 107 ms | 3.26 / 2.61 |

(A's TTLA equals its TTFA.) What this shows:

- One stream starts in ~87-91 ms on both models, from 560-800 ms, and every chunk beat its playback by at least 106 ms; streaming costs ~5% more GPU per second of audio (the front re-runs per chunk, and the conv stack re-decodes 6 context frames per chunk).
- In B a stream holds the model until it ends: the short line among long ones waits 4.4 s (0.6B) / 7.7 s (1.7B) for its first audio, the second request in `late` waits 1.2 s / 3.0 s, and burst8's worst first audio is 4.1 s / 6.1 s. Only batching (D) fixes that.
- B uses less VRAM than A (0.6B 1.67 vs 1.94-2.34 GiB): the one-call path decodes a whole line, reference included, in one conv stack.
- Per chunk, the graphed front costs 1.6-1.9 ms of GPU and the conv stack 10.1 ms median (13.7 ms for a 22-frame window); 0.6B per second of audio: ~90 ms talker, ~15 ms decode.
- Over HTTP (a throwaway `muse _worker` on port 8123 from this branch, no gateway): one-call 558 ms median, stream first audio 90 ms median / 92 ms worst, 6 chunks per line.

### Time to first audio: experiments (0.6B unless said)

| experiment | TTFA median | kept |
| --- | ---: | --- |
| baseline stream: eager decoder, first 4 then 16 frames, context 6 | 123 ms (1.7B 165 ms, contaminated) | |
| front and conv stack as CUDA graphs per padded shape (front 23 ms of launches -> 1.9 ms) | 104 ms (1.7B 111 ms) | yes |
| first chunk 2 frames, then straight to 16 | 0.6B 89; 1.7B 91 ms but a chunk 17 ms late (16 frames take ~170 ms there, 2 frames play for 160 ms) | no |
| first chunk 2 frames, doubling to 16 | 87-89 ms (1.7B 91 ms), margin >= 106 ms | yes |
| first chunk 1 frame, doubling | 88 ms (worst 108), margin 52 ms | no: no gain |
| conv context 4 instead of 6 (B, GPU ms per audio s) | 110.2 -> 108.3 | no: -1.7% GPU for ~0.8 dB at the floor |
| steady chunk 32 instead of 16 | 108.6 | no |
| both | 107.0 (-2.9%) | no: small next to the talker's ~90 ms/s; see stage 2 notes |

Where the 87 ms goes (0.6B): capture (prompt building) 5 ms, prefill 61 ms, 2 frames 14 ms, decode 8 ms. The prefill is upstream's eager path over int8 layers (dequantized per call); changing it would change the one-call path's numerics, so it was left alone. It is the next lever, and it applies to both paths.

### Not switched on: the one-call path without the reference's conv stack

`SplitDecoder.decode(codes, skip=ref)` against qwen-tts's decode of the same clone codes: narrator (49 reference frames) 106 -> 90 ms, `wispwood_nettle` (63) 118 -> 86 ms, a tiled 30 s reference (375 frames) 268 -> 87-96 ms, at 37-43 dB against upstream's audio (the 30 s case is against upstream's own 300-frame chunked decode). These times were taken while another process used the GPU for about two minutes and need re-taking before any decision. Not enabled.

### For stage 2

- `FastTalker.request(kwargs, ref_codes=None) -> TalkerRequest | None`; `FastTalker.start(req) -> TalkerRun`; `FastTalker.step(run) -> codes [1, 16] | None`; `FastTalker.capture(call) -> kwargs`; `FastTalker.disable(message)`. Batch 1 throughout: one `_TalkerState`, one CUDA graph per (capacity, params), a run is retired by the next start. `TalkerRun` keeps `pos` / `produced` / `done` on the host; the per-row state (position, text cursor `n`, repetition set `seen`, stop flag `unfinished`, token, hidden) lives in `_TalkerState`'s batch-1 tensors.
- `SplitDecoder.front(codes [B, Q, T])` and `conv_stack(hidden [B, C, T])` take several streams; rows equal single calls within the bf16 floor (37.4 dB, not bit-identical: batched GEMMs round differently). `_shaped(kind, fn, x, size)` pads the last axis and graphs per full shape, batch included. `StreamDecoder` and `SplitDecoder.window` / `stream_window` are one stream at a time (`[0, 0]` indexing, one start / end).
- Batched decode, measured (0.6B, graphed, GPU ms per call at 1 / 2 / 4 / 8 / 16 streams): front over 128 frames 1.9 / 2.8 / 4.5 / 7.7 / 13.3; conv stack over 24 frames 13.6 / 24.5 / 45.2 / 87.5 / 168.0. The front batches well; the conv stack is compute-bound and scales linearly, ~7 ms per second of audio per stream, of which about a third is re-decoding the 6 context frames of each 16-frame chunk. A conv stack with carried state (excluded here) or longer chunks is where decode capacity is.
- The route's lock is still `threading.Lock`; the harness takes it with `acquire()` / `release()` only, so a semaphore drops in. `compare.py` already passes `max_batch_size` to the model (ignored today) and has C / D defined.
