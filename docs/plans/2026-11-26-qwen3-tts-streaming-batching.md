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

### Review follow-ups (stage 1)

- **The 36 s line at 29.8 dB.** Checked against the unchunked one-pass decode of the same codes (two seeds, 37-41 s lines). Where both paths share the codes: the stream is 37.8 dB from the one-pass decode before and after frame 300; the one-call audio is 40.1 dB before qwen-tts's second 300-frame chunk and 27.7 dB after it, 30.2 dB overall. So the one-call path is the less faithful one: `chunked_decode` re-starts at frame 300 with 25 frames of context.
- **Same seed, different codes, on that line.** In the other seed the stream and the one-call codes diverged (stream -3 dB from the one-pass audio of the one-call codes). Cause: the first long line grows the talker's KV cache past 512 tokens, the regrown cache's graph is captured, and the capture's two warm-up frames draw from the CUDA RNG. The next request with the same seed finds the graph already captured, so its draws are offset. The codes-equal gate holds whenever both runs see the same graph cache (all stage-1 gate runs did); a first request after any capture (new KV capacity or sampling parameters) consumes extra draws. Same on `main`. Not fixed (it would change the one-call path's random stream); noted for anyone gating on seeds.

## Stage 2 results: batching (2026-11-27)

### Weight path, measured (step 1)

`sb/rows_probe.py`, idle RTX 3060, 0.6B shapes, a frame's 108 layer products, ms:

| rows | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| int8 GEMV kernel (forced wider past 4) | 5.59 | 7.00 | 12.08 | 16.63 | 26.89 | 58.26 |
| bf16 cuBLAS (float weights) | 11.76 | 12.94 | 12.95 | 13.00 | 13.12 | 17.39 |
| int8 dequantize + cuBLAS | 27.39 | 28.10 | 28.10 | 28.17 | 28.29 | 32.68 |

The contaminated reading held: int8 to ~4 rows, float cuBLAS flat to 16. Wide frames read float weights kept beside the int8 (`keep_float`, only when `max_batch_size` > 2): resident 2.0 -> 2.5 GiB on the 0.6B, 3.3 -> 5.3 GiB on the 1.7B. Because the code predictor's first step has two positions per row and the int8 GEMV takes at most four rows, buckets 1-2 read int8 and 4+ read float (`INT8_MAX_ROWS = 2`).

### What was built

- `runtimes/qwen3_tts_batch.py`: `BatchedTalker` (rows of one decode state; join = that line's prefill alone into a free row; leave at its own EOS with the last row moved into the hole; buckets 1, 2, 4, 8, 16, 32; one CUDA graph per bucket, KV capacity and sampling parameters; per-row position, text cursor, repetition set, stop flag, RoPE angle and live mask; the state's rows grow by bucket, so VRAM follows the live streams; the trailing text buffer per row has room to append). The one-row bucket calls `FastTalker._frame` on a view of row 0: today's frame. `Engine`: one thread owning the GPU (prepare = qwen-tts prompt building, admit, frames, per-stream chunk decode, whole-line decode, pacing to `stream_lead_seconds`, cancel, a `priority` carried and unused, `run_on_engine` for other GPU work), no web framework imports.
- Runtime: `max_batch_size` (hidden; 1 = today's code path, unchanged) and `stream_lead_seconds` (1.5). Above 1 on CUDA, `_inference_lock` is a `Semaphore(max_batch_size)`, both `synthesize` and `synthesize_stream` go through the engine, prompt building runs on its thread, and after an engine failure the model serves one request at a time (on the engine thread while it lives, else under a lock).
- Prefill for joining lines (step 6): the prompt padded at the end to a multiple of 32 tokens, one CUDA graph per padded length, float weights when kept, real positions' K/V copied into the row. 61 -> 8.3 ms per join (0.6B, 58 tokens); int8-only (max_batch_size 2) 12.3 ms.

### Gates

| gate | 0.6B | 1.7B | pass |
| --- | --- | --- | --- |
| `max_batch_size` 1: `run.py` / `run17.py` `INT8=1`; stage-1 stream gates (12 takes) | pass, frame 7.219 ms, every gate as before; codes equal 12/12, SNR min 33.8 dB, local seams median 1.02 | pass, frame 10.198 ms, KLs as before (its wall RTF read 0.187, host noise; the GPU frame time is unchanged) | yes |
| rows independent within a bucket (greedy; same line, different companions) | bucket 2 and 8: identical | identical | yes |
| one-row bucket == solo FastTalker (today's prefill) | identical, 34 / 34 frames | identical | yes |
| graphed == masked eager per bucket (same prefill) | buckets 1-8 identical with today's prefill; with the graphed padded prefill vs the same prefill run eagerly, identical for buckets 2-8 and the first 31 of 34 frames in bucket 1 | identical with today's prefill | yes (bucket 1 with the graphed prefill: no, from frame 31) |
| teacher-forced talker KL vs today's int8 one-row path (limit 2.5e-3) | bucket 2 (int8) 0.89e-3, 4 / 8 (float) 1.45e-3 / 1.48e-3; with the engine's padded float prefill 1.44e-3 / 1.92e-3 | 0.58e-3, 1.00e-3 / 1.00e-3; padded prefill 1.70e-3 / 2.16e-3 | yes |
| same weight path: float bucket 8 vs bucket 4 | 1.7e-5 | 0.0 | yes |
| first talker token KL, padded graphed prefill vs eager int8 prefill | float 5.8e-3 (max 2.1e-2), int8-only 2.8e-3; greedy first token equal 24/24 | | reported (no limit; the per-frame KL above is the gated one) |
| short line among long ones within 1.5x of solo | C: 789 vs 324 ms = 2.4x | | **no** (C) |
| | D: 933 vs 721 ms = 1.29x (paced) | | yes (D) |
| ASR errors / UTMOS, 48 batched takes (all 24 lines submitted together) vs solo | 2 / 628 vs 2; 4.451 vs 4.473 (se 0.012) | 2 / 628 vs 4-6; 4.410 vs 4.418 | within noise |
| streams under batching: SNR vs own codes' one-pass decode, exact length | min 32.8 dB (median 40.5), lengths exact | min 36.0 (median 38.8), exact | **no** on the 0.6B by 0.2 dB in 1 of 48 |

The 32.8 dB stream is the decoder's own floor on those codes (the per-stream decode is stage 1's, unchanged; stage 1's own minimum was 33.8 over different codes), but I did not relax the gate. The C failure is the cost of a wide frame: three lines in bucket 4 take 19.0 ms per frame against 7.0 ms alone.

### Frames and VRAM

Graphed frame GPU ms (per line) by bucket: 0.6B 7.01 (7.01), 12.19 (6.09), 19.01 (4.75), 21.39 (2.67), 27.19 (1.70), 40.69 (1.27); 1.7B 9.93, 15.99, 26.04, 28.54, 34.57, 48.27 (1.51 at 32). Decode state: 55 MB per row at 512 KV tokens (both models), 1.75 GiB at 32 rows. Where a graphed frame's GPU time goes (torch.profiler over graph replays, 0.6B, stage 3, after narrow products moved back to the int8 GEMV): bucket 1 7.3 ms (int8 GEMV 6.0, norm / RoPE 0.6, attention 0.3, sampling 0.2); bucket 4 18.9 ms (GEMV 11.7, cuBLAS 4.3 for the predictor's 8-row first step, index / copy 1.1, elementwise 1.1, norm / RoPE 0.6); bucket 8 21.9 ms (cuBLAS 18.7, index / copy 1.3, elementwise 1.1); bucket 16 27.8 ms (cuBLAS 24.2, index / copy 1.4, elementwise 1.2). The frame is weight-bound: attention, sampling and torch-op overhead are under 3 ms at every bucket, and the engine's host work per frame (one clone, per-row slices, the stop-flag sync) is 65-125 us. Bucket 4 was 19.6 ms with all its products on float weights.

### Comparison A-D (in process, median of 5 runs, `max_batch_size` 32 for C / D)

First audio median / worst (ms), audio s per wall s, smallest chunk margin, peak GiB. A / B are the stage 1 runs. D paces streams to 1.5 s ahead of playback, so its last-audio times and audio-per-wall follow playback, not the GPU.

0.6B (GPU kernel ms per second of audio over the six `seq` lines: A 104.5, B 110.1, C 103.5, D 109.6):

| workload | A | B | C | D |
| --- | --- | --- | --- | --- |
| seq | 561 / 717, 8.27 | 87 / 94, 8.33, +115 | 504 / 683, 9.08 | 52 / 53, +120, 2.6 GiB |
| burst2 | 945 / 1249 | 387 / 810 | 964 / 1683, 9.98 | 77 / 575, +88 |
| burst4 | 1494 / 2410 | 966 / 1924 | 1505 / 2687, 11.9 | 127 / 606, +39 |
| burst8 | 2497 / 4642, 7.94 | 2059 / 4141, 8.06 | 1757 / 3537, 16.9 | 208 / 701, **-69** |
| burst16 | | | 2628 / 5434, 18.4 | 416 / 1033, **-558** |
| burst32 | | | 4618 / 10235, 21.4, 5.2 GiB | 845 / 1498, **-2779**, 5.2 GiB |
| short among 3 long: short line's last audio | 5742 | 4567 | 789 (solo 324) | 933 (solo 721) |

1.7B (A 140.0, B 148.8, C 137.4, D 145.9 ms per audio s):

| workload | A | B | C | D |
| --- | --- | --- | --- | --- |
| seq | 799 / 1137, 6.66 | 91 / 94, 6.49 | 760 / 1084, 6.96 | 62 / 64, +110 |
| burst8 | 3708 / 6686, 6.50 | 3052 / 6114, 6.42 | 2365 / 4116, 14.1 | 272 / 771, **-28** |
| burst32 | | | 4711 / 8950, 24.3, 8.0 GiB | 969 / 1625, **-1702**, 8.0 GiB |
| short among 3 long: short line's last audio | | 8053 | 940 (solo 427) | 1025 (solo 1292 paced) |

Throughput (C) rises with the burst, 2.6x A's at 32 lines on the 0.6B and 3.6x on the 1.7B, at the same GPU ms per second of audio for a lone line. A lone streamed line starts in 52 ms (0.6B) / 62 ms (1.7B), from 87 / 91 in B, because the joining line's prefill is graphed. These D rows are stage 2's doubling schedule, under which bursts of 8 or more simultaneous streams went late at their start; the burst table was not re-run with stage 3's deadline-driven chunks (the HTTP check below covers one simultaneous start).

### NPC capacity (stages 3 and 3b: deadline-driven chunks)

Stage 2's doubling schedule (2, 4, 8, 16 frames) needs each next chunk generated within the playback of the one before, i.e. an all-in frame under 40 ms; it capped the 0.6B at 8 speakers. The engine (`max_batch_size` > 1 only; `max_batch_size` 1 never builds it) now emits by deadline:

- A playing line's pending frames go out when one more frame plus a decode would not fit in its audio ahead of playback (`emission_due`: lead + `stream_buffer_seconds` - 30 ms < the slowest of the last 32 loop iterations + the decode estimate), or at `stream_chunk_frames` (16).
- A line already behind its playback waits for at least `least_chunk_frames` = ceil(1.5 f / (T - f)) frames (all 16 past f = 0.9 T), f the mean all-in frame time. Without this, a crowd that fell behind decoded 1-2-frame chunks for every line every frame and never recovered (the "mean" first-chunk variant at 24 speakers: decode share 0.68, 43 ms a decode, all-in 105 ms a frame; no OOM, growth or runaway involved). A version that held back lines not yet late made them underrun (5-13 late lines at 24 speakers).
- First chunk (`first_chunk_size`): with fewer than `HEAVY_ROWS` (10) live rows and the mean all-in frame under half a frame's audio, the larger of ceil(1.5 f / (T - f)) on the running mean and a floor covering one slow frame plus a decode, ceil((slowest + decode + 30 ms) / T) + 1; otherwise the long chunk, `first_chunk_frames` on the slowest recent frame (stage 3's rule), which kept 24 speakers on time where every mean-based variant tried put 13-17 lines late. A pass that admits several lines at once is sized for the all-in time measured at the bucket it lands in (`bucket_est`), so a burst is not sized for the idle engine it found.

Workload: N speakers, one of the 24 lines (2.5-6.5 s) each with 1-3 s pauses, starts staggered over 2 s, 20 s per N, all streamed. ok = no chunk after a line's first arrived after its playback time (client buffer counted) and the worst first audio < 1 s. "voices" = mean live rows. Margin = the smallest amount by which a chunk beat its playback time. Idle GPU (checked) unless marked.

0.6B, client buffer 0, `warm_rows` 16, three runs (first audio median / worst ms, min margin, late lines):

| N | voices | run 1 | run 2 | run 3 | the coordinator's mean variant | stage 3 rule |
| ---: | ---: | --- | --- | --- | --- | --- |
| 8 | 3.4 | 106 / 216, +27, 0 | 103 / 213, +27, 0 | 95 / 219, +25, 0 | 116 / 360, +12, 0 | 96 / 195, +32, 0 |
| 12 | 5.1-5.4 | 146 / 729, +25, 0 | 114 / 593, +31, 0 | 121 / 681, +11, 0 | 137 / 254, +13, 0 | 159 / 587, +22, 0 |
| 16 | 7.2-7.4 | 165 / 799, +5, 0 | 157 / 733, +28, 0 | 146 / 737, +1, 0 | 161 / 262, +32, 0 | 537 / 867, -2, 1 |
| 24 | 12.1-12.8 | 831 / 1346, -20, 6 | 717 / 1190, +10, 0 | 714 / 1125, +12, 0 | 989 / 3493, -5636, 46 | 929 / 1228, +26, 0 |
| 32 (`warm_rows` 32) | 17.2 | 1119 / 1709, -844, 4 | | | | |

Against the target: N = 8 met on all three runs; N = 24 met on two of three (0 late, first audio no worse than stage 3's rule); N = 12 and 16 match the mean variant on median first audio and lateness, but **not on worst first audio** (593-799 ms against its 254-262). The worst cases are lines sized as part of a multi-line pass (several speakers' next lines landing in the same pass), which the burst rule sizes for the measured all-in time at the bucket they join (long chunks); the mean variant had no burst handling. I did not get both.

0.6B, client buffer 200 ms, `warm_rows` 32, one run:

| N | voices | first audio med / worst | min margin | late lines |
| ---: | ---: | --- | ---: | ---: |
| 8 | 3.4 | 96 / 165 | +38 | 0 / 51 |
| 12 | 5.0 | 116 / 593 | +34 | 0 / 71 |
| 16 | 7.3 | 146 / 795 | +28 | 0 / 87 |
| 24 | 12.7 | 839 / 1310 | +12 | 0 / 108 |
| 32 | 18.0 | 1268 / 2097 | -1157 | 10 / 111 |

**0.6B capacity, final rule: 16 speakers (7.4 voices) with no client buffer on every run, 24 speakers (12.5 voices) on two of three runs (first audio then 0.7-0.8 s median, 1.1-1.3 s worst); with 200 ms buffer 24 speakers without a late line. 32 speakers is past real time.** One client: 53 ms first audio (stage 3 measurement, unchanged path). The 1.7B was not re-run with the stage 3b rule. Under stage 3's rule (buffer 0, a muse-local 0.6B worker resident and idle): 16 speakers (8.8 voices) with no late line, first audio 361 / 904 ms; 24 speakers 25 lines late.

All-in frame time (engine loop: frame, per-row takes, the decodes due) sets the ceiling, which must stay under 80 ms: 0.6B 8.6 ms alone, 21 ms at 3.4 voices, 25-29 at 5, 31-35 at 7-8, 40-48 at 12-13, 52-57 at 17-18.

**Quality under deadline chunking (stage 3b).** The 24 lines x 2 seeds streamed one at a time through the engine while a 12-speaker crowd kept it loaded (mean live rows 6.3 on the 0.6B, 8.4 on the 1.7B; every chunk size from 1 to 16 frames occurred). Against the solo takes of stage 1 (same voices, same lines):

| | 0.6B crowd | 0.6B solo | 1.7B crowd | 1.7B solo |
| --- | --- | --- | --- | --- |
| ASR word errors / 628 | 2 | 2 | 3 | 4-6 |
| UTMOS mean (se), min | 4.463 (0.009), 4.23 | 4.473 (0.009) | 4.399 (0.017), 3.92 | 4.418 (0.011) |
| SNR vs one-pass decode of own codes, min / median | 36.7 / 40.7 dB | 33.8 / 40.6 | 34.8 / 38.3 dB | 34.4 / 38.8 |
| exact length | 48 / 48 | | 48 / 48 | |

Within noise everywhere; the 1.7B's three errors are two "traveller" spellings and one take heard as "but said" for "I said". (The script aborted at interpreter exit with its crowd threads still inside the engine, after writing every result; a harness artefact.)

**Growth hiccups and warm-up.** A long streamed line plays while 8 more join one per second. Without warm-up the first growth to 2 and to 4 rows (state rebuilt, graphs recaptured) made the playing line 256 ms and 188 ms late; with `warm_rows` 16 no line was late at any join. VRAM after load: 2.51 GiB (no warm-up), 3.12 GiB (8 rows), 3.56 GiB (16 rows). With `warm_rows` 8, the join to 2 rows was still 44 ms late once (not explained).

**Runaway lines (stage 3b: capped).** Twice in the stage 3 runs a sampled line did not stop (qwen-tts's `max_new_tokens` is 4096 frames), pushed one row past 512 KV tokens, and growing all 32 rows to 1024 ran out of memory (handled: only that line failed, but the rebuild stalled every stream). Every line on every path now gets `max_new_tokens = frame_limit(text)` = 10 frames a character + 40, at most 4096; a line that reaches it ends there, as if it had stopped, with a warning. (Stage 3b first set 3 a character from English prose alone; review then measured Chinese and Japanese at 2.3-2.5 a character, a price with a date at up to 4.5 and a bare digit string at 5.5, and at 3 a character one of three takes of `Dr. Smith paid £3,499.99 on 03/04/2025.` would have been cut at 157 of 174 frames. A 60-character line that runs away now stops after 51 s instead of 5.5 min.) Measured over 96 takes (24 lines, 2 seeds, both models): at most 1.06 frames a character (0.61-1.06, the highest on the shortest lines); a 750-character paragraph 0.62-0.68. The stage-1 code-equality gate still holds at `max_batch_size` 1 (12/12 takes on each model), so ordinary English lines never reach the limit.

**HTTP (stage 3b, idle GPU).** An in-process app (muse's speech router) on a free port, `max_batch_size` 16, `warm_rows` 16; all clients start at the same instant; one extra streaming client disconnects after its first chunk.

| clients | stream first audio med / worst | stream min margin | whole lines med / worst | after |
| --- | --- | ---: | --- | --- |
| 8 streams + 8 whole + 1 dropped | 417 / 444 ms | -1378 ms | 4.3 / 6.5 s | 0 live rows, engine up |
| 4 streams + 4 whole + 1 dropped | 273 / 298 ms | -50 ms | 2.7 / 3.3 s | 0 live rows, engine up |

(Stage 3, before burst sizing: 8 + 8 + 1 at 522 / 644 ms and -3.3 s; 4 + 4 + 1 at 238 / 254 ms and -234 ms, the latter possibly contaminated.) Burst sizing cut the lateness by 60-80% but nine or seventeen lines starting in the same instant still go late once at zero buffer: a whole line generates unpaced at full speed beside the streams, and the first frames after a 17-line join run at the 32-row bucket.

**Under contention, no MPS** (the owner's NPC GGUF on `llama-server`, 200-token completions back to back; buffer 0):

| N | voices | first audio med / worst | min margin | late lines | ok |
| ---: | ---: | --- | ---: | ---: | --- |
| 4 | 1.9 | 81 / 187 | +31 | 0 / 22 | yes |
| 8 | 3.9 | 157 / 613 | +17 | 0 / 45 | yes |
| 12 | 5.8 | 350 / 880 | +13 | 0 / 61 | yes |
| 16 | 8.5 | 750 / 1290 | +9 | 0 / 71 | first audio only |

The language model ran at 370 tok/s alone and a median 286 (min 45) tok/s during these runs. A second run of the same sweep gave N = 16 with 15 late lines (one line went to 2.4 s first audio), so treat 12 as the capacity beside the language model (stage 2: 4).

**MPS.** My attempt failed because the daemon's control socket path under the scratchpad exceeds the ~107-character Unix socket limit. The coordinator ran the sweep from `/run/user/$UID/npc-mps/`: clean capacity is 8 speakers with or without MPS; under MPS the heavy end degrades gracefully (N = 24: 14.3 against 6.9 audio-seconds per second) and the language model alone is slower (275 against 361 tok/s).

### Experiment log (stage 2)

| experiment | result | kept |
| --- | --- | --- |
| state sized for `max_rows` up front | the gate script ran out of VRAM (1.75 GiB of KV per state at 32 rows) | no: rows now grow by bucket |
| graphed padded prefill, float weights (`max_batch_size` > 2) | join 61 -> 8.3 ms; lone streamed line's first audio 87 -> 52 ms | yes |
| the same on int8 only (`max_batch_size` 2) | 12.3 ms | (the path used at 2) |
| `INT8_MAX_ROWS` 4 instead of 2 | not tried: at 4 rows the predictor's first step has 8 positions, past the GEMV's 4-row limit, which falls back to dequantize + cuBLAS (the slowest path measured) | |

### Not done

- **Batched decode of the due chunks** (decoding several streams' conv stacks in one call). The conv stack is close to linear in streams (stage 1: 13.6 ms for one 24-frame window, 45 ms for four, 168 ms for sixteen), and with deadline-driven chunks most chunks are long, so the per-call saving is small; decode is 13-28% of the engine's time up to 12 voices. Not built.
- **Frames for a subset of rows** (a narrow frame for the rows behind while the rest are ahead). Feasible in principle: the graphs already run rows [0, k) of a state whose live rows are compacted to the front, so a "behind" subset would have to be packed to the front (row swaps cost one K/V copy per layer, ~29 MB a row at 512 tokens) or the frame given a per-row active mask (every state update - KV write, position, text cursor, repetition set - would need gating, and the frame would still cost its full bucket). The gain is for a line joining a busy engine; with prefill at 8 ms and the first chunk sized to the load, I don't think it pays.
- **The KV cache grows for all rows together**, so one runaway line costs every row 2x the KV (see above); a per-line frame limit is the simple fix.
- **The first chunk under a burst** is sized from the frame time measured before the burst; predicting it from the rows about to join would make simultaneous starts safe.
- The RNG offset from graph capture (stage 1 review follow-ups) is not fixed.
