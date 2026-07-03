# Concurrency stress test: results and findings

Empirical stress test of a live muse gateway under concurrent load, run
2026-07-02 against the `.204` GPU box (frodo, 12 GB VRAM / 64 GB RAM),
muse v0.50.2. Harness: `stress_test.py` (async httpx, five scenarios).
Raw data: `results.json`, `loaded_snapshots.json`. Generated artifacts
(PNG / WAV): `artifacts/`.

## TL;DR (what we learned)

1. **A cold load freezes ALL traffic for its full duration.** The single
   biggest finding, proven cleanly by scenario D: 6 requests to an
   already-hot, tiny TTS model, submitted 0.25s into a 37s cold load of an
   unrelated image model, each took ~35s instead of their normal ~1-2s.
   The stall equals the cold-load time. Cause: `director.acquire` is a
   synchronous call on the gateway's single async event loop, so a cold
   load blocks the loop and every other request queues behind it, hot or
   not. This confirms the caveat raised during the earlier code walkthrough.

2. **Concurrent requests to the SAME model serialize inside the worker.**
   Scenario A produced a perfect staircase: 6 concurrent chat requests
   finished at 10.9, 28.8, 42.5, 56.5, 72.2, 87.8 seconds, each ~14s after
   the previous. The worker does not parallelize same-model inference
   (llama.cpp generation is single-threaded per model; the GPU serializes).
   Concurrency buys almost nothing here (wall 87.8s vs sequential 98.1s)
   while inflating the slowest requester's latency 5.7x.

3. **Under VRAM pressure the system degrades GRACEFULLY, not catastrophically.**
   When more models are requested than fit, the in-flight ones are pinned
   by refcount and cannot be evicted, so surplus requests get a clean
   `503 model_too_large_for_device` instead of an OOM or a crash. Scenario
   C: 3 of 5 image models succeeded, 2 rejected cleanly. Scenario E
   (11-model herd): 7 succeeded, 4 rejected cleanly. Zero crashes, zero
   OOMs, zero corrupted responses across the whole run.

4. **Different models DO run in parallel once loaded** (separate worker
   processes, separate GPU contexts). The bottleneck is never the compute
   overlap; it is (a) the serialized cold loads on the blocked event loop
   and (b) VRAM capacity.

## Setup

- Server: `http://192.168.0.204:8000`, one in-process gateway (single
  asyncio event loop) fronting per-model worker subprocesses.
- 11 models exercised across chat, embeddings, rerank, moderation,
  summarization, TTS, and 5 image-generation models.
- Light models (embed ~1.5, rerank ~2.2, moderation ~0.5, summarize ~1.6,
  kokoro ~0.4, chat ~2.6 GB) co-reside within 12 GB. Image models
  (sd-turbo ~3-4, sdxl-turbo ~7, pixel-art-xl ~6.7 GB) cannot co-reside
  with much else, so they force eviction.
- Latencies are client-side wall time. "dispatch offset" is when each
  request left the client relative to the batch start.

## Scenario A: same-model concurrency (6x qwen3.5-4b-q4, hot)

| mode | per-request latencies (s) | wall (s) |
|---|---|---|
| concurrent (6 at once) | 10.9, 28.8, 42.5, 56.5, 72.2, 87.8 | 87.8 |
| sequential (control)   | 15.4, 15.7, 16.6, 16.2, 17.2, 17.0 | 98.1 |

The concurrent latencies form a staircase with a ~14s step. That is the
signature of **serialization**: the worker processes one generation at a
time, so requester N waits for N generations. The `asyncio.to_thread`
dispatch in the chat route lets the requests overlap at the Python I/O
level, but the actual llama.cpp generation loop is single-threaded and
GPU-bound, so it is a queue.

Takeaway: firing N concurrent requests at one model does not speed the
batch up. It just moves the wait from the client to the server and makes
the tail latency N times the single-request latency. For throughput on one
model you need either a batching runtime or multiple worker replicas;
muse currently runs one worker per model.

## Scenario B: different light models, all at once (6 models)

All six finished in a tight cluster at 59.6 - 69.3s, despite being
different models on different workers.

| model | latency (s) |
|---|---|
| qwen3-embedding-0.6b | 60.4 |
| bge-reranker-v2-m3 | 59.6 |
| text-moderation | 60.7 |
| bart-large-cnn | 61.8 |
| kokoro-82m | 60.4 |
| qwen3.5-4b-q4 | 69.3 |

If the cold loads had run in parallel, the batch would finish in about one
load time (~10-15s). Instead every request paid ~60s: the cumulative time
to load all six models **one after another**. This is the same
event-loop-blocking effect as scenario D, seen from a different angle. The
loads serialized on the blocked loop; once all six workers were up, the
actual inference was fast and overlapped, so all six completed near
together. Net cost of serialized loading here: roughly 4x.

## Scenario C: eviction storm (all 5 image models at once)

| result | model | latency (s) | note |
|---|---|---|---|
| OK   | sd-turbo | 116.9 | png 512x512 |
| OK   | sd_pixelart_spritesheet_generator | 121.5 | png 512x512 |
| OK   | all-in-one-pixel-model | 115.0 | png 512x512 |
| FAIL | sdxl-turbo | 29.0 | 503 model_too_large_for_device |
| FAIL | pixel-art-xl | 29.0 | 503 model_too_large_for_device |

Loaded set: `[8 light models]` before -> `[all-in-one, sd-turbo,
sd_pixelart]` after. The entire 8-model light working set was evicted to
fit 3 image models. The two that failed hit the eviction loop when every
resident model had refcount > 0 (their requests were in flight and pinned
them), so there were no evictable candidates and they got a fast, clean
503 at ~29s rather than waiting out a load or OOMing. This is the refcount
pin working exactly as designed: in-flight work is never evicted out from
under itself.

## Scenario D: cold-load-blocks-the-loop probe (the key experiment)

Design: warm kokoro-82m (tiny TTS, hot). At t=0 fire ONE request to a cold
sdxl-turbo. At t=0.25s fire 6 requests to the hot kokoro. If the event loop
is blocked by the synchronous cold `acquire`, the hot requests cannot be
serviced until the load finishes.

| model | dispatch @ | latency (s) | notes |
|---|---|---|---|
| sdxl-turbo (cold) | 0.00s | 37.3 | the blocking load |
| kokoro-82m (hot) | 0.25s | 35.1 | should be ~1-2s |
| kokoro-82m (hot) | 0.25s | 35.2 | |
| kokoro-82m (hot) | 0.25s | 35.4 | |
| kokoro-82m (hot) | 0.25s | 35.7 | |
| kokoro-82m (hot) | 0.25s | 35.9 | |
| kokoro-82m (hot) | 0.25s | 35.7 | |

**Definitive.** Six requests to an already-loaded, sub-second model each
took ~35s, finishing within a 0.7s cluster right as the cold load
completed. Their latency equals the cold-load duration minus the 0.25s
stagger. The requests sat unprocessed in the kernel accept queue while the
gateway's only event-loop thread was stuck inside the synchronous
`director.acquire` for sdxl-turbo, then were all released and serviced the
instant it returned.

This is the concrete cost of the one architectural sharp edge: **on this
deployment, any cold load stalls every concurrent request, including
requests to unrelated hot models, for the entire load duration** (here 35s
for a 7 GB SDXL model; it would be minutes for a video model).

## Scenario E: thundering herd (one request to every model, 11 at once)

7 OK, 4 FAIL. All latencies clustered at 88 - 122s (serialized cold loads,
same as B and D). Every failure was a `503 model_too_large_for_device` on
a surplus image model that could not get a VRAM slot; the light models plus
one image model (sd-turbo) all succeeded.

| result | count | models |
|---|---|---|
| OK | 7 | chat, embed, rerank, moderation, summarize, kokoro, sd-turbo |
| FAIL 503 | 4 | sdxl-turbo, pixel-art-xl, sd_pixelart, all-in-one |

Even at maximum chaos (11 simultaneous cold requests, 5 of them image
models fighting over 12 GB) the server stayed up, answered everything it
could fit, and rejected the rest cleanly. No crash, no OOM, no partial or
corrupted responses.

## Secondary finding: one error code covers two very different situations

`503 model_too_large_for_device` is returned both for a model that can
never fit (genuinely too big for the card) and for a model that fits fine
but lost a transient race for VRAM (all slots pinned by in-flight requests,
scenarios C and E). The message text distinguishes them ("no evictable
candidates remain..."), but the machine-readable `code` does not. A client
cannot tell "retry in a few seconds and it will work" from "this will never
work." A distinct transient code (or a `Retry-After` header) would let
clients back off and retry the first case instead of giving up.

## Recommendations (not yet applied)

1. **Take the cold load off the event loop.** Wrap the acquire in
   `await asyncio.to_thread(state.director.acquire, ...)` in
   `gateway.py:_route_via_director`. The director's own RLock + in-flight
   Events already make it safe to call from a worker thread, so this is
   close to a one-line change. It would eliminate finding #1 entirely:
   cold loads would run off-loop and hot traffic would flow during a load.
   This is the highest-value fix by far; scenario D shows a 20-30x latency
   penalty on unrelated hot requests today.

2. **(Optional) Distinguish transient VRAM contention from true
   over-capacity** with a separate 503 code and/or `Retry-After`, so
   clients can retry the winnable case.

3. **(Optional) Per-model request fairness.** Same-model requests serialize
   (scenario A); that is inherent to a single-threaded generation runtime,
   but an explicit bounded queue with FIFO ordering would make tail latency
   predictable and let the worker shed load (429) instead of growing an
   unbounded staircase under sustained pressure.

None of these are correctness bugs. The system is safe and degrades
gracefully today; these are latency and ergonomics improvements.

## Reproduce

```bash
python examples/concurrent/stress_test.py http://192.168.0.204:8000
# writes results.json, loaded_snapshots.json, artifacts/*.png|wav
```
