# Request Queueing Design (per-model concurrency + capacity wait)

- **Date:** 2026-07-08
- **Status:** approved design, pre-implementation
- **Owner:** gateway / LoadDirector (single-node `muse serve`)

## Problem

Two "at capacity" situations today degrade badly under load:

1. **No per-model concurrency limit.** A loaded model accepts unlimited
   simultaneous requests. On the CPU box (192.168.0.102, 12 cores), five
   concurrent hits to the 32B GGUF all fight over the same cores and each
   crawls; the box saturates and unrelated requests on the node time out.
2. **Capacity 503 instead of waiting.** When a cold model needs room and
   every loaded model is in-use (refcount > 0), `_evict_lru_until_fits`
   503s immediately (`model_too_large_for_device`). It does not wait for
   an in-flight request to finish even when one will release capacity in
   seconds. Anima-style bursts (image + music + chat back-to-back) hit
   this on the 12 GB card.

Non-goals (explicitly out of scope):

- Cross-node / federation queue awareness. The coordinator's
  `select_node` already reads `in_flight`; making it queue-depth-aware is
  a follow-on under the separate "double-check federation" work item.
  This design is single-node.
- External brokers (Redis/Celery). Wrong scale for a 1-2 box deployment
  and breaks the synchronous OpenAI request/response contract.
- Priorities / preemption / request classes. FIFO only in v1.

## Decisions (locked with user)

| Question | Decision |
|---|---|
| Scope | BOTH mechanisms: per-model concurrency cap (foundation) + capacity admission wait (on top) |
| Client experience while queued | HOLD the connection (OpenAI/vLLM/TGI style); no 429s, no client changes |
| Default concurrency cap | UNLIMITED (today's behavior); opt-in via `capabilities.max_concurrency`; explicit over auto |
| Queue-timeout status | 503 (matches today's capacity contract; OpenAI SDKs treat it as retryable) |

## Architecture (Approach 1: gateway-orchestrated async queue)

The hard constraint: the gateway is a single asyncio event loop, and
`director.acquire` already runs OFF the loop via `asyncio.to_thread`
(#318), with same-model cold-load coalescing keeping N-1 waiters ON the
loop (#319). Any new waiting must follow the same rule: **waiters park on
the loop with asyncio primitives, never in ThreadPoolExecutor threads.**
A director-internal blocking queue (threading.Condition in `acquire`)
would park one pool thread per waiter for the full wait -- with a
`max_concurrency=1` 32B at ~40s/request, a burst of 30 waiters exhausts
the default pool and stalls unrelated hot traffic. That approach is
rejected; all queueing lives in the gateway.

```
request -> extract model
        -> [1] ConcurrencyGate.slot(model)   (asyncio.Semaphore, ON-loop wait)
        -> [2] acquire-with-capacity-wait    (off-loop acquire; on transient
               capacity 503, ON-loop await of a capacity-freed Event, retry
               under one deadline)
        -> forward / stream
        -> finally: director.release + gate release
```

### Component 1: ConcurrencyGate (new, `muse/cli_impl/concurrency_gate.py`)

- Holds `{model_id: asyncio.Semaphore}` plus `{model_id: int}` live
  waiter counts. Semaphores are created lazily on first request for a
  model, sized from the effective cap.
- **Effective cap resolution:** `capabilities.max_concurrency` if
  declared, else `server.default_max_concurrency`, else unlimited.
  `0`/`None`/absent = unlimited = NO semaphore, no gating, zero overhead
  -- today's behavior exactly.
- Acquire is `asyncio.wait_for(sem.acquire(), remaining_deadline)`.
  CPython semaphores wake waiters FIFO, giving fairness for free.
- Release happens in the same `finally` that releases the director
  refcount (streaming: relay-finally; buffered: after `aread`).
  BaseException-wide, mirroring the #319-era release discipline.
- `max_queue_depth` check happens before parking: if the model's live
  waiter count already exceeds the bound, fast-fail 503
  (`queue_full`) instead of accumulating connections.
- Cap changes (manifest edit / re-pull) apply on supervisor restart.
  Semaphores are not resized live in v1.

### Component 2: capacity-wait (gateway retry loop + director notifier)

- The director's capacity error gets a `retryable` flag:
  - `retryable=True`: transient -- no evictable candidates because
    loaded models are in-use (refcount > 0). Capacity WILL free when a
    request finishes.
  - `retryable=False`: permanent -- the model cannot fit even an
    evicted-empty device. (The request-path `revalidate_servability`
    check already fast-fails most of these before the director; the flag
    covers the remainder.)
- The gateway wraps the off-loop `acquire` in a bounded retry loop:
  on a `retryable=True` capacity 503, `await` a gateway-owned
  `asyncio.Event` (armed before the failed attempt to avoid a
  missed-wakeup race), clear it, re-acquire. One deadline
  (`server.queue_timeout_seconds`) covers gate-wait + capacity-wait +
  retries combined. Deadline exhausted -> 503 with the queue reason.
- **Notifier:** the director gets a `capacity_listener` callback slot.
  The supervisor wires it at boot to
  `loop.call_soon_threadsafe(event.set)`. The director calls it (fire
  and forget, try/except) after every `release()` that drops a refcount
  to 0 and after every eviction commit. Waiters wake, retry, and either
  fit now or park again. Thundering-herd cost is bounded: each retry is
  one `_decide` pass under the director lock.
- `retryable=False` surfaces immediately -- never spins the loop.

### Component 3: config knobs (three new `muse.core.config` rows)

| Key | Env | Type | Default | Meaning |
|---|---|---|---|---|
| `server.default_max_concurrency` | `MUSE_DEFAULT_MAX_CONCURRENCY` | int | `0` (unlimited) | Cap for models that do not declare `capabilities.max_concurrency` |
| `server.queue_timeout_seconds` | `MUSE_QUEUE_TIMEOUT_SECONDS` | float | `300.0` | Max total held-connection wait (gate + capacity + retries) |
| `server.max_queue_depth` | `MUSE_MAX_QUEUE_DEPTH` | int | `0` (unbounded) | Per-model waiter bound; exceeded -> immediate 503 `queue_full` |

Per-model `capabilities.max_concurrency` (int) is a new advisory
manifest capability; the curated 32B entry
(`qwen2.5-32b-instruct-gguf-q4-k-m`, on the CPU box's catalog -- set
there via catalog manifest, not curated.yaml, since it was
resolver-pulled) documents `max_concurrency: 1` as the flagship use.

All three resolve through the config registry (env / config.yaml /
`muse config set`), lenient reads on the request path.

### Component 4: observability

- Telemetry `request` event gains `queued_ms` (total time parked in
  gate + capacity waits; 0 when never parked). New nullable column,
  same sparse-table pattern.
- `/v1/admin/memory` per-model breakdown gains `queue_depth` (live
  waiter count from the gate) beside `refcount`.
- `/v1/telemetry/summary` gains the same per-model `queue_depth`, so
  the dashboard can plot it and the federation coordinator can later
  consume it (the seam for queue-aware routing).
- Queue-timeout 503 body: `{"error": {"code": "queue_timeout",
  "message": "waited <N>s for model '<id>' (queue depth <D>)", ...}}`.
  Queue-full 503 body: code `queue_full`.

### Error semantics summary

| Situation | Status | Code |
|---|---|---|
| Waited past `queue_timeout_seconds` | 503 | `queue_timeout` |
| Waiters exceed `max_queue_depth` | 503 | `queue_full` |
| Permanent can't-fit (unchanged) | 503 | `model_too_large_for_device` / `model_unservable` |

## Testing

- **Unit -- ConcurrencyGate:** slot acquire/release balance; FIFO wake
  order; unlimited models bypass entirely; timeout -> 503 `queue_timeout`;
  depth bound -> 503 `queue_full`; release on BaseException.
- **Unit -- capacity-wait:** transient 503 -> park -> director release
  fires notifier -> retry succeeds; permanent 503 surfaces immediately;
  deadline exhaustion -> 503; notifier armed-before-attempt (no missed
  wakeup); `call_soon_threadsafe` used (fake loop asserts).
- **Unit -- director:** `retryable` flag set correctly on both capacity
  branches; notifier called on release-to-zero and eviction-commit;
  notifier exceptions swallowed.
- **Slow e2e:** a `max_concurrency=1` fake model, N=4 concurrent
  requests -> all 200, serialized (assert via per-request
  start/end timestamps), none 503 within the timeout window.
- **Live (GPU box, post-deploy):** 32B with cap 1 under 3 concurrent
  requests -> responses arrive serially, `/v1/admin/memory` shows
  `queue_depth` during the burst, `queued_ms` lands in telemetry.

## Rollout

Ship as v0.55.0. Backward compatible: with no caps declared and default
config, behavior is identical to v0.54.5 except that transient capacity
503s become bounded waits (strictly an improvement for clients; anyone
relying on instant capacity-503s can set `queue_timeout_seconds: 0`,
which degrades to today's immediate-503 behavior).
