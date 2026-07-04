# Observability dashboard (single-node) -- design

**Goal:** Each muse node serves a self-contained web dashboard at `GET
/dashboard` showing (1) live node state, (2) historical metrics as charts,
and (3) real-time per-model log streaming -- so an operator can open a browser
at any node and see what it is doing. Single-node now; built federation-forward
so a later cluster dashboard aggregates peers with the same shape.

First sub-project of the federation arc (observability is the load-signal
substrate federation's smart routing will consume, and the visible payoff).

## Problem

muse has no operability surface. To see what a node is doing you read the raw
`muse-serve.log` (all workers melted together, since `spawn_worker` does a bare
`subprocess.Popen(cmd)` and workers inherit the supervisor's stdout). There is
no history (loads/evictions, request rate, latency, VRAM over time), no
per-model log view, and nothing a browser can render. The admin API exposes
point-in-time JSON but no time series and no UI.

## Goals / non-goals

In scope:
- Structured telemetry events -> SQLite, with bounded rolling retention.
- Per-worker stdout/stderr capture into in-memory ring buffers + live SSE tail.
- A self-contained (no CDN / external services) HTML dashboard at `/dashboard`
  with live state, charts, and a log panel.
- Everything gated by `admin.token` (closed by default), safe to internet-expose.
- Data endpoints shaped so a future cluster dashboard can fan out to peers.

Out of scope (deferred, YAGNI):
- Searchable/persisted historical log store (Loki/ELK-scale). Logs are live
  tail + a short recent buffer only.
- Alerting / notifications.
- Multi-node aggregation (this is single-node; federation is a later cycle).
- Auth on the generation endpoints (`/v1/chat/completions` etc.). Noted as a
  separate future decision for internet exposure; not part of this work.

## Architecture

One small package, `muse.observability`, with clean seams:

```
director / gateway ---> telemetry.record(event)   (fire-and-forget, non-blocking)
                              |
supervisor (always-on) owns:  +-- TelemetryRecorder --> SQLite telemetry.db
                              +-- periodic Sampler (VRAM/RAM/loaded/in-flight)
                              +-- LogHub: per-worker ring buffers + SSE pub/sub
                                        ^
spawn_worker pipes each worker's stdout/stderr into the LogHub

gateway FastAPI app mounts the dashboard router:
  GET /dashboard                      self-contained HTML
  GET /v1/telemetry/summary           live state JSON
  GET /v1/telemetry/series?metric&window   chart data
  GET /v1/telemetry/logs/{model_id}   SSE log stream
  (all gated by admin.token)
```

Design principle: the hot paths (director, gateway) only call
`telemetry.record(...)` -- a fire-and-forget enqueue that never blocks a request
and never raises into the caller. They know nothing about SQLite, threads, or
SSE. `muse.observability` is import-light (stdlib + sqlite3; no torch/fastapi at
module top so `muse --help` stays fast). The dashboard router imports fastapi
lazily like the other routers.

Files:
- `src/muse/observability/__init__.py` -- public API: `record(event)`,
  `get_recorder()`, `LogHub`, `build_dashboard_router(state)`.
- `src/muse/observability/events.py` -- event dataclasses + types.
- `src/muse/observability/store.py` -- `TelemetryStore` (SQLite: schema,
  batched writes, retention prune, series queries).
- `src/muse/observability/recorder.py` -- `TelemetryRecorder` (bounded queue +
  background flush thread; drops on overflow rather than blocking).
- `src/muse/observability/sampler.py` -- periodic sampler thread.
- `src/muse/observability/logs.py` -- `LogHub` (per-model ring buffer +
  pub/sub for SSE).
- `src/muse/observability/dashboard.py` -- `build_dashboard_router` + the
  inline HTML (`DASHBOARD_HTML`).
- `src/muse/observability/auth.py` OR reuse `muse.admin.auth` -- token check
  usable both as a header dependency and a query-param check for SSE.

## Telemetry event model

Emit STRUCTURED events; never reconstruct metrics by parsing logs. Event types:

- `model_load` -- {model_id, pool ("cuda"/"cpu"/...), gb, cold_load_seconds}
- `model_evict` -- {model_id, pool, reason ("lru"/"idle_timeout"/"admin"/...)}
- `request` -- {model_id, modality, latency_ms, status (int), stream (bool)}
  recorded once, on completion, by the gateway forward wrapper.
- `sample` -- {free_vram_gb, free_ram_gb, gpu_used_gb, loaded_count,
  in_flight_count} written every `telemetry.sample_interval_seconds`.

Storage: one wide sparse `events` table (avoids per-type tables and avoids a
stringly-typed JSON blob):

```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,           -- epoch seconds
  type TEXT NOT NULL,         -- model_load | model_evict | request | sample
  model_id TEXT,
  pool TEXT,
  gb REAL, latency_ms REAL, status INTEGER, reason TEXT,
  cold_load_seconds REAL, stream INTEGER,
  free_vram_gb REAL, free_ram_gb REAL, gpu_used_gb REAL,
  loaded_count INTEGER, in_flight_count INTEGER, modality TEXT
);
CREATE INDEX idx_events_ts   ON events(ts);
CREATE INDEX idx_events_type ON events(ts, type);
```

Location: `<catalog_dir>/telemetry.db` (honors `MUSE_CATALOG_DIR`). WAL mode
for concurrent read (dashboard) + write (recorder).

Retention: a prune runs on the flush thread (e.g. every flush, cheaply) or on
an interval: `DELETE FROM events WHERE ts < now - retention_days*86400`, then a
periodic `PRAGMA wal_checkpoint`/occasional `VACUUM` guard against unbounded
file growth.

Recorder: `record(event)` puts the event on a bounded `queue.Queue`
(non-blocking `put_nowait`; on full, drop + increment a dropped counter -- the
hot path must never block on telemetry). A single daemon flush thread drains
the queue in batches and does one executemany insert per batch (few fsyncs).
`get_recorder()` returns a process singleton; a no-op recorder is used when
`telemetry.enabled` is false so call sites are unconditional.

Series queries (`store.series(metric, window, bucket)`): pre-aggregate in SQL
into time buckets -- request rate (count per bucket), latency p50/p95 (SQLite
has no percentile; compute via ordered subquery or approximate with
avg+max for v1, exact percentile is a documented follow-up), VRAM/RAM
over time (avg of `sample` rows per bucket), load/evict counts per bucket.
Returns `{metric, window, points: [{t, ...}]}` JSON.

## Log capture + streaming (Approach 1)

Change `supervisor.spawn_worker`:
`subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`.
Per worker, start a daemon reader thread that, for each line:
1. appends to `LogHub`'s per-`model_id` ring buffer
   (`collections.deque`, bounded by ~`telemetry.log_buffer_kb` of text),
2. publishes the line to any live SSE subscribers for that model,
3. re-emits to the supervisor's own stdout prefixed `[model_id] ...` so the
   aggregate `muse-serve.log` and the operator's terminal still see everything.

`LogHub`:
- `buffers: dict[model_id -> deque]` (recent lines, byte-bounded).
- `subscribe(model_id) -> queue`, `unsubscribe(...)`, `publish(model_id, line)`
  -- a simple in-process pub/sub; each SSE connection is one subscriber queue.
- `snapshot(model_id) -> list[str]` -- the recent buffer, sent first when a
  client opens the panel, then live lines follow.
- Buffer for a model is created on first spawn and dropped when the worker is
  removed (or retained a short grace so you can read a crashed worker's tail).

Reader-thread lifecycle: the thread exits when the pipe closes (worker died);
the auto-restart monitor already respawns workers, and a respawn starts a fresh
reader. No blocking of the supervisor's main loop (reader is its own daemon
thread; the pipe is always drained so the worker never blocks on a full pipe).

## Dashboard UI

`GET /dashboard` returns one self-contained HTML document: inline CSS + JS, no
external requests (works behind a strict CSP and when internet-exposed). Three
panels:

1. **Live state** -- loaded models (with pool + gb + last-used), VRAM/RAM
   gauges, in-flight count. Polls `/v1/telemetry/summary` (or a light SSE) every
   couple seconds.
2. **Charts** -- request rate, latency, VRAM-over-time, load/evict timeline,
   from `/v1/telemetry/series`. Rendered as inline SVG by a tiny built-in
   helper (line/area/bar from a points array). No chart-library dependency.
3. **Logs** -- pick a loaded model -> opens an `EventSource` to
   `/v1/telemetry/logs/{model_id}`; shows the recent buffer then live-tails.

Auth in the browser: on first load the page prompts for the admin token, stores
it in `sessionStorage`, sends it as `Authorization: Bearer <t>` on fetch/XHR,
and appends it as `?access_token=<t>` on the `EventSource` URL (because
`EventSource` cannot set headers). A 401 clears the stored token and re-prompts.

## Endpoints (all closed by default; require `admin.token`)

| endpoint | returns |
|---|---|
| `GET /dashboard` | self-contained HTML |
| `GET /v1/telemetry/summary` | `{node, loaded:[...], pools:{...}, in_flight, dropped_events}` |
| `GET /v1/telemetry/series?metric=&window=&bucket=` | `{metric, window, points:[...]}` |
| `GET /v1/telemetry/logs/{model_id}` | `text/event-stream` (recent buffer, then live) |

Auth: a shared check usable as a FastAPI header dependency (for JSON/HTML) and
as a query-param validator (for SSE `access_token`), both constant-time
(`secrets.compare_digest`). Behavior, matching the admin gate's closed-by-default
posture (`muse.admin.auth`):
- **No `admin.token` configured** -> `503` `dashboard_closed` (never open; the
  dashboard is not world-readable even without a token set -- the operator must
  configure a token to view it, the safe default for an internet-exposed box).
- **Token configured, request missing the bearer / `access_token`** -> `401`.
- **Token configured, wrong value** -> `403` (constant-time compare).
All in the OpenAI-shape `{"error":{code,message,type}}` envelope. The
`/dashboard` HTML page itself may load (it is a static shell that then prompts
for the token and calls the gated data endpoints), OR also gate -- v1 gates the
data endpoints and lets the shell load so the browser can show a clean login
prompt rather than a raw 401 body.

## Config (new `telemetry` group in the settings registry)

- `telemetry.enabled` (bool, default true) -- master switch; false installs a
  no-op recorder + hides the routes.
- `telemetry.retention_days` (int, default 7) -- rolling window for `events`.
- `telemetry.log_buffer_kb` (int, default 64) -- per-model recent-log ring size.
- `telemetry.sample_interval_seconds` (float, default 10.0) -- sampler cadence.

## Federation-forward shape

`/v1/telemetry/summary` includes a `node` identity (host/url). The future
cluster dashboard aggregates peers' `/v1/telemetry/summary` + `/series` exactly
like the existing `/v1/models` fan-out, and the log panel can target
`{node, model_id}`. No cluster logic in this cycle; just the shape.

## Testing

- `store`: schema init, batched insert, retention prune boundary, series
  bucketing/aggregation (deterministic ts via injected clock; no `Date.now()`
  reliance -- pass timestamps in).
- `recorder`: enqueue is non-blocking; overflow drops + counts, never raises;
  flush thread batches; no-op recorder when disabled.
- `logs`: ring buffer byte-bound + eviction; pub/sub delivers to subscribers;
  snapshot-then-live ordering; unsubscribe cleanup.
- `sampler`: emits `sample` events at the configured cadence (mock memory_probe;
  no real GPU).
- `dashboard router`: each endpoint returns the right shape; auth gate returns
  401 without a token and 200 with it; SSE `access_token` query-param path
  authenticates; `/dashboard` HTML has no external URLs (assert no `http`
  external refs / self-contained).
- `spawn_worker` log piping: a fake worker process emitting lines lands in the
  LogHub buffer and reaches a subscriber (in-process, mocked Popen).
- Registry meta-test still holds (new telemetry settings go through the config
  registry; no stray env reads).

Follow the existing FakeModel / in-process patterns; no real weights or GPU.

## Acceptance criteria

- Opening `/dashboard` (with a valid token) shows loaded models, live VRAM/RAM,
  charts backed by real recorded events, and a live-tailing log panel per model.
- Every model_load/evict, request completion, and periodic sample is recorded;
  the hot path is never blocked by telemetry (verified: overflow drops).
- Per-model logs stream live AND still appear (prefixed) in the aggregate log.
- With `admin.token` set, all telemetry endpoints reject unauthenticated
  callers (header and SSE query-param paths); with no token, the endpoints are
  closed, never open.
- `telemetry.enabled=false` removes all overhead (no-op recorder, no routes).
- Full fast lane green; `muse --help` still instant (no heavy imports added to
  the import path).

## Open follow-ups (documented, not built here)

- Exact latency percentiles (v1 may approximate).
- Retained/searchable logs, alerting, multi-node aggregation, generation-endpoint
  auth -- all deferred to their own cycles.
