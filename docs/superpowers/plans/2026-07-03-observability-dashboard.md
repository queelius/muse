# Observability Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `muse.observability` package that records structured telemetry events to SQLite, captures per-worker logs into ring buffers with live SSE tail, and serves a self-contained web dashboard at `GET /dashboard` (live state + charts + logs), gated by `admin.token`.

**Architecture:** Hot paths (director, gateway) call a fire-and-forget `observability.record(event)` that enqueues to a background flush thread writing SQLite; the supervisor runs the recorder + a periodic sampler + a `LogHub`; `spawn_worker` pipes each worker's stdout/stderr into the LogHub; the gateway FastAPI app mounts a dashboard router. Every unit is independently testable with no GPU / no weights.

**Tech Stack:** Python 3.10+, stdlib `sqlite3` + `queue` + `threading`, FastAPI (SSE via `sse-starlette` already a dep), the existing `muse.core.config` registry, `muse.core.memory_probe`, `muse.admin.auth`.

**Reference spec:** `docs/superpowers/specs/2026-07-03-observability-dashboard-design.md`. Branch: `feature/observability-dashboard`.

## Global Constraints

- **ASCII only** in all file content AND commit messages. NO em-dash characters (a hook rejects them). Use `--`, colon, comma, parentheses.
- **TDD red-green** every task: failing test, watch it fail, minimal code, watch it pass, commit.
- **Commit locally after each task.** Do NOT push, bump the version, tag, or release -- gated on a later explicit "go".
- **Commit trailers** (both, exactly):
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV`
- **Import-light:** `muse.observability.*` imports only stdlib at module top (sqlite3, queue, threading, collections, time, json, logging). NO torch/transformers. FastAPI/sse-starlette imported lazily inside `dashboard.py`'s `build_dashboard_router` (like the other routers). `muse --help` must stay instant.
- **Never block the hot path:** `record()` must never raise into the caller and never block; on queue-full it drops and counts.
- **Never log/echo the admin token.**
- **Fast lane green:** `pytest tests/ -q -m "not slow"` after every task.
- **Config-registry discipline:** new settings go through `muse.core.config.SETTINGS`; no stray `os.environ.get("MUSE_` (the meta-test `tests/core/test_no_stray_env_reads.py` enforces this).

## File Structure

- `src/muse/observability/__init__.py` -- public API re-exports: `record`, `get_recorder`, `reset_recorder`, `Event` types, `LogHub`, `build_dashboard_router`.
- `src/muse/observability/events.py` -- event dataclasses + `EVENT_COLUMNS` + `event_to_row`.
- `src/muse/observability/store.py` -- `TelemetryStore` (SQLite).
- `src/muse/observability/recorder.py` -- `TelemetryRecorder`, `_NoopRecorder`, module singleton `record`/`get_recorder`/`reset_recorder`/`init_recorder`.
- `src/muse/observability/logs.py` -- `LogHub`.
- `src/muse/observability/sampler.py` -- `Sampler`.
- `src/muse/observability/dashboard.py` -- `build_dashboard_router(state)` + `DASHBOARD_HTML`.
- Modify: `src/muse/core/config.py` (telemetry settings), `src/muse/cli_impl/load_director.py` (emit load/evict), `src/muse/cli_impl/gateway.py` (emit request + mount router), `src/muse/cli_impl/supervisor.py` (lifecycle + `spawn_worker` piping).
- Tests under `tests/observability/` + additions to `tests/cli_impl/`.

---

### Task 1: telemetry config settings

**Files:** Modify `src/muse/core/config.py`; Test `tests/core/test_config_registry.py`.

**Interfaces:** Produces registry keys `telemetry.enabled` (bool, True), `telemetry.retention_days` (int, 7), `telemetry.log_buffer_kb` (int, 64), `telemetry.sample_interval_seconds` (float, 10.0).

- [ ] **Step 1: failing test** -- append to `tests/core/test_config_registry.py`:
```python
def test_telemetry_settings_present():
    from muse.core import config as cfg
    for key, default in [
        ("telemetry.enabled", True),
        ("telemetry.retention_days", 7),
        ("telemetry.log_buffer_kb", 64),
        ("telemetry.sample_interval_seconds", 10.0),
    ]:
        assert key in cfg.SETTINGS_BY_KEY, key
        assert cfg.SETTINGS_BY_KEY[key].default == default
```
- [ ] **Step 2: run, expect FAIL** -- `pytest tests/core/test_config_registry.py::test_telemetry_settings_present -q`
- [ ] **Step 3: implement** -- add a `# --- telemetry ---` block to `SETTINGS` in `config.py` (after the `server` group):
```python
    Setting("telemetry.enabled", "MUSE_TELEMETRY_ENABLED",
            "bool", True, "telemetry",
            "Record telemetry events + serve the /dashboard observability UI."),
    Setting("telemetry.retention_days", "MUSE_TELEMETRY_RETENTION_DAYS",
            "int", 7, "telemetry", "Rolling retention window for telemetry events."),
    Setting("telemetry.log_buffer_kb", "MUSE_TELEMETRY_LOG_BUFFER_KB",
            "int", 64, "telemetry", "Per-model recent-log ring-buffer size (KB)."),
    Setting("telemetry.sample_interval_seconds", "MUSE_TELEMETRY_SAMPLE_INTERVAL_SECONDS",
            "float", 10.0, "telemetry", "Seconds between VRAM/RAM/loaded samples."),
```
- [ ] **Step 4: run, expect PASS**; also `pytest tests/core/ -q -m "not slow"`.
- [ ] **Step 5: commit** -- `feat(observability): telemetry config settings`

---

### Task 2: event model (`events.py`)

**Files:** Create `src/muse/observability/__init__.py` (empty for now), `src/muse/observability/events.py`; Test `tests/observability/test_events.py`.

**Interfaces:** Produces `EVENT_COLUMNS: tuple[str,...]` (the sparse column order, minus `id`), and `event_to_row(type: str, ts: float, **fields) -> dict` returning a dict with every column key present (None where absent) plus `ts` and `type`. Column set: `ts, type, model_id, pool, gb, latency_ms, status, reason, cold_load_seconds, stream, free_vram_gb, free_ram_gb, gpu_used_gb, loaded_count, in_flight_count, modality`.

- [ ] **Step 1: failing test** (`tests/observability/test_events.py`):
```python
from muse.observability.events import EVENT_COLUMNS, event_to_row

def test_event_to_row_fills_all_columns_with_none():
    row = event_to_row("request", 1.0, model_id="m", latency_ms=12.5, status=200)
    assert set(row) == set(EVENT_COLUMNS)
    assert row["type"] == "request" and row["ts"] == 1.0
    assert row["model_id"] == "m" and row["latency_ms"] == 12.5 and row["status"] == 200
    assert row["pool"] is None and row["free_vram_gb"] is None  # unset -> None

def test_event_to_row_rejects_unknown_field():
    import pytest
    with pytest.raises(ValueError):
        event_to_row("request", 1.0, bogus=1)
```
- [ ] **Step 2: run, expect FAIL** (ModuleNotFoundError).
- [ ] **Step 3: implement** `events.py`:
```python
from __future__ import annotations
from typing import Any

EVENT_COLUMNS: tuple[str, ...] = (
    "ts", "type", "model_id", "pool", "gb", "latency_ms", "status", "reason",
    "cold_load_seconds", "stream", "free_vram_gb", "free_ram_gb", "gpu_used_gb",
    "loaded_count", "in_flight_count", "modality",
)
_FIELD_COLUMNS = frozenset(EVENT_COLUMNS) - {"ts", "type"}


def event_to_row(type: str, ts: float, **fields: Any) -> dict[str, Any]:
    """Build a full sparse row dict (every column present, None where unset)."""
    unknown = set(fields) - _FIELD_COLUMNS
    if unknown:
        raise ValueError(f"unknown telemetry field(s): {sorted(unknown)}")
    row: dict[str, Any] = {c: None for c in EVENT_COLUMNS}
    row["ts"] = ts
    row["type"] = type
    row.update(fields)
    return row
```
Leave `__init__.py` empty (populated in Task 8/9 re-exports as they land; final re-export list added in the last wiring task).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): telemetry event model`

---

### Task 3: SQLite store (`store.py`)

**Files:** Create `src/muse/observability/store.py`; Test `tests/observability/test_store.py`.

**Interfaces:**
- Consumes: `EVENT_COLUMNS`, `event_to_row` (Task 2).
- Produces: `class TelemetryStore`:
  - `__init__(self, path: str | pathlib.Path)` -- opens/creates the db (WAL), creates schema.
  - `insert_many(self, rows: list[dict]) -> None` -- one `executemany`.
  - `prune(self, older_than_ts: float) -> int` -- delete + return rows removed.
  - `series(self, metric: str, since_ts: float, bucket_seconds: float) -> dict` -- aggregated points.
  - `summary_counts(self) -> dict` -- small helper for tests/summary (e.g. total events).
  - `close(self)`.
  - `metric` supported values: `request_rate`, `latency`, `vram`, `ram`, `load_evict`.

- [ ] **Step 1: failing test** (`tests/observability/test_store.py`):
```python
import pytest
from muse.observability.store import TelemetryStore
from muse.observability.events import event_to_row

@pytest.fixture
def store(tmp_path):
    s = TelemetryStore(tmp_path / "t.db")
    yield s
    s.close()

def test_insert_and_request_rate_bucketing(store):
    rows = [event_to_row("request", ts, model_id="m", latency_ms=10.0, status=200)
            for ts in (100.0, 101.0, 102.0, 160.0)]
    store.insert_many(rows)
    out = store.series("request_rate", since_ts=0.0, bucket_seconds=60.0)
    assert out["metric"] == "request_rate"
    # bucket [60,120) has 3 requests, [120,180) has 1
    counts = {p["t"]: p["count"] for p in out["points"]}
    assert counts[120.0] == 3 and counts[180.0] == 1

def test_latency_series(store):
    store.insert_many([event_to_row("request", 61.0, model_id="m", latency_ms=x, status=200)
                       for x in (10.0, 20.0, 30.0)])
    out = store.series("latency", since_ts=0.0, bucket_seconds=60.0)
    p = out["points"][0]
    assert p["avg"] == 20.0 and p["max"] == 30.0

def test_prune(store):
    store.insert_many([event_to_row("sample", ts, free_vram_gb=1.0) for ts in (10.0, 5000.0)])
    removed = store.prune(older_than_ts=100.0)
    assert removed == 1
    assert store.summary_counts()["total"] == 1
```
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** `store.py`. Schema exactly as the spec (sparse `events` table + `idx_events_ts`, `idx_events_type`); `PRAGMA journal_mode=WAL`. `insert_many` uses `executemany` over `EVENT_COLUMNS` order. `series` dispatches per `metric`:
  - `request_rate`: `SELECT CAST(ts/:b AS INT)*:b + :b AS t, COUNT(*) count WHERE type='request' GROUP BY t` (bucket label = bucket end).
  - `latency`: same bucketing, `AVG(latency_ms) avg, MAX(latency_ms) max` for `type='request'`.
  - `vram` / `ram`: `AVG(free_vram_gb)`/`AVG(free_ram_gb)` for `type='sample'`.
  - `load_evict`: counts of `type IN ('model_load','model_evict')` split into `loads`/`evicts` per bucket.
  Return `{"metric": metric, "points": [...]}`. Raise `ValueError` for an unknown metric.
  Open the connection with `check_same_thread=False` and guard writes with an internal `threading.Lock` (the recorder flush thread writes; the request thread reads).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): SQLite telemetry store`

---

### Task 4: recorder (`recorder.py`)

**Files:** Create `src/muse/observability/recorder.py`; Test `tests/observability/test_recorder.py`.

**Interfaces:**
- Consumes: `TelemetryStore`, `event_to_row`.
- Produces:
  - `class TelemetryRecorder`: `__init__(store, *, max_queue=10000, flush_interval=0.5)`; `record(type, **fields)` (non-blocking, uses `time.time()` for ts, `put_nowait`, drop+`self.dropped += 1` on Full); `flush(self)` (drain + `insert_many`); `start(self)` / `stop(self)` (daemon flush thread); `dropped: int`.
  - `class _NoopRecorder`: `record(*a, **k)` no-op, `dropped=0`, `start`/`stop`/`flush` no-op.
  - Module singleton: `init_recorder(store, *, enabled=True) -> None` (sets `_RECORDER` to a real or noop recorder and `start()`s it); `get_recorder()` (returns current or a `_NoopRecorder` if uninit); `record(type, **fields)` (delegates to `get_recorder().record`); `reset_recorder()` (stop + clear; test hook).

- [ ] **Step 1: failing test** (`tests/observability/test_recorder.py`):
```python
import time, pytest
from muse.observability.store import TelemetryStore
from muse.observability import recorder as rec

@pytest.fixture(autouse=True)
def _reset():
    rec.reset_recorder(); yield; rec.reset_recorder()

def test_record_enqueues_and_flush_writes(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, flush_interval=0.05)
    r.record("request", model_id="m", latency_ms=5.0, status=200)
    r.flush()
    assert store.summary_counts()["total"] == 1
    r.stop(); store.close()

def test_overflow_drops_not_raises(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, max_queue=2)
    for _ in range(10):
        r.record("sample", free_vram_gb=1.0)   # must never raise
    assert r.dropped >= 1
    r.stop(); store.close()

def test_module_record_is_noop_until_init(tmp_path):
    rec.record("request", model_id="m")   # no recorder yet -> silent no-op, no raise
    assert rec.get_recorder().dropped == 0

def test_disabled_init_is_noop(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    rec.init_recorder(store, enabled=False)
    rec.record("request", model_id="m")
    assert store.summary_counts()["total"] == 0
```
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** `recorder.py`. `record` wraps `event_to_row(type, time.time(), **fields)` and `put_nowait`; the flush thread loops on a stop `Event`, draining up to N per cycle. Wrap the flush-thread body in a broad `except Exception: logger.warning(...)` so a store error never kills the thread. `get_recorder()` returns a process-global `_RECORDER` or a shared `_NoopRecorder()`.
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): fire-and-forget telemetry recorder`

---

### Task 5: log hub (`logs.py`)

**Files:** Create `src/muse/observability/logs.py`; Test `tests/observability/test_logs.py`.

**Interfaces:** Produces `class LogHub`:
- `__init__(self, *, buffer_bytes=65536)`.
- `append(self, model_id: str, line: str) -> None` -- add to that model's ring buffer (byte-bounded: evict oldest lines until under `buffer_bytes`) AND publish to subscribers.
- `snapshot(self, model_id: str) -> list[str]` -- current buffer.
- `subscribe(self, model_id: str) -> queue.Queue` / `unsubscribe(self, model_id, q)`.
- `drop(self, model_id: str) -> None` -- remove a model's buffer + subscribers.

- [ ] **Step 1: failing test** (`tests/observability/test_logs.py`):
```python
from muse.observability.logs import LogHub

def test_snapshot_and_byte_bound():
    hub = LogHub(buffer_bytes=20)
    for i in range(10):
        hub.append("m", f"line{i}")   # each ~5-6 bytes; only most-recent fit
    snap = hub.snapshot("m")
    assert snap and snap[-1] == "line9"
    assert sum(len(s) for s in snap) <= 20

def test_pubsub_delivers_new_lines():
    hub = LogHub()
    q = hub.subscribe("m")
    hub.append("m", "hello")
    assert q.get_nowait() == "hello"
    hub.unsubscribe("m", q)
    hub.append("m", "after")
    assert q.qsize() == 0  # unsubscribed -> no more
```
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** `logs.py` -- `dict[model_id -> collections.deque]` for buffers with a running byte count, `dict[model_id -> set[queue.Queue]]` for subscribers; all mutations under one `threading.Lock`. `append` publishes with `put_nowait` inside a try/except (a slow/full subscriber must not block the reader thread; drop for that subscriber).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): per-model log hub (ring buffer + pub/sub)`

---

### Task 6: sampler (`sampler.py`)

**Files:** Create `src/muse/observability/sampler.py`; Test `tests/observability/test_sampler.py`.

**Interfaces:**
- Consumes: `record` (module fn), `muse.core.memory_probe` (`gpu_free_gb`, `cpu_free_gb` -- confirm exact names in the module), a `state`-like object exposing `director.loaded` (dict) and an in-flight count.
- Produces: `class Sampler`: `__init__(self, *, interval, loaded_fn, inflight_fn, record_fn=record)`; `start()`/`stop()` (daemon thread); `sample_once(self)` -- reads memory + counts and calls `record_fn("sample", free_vram_gb=..., free_ram_gb=..., loaded_count=..., in_flight_count=...)`.

- [ ] **Step 1: failing test** (`tests/observability/test_sampler.py`):
```python
from muse.observability.sampler import Sampler

def test_sample_once_records(monkeypatch):
    import muse.observability.sampler as smod
    monkeypatch.setattr(smod, "gpu_free_gb", lambda: 3.0)
    monkeypatch.setattr(smod, "cpu_free_gb", lambda: 20.0)
    seen = []
    s = Sampler(interval=999, loaded_fn=lambda: {"m": object()},
                inflight_fn=lambda: 2, record_fn=lambda t, **k: seen.append((t, k)))
    s.sample_once()
    assert seen[0][0] == "sample"
    k = seen[0][1]
    assert k["free_vram_gb"] == 3.0 and k["loaded_count"] == 1 and k["in_flight_count"] == 2
```
- [ ] **Step 2: run, expect FAIL** (adjust `gpu_free_gb`/`cpu_free_gb` import to the real memory_probe names; if `cpu_free_gb` does not exist, use the real function -- read `src/muse/core/memory_probe.py` first and match).
- [ ] **Step 3: implement** `sampler.py` importing the real memory-probe helpers at module top (they are stdlib+psutil/pynvml-guarded, safe), with a daemon loop `while not stop.wait(interval): self.sample_once()` wrapped in try/except so a probe error never kills the thread.
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): periodic VRAM/RAM/loaded sampler`

---

### Task 7: dashboard auth helper

**Files:** Create `src/muse/observability/dashboard_auth.py` (small) OR add to `dashboard.py`; Test `tests/observability/test_dashboard_auth.py`.

**Interfaces:** Produces `check_dashboard_token(bearer: str | None, access_token: str | None) -> None` that raises the same OpenAI-shape `HTTPException` the admin gate uses:
- no `admin.token` configured -> 503 `dashboard_closed`.
- token configured, neither bearer nor access_token supplied -> 401 `missing_token`.
- supplied value mismatches (constant-time) -> 403 `invalid_token`.
- match -> return None.
Reuse `muse.admin.auth._err` / `error_type_for_status` for the envelope; read `admin.token` via `config.get("admin.token")`.

- [ ] **Step 1: failing test** covering the four branches (no token -> 503; missing -> 401; wrong -> 403; right via bearer OR via access_token -> ok). Use `monkeypatch.setenv("MUSE_ADMIN_TOKEN", ...)` + `config.reset_config()`.
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** using `secrets.compare_digest`; strip the configured token (match `auth.py`'s trailing-newline defense). A FastAPI dependency wrapper `require_dashboard_auth(authorization: str | None = Header(None), access_token: str | None = Query(None))` calls it.
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): dashboard auth (header + SSE query param, closed by default)`

---

### Task 8: dashboard router + HTML (`dashboard.py`)

**Files:** Create `src/muse/observability/dashboard.py`; Test `tests/observability/test_dashboard_router.py`.

**Interfaces:**
- Consumes: `TelemetryStore` (via `state`), `LogHub` (via `state`), `require_dashboard_auth`, `get_recorder`.
- Produces: `build_dashboard_router(state) -> APIRouter` with `GET /dashboard` (returns `DASHBOARD_HTML`, un-gated shell), `GET /v1/telemetry/summary`, `GET /v1/telemetry/series`, `GET /v1/telemetry/logs/{model_id}` (SSE). `state` must expose `state.telemetry_store`, `state.log_hub`, `state.director.loaded`, and a node id (`state.node_url` or host). `DASHBOARD_HTML: str`.

- [ ] **Step 1: failing test** (`tests/observability/test_dashboard_router.py`) -- build a `FastAPI()`, include the router with a fake `state` (a `SimpleNamespace` with a tmp `TelemetryStore`, a `LogHub`, and a `director` with `.loaded={}`), drive with `TestClient`:
```python
def test_summary_requires_token(client_no_token):
    assert client_no_token.get("/v1/telemetry/summary").status_code == 503

def test_summary_ok_with_token(client_with_token):
    r = client_with_token.get("/v1/telemetry/summary",
                              headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    assert "loaded" in r.json() and "node" in r.json()

def test_series_ok(client_with_token):
    r = client_with_token.get("/v1/telemetry/series?metric=request_rate&window=3600",
                              headers={"Authorization": "Bearer t"})
    assert r.status_code == 200 and r.json()["metric"] == "request_rate"

def test_dashboard_html_is_self_contained(client_with_token):
    body = client_with_token.get("/dashboard").text
    # no external resource loads (works internet-exposed / strict CSP)
    import re
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', body)
```
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** `dashboard.py`. `summary` returns `{"node": <id>, "loaded": [ {model_id, pool, gb, last_used} for each state.director.loaded ], "in_flight": ..., "dropped_events": get_recorder().dropped}`. `series` maps `window` seconds -> `since_ts = time.time() - window`, picks a bucket (e.g. `window/60`), calls `store.series`. `logs/{model_id}` returns an `EventSourceResponse` (from `sse_starlette.sse`) that first yields the `hub.snapshot(model_id)` lines then streams from `hub.subscribe(model_id)` until client disconnect (unsubscribe in a finally). Gate summary/series/logs with `require_dashboard_auth` (logs reads the `access_token` query param). `DASHBOARD_HTML` is a complete self-contained page: inline CSS + JS, a token prompt storing to `sessionStorage`, `fetch` with the bearer header for summary/series (poll every 2s), an inline SVG line-chart helper, and an `EventSource("/v1/telemetry/logs/<m>?access_token=<t>")` log panel. Keep it minimal but functional (no external URLs).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** -- `feat(observability): dashboard router + self-contained HTML`

---

### Task 9: director instrumentation (load / evict events)

**Files:** Modify `src/muse/cli_impl/load_director.py`; Test `tests/cli_impl/test_load_director_telemetry.py`.

**Interfaces:** Consumes `observability.record`. At the point a cold load commits (after `_load_and_commit` success) call `record("model_load", model_id=..., pool=..., gb=..., cold_load_seconds=...)`; at each eviction (in `_evict_lru_until_fits` / idle path where a `LoadEntry` is removed) call `record("model_evict", model_id=..., pool=..., reason=...)`. Import `from muse.observability import record` at module top (import-light, safe).

- [ ] **Step 1: failing test** -- monkeypatch `muse.cli_impl.load_director.record` to capture calls; drive a load and an eviction (reuse existing director test fixtures) and assert a `model_load` and a `model_evict` event were recorded with the right model_id/pool.
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** the two `record(...)` calls at the identified sites. They are fire-and-forget; do not let them alter control flow.
- [ ] **Step 4: run, expect PASS**; `pytest tests/cli_impl/ -q -m "not slow"`.
- [ ] **Step 5: commit** -- `feat(observability): record model load/evict events from the director`

---

### Task 10: gateway instrumentation + mount router

**Files:** Modify `src/muse/cli_impl/gateway.py`; Test `tests/cli_impl/test_gateway_telemetry.py`.

**Interfaces:** Consumes `observability.record`, `build_dashboard_router`. In the forward-with-release path, on response completion record `record("request", model_id=..., modality=<derived from path>, latency_ms=..., status=..., stream=...)` (measure latency around the forward; capture status from the response). In `build_gateway`, when `config.get("telemetry.enabled")`, mount `build_dashboard_router(state)`.

- [ ] **Step 1: failing test** -- monkeypatch `gateway.record`; drive a forwarded request through the test harness and assert one `request` event with a numeric `latency_ms` and the response status. Separately assert the dashboard router is mounted (a `GET /dashboard` on the built app returns 200).
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement.** Wrap the existing forward timing; derive modality from the request path prefix (e.g. `/v1/chat/completions` -> `chat/completion`) via a small static map or the path. Mount the router guarded by the telemetry flag.
- [ ] **Step 4: run, expect PASS**; whole fast lane.
- [ ] **Step 5: commit** -- `feat(observability): record request events + mount dashboard router in the gateway`

---

### Task 11: supervisor lifecycle + worker log piping

**Files:** Modify `src/muse/cli_impl/supervisor.py`; Test `tests/cli_impl/test_supervisor_telemetry.py`.

**Interfaces:** Consumes `TelemetryStore`, `init_recorder`, `Sampler`, `LogHub`. On boot (in `run_supervisor`, when `telemetry.enabled`): create `TelemetryStore(<catalog_dir>/telemetry.db)`, `init_recorder(store)`, create a `LogHub(buffer_bytes=log_buffer_kb*1024)`, start a `Sampler(interval=sample_interval, loaded_fn=lambda: state.director.loaded, inflight_fn=...)`, and attach `state.telemetry_store` + `state.log_hub` so the gateway router can read them. Add a retention-prune tick (reuse the idle-sweep thread or a small periodic call: `store.prune(time.time() - retention_days*86400)`). Change `spawn_worker`: `Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)` and start a daemon reader thread that for each line calls `state.log_hub.append(model_id, line)` and re-emits `print(f"[{model_id}] {line}", end="")` so the aggregate log still shows it. Drop the model's LogHub buffer when the worker is removed.

- [ ] **Step 1: failing test** -- (a) a helper that, given a fake `Popen` whose `stdout` yields two lines, runs the reader loop and asserts both lines land in a `LogHub`; (b) `run_supervisor` boot (in-process, mocked) with `telemetry.enabled` sets `state.telemetry_store` and `state.log_hub` and starts a recorder. Mock the subprocess; no real worker.
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement.** Factor the reader loop into a small testable function `def _pump_worker_logs(proc, model_id, hub)`. Guard everything behind the telemetry flag so `telemetry.enabled=false` keeps today's behavior (bare `Popen(cmd)`, no LogHub).
- [ ] **Step 4: run, expect PASS**; whole fast lane.
- [ ] **Step 5: commit** -- `feat(observability): supervisor telemetry lifecycle + per-worker log piping`

---

### Task 12: public API re-exports, docs, CONFIG.md

**Files:** Modify `src/muse/observability/__init__.py`, `CLAUDE.md`, `README.md`, `docs/CONFIG.md`; Test `tests/observability/test_public_api.py`.

- [ ] **Step 1: failing test** -- `from muse.observability import record, get_recorder, reset_recorder, init_recorder, LogHub, build_dashboard_router, TelemetryStore` all import.
- [ ] **Step 2: run, expect FAIL.**
- [ ] **Step 3: implement** the `__init__.py` re-exports. Add: a `## Observability` section to `CLAUDE.md` (the package, the event model, log piping, `/dashboard`, the telemetry settings, closed-by-default auth); a short README subsection; and the four `telemetry.*` rows to the `docs/CONFIG.md` settings tables (new `telemetry` group).
- [ ] **Step 4: run, expect PASS**; final `pytest tests/ -q -m "not slow"`.
- [ ] **Step 5: commit** -- `docs(observability): public API + CLAUDE.md/README/CONFIG.md`

---

## Self-Review

- **Spec coverage:** telemetry store (T3), recorder fire-and-forget + drop (T4), event model (T2), sampler (T6), log capture Approach-1 + piping (T5+T11), dashboard HTML + endpoints (T8), auth closed-by-default (T7), config knobs (T1), instrumentation (T9 director, T10 gateway, T11 supervisor), federation-forward `node` id in summary (T8), docs (T12). Every spec section maps to a task.
- **Import-light:** observability modules import stdlib only at top; FastAPI/sse-starlette lazy in dashboard.py (T8); director/gateway/supervisor already import heavy libs so `from muse.observability import record` there is fine.
- **Type consistency:** `record(type, **fields)` / `event_to_row(type, ts, **fields)` / `EVENT_COLUMNS` / `TelemetryStore.series(metric, since_ts, bucket_seconds)` / `LogHub.append/snapshot/subscribe/unsubscribe/drop` / `build_dashboard_router(state)` used identically across tasks.
- **No placeholders:** each task carries real test + implementation code or the exact modification + covering test.
- **Deferred (not in this plan):** searchable log store, alerting, multi-node aggregation, generation-endpoint auth, exact latency percentiles (v1 uses avg+max per T3).
