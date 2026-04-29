# Admin REST API + `muse models info` polish (v0.28.0)

**Date:** 2026-04-28
**Driver:** ship runtime model control via HTTP under `/v1/admin/*`,
plus a polished `muse models info` CLI. Closes task #148.

This is **not** a modality. It is infrastructure: a separate concern
from generation. `/v1/admin/*` lets users (and future MCP clients via
task #149) enable, disable, probe, pull, and remove models without
restarting `muse serve`. The `muse models info` polish is a CLI-side
companion that surfaces both static catalog metadata and live worker
state in an organized layout.

## Goal

1. Mount eleven endpoints under `/v1/admin/*` on the gateway:
   - `POST /v1/admin/models/{id}/enable` (async, 202+job_id)
   - `POST /v1/admin/models/{id}/disable` (sync)
   - `POST /v1/admin/models/{id}/probe` (async)
   - `POST /v1/admin/models/{id}/pull` (async)
   - `DELETE /v1/admin/models/{id}?purge=bool` (sync)
   - `GET /v1/admin/models/{id}/status` (sync)
   - `GET /v1/admin/memory` (sync)
   - `GET /v1/admin/workers` (sync)
   - `POST /v1/admin/workers/{port}/restart` (sync)
   - `GET /v1/admin/jobs/{job_id}` (sync)
   - `GET /v1/admin/jobs` (sync)
2. All admin endpoints require `Authorization: Bearer <token>` matching
   the `MUSE_ADMIN_TOKEN` environment variable. With no token configured,
   admin endpoints reject every request with `503 admin_disabled`. This
   is closed-by-default: muse-as-root could be hijacked otherwise.
3. Async operations (enable, pull, probe) return `202 Accepted` with
   `{"job_id": "...", "status": "pending"}`. Status is polled via
   `GET /v1/admin/jobs/{job_id}`. Jobs persist in-memory in the gateway
   for ten minutes after completion; older jobs are reaped.
4. Sync operations (disable, remove, status, memory, workers, restart)
   return immediately with the operation result.
5. Refactor supervisor: introduce a `SupervisorState` singleton holding
   the worker list + device + a global lock. Admin endpoints read and
   mutate this state. The existing restart-monitor thread keeps working
   on the same `specs` list; nothing about the monitor changes.
6. Polish `muse models info <id>` to organized sections:
   header (id + status), basics (modality / hf_repo / description /
   license / source), storage (weights / venv), memory (annotated /
   measured), capabilities (per-modality known-flag display), worker
   status (live data when available; "not running" otherwise).
7. Add an `AdminClient` Python helper (`muse.admin.client`) so
   programmatic users get the same surface without raw HTTP.

## Non-goals

- **Cross-process arbitration.** Admin operations assume one supervisor
  process. Multi-host muse clusters are out of scope.
- **Persistent jobs.** JobStore is in-memory. Restarting `muse serve`
  drops in-flight job records. A pull that survives a restart will land
  in the catalog (the subprocess updates catalog.json directly), but
  callers polling its job_id after restart get a 404.
- **WebSocket progress streams.** Pull and probe progress is observable
  via repeated polling. Streaming progress is filed for v1.next.
- **OAuth / per-user tokens / RBAC.** One bearer token gates all admin
  access. Production deployments fronting muse with nginx + their own
  auth layer can drop the token check via a reverse-proxy header rewrite,
  but muse itself does not implement OAuth.
- **MCP integration.** Task #149 is a separate effort that adds an MCP
  server consuming this admin API plus the inference routes.
- **Per-model lock granularity.** One global lock guards SupervisorState
  mutations for v1. If admin contention becomes measurable, swap to a
  per-model lock without a wire change.
- **enable-without-load mode.** The state model is one axis: `enabled`.
  enable means catalog.enabled=true AND loaded into a worker. disable
  means catalog.enabled=false AND unloaded. There is no "loaded but
  marked-disabled" intermediate.
- **Admin metrics endpoint.** No `/v1/admin/metrics` (Prometheus,
  request count, etc.). Filed for a future observability pass.

## Architecture

```
HTTP Authorization: Bearer <MUSE_ADMIN_TOKEN>
    |
    v
Gateway FastAPI app (port 8000)
    |
    +-- existing /v1/* generation routes (unchanged)
    +-- NEW /v1/admin/* admin routes
            |
            +-- auth dependency (verify_admin_token)
            +-- /models/{id}/enable -> AdminOps.enable_model -> JobStore + Thread
            +-- /models/{id}/disable -> AdminOps.disable_model (sync)
            +-- /models/{id}/probe -> AdminOps.probe_model -> JobStore + subprocess
            +-- /models/{id}/pull -> AdminOps.pull_model -> JobStore + subprocess
            +-- DELETE /models/{id} -> AdminOps.remove_model (sync)
            +-- /models/{id}/status -> read SupervisorState + catalog
            +-- /memory -> psutil + GPU-mem read of each worker
            +-- /workers -> list workers from SupervisorState
            +-- /workers/{port}/restart -> SIGTERM; restart-monitor handles bringup
            +-- /jobs/{id} -> JobStore.get
            +-- /jobs -> JobStore.list
```

`SupervisorState` is a module-level singleton in
`muse.cli_impl.supervisor`. The supervisor sets it during
`run_supervisor`; the gateway reads it via
`muse.cli_impl.supervisor.get_supervisor_state`. When admin endpoints
load before a supervisor is running (rare; e.g. tests), they get a
sentinel state that returns clear errors instead of crashing.

## Wire contracts

All admin endpoints accept and return JSON. All errors use the OpenAI
envelope shape:

```json
{"error": {"code": "...", "message": "...", "type": "invalid_request_error"}}
```

### POST /v1/admin/models/{id}/enable

Body: `{}` (or omitted).

Effects (asynchronous):
1. Set `catalog.enabled = True` for `{id}`.
2. Reload supervisor's planned worker grouping.
3. If a new worker venv-group must be spawned, spawn it.
4. If the model can join an existing worker, restart that worker
   with the augmented model list.
5. Wait for `/health` of the affected worker.
6. Mark job done with `{"worker_port": ..., "loaded": true}`.

Response (immediate):
```json
{"job_id": "abc123", "status": "pending"}
```

HTTP 202 Accepted on success, 404 if the model id is unknown.

### POST /v1/admin/models/{id}/disable

Body: `{}`.

Effects (synchronous):
1. Set `catalog.enabled = False` for `{id}`.
2. Find the worker hosting the model; if multiple models share that
   worker, restart it without `{id}`. If `{id}` was the only model in
   that worker, terminate the worker outright.

Response (immediate, sync):
```json
{
  "model_id": "soprano-80m",
  "loaded": false,
  "worker_terminated": true,
  "remaining_models_in_worker": []
}
```

HTTP 200 on success, 404 if the model id is unknown, 409 if the model
was already disabled.

### POST /v1/admin/models/{id}/probe

Body: `{"no_inference": false, "device": null}`.

Effects (asynchronous): equivalent to `muse models probe <id>`. Spawns
a subprocess in the model's venv; persists the resulting measurement
record into `catalog.json` under `measurements.<device>`.

Response: `202 Accepted` + `{"job_id": "...", "status": "pending"}`.

### POST /v1/admin/models/{id}/pull

Body: `{}`. The path id is the resolver URI or curated alias to pull,
URL-encoded. Practical clients call this with the raw identifier in the
JSON body instead, since `hf://` URIs do not survive path encoding well:

```http
POST /v1/admin/models/_/pull
Content-Type: application/json

{"identifier": "hf://Qwen/Qwen3-8B-GGUF@q4_k_m"}
```

The path id `_` is the documented placeholder when the body carries
the identifier. This avoids a special path matcher.

Effects (asynchronous): subprocess invokes `muse pull <identifier>`
and tails its stdout/stderr into `job.log_lines`. On success, the
catalog has the new entry; on failure, the job's `error` field carries
the subprocess stderr.

Response: `202 Accepted` + `{"job_id": "...", "status": "pending"}`.

### DELETE /v1/admin/models/{id}?purge=true

Effects (synchronous):
1. If the model is currently loaded into a worker, refuse with
   `409 model_loaded`. Caller must `disable` first.
2. Remove the catalog entry. If `purge=true`, also `rmtree` the venv.

Response: `200` + `{"model_id": "...", "removed": true, "purged": true}`.

### GET /v1/admin/models/{id}/status

Response (synchronous):
```json
{
  "model_id": "soprano-80m",
  "modality": "audio/speech",
  "enabled": true,
  "loaded": true,
  "worker_port": 9001,
  "worker_pid": 47821,
  "worker_uptime_seconds": 8123,
  "worker_status": "running",
  "restart_count": 0,
  "last_error": null,
  "measurements": {"cuda": {"weights_bytes": 322, ...}}
}
```

When the model is in the catalog but not loaded, `loaded=false` and the
worker fields are null.

### GET /v1/admin/memory

Response: per-device aggregate plus per-model breakdown:
```json
{
  "gpu": {
    "device": "cuda:0",
    "used_gb": 12.4,
    "total_gb": 24.0,
    "headroom_gb": 11.6,
    "models": [
      {"model_id": "qwen3.5-9b-q4", "weights_gb": 5.6, "peak_gb": 6.8},
      {"model_id": "sd-turbo", "weights_gb": 4.0, "peak_gb": 4.6}
    ]
  },
  "cpu": {
    "used_gb": 6.1,
    "total_gb": 64.0,
    "models": [{"model_id": "kokoro-82m", "weights_gb": 0.4}]
  }
}
```

The GPU section is `null` when no GPU is present or `pynvml` is not
installed. Per-model bytes come from the persisted probe measurement
when present, falling back to the manifest's `capabilities.memory_gb`
annotation.

### GET /v1/admin/workers

Response: list of worker spec records:
```json
{
  "workers": [
    {
      "port": 9001,
      "models": ["soprano-80m", "kokoro-82m"],
      "pid": 47821,
      "uptime_seconds": 8123,
      "restart_count": 0,
      "status": "running"
    }
  ]
}
```

### POST /v1/admin/workers/{port}/restart

Effects (synchronous): SIGTERM the worker by its port. The supervisor's
existing restart-monitor thread sees the death and respawns it on the
next tick. Returns immediately with the pre-restart spec; clients can
poll `/v1/admin/workers` to observe the bringup.

Response: `200` + `{"port": 9001, "signal": "SIGTERM", "message": "..."}`.

### GET /v1/admin/jobs/{job_id}

Response (synchronous):
```json
{
  "job_id": "abc123",
  "op": "enable",
  "model_id": "soprano-80m",
  "state": "done",
  "started_at": "2026-04-28T22:11:00Z",
  "finished_at": "2026-04-28T22:11:18Z",
  "result": {"worker_port": 9001, "loaded": true},
  "error": null,
  "log_lines": ["spawning worker...", "ready"]
}
```

State machine: `pending -> running -> (done | failed)`. `failed` carries
a non-null `error` field; `done` carries a non-null `result` field.

`404 job_not_found` when the job_id is unknown or has already been
reaped (after ten minutes).

### GET /v1/admin/jobs

Response: list of recent jobs (most recent first, capped at 100):
```json
{
  "jobs": [
    {"job_id": "abc123", "op": "enable", "model_id": "soprano-80m",
     "state": "done", "started_at": "..."},
    ...
  ]
}
```

## Auth model

The token check is implemented as a FastAPI dependency:

```python
def verify_admin_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("MUSE_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(503, detail={"error": {"code": "admin_disabled", ...}})
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail={"error": {"code": "missing_token", ...}})
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(403, detail={"error": {"code": "invalid_token", ...}})
```

Five behavior cases (all tested):
- token unset, request has Bearer header -> 503 admin_disabled
- token unset, request has no header -> 503 admin_disabled
- token set, request has no header -> 401 missing_token
- token set, request has malformed header (not "Bearer X") -> 401 missing_token
- token set, request has wrong bearer -> 403 invalid_token
- token set, request has correct bearer -> dependency returns None, route runs

The token is never echoed in error messages, log lines, or job records.
A test asserts that an HTTP error response with `MUSE_ADMIN_TOKEN=secret`
set never contains the literal "secret" string in its body.

## Job lifecycle

`Job` dataclass:
```python
@dataclass
class Job:
    job_id: str
    op: str               # "enable" | "pull" | "probe"
    model_id: str
    state: str            # "pending" | "running" | "done" | "failed"
    started_at: str       # ISO timestamp
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None  # not serialized
```

`JobStore` is a singleton in `muse.admin.jobs`:
- `create(op, model_id) -> Job` returns a pending job, registers it.
- `update(job_id, **fields)` sets attributes.
- `get(job_id) -> Job | None` returns the live job.
- `list_recent(limit=100)` returns most recent jobs.
- `reap(now)` drops jobs older than ten minutes after `finished_at`.
- `shutdown()` joins all live threads with a five-second timeout.

State transitions:
- `pending`: created; thread not yet running. The store records it
  immediately so `GET /jobs/{id}` works between the 202 response and
  the thread actually starting.
- `running`: worker thread has begun executing the operation.
- `done`: completion path; `result` populated, `finished_at` set.
- `failed`: exception path; `error` populated, `finished_at` set.

Reaping: a tiny background thread (or a check-on-read pass) prunes jobs
where `finished_at < now - 10min`. To keep the implementation simple,
v1 prunes on every `list_recent` call (lazy reap). A dedicated reaper
thread is filed as a v1.next nicety.

Thread lifecycle on shutdown:
The gateway registers a `shutdown` event handler that calls
`JobStore.shutdown()`. If a job is mid-pull when shutdown arrives, the
subprocess is sent SIGTERM and the thread joins with a five-second
timeout (then the process exits regardless).

## Supervisor refactor

Today's `run_supervisor` builds `specs: list[WorkerSpec]` in its local
scope. Admin endpoints need access to that list to enable/disable models.

Add a `SupervisorState` dataclass to `src/muse/cli_impl/supervisor.py`:

```python
@dataclass
class SupervisorState:
    workers: list[WorkerSpec] = field(default_factory=list)
    device: str = "auto"
    started_at: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


_state: SupervisorState | None = None


def get_supervisor_state() -> SupervisorState:
    """Return current state. Returns an empty sentinel when no
    supervisor has set state (e.g. running gateway in test harness)."""
    return _state if _state is not None else SupervisorState()


def set_supervisor_state(state: SupervisorState) -> None:
    global _state
    _state = state


def clear_supervisor_state() -> None:
    """Test hook."""
    global _state
    _state = None
```

`run_supervisor`:
1. Builds `specs` exactly as today.
2. Wraps them in `SupervisorState` and registers via `set_supervisor_state`.
3. Spawns workers, runs the monitor (uses `state.workers` directly).
4. On exit, calls `clear_supervisor_state` and `JobStore.shutdown`.

The monitor thread reads `state.workers` (not a separate list arg).
Existing tests for the monitor stay intact: they already pass `[spec]`
explicitly and that path is preserved.

The lock guards mutations triggered by admin endpoints (enable/disable
adding/removing workers). The monitor takes the same lock when spawning
restarts so it does not race with a concurrent admin-triggered restart
of the same port.

## `muse models info` polish

Today's output is a flat dict-dump. The redesign organizes it into
sections with a header line carrying the live status:

```
$ muse models info sdxl-turbo
sdxl-turbo                              [enabled, loaded on worker port 9003]

  modality:        image/generation
  hf_repo:         stabilityai/sdxl-turbo
  description:     Diffusers text-to-image: stabilityai/sdxl-turbo
  license:         CreativeML Open RAIL++-M
  source:          curated alias -> hf://stabilityai/sdxl-turbo

Storage:
  weights:         7.0 GB (on disk)
  venv:            /home/spinoza/.muse/venvs/sdxl-turbo

Memory (GPU):
  annotated peak:  8.0 GB
  measured:        weights 7.0 GB, peak 7.6 GB at 1024x1024 (probed 2026-04-28)

Capabilities:
  text-to-image:   yes
  img2img:         yes
  inpainting:      yes
  variations:      yes
  default size:    1024x1024
  default steps:   1
  device pref:     auto

Worker status:
  pid:             47821
  uptime:          2h 15m
  status:          running
  restart count:   0
  last error:      none
```

Per-modality known-flag display uses a small declarative dict in
`muse.cli_impl.models_info_display`:

```python
KNOWN_CAPABILITIES = {
    "image/generation": {
        "supports_text_to_image": ("text-to-image", _yes_no),
        "supports_img2img": ("img2img", _yes_no),
        "supports_inpainting": ("inpainting", _yes_no),
        "supports_variations": ("variations", _yes_no),
        "default_size": ("default size", str),
        "default_steps": ("default steps", str),
        "device": ("device pref", str),
    },
    "audio/speech": {
        "voices": ("voices", lambda v: ", ".join(v) if isinstance(v, list) else str(v)),
        "sample_rate": ("sample rate", str),
        "device": ("device pref", str),
    },
    # etc.
}
```

Unknown capability keys fall through to a single "(other)" line:
`(other capabilities: trust_remote_code=True, gguf_file=foo.gguf)`.
This matches the structured-data principle in our working style notes:
the display is derived from the manifest, not maintained as a separate
parallel allowlist that grows per modality.

When `muse models info` is invoked while no supervisor is running,
the "Worker status" section reads `not running`. Detection works by
hitting `GET /v1/admin/workers` on the local gateway with a short
timeout (2 seconds) plus the `MUSE_ADMIN_TOKEN` env var if set; on
failure (connection refused, 401, 503), fall back to the offline view.

## AdminClient

`muse.admin.client.AdminClient` is a thin Python wrapper:

```python
class AdminClient:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
        self.token = token or os.environ.get("MUSE_ADMIN_TOKEN")

    def enable(self, model_id: str) -> dict: ...
    def disable(self, model_id: str) -> dict: ...
    def probe(self, model_id: str, *, no_inference: bool = False, device: str | None = None) -> dict: ...
    def pull(self, identifier: str) -> dict: ...
    def remove(self, model_id: str, *, purge: bool = False) -> dict: ...
    def status(self, model_id: str) -> dict: ...
    def memory(self) -> dict: ...
    def workers(self) -> dict: ...
    def restart_worker(self, port: int) -> dict: ...
    def job(self, job_id: str) -> dict: ...
    def jobs(self) -> dict: ...

    def wait(self, job_id: str, *, timeout: float = 300.0, poll: float = 1.0) -> dict:
        """Block until job is done or failed; return final job record."""
```

`wait` is a convenience for "fire and block": call it after `enable`,
`pull`, or `probe` to turn an async job into a sync call.

## Resilience considerations

1. **Async job thread leakage.** Each `enable`/`pull`/`probe` op spawns
   a thread. Tracked by JobStore; `shutdown()` joins them with a
   five-second timeout. Pull subprocesses get SIGTERM on shutdown.
2. **SupervisorState mutation races.** Concurrent `enable` and `disable`
   on the same model are serialized by the global RLock. v1 takes the
   lock for the entire operation; per-model locks are filed as a future
   refinement.
3. **Worker death during admin ops.** If a worker dies between the
   `loaded` check and the unload call, the disable path treats it as
   already-unloaded and proceeds.
4. **Test coverage.** All admin operations are tested with mocked
   supervisor state (no real subprocess.Popen). The existing slow e2e
   test grows one admin call (an enable + status round-trip) so the
   end-to-end path stays exercised.
5. **`muse models info` outside the supervisor.** Detection is via a
   short-timeout HTTP call to `/v1/admin/workers`. Connection refused,
   401, or 503 all fall back to the offline view. No traceback bleed.
6. **Token leakage.** Tests assert the token is never present in any
   error response body, log line, or job record (every Job is JSON-
   serialized in tests with `MUSE_ADMIN_TOKEN=secret-test-token` set
   and the resulting JSON is asserted to NOT contain "secret-test-token").

## Out of scope

Documented in non-goals above. Worth re-emphasizing:
- No multi-host clustering.
- No persistent jobs.
- No WebSocket progress.
- No OAuth / RBAC.
- No MCP integration in this release (task #149 is the followup).
- No metrics endpoint.

## Migration

Zero migration cost for existing users. The admin API is opt-in by
setting `MUSE_ADMIN_TOKEN`. Without it, the endpoints respond
`503 admin_disabled` and existing inference routes are untouched.

The `muse models info` output is a CLI presentation change. Scripts
that parse the output should not break (no parser of the dict-dump
output exists in muse itself or in any documented example), but if a
user has one, the version bump to v0.28.0 is the heads-up.

## Why this matters

Without the admin API, the user has to:
1. `muse models disable foo`
2. `Ctrl+C` `muse serve`
3. `muse serve` again

That sequence loses every loaded model's warm state and stalls every
in-flight client request. With the admin API, a single
`POST /v1/admin/models/foo/disable` frees the VRAM in seconds, leaving
all other workers alive. Same for enable: a hot-swap with no client
downtime.

The CLI polish is independent and equally valuable: today's
`muse models info` is a dict dump that hides the most important fact
(is this model actually loaded?) behind seven lines of background data.
The redesign puts that fact in the header line and groups the rest by
relevance.
