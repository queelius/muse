# Federation coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A thin `muse federate` coordinator process that fronts a static list of muse nodes and routes each OpenAI-compatible request to the node that can best serve its `model` (loaded > enabled, in-flight tie-break), so one endpoint fronts a whole cluster.

**Architecture:** A new pure-logic package `muse.federation` (node membership, node-state model, router selection, poll+cache registry) plus a FastAPI coordinator app in `muse.cli_impl.federation` that reuses the existing gateway's `extract_model_from_request`, `_forward` (plain forward + SSE relay), and `_openai_error`. Nodes are unmodified muse servers; the coordinator reads their public `/v1/models` + `/health` (+ gated `/v1/telemetry/summary` when a per-node token is configured).

**Tech Stack:** Python 3.11, FastAPI, httpx (async), typer CLI, the muse config registry. Reuses `muse.cli_impl.gateway` helpers.

## Global Constraints

- ASCII only in all source, tests, docs, and commit messages. NO em-dash (a repo hook rejects em-dashes).
- TDD red-green per task; run the named test to FAIL before implementing.
- Import-light: `muse.federation.*` pure-logic modules import stdlib + httpx only (NO torch/diffusers). The coordinator app imports fastapi (fine; it is a server entrypoint, never on the `muse --help` path).
- Reuse, do not duplicate: forward + SSE relay is `muse.cli_impl.gateway._forward(request, target_url, timeout)`; model extraction is `gateway.extract_model_from_request(request)`; error envelope is `gateway._openai_error(status, code, message, error_type=...)`. Do not re-implement these.
- Every task ends green on its own test file, then the fast lane `pytest tests/ -q -m "not slow"` with no regressions.
- Commit trailer on every commit, exactly:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV`

---

### Task 1: config settings + node membership (`muse.federation.nodes`)

**Files:** Create `src/muse/federation/__init__.py` (empty for now), `src/muse/federation/nodes.py`; Modify `src/muse/core/config.py`; Test `tests/federation/test_nodes.py`.

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class NodeSpec: url: str; name: str; token: str | None = None` (url normalized: strip trailing slash).
  - `load_nodes(cli_nodes: list[str] | None = None, config_path: str | Path | None = None) -> list[NodeSpec]` -- merges nodes from CLI url strings (`"http://h:8000"` or `"name=http://h:8000"`) and/or a yaml file with shape `nodes: [{url, name?, token?}, ...]`. Dedup by url (first wins). Derives a default `name` from the host if absent.
- Config rows added to `SETTINGS` (after the telemetry group), style-matching existing rows:
  - `federation.refresh_interval_seconds` (`MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS`, float, 3.0, group "federation", "Seconds between coordinator polls of each node's state.")
  - `federation.forward_timeout_seconds` (`MUSE_FEDERATION_FORWARD_TIMEOUT_SECONDS`, float, 300.0, "federation", "Per-request timeout when the coordinator forwards to a node.")
  - `federation.config_file` (`MUSE_FEDERATION_CONFIG`, opt_str, None, "federation", "Path to the coordinator node-list yaml (default <catalog_dir>/federation.yaml).")

- [ ] **Step 1: failing test** (`tests/federation/test_nodes.py`):
```python
from pathlib import Path
from muse.federation.nodes import NodeSpec, load_nodes

def test_cli_nodes_plain_and_named():
    nodes = load_nodes(cli_nodes=["http://a:8000/", "b=http://b:8000"])
    assert nodes[0] == NodeSpec(url="http://a:8000", name="a", token=None)  # trailing slash stripped, name from host
    assert nodes[1] == NodeSpec(url="http://b:8000", name="b", token=None)

def test_yaml_nodes_and_dedup(tmp_path):
    p = tmp_path / "federation.yaml"
    p.write_text("nodes:\n  - {url: http://a:8000, name: alpha, token: t1}\n  - url: http://a:8000\n")
    nodes = load_nodes(config_path=p)
    assert len(nodes) == 1 and nodes[0].name == "alpha" and nodes[0].token == "t1"  # dedup by url, first wins

def test_config_setting_defaults():
    from muse.core import config
    assert config.get("federation.refresh_interval_seconds") == 3.0
```

- [ ] **Step 2: run to FAIL** (`pytest tests/federation/test_nodes.py -v`).
- [ ] **Step 3: implement** `nodes.py` (stdlib + `yaml` which muse already depends on for curated/config) and the three `Setting` rows. Name-from-host: `urlparse(url).hostname or url`.
- [ ] **Step 4: run to PASS**, then `pytest tests/ -q -m "not slow"`.
- [ ] **Step 5: commit** -- `feat(federation): node membership model + coordinator config settings`.

---

### Task 2: node-state model + refresh reducer (`muse.federation.state`)

**Files:** Create `src/muse/federation/state.py`; Test `tests/federation/test_state.py`.

**Interfaces:**
- Consumes: `NodeSpec`.
- Produces:
  - `@dataclass class ModelAvail: loaded: bool; enabled: bool` (per-model on a node).
  - `@dataclass class NodeState: spec: NodeSpec; reachable: bool; models: dict[str, ModelAvail]; in_flight: int | None; last_poll_ts: float`.
  - `build_node_state(spec, *, models_payload, health_payload, summary_payload, now) -> NodeState` -- pure. `models_payload` is the parsed `/v1/models` dict (`{"data": [{id, loaded, ...}]}`) or None (fetch failed). `health_payload` parsed `/health` or None. `summary_payload` parsed `/v1/telemetry/summary` or None (no token / failed). Rules:
    - reachable = `models_payload is not None` (a node whose /v1/models we cannot read is unroutable).
    - models: for each entry in `models_payload["data"]`, `ModelAvail(loaded=bool(entry.get("loaded")), enabled=True)` (a listed model is enabled/serviceable; the node's own gateway lists only enabled catalog models). Missing/empty data -> `{}`.
    - in_flight = `summary_payload.get("in_flight")` if summary_payload else None.
    - last_poll_ts = now.

- [ ] **Step 1: failing test** (`tests/federation/test_state.py`):
```python
from muse.federation.nodes import NodeSpec
from muse.federation.state import build_node_state

SPEC = NodeSpec(url="http://a:8000", name="a")

def test_reachable_with_models_and_loaded_flags():
    st = build_node_state(SPEC,
        models_payload={"data": [{"id": "m1", "loaded": True}, {"id": "m2", "loaded": False}]},
        health_payload={"status": "ok"}, summary_payload={"in_flight": 2}, now=100.0)
    assert st.reachable is True
    assert st.models["m1"].loaded is True and st.models["m1"].enabled is True
    assert st.models["m2"].loaded is False
    assert st.in_flight == 2 and st.last_poll_ts == 100.0

def test_unreachable_when_models_none():
    st = build_node_state(SPEC, models_payload=None, health_payload=None, summary_payload=None, now=5.0)
    assert st.reachable is False and st.models == {} and st.in_flight is None

def test_no_summary_means_in_flight_none():
    st = build_node_state(SPEC, models_payload={"data": [{"id": "m"}]}, health_payload={"status": "ok"},
                          summary_payload=None, now=1.0)
    assert st.in_flight is None and st.models["m"].loaded is False  # no loaded key -> False
```

- [ ] **Step 2: run to FAIL.**
- [ ] **Step 3: implement** `state.py` (pure; stdlib only).
- [ ] **Step 4: run to PASS**, then fast lane.
- [ ] **Step 5: commit** -- `feat(federation): node-state model + pure refresh reducer`.

---

### Task 3: router selection (`muse.federation.router`)

**Files:** Create `src/muse/federation/router.py`; Test `tests/federation/test_router.py`.

**Interfaces:**
- Consumes: `NodeState`.
- Produces: `select_node(model_id: str, states: list[NodeState], *, rr_counter: dict | None = None) -> NodeState | None`. Policy:
  1. candidates = reachable states whose `.models` contains `model_id` (a listed model is enabled/serviceable).
  2. if none -> return None.
  3. prefer loaded: if any candidate has `models[model_id].loaded`, restrict to those.
  4. tie-break among the winning set: pick min `in_flight` (treat None as +inf so a node with a known-low load beats an unknown); if still tied (or all None), round-robin via `rr_counter` (a mutable dict keyed by model_id holding a rotating index) else the first by stable order (sort candidates by `spec.url` for determinism before RR).

- [ ] **Step 1: failing test** (`tests/federation/test_router.py`):
```python
from muse.federation.nodes import NodeSpec
from muse.federation.state import NodeState, ModelAvail
from muse.federation.router import select_node

def _st(url, model, loaded, in_flight=None, reachable=True):
    return NodeState(spec=NodeSpec(url=url, name=url), reachable=reachable,
                     models={model: ModelAvail(loaded=loaded, enabled=True)} if model else {},
                     in_flight=in_flight, last_poll_ts=0.0)

def test_none_when_no_candidate():
    assert select_node("m", [_st("http://a", None, False)]) is None

def test_loaded_beats_enabled_only():
    a = _st("http://a", "m", loaded=False); b = _st("http://b", "m", loaded=True)
    assert select_node("m", [a, b]).spec.url == "http://b"

def test_in_flight_tiebreak_among_loaded():
    a = _st("http://a", "m", loaded=True, in_flight=5); b = _st("http://b", "m", loaded=True, in_flight=1)
    assert select_node("m", [a, b]).spec.url == "http://b"

def test_unreachable_excluded():
    a = _st("http://a", "m", loaded=True, reachable=False); b = _st("http://b", "m", loaded=False)
    assert select_node("m", [a, b]).spec.url == "http://b"

def test_round_robin_when_all_equal():
    a = _st("http://a", "m", loaded=True); b = _st("http://b", "m", loaded=True)
    rr = {}
    first = select_node("m", [a, b], rr_counter=rr).spec.url
    second = select_node("m", [a, b], rr_counter=rr).spec.url
    assert {first, second} == {"http://a", "http://b"}  # rotates
```

- [ ] **Step 2: run to FAIL.**
- [ ] **Step 3: implement** `router.py` (pure). RR: `idx = rr_counter.get(model_id, 0); choice = sorted_candidates[idx % len]; rr_counter[model_id] = idx + 1`.
- [ ] **Step 4: run to PASS**, then fast lane.
- [ ] **Step 5: commit** -- `feat(federation): model-locality router with in-flight tie-break`.

---

### Task 4: node registry (poll + cache, async) (`muse.federation.registry`)

**Files:** Create `src/muse/federation/registry.py`; Test `tests/federation/test_registry.py`.

**Interfaces:**
- Consumes: `NodeSpec`, `build_node_state`, config.
- Produces: `class NodeRegistry`:
  - `__init__(self, nodes: list[NodeSpec], *, refresh_interval: float, clock=time.monotonic, fetch=None)`. `fetch` is an injectable async callable `fetch(url, token) -> tuple[models_payload|None, health_payload|None, summary_payload|None]` (default uses httpx; tests inject a fake so no network + no real clock dependence).
  - `async refresh_once(self) -> None` -- concurrently fetch all nodes, build states via the reducer, replace the cached snapshot under a lock.
  - `snapshot(self) -> list[NodeState]` -- the cached states (empty list before first refresh).
  - `start(self) -> None` / `async aclose(self) -> None` -- background asyncio task looping `refresh_once` every interval; cancels cleanly on aclose.
  - `node_by_url(self, url) -> NodeSpec | None`.

- [ ] **Step 1: failing test** (`tests/federation/test_registry.py`) -- inject a fake fetch:
```python
import pytest
from muse.federation.nodes import NodeSpec
from muse.federation.registry import NodeRegistry

@pytest.mark.asyncio
async def test_refresh_once_builds_snapshot():
    specs = [NodeSpec(url="http://a:8000", name="a"), NodeSpec(url="http://b:8000", name="b")]
    async def fake_fetch(url, token):
        if "a:" in url: return ({"data": [{"id": "m1", "loaded": True}]}, {"status": "ok"}, {"in_flight": 0})
        return (None, None, None)  # b unreachable
    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()
    snap = {s.spec.name: s for s in reg.snapshot()}
    assert snap["a"].reachable and snap["a"].models["m1"].loaded and snap["a"].in_flight == 0
    assert snap["b"].reachable is False
```
(Use the repo's async-test convention: check whether `pytest-asyncio` is configured -- `grep asyncio pyproject.toml setup.cfg tests/*/conftest.py`. If async tests are NOT already supported in the suite, make `refresh_once` independently drivable via `asyncio.run(...)` inside a plain sync test instead of adding a new pytest plugin dependency. Prefer NOT adding a new test dependency.)

- [ ] **Step 2: run to FAIL.**
- [ ] **Step 3: implement** `registry.py`. Default httpx fetch: GET `<url>/v1/models` + `<url>/health` always; GET `<url>/v1/telemetry/summary` with `Authorization: Bearer <token>` ONLY if `token`; any error -> that payload is None (the reducer maps models_payload None -> unreachable). Use a short per-fetch timeout (a couple seconds) so a dead node does not stall the whole refresh; gather concurrently.
- [ ] **Step 4: run to PASS**, then fast lane.
- [ ] **Step 5: commit** -- `feat(federation): node registry with concurrent poll-and-cache`.

---

### Task 5: coordinator app (`muse.cli_impl.federation`)

**Files:** Create `src/muse/cli_impl/federation.py`; Test `tests/cli_impl/test_federation_app.py`.

**Interfaces:**
- Consumes: `NodeRegistry`, `select_node`; reuses `gateway.extract_model_from_request`, `gateway._forward`, `gateway._openai_error`.
- Produces: `build_coordinator(registry: NodeRegistry, *, timeout: float, rr_counter: dict | None = None) -> FastAPI`. Routes:
  - `GET /health` -> `{"status": "ok"|"degraded", "nodes": [{name, url, reachable, model_count, in_flight}]}`; ok iff >=1 reachable.
  - `GET /v1/models` -> OpenAI-shape `{"object": "list", "data": [...]}`: union of model ids across reachable snapshots, one entry per id with `id`, `object: "model"`, `owned_by: "muse"`, plus muse extras `loaded: bool` (any node has it loaded) and `nodes: [names that have it]`.
  - `GET /v1/federation/nodes` -> per-node operator view (name, url, reachable, loaded model list, in_flight, last_poll age in seconds).
  - catch-all `@app.api_route("/{full_path:path}", methods=[GET, POST])` -> extract model; if None -> 400 `model_required`; `select_node`; None -> 404 `model_not_available`; else `_forward(request, f"{node.url}/{full_path}", timeout)`. On a forward ConnectError/timeout, try the next-best candidate ONCE (recompute candidates excluding the failed url), else 502 `no_node_available`. Do NOT mount the catch-all before the explicit routes (registration order: explicit first).

- [ ] **Step 1: failing test** (`tests/cli_impl/test_federation_app.py`) -- a fake registry with canned snapshot + monkeypatch `federation._forward` to a stub that records the target_url and returns a Response echoing which node:
```python
from types import SimpleNamespace
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
import muse.cli_impl.federation as fed
from muse.federation.nodes import NodeSpec
from muse.federation.state import NodeState, ModelAvail

def _snap():
    a = NodeState(NodeSpec("http://a:8000","a"), True, {"m1": ModelAvail(True, True)}, 0, 0.0)
    b = NodeState(NodeSpec("http://b:8000","b"), True, {"m2": ModelAvail(True, True)}, 0, 0.0)
    return [a, b]

class FakeReg:
    def snapshot(self): return _snap()

def test_routes_to_node_with_model(monkeypatch):
    seen = {}
    async def fake_forward(request, target_url, timeout):
        seen["url"] = target_url
        return JSONResponse({"ok": True})
    monkeypatch.setattr(fed, "_forward", fake_forward)
    app = build := fed.build_coordinator(FakeReg(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "m2", "messages": []})
    assert r.status_code == 200 and seen["url"] == "http://b:8000/v1/chat/completions"

def test_404_when_no_node_has_model(monkeypatch):
    app = fed.build_coordinator(FakeReg(), timeout=5)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_available"

def test_v1_models_union():
    c = TestClient(fed.build_coordinator(FakeReg(), timeout=5))
    ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
    assert ids == {"m1", "m2"}

def test_health_aggregate():
    assert TestClient(fed.build_coordinator(FakeReg(), timeout=5)).get("/health").json()["status"] == "ok"
```
(Add a failover test: a snapshot with two nodes having `m`, a fake_forward that raises `httpx.ConnectError` for the first url and succeeds for the second; assert the response is 200 and the second url was used. Import the real `_forward` name into `federation.py` as a module attribute so the monkeypatch on `fed._forward` takes effect.)

- [ ] **Step 2: run to FAIL.**
- [ ] **Step 3: implement** `federation.py`. `from muse.cli_impl.gateway import extract_model_from_request, _forward, _openai_error` at module top (bind `_forward` as a module name so tests can patch it). Failover: wrap the `_forward` await in try/except on `(httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)`; on failure pick another candidate for the same model (exclude the failed url) and forward once more; exhausted -> `_openai_error(502, "no_node_available", ...)`.
- [ ] **Step 4: run to PASS**, then fast lane.
- [ ] **Step 5: commit** -- `feat(federation): coordinator app (model-routing + union /v1/models + /health)`.

---

### Task 6: `muse federate` CLI + runner + docs

**Files:** Modify `src/muse/cli.py` (add `federate` command), `src/muse/cli_impl/federation.py` (add `run_coordinator`); Modify `CLAUDE.md`, `README.md`, `docs/CONFIG.md`; Test `tests/cli_impl/test_federation_cli.py`.

**Interfaces:**
- Produces: `run_coordinator(*, host: str, port: int, cli_nodes: list[str] | None, config_path: str | None) -> int` -- `load_nodes(...)`, build a `NodeRegistry` (interval from `config.get("federation.refresh_interval_seconds")`), `registry.start()`, `build_coordinator(registry, timeout=config.get("federation.forward_timeout_seconds"))`, run uvicorn (reuse `gateway.run_uvicorn` if present, else `uvicorn.run`), `registry.aclose()` on shutdown. Returns 0.
- CLI: `@app.command()` `federate(host: str = "0.0.0.0", port: int = 8100, node: list[str] = Option([]), config: str = Option(None))` in `cli.py` delegating to `run_coordinator`. Empty node list AND no config file -> exit 2 with a clear message ("no nodes: pass --node or --config").

- [ ] **Step 1: failing test** (`tests/cli_impl/test_federation_cli.py`): use typer's `CliRunner` to assert `muse federate` with no nodes exits nonzero with the message; and (monkeypatching `run_coordinator` to a no-op returning 0) that `--node http://a:8000` parses and calls `run_coordinator` with `cli_nodes=["http://a:8000"]`.
- [ ] **Step 2: run to FAIL.**
- [ ] **Step 3: implement** the command + `run_coordinator`. Docs: `CLAUDE.md` federation section (topology, routing policy, the `federation.*` settings, model-locality v1 + deferred v2, node-list yaml shape, that it reuses the gateway forward and needs no node changes); `README.md` short subsection with the `muse federate` invocation + OpenAI-client-points-at-coordinator example; `docs/CONFIG.md` new `### federation` group table (the three rows from Task 1).
- [ ] **Step 4: run to PASS**, then fast lane.
- [ ] **Step 5: commit** -- `feat(federation): muse federate CLI + runner + docs`.

---

### Task 7: slow e2e + package public API

**Files:** Create `tests/cli_impl/test_federation_e2e.py` (marked `@pytest.mark.slow`); Modify `src/muse/federation/__init__.py` (re-exports); Test `tests/federation/test_public_api.py`.

**Interfaces:**
- `muse.federation.__init__` re-exports the pure-logic public names: `NodeSpec, load_nodes, NodeState, ModelAvail, build_node_state, select_node, NodeRegistry`. Keep it import-light (all are stdlib+httpx-backed; NO fastapi in the package `__init__` -- `build_coordinator`/`run_coordinator` live in `cli_impl.federation` and are NOT re-exported here, so importing `muse.federation` never drags in fastapi).

- [ ] **Step 1: failing test** (`tests/federation/test_public_api.py`): `from muse.federation import NodeSpec, load_nodes, build_node_state, select_node, NodeRegistry` all import; a subprocess import-light check asserting `fastapi` is NOT in `sys.modules` after `import muse.federation`.
- [ ] **Step 2: run to FAIL** (empty `__init__`).
- [ ] **Step 3: implement** the re-exports. Then write the slow e2e: stand up TWO in-process fake muse-node apps (tiny FastAPI apps exposing `/v1/models` returning distinct model ids with `loaded: true`, `/health`, and an echo `/v1/chat/completions` returning which node), mount them on two ephemeral ports via `uvicorn`/`httpx.ASGITransport` or a threaded `TestServer`; build a real `NodeRegistry` pointed at them, `refresh_once`, build the coordinator, and assert a request for node-A's model reaches node A and node-B's model reaches node B, plus failover when one node app is stopped. If a full two-server socket setup is too heavy for the in-process slow lane, drive the coordinator with a real `NodeRegistry` whose `fetch` hits two `httpx.ASGITransport`-wrapped fake node apps (no real sockets) -- state which you did.
- [ ] **Step 4: run** `pytest tests/federation/ tests/cli_impl/test_federation_e2e.py -q` green, then the full fast lane, then `pytest tests/ -q` (includes the slow test) once.
- [ ] **Step 5: commit** -- `feat(federation): public API re-exports + slow e2e coordinator-over-two-nodes`.

---

## Self-Review

- **Spec coverage:** coordinator topology (T5/T6), static membership + config (T1), node-state from public /v1/models + optional gated summary (T2/T4), model-locality routing + in-flight tie-break (T3), forward + one-shot failover reusing gateway `_forward` (T5), aggregated /v1/models + /health + /v1/federation/nodes (T5), CLI + docs (T6), e2e (T7). Every spec section maps to a task. Deferred items (dynamic membership, hardware-aware routing, aggregated dashboard, coordinator auth, peer mesh) are explicitly out of scope and not tasked.
- **Type consistency:** `NodeSpec` / `NodeState` / `ModelAvail` / `build_node_state` / `select_node` / `NodeRegistry.snapshot` used identically across T2-T7. `_forward(request, target_url, timeout)` and `extract_model_from_request(request)` match the real gateway signatures verified before planning.
- **Import hygiene:** `muse.federation` package stays fastapi-free (T7 asserts it); the coordinator app lives in `cli_impl.federation`.
- **No new test dependency:** T4 flags the async-test decision (use `asyncio.run` in a sync test rather than adding pytest-asyncio if the suite lacks it).
