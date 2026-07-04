# muse federation (coordinator) -- design

> **Status:** DRAFT for Alex's review. The core architectural choices below were
> made on best judgment while you were away (you'd said "we're ready" and picked
> the recommended options all through the observability arc). Every decision I made
> for you is called out in **[DECISION]** callouts -- redirect any of them on review
> and I'll revise before we plan/build. Nothing here is implemented yet.

**Goal:** Point one OpenAI-compatible endpoint at a cluster of muse nodes and have
each request dispatched to whichever node can best serve its `model` -- so a client
sees "one server with all the GPUs" instead of N separate boxes. This is the second
sub-project of the federation arc; it consumes the observability substrate (each
node's `/v1/models` and `/v1/telemetry/summary`) shipped in v0.53.0.

**Second sub-project of the federation arc** (observability was the first: it built
the per-node state a router needs; this builds the router).

---

## Context

Today each muse node is an island: a client must know which box has the model it
wants and hit that box directly. muse already has the internal shape a cross-node
router needs -- the per-node gateway routes by the request-body `model` field to a
local worker; federation is that same pattern lifted one level up (route to a
*node* instead of a *worker*). The observability work exposed the state to route on:
`GET /v1/models` (public: per-model `loaded`/`enabled`/`owned_by`) and
`GET /v1/telemetry/summary` (admin-gated: `{node, loaded, in_flight}`).

---

## Topology: a dedicated coordinator

**[DECISION] A thin coordinator process fronts a static list of node URLs.** Clients
hit the coordinator; it picks a node and forwards. Nodes stay unmodified muse
servers -- no peer-awareness, no loop-prevention, no change to the hot gateway path
we just hardened. (The alternative -- a peer mesh where each gateway delegates to
peers -- is more resilient but complicates every node's request path; deferred.)

```
                 clients (OpenAI SDK, curl, ...)
                          |
                 [ muse federate ]  :8100        <- new coordinator (this project)
                  /       |       \
             node A     node B     node C         <- unmodified muse servers
            (frodo)    (64GB box)   (...)          each already lazy-loads + evicts
```

The coordinator is a FastAPI app, structurally a sibling of the existing gateway
(`muse.cli_impl.gateway`), living in a new `muse.cli_impl.federation` (+ a
`muse.federation` package for the node-registry / router logic). It reuses the
gateway's request-forward + SSE-relay machinery (extracted/shared where clean).

---

## Node membership (static, v1)

**[DECISION] Nodes are a static list, configured, not auto-discovered.** Two input
paths, both fine:
- CLI: `muse federate --port 8100 --node http://192.168.0.204:8000 --node http://192.168.0.50:8000`
- File: `~/.muse/federation.yaml` -- a list of `{url, name?, token?}` entries.

Each node entry may carry an OPTIONAL admin `token` (only needed to read that node's
gated `/v1/telemetry/summary` for the in-flight tie-break; omit it and the coordinator
routes off the public `/v1/models` alone -- see Routing).

Dynamic membership (a node registering/deregistering itself, gossip, health-based
ejection) is **out of scope for v1** -- a static cluster is the common case and keeps
the first cut simple.

---

## Node-state refresh (poll + cache)

The coordinator keeps a cached view of each node, refreshed by a background task
every `federation.refresh_interval_seconds` (default 3s), so routing is a fast local
lookup (no per-request fan-out):

- `GET <node>/v1/models` (public) -> which models the node has, and `loaded` per model.
- `GET <node>/health` -> reachable / degraded.
- `GET <node>/v1/telemetry/summary` (ONLY if a token is configured for that node) ->
  `in_flight` load count, for the tie-break.

A node that fails its poll is marked `unreachable` and skipped by the router until it
recovers. The refresh is stale-tolerant: a routing decision made against a slightly
stale cache that turns out wrong (the node cold-loads, or 404s) is handled at forward
time (retry/failover), not by demanding a fresh poll per request.

---

## Routing policy (model-locality, v1)

**[DECISION] v1 routes by model-locality with a cheap load-aware tie-break.** For a
request naming model `M`:

1. **Candidates:** nodes whose cached `/v1/models` lists `M` as `enabled`.
2. **Prefer loaded:** among candidates, those with `M` currently `loaded` win over
   those that would have to cold-load it (avoids paying a cold load when a warm node
   exists).
3. **Tie-break:** among equally-good candidates, pick the lowest `in_flight`
   (from telemetry, if tokens are configured) else round-robin. This is a taste of
   load-aware routing using the observability data, without full queue modeling.
4. **None:** no node has `M` enabled -> 404 `model_not_available` (OpenAI-shape).

**Deferred to v2 (noted, not built):** hardware-aware routing (route a big model to
the node whose GPU fits it -- needs per-node capacity in the state view), true
queue-depth / latency-aware routing (needs richer telemetry than in_flight), and
sticky sessions. v1 deliberately ships the locality win first.

---

## Request forwarding + failover

The coordinator extracts `model` from the request body (POST) or query (GET) -- the
exact logic the gateway already uses -- picks a node, and forwards, streaming the
response back unchanged (buffered and SSE, reusing the gateway's relay). It forwards
the client's headers (so a node that gates inference, if ever configured, still sees
auth) minus hop-by-hop headers.

**Failover:** if the chosen node's forward fails at connection time (node just went
down, or 503 model_unservable), the coordinator tries the next-best candidate for `M`
once, then returns 502 `no_node_available` if all fail. (A failure MID-STREAM is not
retried -- the client already has partial output; same principle as the gateway's
worker-forward.)

---

## Auth

**[DECISION] v1: the coordinator's inference routes are open, mirroring a node's
gateway** (a node's `/v1/chat/completions` etc. are not gated today; only admin +
dashboard are). Per-node admin tokens, if configured, are used ONLY by the
coordinator's outbound polling to read gated telemetry -- they are never exposed to
clients. Gating the coordinator's own inference surface (for internet exposure) is
deferred, consistent with how generation-endpoint auth was deferred on a single node.

The coordinator MAY expose an aggregated admin/telemetry surface later; v1 does not.

---

## Aggregated read endpoints

The coordinator exposes, for client + operator convenience:
- `GET /v1/models` -- the UNION of all nodes' models, de-duplicated by id, each
  annotated with which nodes have it and whether any has it loaded. OpenAI-shape
  compatible (a client listing models sees the whole cluster's catalog).
- `GET /health` -- aggregate: healthy if >=1 node is reachable; per-node breakdown in
  the body.
- `GET /v1/federation/nodes` -- operator view: each node's url/name, reachable state,
  loaded models, in_flight (if known), last-poll age.

**[DECISION] A federated /dashboard is deferred.** v1 links out: the coordinator's
`/v1/federation/nodes` (and a tiny optional HTML index) points at each node's own
`/dashboard`. A true aggregated dashboard (cluster-wide charts) is a natural v2 once
the per-node telemetry is proven in the field.

---

## CLI + wire surface

```
muse federate --port 8100 \
    --node http://192.168.0.204:8000 \
    --node http://192.168.0.50:8000
# or: muse federate --config ~/.muse/federation.yaml
```

New config group `federation.*` in the settings registry:
- `federation.refresh_interval_seconds` (default 3.0)
- `federation.forward_timeout_seconds` (default 300 -- long, for slow generations)
- `federation.nodes` (file/CLI; the node list is not a scalar setting but a
  structured list -- likely its own `federation.yaml`, not a single env var).

Client usage is unchanged OpenAI-compat, just pointed at the coordinator:
`OpenAI(base_url="http://coordinator:8100/v1", api_key="not-used")`.

---

## Scope

**In (v1):**
- Coordinator process fronting a static node list.
- Poll+cache node state from public `/v1/models` + `/health` (+ gated telemetry if
  tokens given).
- Model-locality routing (loaded > enabled) with in_flight/round-robin tie-break.
- Forward + SSE relay + one-shot failover.
- Aggregated `/v1/models`, `/health`, `/v1/federation/nodes`.

**Out (v2+, noted):**
- Dynamic node registration / gossip / health-ejection.
- Hardware-aware + true queue-depth/latency routing (folds in the deferred review
  findings: multi-GPU probe #32, budget-fallback #33/#4).
- Aggregated cluster dashboard.
- Coordinator-side inference auth.
- Peer-mesh delegation (the alternative topology).

---

## Testing approach

- Unit: the router's node-selection given a fabricated node-state view (loaded vs
  enabled vs unreachable, tie-break, none-available -> 404). Pure function, no network.
- Unit: the state-refresh reducer given fake `/v1/models` + `/health` payloads.
- Router/forward: a FastAPI test app + fake backend nodes (TestClient / stub httpx)
  proving a request for model M lands on the node that has M, and fails over.
- Slow e2e (opt-in): a coordinator in front of two real in-process muse gateways.
- No changes to node code, so the existing node suites are the regression guard.

---

## Open decisions for Alex (the [DECISION] callouts, collected)

1. **Topology:** dedicated coordinator (chosen) vs peer mesh.
2. **Membership:** static config (chosen) vs any dynamic registration in v1.
3. **Routing:** model-locality + in_flight tie-break (chosen) vs pull hardware-aware
   into v1.
4. **Auth:** open inference surface (chosen) vs gate the coordinator now.
5. **Dashboard:** link-out to per-node dashboards (chosen) vs build an aggregated one
   in v1.

If these sit right, the next step is a task-by-task implementation plan (same
subagent-driven TDD flow as observability).
