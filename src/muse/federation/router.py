"""Model-locality routing for the muse federation coordinator.

`select_node` picks which remote node a request for a given model_id
should be forwarded to, given the coordinator's polled `NodeState` list.
Pure module: stdlib only, no fastapi/httpx/torch. No network calls, no
clock reads -- the caller supplies everything it needs.

Policy:
  1. candidates = reachable states whose `.models` contains model_id.
  2. no candidates -> None.
  3. prefer loaded: if any candidate has models[model_id].loaded, the
     winning set narrows to those (a merely-enabled node loses to a
     loaded one).
  4. tie-break among the winning set:
     - by min in_flight, treating None as +infinity (a node with a
       known-low in_flight beats an unknown-None; if every candidate is
       None they all tie here).
     - if still tied, round-robin: sort the tied set by spec.url first
       (determinism), then pick index rr_counter.get(model_id, 0) %
       len(...), and increment rr_counter[model_id]. If rr_counter is
       None, just return the first by sorted url (deterministic, no
       rotation).
"""

from __future__ import annotations

from muse.federation.state import NodeState


def select_node(
    model_id: str,
    states: list[NodeState],
    *,
    rr_counter: dict | None = None,
) -> NodeState | None:
    candidates = [
        state
        for state in states
        if state.reachable and model_id in state.models
    ]
    if not candidates:
        return None

    if any(state.models[model_id].loaded for state in candidates):
        candidates = [state for state in candidates if state.models[model_id].loaded]

    def _in_flight_key(state: NodeState) -> float:
        return state.in_flight if state.in_flight is not None else float("inf")

    min_in_flight = min(_in_flight_key(state) for state in candidates)
    tied = [state for state in candidates if _in_flight_key(state) == min_in_flight]

    if len(tied) == 1:
        return tied[0]

    tied_sorted = sorted(tied, key=lambda state: state.spec.url)
    if rr_counter is None:
        return tied_sorted[0]

    idx = rr_counter.get(model_id, 0)
    choice = tied_sorted[idx % len(tied_sorted)]
    rr_counter[model_id] = idx + 1
    return choice
