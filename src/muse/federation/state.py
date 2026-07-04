"""Node-state model + pure refresh reducer for the muse federation coordinator.

`build_node_state` folds three polled payloads (`/v1/models`, `/health`,
`/v1/telemetry/summary`) for one node into a `NodeState` snapshot. It is
pure: no network calls, no clock reads. The caller fetches the payloads
and passes the current time in via `now`, so this module stays stdlib
only (dataclasses) and is trivially testable without mocking a clock or
an HTTP client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from muse.federation.nodes import NodeSpec


@dataclass
class ModelAvail:
    loaded: bool
    enabled: bool


@dataclass
class NodeState:
    spec: NodeSpec
    reachable: bool
    models: dict[str, ModelAvail] = field(default_factory=dict)
    in_flight: int | None = None
    last_poll_ts: float = 0.0


def build_node_state(
    spec: NodeSpec,
    *,
    models_payload: dict | None,
    health_payload: dict | None,
    summary_payload: dict | None,
    now: float,
) -> NodeState:
    """Fold polled payloads for one node into a NodeState snapshot.

    - reachable = models_payload is not None (a node whose /v1/models we
      cannot read is unroutable).
    - models: one ModelAvail per entry in models_payload["data"], keyed by
      entry["id"]. Missing/absent "data" (or models_payload is None)
      yields an empty dict.
    - in_flight = summary_payload.get("in_flight") if summary_payload is
      not None, else None.
    - last_poll_ts = now (passed in, never read from a clock here).
    """
    reachable = models_payload is not None

    models: dict[str, ModelAvail] = {}
    if models_payload is not None:
        for entry in models_payload.get("data") or []:
            models[entry["id"]] = ModelAvail(
                loaded=bool(entry.get("loaded")),
                enabled=True,
            )

    in_flight = summary_payload.get("in_flight") if summary_payload else None

    return NodeState(
        spec=spec,
        reachable=reachable,
        models=models,
        in_flight=in_flight,
        last_poll_ts=now,
    )
