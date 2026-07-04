"""Muse federation coordinator: public package API.

Re-exports the pure-logic building blocks of the federation coordinator:
node membership (`NodeSpec`, `load_nodes`), polled node state
(`NodeState`, `ModelAvail`, `build_node_state`), model-locality routing
(`select_node`), and the async poller (`NodeRegistry`). All of these are
stdlib + httpx + yaml backed; none of them import fastapi or any ML
dependency.

Deliberately NOT re-exported here: `build_coordinator` and
`run_coordinator` (in `muse.cli_impl.federation`), which assemble the
FastAPI app and therefore import fastapi plus the gateway module. Keeping
those out of this package's `__init__` means `import muse.federation`
never drags fastapi into `sys.modules`, so callers that only need the
pure routing/state logic (e.g. a lightweight client or a unit test) stay
import-light. See `tests/federation/test_public_api.py` for the
regression check.
"""

from __future__ import annotations

from muse.federation.nodes import NodeSpec, load_nodes
from muse.federation.registry import NodeRegistry
from muse.federation.router import select_node
from muse.federation.state import ModelAvail, NodeState, build_node_state

__all__ = [
    "NodeSpec",
    "load_nodes",
    "NodeState",
    "ModelAvail",
    "build_node_state",
    "select_node",
    "NodeRegistry",
]
