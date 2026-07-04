"""Public API contract for the `muse.federation` package.

Two things pinned here:

1. The pure-logic names re-exported from `muse.federation` (not
   `muse.cli_impl.federation`, which additionally exports
   `build_coordinator` / `run_coordinator` and pulls in fastapi).
2. Import hygiene: `import muse.federation` must never drag fastapi into
   `sys.modules`. Checked in a subprocess so an already-imported fastapi
   from some other test in the same process can't hide a regression.
"""

import subprocess
import sys


def test_public_names_import():
    from muse.federation import (
        ModelAvail,
        NodeRegistry,
        NodeSpec,
        NodeState,
        build_node_state,
        load_nodes,
        select_node,
    )

    assert NodeSpec is not None
    assert load_nodes is not None
    assert NodeState is not None
    assert ModelAvail is not None
    assert build_node_state is not None
    assert select_node is not None
    assert NodeRegistry is not None


def test_all_matches_expected_names():
    import muse.federation as federation

    assert set(federation.__all__) == {
        "NodeSpec",
        "load_nodes",
        "NodeState",
        "ModelAvail",
        "build_node_state",
        "select_node",
        "NodeRegistry",
    }


def test_import_does_not_pull_in_cli_impl_federation():
    """`build_coordinator` / `run_coordinator` are NOT re-exported here;
    they live in `muse.cli_impl.federation`, which imports fastapi + the
    gateway."""
    import muse.federation as federation

    assert not hasattr(federation, "build_coordinator")
    assert not hasattr(federation, "run_coordinator")


def test_import_is_fastapi_free_in_subprocess():
    """Clean-interpreter check: `import muse.federation` alone must never
    pull fastapi into sys.modules. httpx MAY be present (NodeRegistry's
    default fetch uses it), only fastapi is disallowed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import muse.federation, sys; "
            "assert 'fastapi' not in sys.modules, "
            "'fastapi leaked into muse.federation import'",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess import check failed\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
