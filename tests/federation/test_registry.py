from muse.federation.nodes import NodeSpec
from muse.federation.registry import NodeRegistry


async def test_refresh_once_builds_snapshot():
    specs = [NodeSpec(url="http://a:8000", name="a"), NodeSpec(url="http://b:8000", name="b")]

    async def fake_fetch(url, token):
        if "a:" in url:
            return ({"data": [{"id": "m1", "loaded": True}]}, {"status": "ok"}, {"in_flight": 0})
        return (None, None, None)  # b unreachable

    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()
    snap = {s.spec.name: s for s in reg.snapshot()}
    assert snap["a"].reachable and snap["a"].models["m1"].loaded and snap["a"].in_flight == 0
    assert snap["b"].reachable is False


async def test_refresh_once_isolates_node_that_raises():
    """One node's fetch RAISES (proves gather-level per-node isolation:
    a bad node must not abort the refresh for a healthy sibling)."""
    specs = [NodeSpec(url="http://good:8000", name="good"), NodeSpec(url="http://bad:8000", name="bad")]

    async def fake_fetch(url, token):
        if "bad:" in url:
            raise RuntimeError("boom: differently-shaped /v1/models response")
        return ({"data": [{"id": "m1", "loaded": True}]}, {"status": "ok"}, {"in_flight": 0})

    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()  # must not raise/abort
    snap = {s.spec.name: s for s in reg.snapshot()}

    # healthy node's state is present, reachable, and has its model
    assert snap["good"].reachable is True
    assert snap["good"].models["m1"].loaded is True

    # bad node degrades to an unreachable NodeState instead of aborting
    assert snap["bad"].reachable is False
    assert snap["bad"].models == {}
    assert snap["bad"].in_flight is None


async def test_refresh_once_skips_id_less_entry_but_stays_reachable():
    """A node returns a well-formed 200 whose /v1/models entries lack an
    "id" key (a differently-shaped/older muse). The node should stay
    reachable with the malformed entry skipped, not raise and abort."""
    specs = [NodeSpec(url="http://a:8000", name="a")]

    async def fake_fetch(url, token):
        return ({"data": [{"name": "x"}]}, {"status": "ok"}, None)

    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()
    snap = {s.spec.name: s for s in reg.snapshot()}

    assert snap["a"].reachable is True
    assert snap["a"].models == {}


def test_poll_timeout_binds_into_default_fetch():
    """NodeRegistry(poll_timeout=X) binds X into the default httpx fetch."""
    from functools import partial
    from muse.federation.nodes import NodeSpec
    from muse.federation.registry import NodeRegistry, _default_fetch
    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=3.0,
        poll_timeout=7.5,
    )
    assert isinstance(reg._fetch, partial)
    assert reg._fetch.func is _default_fetch
    assert reg._fetch.keywords["timeout"] == 7.5


def test_poll_timeout_defaults_when_unset():
    """No poll_timeout -> falls back to the module default (>= node aggregation timeout)."""
    from functools import partial
    from muse.federation.nodes import NodeSpec
    from muse.federation.registry import NodeRegistry, _FETCH_TIMEOUT_SECONDS
    reg = NodeRegistry([NodeSpec(url="http://a:8000", name="a")], refresh_interval=3.0)
    assert isinstance(reg._fetch, partial)
    assert reg._fetch.keywords["timeout"] == _FETCH_TIMEOUT_SECONDS
    assert _FETCH_TIMEOUT_SECONDS >= 5.0  # above the 5s node aggregation timeout


def test_injected_fetch_is_used_as_is():
    """An injected fetch is NOT wrapped (preserves the test-injection contract)."""
    from muse.federation.nodes import NodeSpec
    from muse.federation.registry import NodeRegistry
    async def fake(url, token): return (None, None, None)
    reg = NodeRegistry([NodeSpec(url="http://a:8000", name="a")], refresh_interval=3.0, fetch=fake)
    assert reg._fetch is fake


def test_federation_poll_timeout_config_default():
    from muse.core import config
    assert config.get("federation.poll_timeout_seconds") == 10.0
