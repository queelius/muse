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
