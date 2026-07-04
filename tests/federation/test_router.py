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
