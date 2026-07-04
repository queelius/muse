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


def test_entry_missing_id_is_skipped_not_raised():
    """A data entry lacking a usable "id" (a differently-shaped/older
    muse node) is skipped rather than raising KeyError; the node stays
    reachable and the well-formed entries survive."""
    st = build_node_state(SPEC,
        models_payload={"data": [{"name": "x"}, {"id": "m1", "loaded": True}]},
        health_payload={"status": "ok"}, summary_payload=None, now=7.0)
    assert st.reachable is True
    assert "m1" in st.models and st.models["m1"].loaded is True
    assert len(st.models) == 1  # the id-less entry is absent, not raised
