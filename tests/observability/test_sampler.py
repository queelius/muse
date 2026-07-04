import threading

from muse.observability.sampler import Sampler


def test_sample_once_records(monkeypatch):
    import muse.observability.sampler as smod
    monkeypatch.setattr(smod, "gpu_free_gb", lambda: 3.0)
    monkeypatch.setattr(smod, "cpu_free_gb", lambda: 20.0)
    seen = []
    s = Sampler(interval=999, loaded_fn=lambda: {"m": object()},
                inflight_fn=lambda: 2, record_fn=lambda t, **k: seen.append((t, k)))
    s.sample_once()
    assert seen[0][0] == "sample"
    k = seen[0][1]
    assert k["free_vram_gb"] == 3.0 and k["loaded_count"] == 1 and k["in_flight_count"] == 2


def test_shared_stop_event_stops_the_loop(monkeypatch):
    # Constructing with an external stop_event (the shape run_supervisor's
    # _init_telemetry uses, passing state.stop_event) must let that shared
    # event unblock the sampler loop, same as IdleSweeper's stop_event param.
    import muse.observability.sampler as smod
    monkeypatch.setattr(smod, "gpu_free_gb", lambda: 1.0)
    monkeypatch.setattr(smod, "cpu_free_gb", lambda: 1.0)

    shared = threading.Event()
    s = Sampler(
        interval=0.01,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        record_fn=lambda t, **k: None,
        stop_event=shared,
    )
    assert s._stop is shared

    s.start()
    shared.set()
    s._thread.join(timeout=2.0)
    assert not s._thread.is_alive()


def test_no_arg_construction_still_works():
    # Backward compatibility: omitting stop_event still gives the sampler
    # its own private Event, and start/stop still function.
    s = Sampler(
        interval=0.01,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        record_fn=lambda t, **k: None,
    )
    assert isinstance(s._stop, threading.Event)

    s.start()
    s.stop()
    assert s._thread is None
