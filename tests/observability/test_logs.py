from muse.observability.logs import LogHub

def test_snapshot_and_byte_bound():
    hub = LogHub(buffer_bytes=20)
    for i in range(10):
        hub.append("m", f"line{i}")   # each ~5-6 bytes; only most-recent fit
    snap = hub.snapshot("m")
    assert snap and snap[-1] == "line9"
    assert sum(len(s) for s in snap) <= 20

def test_pubsub_delivers_new_lines():
    hub = LogHub()
    q = hub.subscribe("m")
    hub.append("m", "hello")
    assert q.get_nowait() == "hello"
    hub.unsubscribe("m", q)
    hub.append("m", "after")
    assert q.qsize() == 0  # unsubscribed -> no more
