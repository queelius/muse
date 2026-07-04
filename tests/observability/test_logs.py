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

def test_eviction_counts_utf8_bytes_not_chars():
    # 3 emoji chars, but each is a 4-byte UTF-8 sequence -> 12 bytes total.
    # Under (buggy) char-count accounting this line would measure as 3
    # bytes and comfortably fit inside buffer_bytes=8 alongside a second
    # short line. Under correct byte-count accounting it measures as 12
    # bytes, already over the bound on its own.
    emoji_line = "\U0001F600\U0001F600\U0001F600"
    assert len(emoji_line) == 3
    assert len(emoji_line.encode("utf-8")) == 12

    hub = LogHub(buffer_bytes=8)
    hub.append("m", emoji_line)

    # Oversized-single-line rule: 12 > 8, but it's the only line, so it
    # is retained rather than evicted to empty.
    snap = hub.snapshot("m")
    assert snap == [emoji_line]

    # A second short ASCII line pushes the running byte count to 14
    # (12 + 2), which is > 8 with more than one line buffered, so the
    # emoji line must be evicted. Under char-count accounting the running
    # count would have been 3 + 2 = 5 <= 8, and the emoji line would have
    # wrongly survived.
    hub.append("m", "hi")
    snap = hub.snapshot("m")
    assert emoji_line not in snap
    assert snap == ["hi"]
