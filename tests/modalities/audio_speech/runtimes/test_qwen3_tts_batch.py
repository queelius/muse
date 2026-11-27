"""Batched Qwen3-TTS talker and engine (``runtimes/qwen3_tts_batch.py``).

Rows of one decode state share each 12 Hz frame: a request joins at a frame
boundary (its prefill runs alone, into its row) and leaves at its own EOS.
Checked on the tiny fake talker from ``test_qwen3_tts_fast`` against each
line's solo run and the independent full-recompute reference.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from muse.modalities.audio_speech.runtimes import qwen3_tts_batch as batch  # noqa: E402
from muse.modalities.audio_speech.runtimes import qwen3_tts_fast as fast  # noqa: E402

from .test_qwen3_tts_fast import (  # noqa: E402
    EMBED, EOS, FakeTalker, _talker_kwargs, reference_talker,
)


@pytest.fixture
def talker():
    torch.manual_seed(5)
    model = FakeTalker()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    return model


def _fast(talker):
    cp = fast.FastCodePredictor(talker.code_predictor, torch=torch)
    return fast.FastTalker(talker, cp, torch=torch)


def _prompt(seed, length=5, text=3):
    g = torch.Generator().manual_seed(seed)
    return dict(inputs_embeds=torch.randn(1, length, EMBED, generator=g),
                trailing_text_hidden=torch.randn(1, text, EMBED, generator=g),
                tts_pad_embed=torch.randn(1, 1, EMBED, generator=g))


def _req(ft, seed, max_new=12, **kw):
    return ft.request(_talker_kwargs(_prompt(seed, **kw), max_new_tokens=max_new))


def _solo(ft, req):
    run = ft.start(req)
    frames = []
    while (codes := ft.step(run)) is not None:
        frames.append(codes)
    return frames


def _drive(bt, reqs, joins=None):
    """Run requests through a BatchedTalker; ``joins`` maps index -> frame at
    which it joins (default all at 0). Returns each request's frames."""
    joins = joins or {}
    out = {i: [] for i in range(len(reqs))}
    waiting = sorted(range(len(reqs)), key=lambda i: joins.get(i, 0))
    rows: dict = {}
    frame = 0
    while waiting or rows:
        while waiting and joins.get(waiting[0], 0) <= frame:
            i = waiting.pop(0)
            rows[bt.join(reqs[i])] = i
        while done := [r for r in rows if bt.finished(r)]:
            bt.leave(done[0])
            rows = {bt.moved(r): i for r, i in rows.items() if r != done[0]}
        if not rows:
            frame += 1
            continue
        codes = bt.frame()
        for r, i in rows.items():
            out[i].append(codes[r:r + 1].clone())
        frame += 1
    return out


def _same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def test_rows_reproduce_each_lines_solo_run(talker):
    ft = _fast(talker)
    reqs = [_req(ft, s, max_new=m) for s, m in ((1, 12), (2, 7), (3, 10))]
    with torch.no_grad():
        solo = [_solo(ft, r) for r in reqs]
        bt = batch.BatchedTalker(ft, max_rows=4)
        got = _drive(bt, reqs)
        refs = [reference_talker(talker, **_prompt(s), max_new=m)[0]
                for s, m in ((1, 12), (2, 7), (3, 10))]
    for i in range(3):
        assert _same(got[i], solo[i]) and _same(got[i], refs[i]), i
    assert talker.upstream_calls == []


def test_rows_join_late_and_leave_at_their_own_end(talker):
    ft = _fast(talker)
    specs = ((1, 12), (2, 5), (3, 9), (4, 6))
    reqs = [_req(ft, s, max_new=m) for s, m in specs]
    with torch.no_grad():
        solo = [_solo(ft, r) for r in reqs]
        got = _drive(batch.BatchedTalker(ft, max_rows=4), reqs, joins={1: 2, 2: 3, 3: 7})
    for i in range(4):
        assert _same(got[i], solo[i]), i


def test_a_lines_tokens_do_not_depend_on_what_shares_its_frames(talker):
    ft = _fast(talker)
    target = _req(ft, 1)
    with torch.no_grad():
        alone = _drive(batch.BatchedTalker(ft, max_rows=4), [target])[0]
        for seeds in ((2,), (3, 4), (5, 6, 7)):
            others = [_req(ft, s, max_new=15) for s in seeds]
            got = _drive(batch.BatchedTalker(ft, max_rows=4), [target] + others)[0]
            assert _same(got, alone), seeds


def test_eos_and_min_new_tokens_per_row(talker):
    with torch.no_grad():
        talker.codec_head.weight[EOS] = 50 * talker.codec_head.weight[0].sign()
        talker.model.norm.weight.abs_()
    ft = _fast(talker)
    a = ft.request(_talker_kwargs(_prompt(1), min_new_tokens=3))
    b = ft.request(_talker_kwargs(_prompt(2), min_new_tokens=6))
    with torch.no_grad():
        solo = [_solo(ft, a), _solo(ft, b)]
        got = _drive(batch.BatchedTalker(ft, max_rows=2), [a, b])
    lengths = [len(x) for x in solo]
    assert lengths[0] >= 3 and lengths[1] >= 6 and max(lengths) < 11  # EOS, after min_new
    assert lengths[0] != lengths[1]  # each row leaves at its own EOS
    assert _same(got[0], solo[0]) and _same(got[1], solo[1])


def test_cache_grows_for_every_row(talker, monkeypatch):
    monkeypatch.setattr(batch, "CAPACITY", 4)
    ft = _fast(talker)
    reqs = [_req(ft, 1, max_new=10), _req(ft, 2, max_new=8, length=3)]
    with torch.no_grad():
        solo = [_solo(ft, r) for r in reqs]
        bt = batch.BatchedTalker(ft, max_rows=2)
        got = _drive(bt, reqs)
    assert _same(got[0], solo[0]) and _same(got[1], solo[1])
    assert bt.capacity == 16


def test_bucket_one_is_the_fast_talkers_own_frame(talker, monkeypatch):
    """A lone row runs FastTalker._frame itself (today's graph), not the
    batched frame."""
    ft = _fast(talker)
    calls = []
    real = ft._frame
    monkeypatch.setattr(ft, "_frame", lambda *a, **k: calls.append(1) or real(*a, **k))
    bt = batch.BatchedTalker(ft, max_rows=4)
    with torch.no_grad():
        _drive(bt, [_req(ft, 1, max_new=4)])
    assert len(calls) == 3 and bt.buckets_used == {1}


def test_buckets_round_up(talker):
    ft = _fast(talker)
    bt = batch.BatchedTalker(ft, max_rows=8)
    with torch.no_grad():
        _drive(bt, [_req(ft, s, max_new=4) for s in range(3)])
    assert bt.buckets_used == {4} and bt.st.rows == 4  # rows grow by bucket, not max_rows
    assert batch.bucket_for(1) == 1 and batch.bucket_for(3) == 4 and batch.bucket_for(9) == 16


def test_padded_prefill_gives_the_same_lines(talker, monkeypatch):
    """The CUDA-graph prefill pads the prompt at the end to a step of
    PREFILL_STEP tokens and copies the real positions' K/V into the row; the
    talker is causal, so the line is unchanged. Checked eagerly here."""
    monkeypatch.setattr(batch, "PREFILL_STEP", 8)
    ft = _fast(talker)
    reqs = [_req(ft, s, max_new=9, length=l) for s, l in ((1, 5), (2, 3), (3, 7))]
    with torch.no_grad():
        solo = [_solo(ft, r) for r in reqs]
        bt = batch.BatchedTalker(ft, max_rows=4, padded_prefill=True)
        got = _drive(bt, reqs)
    for i in range(3):
        assert _same(got[i], solo[i]), i
    assert bt.prefill_lengths == {8}


def test_different_parameters_cannot_share_a_frame(talker):
    ft = _fast(talker)
    bt = batch.BatchedTalker(ft, max_rows=4)
    a = _req(ft, 1)
    b = ft.request(_talker_kwargs(_prompt(2), repetition_penalty=2.0))
    with torch.no_grad():
        bt.join(a)
        assert not bt.accepts(b) and bt.accepts(_req(ft, 3))


def test_masked_eager_equals_graph_path_shape_for_each_bucket(talker):
    """CPU has no graphs; the batched frame must still be the masked
    whole-cache computation for any bucket (rows past the live ones idle)."""
    ft = _fast(talker)
    reqs = [_req(ft, s, max_new=6) for s in range(5)]
    with torch.no_grad():
        solo = [_solo(ft, r) for r in reqs]
        got = _drive(batch.BatchedTalker(ft, max_rows=8), reqs)
    for i in range(5):
        assert _same(got[i], solo[i]), i


# -- engine ---------------------------------------------------------------------------


class _Decoder:
    """Stand-in speech decoder: a frame's audio is its first code, repeated."""

    def __init__(self):
        self.calls = []

    def stream(self, ref_codes=None):
        dec = self

        class S:
            def __init__(self):
                self.frames, self.done = [], 0

            @property
            def pending(self):
                return len(self.frames) - self.done

            def push(self, f):
                self.frames.append(f)

            def decode(self):
                new = self.frames[self.done:]
                self.done = len(self.frames)
                dec.calls.append(len(new))
                return np.repeat([float(f[0, 0]) for f in new], 3).astype(np.float32)
        return S()


def _whole(codes, ref):
    return np.repeat(codes[:, 0].float().numpy(), 3).astype(np.float32)


def _engine(ft, **kw):
    kw.setdefault("max_rows", 4)
    kw.setdefault("chunk_frames", 4)
    return batch.Engine(ft, decoder=_Decoder(), decode_whole=_whole, torch=torch,
                        first_frames=2, **kw)


def test_engine_streams_and_finishes_many_requests_like_solo_runs(talker):
    ft = _fast(talker)
    specs = [(s, 10 + s) for s in range(6)]  # >= 9 frames: chunks 2, 4, 4, ...
    reqs = [_req(ft, s, max_new=m) for s, m in specs]
    with torch.no_grad():
        want = [np.repeat([float(c[0, 0]) for c in _solo(ft, r)], 3) for r in reqs]
    eng = _engine(ft)
    try:
        handles = [eng.submit(lambda r=r: r, stream=(i % 2 == 0)) for i, r in enumerate(reqs)]
        for i, h in enumerate(handles):
            if i % 2 == 0:
                chunks = list(h)
                sizes = [len(c) // 3 for c in chunks]
                assert sizes[0] == 2 and max(sizes) <= 4  # light load: first_frames, cap chunk_frames
                got = np.concatenate(chunks)
            else:
                got = h.result(timeout=30)
            np.testing.assert_array_equal(got, want[i])
    finally:
        eng.close()


def test_engine_cancel_frees_the_row(talker):
    ft = _fast(talker)
    eng = _engine(ft, max_rows=1)
    try:
        long = eng.submit(lambda: _req(ft, 1, max_new=200), stream=True)
        next(iter(long))
        long.cancel()
        short = eng.submit(lambda: _req(ft, 2, max_new=5), stream=False)
        assert len(short.result(timeout=30)) == 4 * 3
    finally:
        eng.close()


def test_engine_error_fails_the_frame_and_turns_itself_off(talker, monkeypatch):
    ft = _fast(talker)
    eng = _engine(ft)
    try:
        monkeypatch.setattr(eng.talker, "frame", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        h = eng.submit(lambda: _req(ft, 1), stream=False)
        with pytest.raises(RuntimeError, match="boom"):
            h.result(timeout=30)
        assert eng.failed
        with pytest.raises(batch.EngineUnavailable):
            eng.submit(lambda: _req(ft, 2), stream=False)
    finally:
        eng.close()


def test_engine_refuses_unhandled_requests(talker):
    ft = _fast(talker)
    eng = _engine(ft)
    try:
        h = eng.submit(lambda: None, stream=False)
        with pytest.raises(batch.EngineUnavailable):
            h.result(timeout=30)
        assert not eng.failed
    finally:
        eng.close()


def test_engine_paces_streams_to_a_lead(talker):
    """With every stream more than ``lead_seconds`` of audio ahead of its
    playback, the engine generates nothing until one falls back."""
    ft = _fast(talker)
    eng = _engine(ft, lead_seconds=0.5, frame_seconds=1.0)  # each frame is 1 s of audio
    try:
        h = eng.submit(lambda: _req(ft, 1, max_new=200), stream=True)
        it = iter(h)
        next(it)
        time.sleep(0.3)
        frames = eng.frames_run
        time.sleep(0.3)
        assert eng.frames_run == frames  # paced: ~2 s ahead, nothing generated
        h.cancel()
    finally:
        eng.close()


# -- deadline-driven emission -------------------------------------------------------


def test_first_chunk_grows_with_the_load():
    """c1 >= f / (T - f) (with margin): the second chunk, generated during
    the first one's playback, can then be at least as long as the first."""
    first = batch.first_chunk_frames
    assert first(0.007, 0.08, 2, 16) == 2      # light: the configured first chunk
    assert first(0.040, 0.08, 2, 16) == 2      # 1.0 * 1.5 -> 2
    assert first(0.060, 0.08, 2, 16) == 5      # 3.0 * 1.5 -> 5
    assert first(0.075, 0.08, 2, 16) == 16     # capped at chunk_frames
    assert first(0.090, 0.08, 2, 16) == 16     # slower than real time: the cap


def test_emission_is_due_at_the_cap_or_the_deadline():
    due = batch.emission_due
    # pending reached chunk_frames
    assert due(pending=8, sent=True, lead=5.0, first=2, chunk=8, frame_est=0.01, decode_est=0.01)
    # first chunk: when pending reaches the load-aware first size
    assert not due(pending=1, sent=False, lead=0.0, first=2, chunk=8, frame_est=0.01, decode_est=0.01)
    assert due(pending=2, sent=False, lead=0.0, first=2, chunk=8, frame_est=0.01, decode_est=0.01)
    # playing: wait while one more frame plus a decode fits in the lead (with safety)
    assert not due(pending=3, sent=True, lead=0.5, first=2, chunk=8, frame_est=0.02, decode_est=0.01)
    assert due(pending=3, sent=True, lead=0.045, first=2, chunk=8, frame_est=0.02, decode_est=0.01)
    assert not due(pending=0, sent=True, lead=0.0, first=2, chunk=8, frame_est=0.02, decode_est=0.01)


def test_engine_emits_by_deadline_on_a_fake_clock(talker):
    """With a fake clock the listener's playback position is known exactly:
    chunks stay small while the lead is short and grow to the cap once the
    engine is far ahead; the client buffer adds to every deadline."""
    ft = _fast(talker)
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    eng = _engine(ft, clock=now, frame_seconds=1.0, chunk_frames=6)
    try:
        h = eng.submit(lambda: _req(ft, 1, max_new=40), stream=True)
        sizes = [len(c) // 3 for c in h]
    finally:
        eng.close()
    # the fake clock never moves: after the first chunk the lead only grows,
    # so every later chunk waits for the cap
    assert sizes[0] == 2 and all(s == 6 for s in sizes[1:-1]) and sum(sizes) == 39


def test_engine_emits_early_when_the_listener_is_about_to_run_dry(talker):
    ft = _fast(talker)
    clock = {"t": 0.0}
    eng = _engine(ft, clock=lambda: clock["t"], frame_seconds=1.0, chunk_frames=6)
    real_frame = eng.talker.frame

    def frame():  # each frame takes 0.9 s of (fake) time: under 1 s of audio, barely
        clock["t"] += 0.9
        return real_frame()
    eng.talker.frame = frame
    try:
        h = eng.submit(lambda: _req(ft, 1, max_new=20), stream=True)
        sizes = [len(c) // 3 for c in h]
    finally:
        eng.close()
    assert sum(sizes) == 19 and max(sizes[1:]) < 6  # deadline-driven, not waiting for the cap


def test_buffer_seconds_extends_every_deadline(talker):
    ft = _fast(talker)
    clock = {"t": 0.0}
    eng = _engine(ft, clock=lambda: clock["t"], frame_seconds=1.0, chunk_frames=6,
                  buffer_seconds=100.0)
    real_frame = eng.talker.frame

    def frame():
        clock["t"] += 0.9
        return real_frame()
    eng.talker.frame = frame
    try:
        h = eng.submit(lambda: _req(ft, 1, max_new=20), stream=True)
        sizes = [len(c) // 3 for c in h]
    finally:
        eng.close()
    assert sizes[0] == 2 and all(s == 6 for s in sizes[1:-1])  # the buffer covers the wait


def test_whole_lines_decode_in_chunks_as_they_go(talker):
    """A non-streamed line's audio is decoded chunk_frames at a time while it
    generates, not in one long call at its end that would stall every stream."""
    ft = _fast(talker)
    eng = _engine(ft)  # chunk_frames 4
    try:
        h = eng.submit(lambda: _req(ft, 1, max_new=12), stream=False)
        audio = h.result(timeout=30)
        assert len(audio) == 11 * 3
        assert eng.decoder.calls == [4, 4, 3]
    finally:
        eng.close()


# -- growth out of memory -------------------------------------------------------------


class OutOfMemoryError(RuntimeError):
    """Named like torch's, which the engine recognises by name."""


def _oom_past(monkeypatch, rows):
    real = batch._State

    def state(ft, like, n, capacity):
        if n > rows:
            raise OutOfMemoryError("CUDA out of memory (fake)")
        return real(ft, like, n, capacity)
    monkeypatch.setattr(batch, "_State", state)


def test_row_growth_oom_keeps_live_lines_and_caps_the_rows(talker, monkeypatch):
    ft = _fast(talker)
    _oom_past(monkeypatch, 2)
    eng = _engine(ft, max_rows=8)
    try:
        hs = [eng.submit(lambda s=s: _req(ft, s, max_new=10), stream=False) for s in range(4)]
        out = [h.result(timeout=60) for h in hs]  # the 3rd and 4th wait for a free row
        assert all(len(o) == 9 * 3 for o in out)
        assert not eng.failed and eng.talker.max_rows == 2
    finally:
        eng.close()


def test_kv_growth_oom_fails_only_the_line_that_needed_it(talker, monkeypatch):
    monkeypatch.setattr(batch, "CAPACITY", 8)
    ft = _fast(talker)
    real = batch._State

    def state(ft_, like, n, capacity):
        if capacity > 8:
            raise OutOfMemoryError("CUDA out of memory (fake)")
        return real(ft_, like, n, capacity)
    monkeypatch.setattr(batch, "_State", state)
    eng = _engine(ft, max_rows=2)
    try:
        long = eng.submit(lambda: _req(ft, 1, max_new=30, length=3), stream=False)
        short = eng.submit(lambda: _req(ft, 2, max_new=3, length=3), stream=False)
        assert len(short.result(timeout=60)) == 2 * 3
        with pytest.raises(OutOfMemoryError):
            long.result(timeout=60)
        assert not eng.failed
        after = eng.submit(lambda: _req(ft, 3, max_new=3, length=3), stream=False)
        assert len(after.result(timeout=60)) == 2 * 3
    finally:
        eng.close()


def test_other_frame_errors_still_turn_the_engine_off(talker, monkeypatch):
    ft = _fast(talker)
    eng = _engine(ft)
    try:
        monkeypatch.setattr(eng.talker, "frame", lambda: (_ for _ in ()).throw(ValueError("x")))
        with pytest.raises(ValueError):
            eng.submit(lambda: _req(ft, 1), stream=False).result(timeout=30)
        assert eng.failed
    finally:
        eng.close()


# -- warm-up ------------------------------------------------------------------------------


def test_warm_allocates_the_rows_and_leaves_no_lines(talker):
    ft = _fast(talker)
    eng = _engine(ft, max_rows=8)
    try:
        eng.warm(5, prepare=lambda: _req(ft, 1, max_new=6))
        assert eng.talker.st.rows == 8 and eng.talker.live == 0
        assert {1, 2, 4, 8} <= eng.talker.buckets_used
        h = eng.submit(lambda: _req(ft, 2, max_new=4), stream=False)
        assert len(h.result(timeout=30)) == 3 * 3
    finally:
        eng.close()


def test_engine_handle_carries_priority(talker):
    ft = _fast(talker)
    eng = _engine(ft)
    try:
        h = eng.submit(lambda: _req(ft, 1, max_new=3), stream=False, priority=5)
        assert h.priority == 5 and len(h.result(timeout=30)) == 2 * 3
    finally:
        eng.close()


def test_prepare_runs_on_the_engine_thread(talker):
    ft = _fast(talker)
    eng = _engine(ft)
    seen = []
    try:
        h = eng.submit(lambda: seen.append(threading.current_thread().name) or _req(ft, 1, max_new=3),
                       stream=False)
        h.result(timeout=30)
        assert seen == [eng.thread_name]
        assert eng.run_on_engine(lambda: threading.current_thread().name) == eng.thread_name
    finally:
        eng.close()


# -- Qwen3TTSRuntime with max_batch_size --------------------------------------------

from .test_qwen3_tts_stream import FakeQwen, TOL, make  # noqa: E402,F401 (fixture)

TEXTS = ["A line of speech.", "Another, longer line of speech here.", "Short.",
         "The fourth line, which is the longest of them all by far."]


def _run_threads(fn, items):
    out = [None] * len(items)
    errors = []

    def work(i, item):
        try:
            out[i] = fn(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(i, it)) for i, it in enumerate(items)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    return out


def test_max_batch_size_one_is_todays_runtime(make):
    m = make("clone")
    assert isinstance(m._inference_lock, type(threading.Lock())) and m._engine is None


@pytest.mark.parametrize("mode", ["clone", "design"])
def test_batched_runtime_matches_solo_lines(make, mode):
    solo = make(mode)
    want = [solo.synthesize(t).audio for t in TEXTS]
    want_stream = [np.concatenate([c.audio for c in solo.synthesize_stream(t)]) for t in TEXTS]
    m = make(mode, max_batch_size=4)
    try:
        assert isinstance(m._inference_lock, threading.Semaphore) and m._engine is not None

        def one(t):
            with m._inference_lock:
                return m.synthesize(t).audio
        got = _run_threads(one, TEXTS)
        for g, w in zip(got, want):
            assert len(g) == len(w)
            np.testing.assert_allclose(g, w, atol=TOL)

        def stream(t):
            with m._inference_lock:
                return [c.audio for c in m.synthesize_stream(t)]
        chunks = _run_threads(stream, TEXTS)
        for c, w in zip(chunks, want_stream):
            assert len(c) > 1
            np.testing.assert_allclose(np.concatenate(c), w, atol=TOL)
        assert m._engine.frames_run > 0
    finally:
        m._engine.close()


def test_batched_runtime_falls_back_after_an_engine_failure(make, monkeypatch):
    m = make("clone", max_batch_size=4)
    want = make("clone").synthesize(TEXTS[0]).audio
    try:
        monkeypatch.setattr(m._engine.talker, "frame",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            m.synthesize(TEXTS[0])
        assert m._engine.failed
        np.testing.assert_allclose(m.synthesize(TEXTS[0]).audio, want, atol=TOL)  # one at a time
        assert len(list(m.synthesize_stream(TEXTS[0]))) >= 1
    finally:
        m._engine.close()


def test_batched_runtime_on_cpu_has_no_engine(make):
    m = make("clone", device="cpu", max_batch_size=4)
    assert m._engine is None and isinstance(m._inference_lock, type(threading.Lock()))


def test_batch_capability_keys_are_hidden():
    from muse.core.capability_keys import classify

    assert classify("max_batch_size") == "hidden"
    assert classify("stream_lead_seconds") == "hidden"
    assert classify("stream_buffer_seconds") == "hidden"
    assert classify("warm_rows") == "hidden"


def test_runtime_warms_and_passes_the_client_buffer(make):
    m = make("clone", max_batch_size=8, warm_rows=3, stream_buffer_seconds=0.2)
    try:
        assert m._engine.talker.st.rows == 4 and m._engine.talker.live == 0
        assert m._engine.buffer_seconds == 0.2
        assert len(m.synthesize("A line of speech.").audio) > 0
    finally:
        m._engine.close()
    cold = make("clone", max_batch_size=8)
    try:
        assert cold._engine.talker.st is None  # default: no warm-up, nothing allocated
    finally:
        cold._engine.close()
