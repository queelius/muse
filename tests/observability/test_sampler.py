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
