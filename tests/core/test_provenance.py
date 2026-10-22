import json

from muse.core import provenance
from muse.core.provenance import (
    lookup_revision, model_revision, revision_headers, system_fingerprint,
)

SNAP = "/home/u/.muse/weights/models--Systran--faster-whisper-base/snapshots/" + "e" * 40


def test_recorded_revision_wins():
    r = model_revision("m", {"revision": "a" * 40, "local_dir": SNAP, "hf_repo": "o/r"}, {})
    assert r.revision == "a" * 40


def test_snapshot_path_fallback_for_old_pulls():
    r = model_revision("whisper-base", {"local_dir": SNAP, "hf_repo": "Systran/faster-whisper-base"}, {})
    assert r.revision == "e" * 40
    assert [(c.role, c.repo, c.revision) for c in r.components] == [
        ("weights", "Systran/faster-whisper-base", "e" * 40)]
    assert r.artifacts_digest is None


def test_manifest_revision_fallback():
    assert model_revision("m", {}, {"revision": "d" * 40}).revision == "d" * 40


def test_unknown_revision_is_none_not_error():
    r = model_revision("mine", {"local_dir": "/data/my-weights"}, {})
    assert r.revision is None and r.artifacts_digest is None
    assert revision_headers(r) == [(b"x-muse-model", b"mine")]
    assert system_fingerprint(r) is None


def test_lora_adds_base_component_and_digest():
    catalog = {"sdxl-turbo": {"revision": "b" * 40, "hf_repo": "stabilityai/sdxl-turbo"}}
    r = model_revision(
        "pixel-art-xl",
        {"local_dir": "/w/models--nerijs--pixel-art-xl/snapshots/" + "c" * 40,
         "hf_repo": "nerijs/pixel-art-xl"},
        {"capabilities": {"lora_adapter": True, "base_model": "sdxl-turbo"}}, catalog)
    assert [c.role for c in r.components] == ["weights", "base"]
    assert r.components[1].revision == "b" * 40
    assert len(r.artifacts_digest) == 16
    h = dict(revision_headers(r))
    assert h[b"x-muse-model-artifacts"] == r.artifacts_digest.encode()
    assert h[b"x-muse-model-revision"] == b"c" * 40


def test_base_override_wins():
    catalog = {"other": {"revision": "9" * 40, "hf_repo": "x/other"}}
    r = model_revision("l", {"revision": "1" * 40, "hf_repo": "o/l", "base_override": "other"},
                       {"capabilities": {"lora_adapter": True, "base_model": "sdxl-turbo"}}, catalog)
    assert r.components[1].revision == "9" * 40


def test_provenance_items_and_code():
    r = model_revision("m", {"artifact_provenance": [
        {"repo_id": "a/b", "revision": "1" * 40}, {"repo_id": "a/c", "revision": "2" * 40}],
        "revision": "1" * 40}, {"code_revision": "f" * 40, "hf_repo": "a/b"})
    assert [c.role for c in r.components] == ["weights", "weights", "code"]
    assert r.artifacts_digest


def test_fingerprint_format():
    assert system_fingerprint(model_revision("q", {"revision": "0123456789abcdef" + "0" * 24}, {})) == "q@0123456789ab"


def test_lookup_unknown_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    r = lookup_revision("nope")
    assert r.model_id == "nope" and r.revision is None and r.components == ()


def test_lookup_reads_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps({"x": {"revision": "a" * 40, "hf_repo": "o/x",
                                     "manifest": {"model_id": "x", "modality": "audio/speech",
                                                  "hf_repo": "o/x"}}}))
    assert lookup_revision("x").revision == "a" * 40
    cat.write_text(json.dumps({"x": {"revision": "b" * 40, "hf_repo": "o/x",
                                     "manifest": {"model_id": "x", "modality": "audio/speech",
                                                  "hf_repo": "o/x"}}}))
    import os
    os.utime(cat, ns=(10**18, 10**18))
    assert lookup_revision("x").revision == "b" * 40


def test_lookup_revision_cache_never_stores_result_for_old_mtime(monkeypatch, tmp_path):
    """A result computed against an old mtime must not land in the new cache."""
    import threading
    from muse.core import provenance, catalog_store

    path = tmp_path / "catalog.json"
    path.write_text("{}")
    monkeypatch.setattr(catalog_store, "_catalog_path", lambda: path)
    provenance._cache.clear()
    provenance._cache_mtime = None

    entered = threading.Event()
    release = threading.Event()

    def slow_read():
        entered.set()
        release.wait(5)
        return {"m": {"manifest": {}}}

    monkeypatch.setattr(catalog_store, "_read_catalog", slow_read)
    t = threading.Thread(target=provenance.lookup_revision, args=("m",))
    t.start()
    assert entered.wait(5)
    # catalog changes while the slow lookup is in flight
    import os
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    monkeypatch.setattr(catalog_store, "_read_catalog", lambda: {})
    provenance.lookup_revision("other")  # observes the new mtime, resets cache
    release.set()
    t.join(5)
    assert "m" not in provenance._cache
