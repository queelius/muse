"""VoiceCatalog: stdlib reads and metadata edits under a cross-process lock."""
import json
import os
import subprocess
import sys
import textwrap
import time
import wave
from pathlib import Path

import pytest

from muse.modalities.audio_speech.voice_catalog import BUNDLED_VOICES, VoiceCatalog
from muse.modalities.audio_speech.voice_errors import (
    UnknownVoiceError,
    VoiceError,
    VoiceUnreadableError,
)

SRC = Path(__file__).resolve().parents[3] / "src"


def _wav_bytes(seconds: float = 2.0, rate: int = 24000) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


def _write_voice(tmp_path: Path, name: str, meta_extra: dict | None = None) -> Path:
    """Write <tmp>/voices/m/<name>/ with today's voice.json fields; return the root."""
    root = tmp_path / "voices" / "m"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(tmp_path / "voices", 0o700)
    os.chmod(root, 0o700)
    (d / "reference.wav").write_bytes(_wav_bytes())
    meta = {
        "name": name, "transcript": "Ahoy.", "sample_rate": 24000,
        "duration_seconds": 2.0, "sha256": "ab" * 32,
        "created_at": "2026-11-20T00:00:00+00:00", "source_filename": None,
        **(meta_extra or {}),
    }
    (d / "voice.json").write_text(json.dumps(meta))
    return root


def test_fresh_install_lists_only_bundled_and_creates_nothing(tmp_path):
    cat = VoiceCatalog.for_root(tmp_path / "voices" / "m")
    assert cat.names() == ["narrator"]
    assert [e.name for e in cat.entries()] == ["narrator"]
    assert cat.entry("narrator").read_only
    assert not (tmp_path / "voices").exists()


def test_bundled_first_then_saved_sorted(tmp_path):
    _write_voice(tmp_path, "zed")
    root = _write_voice(tmp_path, "abe")
    assert VoiceCatalog.for_root(root).names() == ["narrator", "abe", "zed"]


def test_for_root_is_one_instance_per_root_and_bundled(tmp_path):
    root = tmp_path / "v"
    assert VoiceCatalog.for_root(root) is VoiceCatalog.for_root(root)
    assert VoiceCatalog.for_root(root, bundled={}) is not VoiceCatalog.for_root(root)


def test_for_model_uses_voice_store_root(tmp_path, monkeypatch):
    from muse.core import config

    monkeypatch.setenv("MUSE_HOME", str(tmp_path))
    config.reset_config()
    try:
        cat = VoiceCatalog.for_model("qwen3-tts-0.6b-base")
        assert cat.root == tmp_path / "voices" / "qwen3-tts-0.6b-base"
    finally:
        config.reset_config()


def test_old_voice_json_lists_and_patch_adds_only_sent_fields(tmp_path):
    root = _write_voice(tmp_path, "brineknife_captain", meta_extra={})
    cat = VoiceCatalog.for_root(root)
    assert cat.entry("brineknife_captain").category is None
    e = cat.update("brineknife_captain", {"category": "npc/pirate/captain"})
    assert e.category == "npc/pirate/captain" and e.tags == () and e.updated_at
    meta = json.loads((root / "brineknife_captain" / "voice.json").read_text())
    assert meta["transcript"] == "Ahoy."
    assert meta["created_at"] == "2026-11-20T00:00:00+00:00"
    assert meta["category"] == "npc/pirate/captain"
    assert "notes" not in meta


def test_update_null_clears_and_bad_value_is_rejected(tmp_path):
    root = _write_voice(tmp_path, "g", meta_extra={"notes": "old", "tags": ["a"]})
    cat = VoiceCatalog.for_root(root)
    assert cat.update("g", {"notes": None}).notes is None
    with pytest.raises(VoiceError) as e:
        cat.update("g", {"category": "Bad Cat!"})
    assert e.value.code == "invalid_voice_metadata"
    assert cat.entry("g").tags == ("a",)


def test_update_never_touches_audio(tmp_path):
    root = _write_voice(tmp_path, "g")
    before = (root / "g" / "reference.wav").read_bytes()
    VoiceCatalog.for_root(root).update("g", {"description": "deep"})
    assert (root / "g" / "reference.wav").read_bytes() == before


def test_update_and_delete_bundled_are_read_only(tmp_path):
    cat = VoiceCatalog.for_root(tmp_path / "v")
    for call in (lambda: cat.update("narrator", {"notes": "x"}),
                 lambda: cat.delete("narrator")):
        with pytest.raises(VoiceError) as e:
            call()
        assert (e.value.code, e.value.status) == ("voice_read_only", 409)


def test_delete_removes_voice(tmp_path):
    root = _write_voice(tmp_path, "g")
    cat = VoiceCatalog.for_root(root)
    cat.delete("g")
    assert cat.names() == ["narrator"]
    assert not (root / "g").exists()
    with pytest.raises(UnknownVoiceError):
        cat.delete("g")


def test_audio_bytes_reads_reference(tmp_path):
    root = _write_voice(tmp_path, "g")
    assert VoiceCatalog.for_root(root).audio_bytes("g")[:4] == b"RIFF"
    assert VoiceCatalog.for_root(root).audio_bytes("narrator")[:4] == b"RIFF"


@pytest.mark.parametrize("bad", ["nope", "../etc", "A", "x" * 65, "", ".lock"])
def test_unknown_and_bad_names_are_voice_not_found(tmp_path, bad):
    cat = VoiceCatalog.for_root(_write_voice(tmp_path, "a"))
    for call in (cat.entry, cat.audio_bytes):
        with pytest.raises(UnknownVoiceError):
            call(bad)
    with pytest.raises(UnknownVoiceError):
        cat.update(bad, {"notes": "x"})


def test_corrupt_voice_json_is_skipped_in_list_and_unreadable_on_get(tmp_path):
    root = _write_voice(tmp_path, "bad")
    (root / "bad" / "voice.json").write_text("{")
    cat = VoiceCatalog.for_root(root)
    assert "bad" not in cat.names()
    assert "bad" not in [e.name for e in cat.entries()]
    with pytest.raises(VoiceUnreadableError):
        cat.entry("bad")


def test_symlinked_and_scratch_dirs_are_skipped(tmp_path):
    root = _write_voice(tmp_path, "real")
    (root / "link").symlink_to(root / "real")
    (root / ".tmp-abc").mkdir()
    assert VoiceCatalog.for_root(root).names() == ["narrator", "real"]


def test_entries_reflect_an_edit_from_another_writer(tmp_path):
    root = _write_voice(tmp_path, "g")
    cat = VoiceCatalog.for_root(root)
    assert cat.entry("g").notes is None
    meta = json.loads((root / "g" / "voice.json").read_text())
    meta["notes"] = "edited elsewhere"
    tmp = root / "g" / "voice.json.new"
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, root / "g" / "voice.json")
    assert [e.notes for e in cat.entries() if e.name == "g"] == ["edited elsewhere"]


def _spawn_lock_holder(root: Path, hold_seconds: float) -> subprocess.Popen:
    code = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from muse.modalities.audio_speech.voice_catalog import VoiceCatalog
        cat = VoiceCatalog.for_root(Path({str(root)!r}))
        with cat.lock(exclusive=True):
            print("held", flush=True)
            time.sleep({hold_seconds})
    """)
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                            text=True, env=env)


def test_reader_in_another_process_waits_for_exclusive_lock(tmp_path):
    root = _write_voice(tmp_path, "a")
    child = _spawn_lock_holder(root, hold_seconds=0.6)
    try:
        assert child.stdout.readline().strip() == "held"
        t0 = time.monotonic()
        VoiceCatalog.for_root(root).entries()
        waited = time.monotonic() - t0
    finally:
        child.wait(timeout=10)
    assert waited >= 0.4


def test_nested_lock_reuses_and_upgrade_raises(tmp_path):
    cat = VoiceCatalog.for_root(_write_voice(tmp_path, "a"))
    with cat.lock(exclusive=True):
        assert cat.names()
    with cat.lock(exclusive=False):
        with pytest.raises(RuntimeError, match="upgrade"):
            with cat.lock(exclusive=True):
                pass
    with cat.lock(exclusive=True):  # usable again after the failed upgrade
        pass


def test_bundled_voices_point_at_package_data():
    wav, meta = BUNDLED_VOICES["narrator"]
    assert wav.name == "narrator.wav" and meta.name == "narrator.json"
    assert wav.is_file() and meta.is_file()
