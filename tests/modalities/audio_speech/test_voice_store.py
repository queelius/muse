import hashlib
import json
import wave

import numpy as np
import pytest

from muse.modalities.audio_speech.protocol import (
    UnknownVoiceError, VoiceError, VoiceUnreadableError,
)
from muse.modalities.audio_speech.voices import VoiceStore, load_bundled_voice


def _secs(s=2.0):
    return (np.sin(np.linspace(0, 100, int(24000 * s))) * 0.5).astype("float32")


def test_save_then_get_round_trips(tmp_path):
    store = VoiceStore(tmp_path)
    v = store.save("grimbold", _secs(), 24000, "  Welcome, traveler.  ")
    assert v.transcript == "Welcome, traveler."
    assert v.sample_rate == 24000 and abs(v.duration_seconds - 2.0) < 1e-3
    assert (tmp_path / "grimbold" / "reference.wav").is_file()
    assert store.get("grimbold").sha256 == v.sha256 == hashlib.sha256(v.audio_path.read_bytes()).hexdigest()
    assert store.names() == ["grimbold"]
    meta = json.loads((tmp_path / "grimbold" / "voice.json").read_text())
    assert {"name", "transcript", "sample_rate", "duration_seconds", "sha256",
            "created_at", "source_filename"} <= set(meta)
    assert not v.read_only and len(v.identity) == 2


@pytest.mark.parametrize("name", ["../x", "Grimbold", "", "-a", "a" * 65, "a/b", "a.b"])
def test_invalid_names_rejected_and_nothing_written(tmp_path, name):
    with pytest.raises(VoiceError) as e:
        VoiceStore(tmp_path).save(name, _secs(), 24000, "hi")
    assert e.value.code == "invalid_voice_name" and e.value.status == 400
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seconds", [0.5, 30.5])
def test_duration_bounds(tmp_path, seconds):
    with pytest.raises(VoiceError) as e:
        VoiceStore(tmp_path).save("a", _secs(seconds), 24000, "hi")
    assert e.value.code == "invalid_reference_audio"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("text", ["", "   ", "x" * 1001])
def test_transcript_bounds(tmp_path, text):
    with pytest.raises(VoiceError) as e:
        VoiceStore(tmp_path).save("a", _secs(), 24000, text)
    assert e.value.code == "invalid_transcript"


def test_existing_name_needs_overwrite(tmp_path):
    store = VoiceStore(tmp_path)
    first = store.save("a", _secs(), 24000, "one")
    with pytest.raises(VoiceError) as e:
        store.save("a", _secs(), 24000, "two")
    assert e.value.code == "voice_exists" and e.value.status == 409
    second = store.save("a", _secs(3.0), 24000, "two", overwrite=True)
    assert second.sha256 != first.sha256 and second.transcript == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["a"]


def test_limit_counts_new_names_only(tmp_path):
    store = VoiceStore(tmp_path, max_voices=1)
    store.save("a", _secs(), 24000, "x")
    with pytest.raises(VoiceError) as e:
        store.save("b", _secs(), 24000, "x")
    assert e.value.code == "voice_limit_reached" and e.value.status == 409
    store.save("a", _secs(), 24000, "y", overwrite=True)


def test_bundled_voice_is_read_only(tmp_path):
    ref = VoiceStore(tmp_path / "scratch").save("narrator", _secs(), 24000, "bundled line")
    bundled = load_bundled_voice(
        "narrator", ref.audio_path, ref.audio_path.with_name("voice.json"))
    assert bundled.read_only
    store = VoiceStore(tmp_path / "root", bundled={"narrator": bundled})
    for call in (lambda: store.save("narrator", _secs(), 24000, "x", overwrite=True),
                 lambda: store.delete("narrator")):
        with pytest.raises(VoiceError) as e:
            call()
        assert e.value.code == "voice_read_only" and e.value.status == 400
    store.save("zed", _secs(), 24000, "x")
    assert store.names() == ["narrator", "zed"]
    assert store.get("narrator") is bundled


def test_delete_and_unknown(tmp_path):
    store = VoiceStore(tmp_path)
    store.save("a", _secs(), 24000, "x")
    store.delete("a")
    assert not (tmp_path / "a").exists() and store.names() == []
    store.save("b", _secs(), 24000, "x")
    for call in (lambda: store.get("a"), lambda: store.delete("a")):
        with pytest.raises(UnknownVoiceError) as e:
            call()
        assert e.value.status == 404 and e.value.code == "voice_not_found"
        assert "'a'" in str(e.value) and "b" in str(e.value)


def test_half_written_directory_is_unreadable_not_listed(tmp_path):
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "voice.json").write_text("{}")
    store = VoiceStore(tmp_path)
    with pytest.raises(VoiceUnreadableError):
        store.get("half")
    assert store.names() == []


def test_corrupt_json_is_unreadable(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "voice.json").write_text("{nope")
    (tmp_path / "bad" / "reference.wav").write_bytes(b"x")
    store = VoiceStore(tmp_path)
    with pytest.raises(VoiceUnreadableError):
        store.get("bad")
    assert store.names() == []


def test_non_finite_audio_rejected(tmp_path):
    audio = _secs()
    audio[10] = np.nan
    with pytest.raises(VoiceError) as e:
        VoiceStore(tmp_path).save("a", audio, 24000, "x")
    assert e.value.code == "invalid_reference_audio"


def test_missing_root_lists_bundled_only_and_creates_lazily(tmp_path):
    store = VoiceStore(tmp_path / "nope")
    assert store.names() == [] and not (tmp_path / "nope").exists()
    store.save("a", _secs(), 24000, "x")
    assert (tmp_path / "nope" / "a").is_dir()


def test_wav_is_pcm16_full_scale(tmp_path):
    v = VoiceStore(tmp_path).save("a", np.ones(24000, dtype="float32"), 24000, "x")
    with wave.open(str(v.audio_path)) as w:
        assert w.getsampwidth() == 2 and w.getframerate() == 24000 and w.getnchannels() == 1
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert frames.max() == 32767
