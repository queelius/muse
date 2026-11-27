"""Qwen3-TTS runtime (VoiceDesign + Base clone) and its two bundled scripts.

qwen_tts is faked through the runtime's module-level ``QwenTTS`` sentinel;
no weights, no torch.
"""
from __future__ import annotations

import importlib.resources
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np
import pytest

from muse.modalities.audio_speech.protocol import (
    AudioChunk,
    AudioResult,
    TTSModel,
    UnknownVoiceError,
    VoiceError,
)
from muse.modalities.audio_speech.runtimes import qwen3_tts as rt

DESIGN_ID = "qwen3-tts-1.7b-voicedesign"
BASE_ID = "qwen3-tts-0.6b-base"


class FakeQwenTTS:
    """Stand-in for qwen_tts.Qwen3TTSModel; records every call."""

    instances: list["FakeQwenTTS"] = []
    output = np.zeros(2400, "float32")

    def __init__(self, path, kwargs):
        self.path = path
        self.load_kwargs = kwargs
        self.calls: list[tuple[str, dict]] = []
        FakeQwenTTS.instances.append(self)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(path, kwargs)

    def generate_voice_design(self, **kw):
        self.calls.append(("design", kw))
        return [FakeQwenTTS.output.copy()], 24000

    def create_voice_clone_prompt(self, **kw):
        self.calls.append(("prompt", kw))
        return [object()]

    def generate_voice_clone(self, **kw):
        self.calls.append(("clone", kw))
        return [FakeQwenTTS.output.copy()], 24000

    def named(self, kind):
        return [kw for k, kw in self.calls if k == kind]


@pytest.fixture(autouse=True)
def fake_qwen(monkeypatch, tmp_path):
    FakeQwenTTS.instances = []
    FakeQwenTTS.output = np.zeros(2400, "float32")
    monkeypatch.setattr(rt, "QwenTTS", FakeQwenTTS)
    monkeypatch.setattr(rt, "torch", None)
    monkeypatch.setenv("MUSE_HOME", str(tmp_path / "home"))
    return FakeQwenTTS


def _design(**kw):
    return rt.Qwen3TTSRuntime(
        hf_repo="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", local_dir="/weights/design",
        device="cpu", model_id=DESIGN_ID, mode="design", **kw)


def _clone(**kw):
    return rt.Qwen3TTSRuntime(
        hf_repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base", local_dir="/weights/base",
        device="cpu", model_id=BASE_ID, mode="clone", **kw)


def _narrator_transcript() -> str:
    data = importlib.resources.files("muse.modalities.audio_speech") / "data"
    return json.loads((data / "narrator.json").read_text(encoding="utf-8"))["transcript"]


def _tone(seconds=2.0, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")


def test_loads_from_local_dir_in_float32_on_cpu():
    m = _design()
    fake = FakeQwenTTS.instances[0]
    assert fake.path == "/weights/design"
    assert fake.load_kwargs["device_map"] == "cpu"
    assert m.model_id == DESIGN_ID
    assert m.sample_rate == 24000
    assert isinstance(m, TTSModel)
    assert hasattr(m._inference_lock, "acquire")


def test_dtype_is_bf16_on_cuda_and_fp32_on_cpu(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(rt, "torch", SimpleNamespace(
        bfloat16="BF16", float16="FP16", float32="FP32"))
    rt.Qwen3TTSRuntime(hf_repo="r", device="cuda", model_id=DESIGN_ID, mode="design")
    _design()
    on_cuda, on_cpu = FakeQwenTTS.instances
    assert on_cuda.path == "r"  # no local_dir: load from the repo id
    assert on_cuda.load_kwargs == {
        "device_map": "cuda:0", "dtype": "BF16", "attn_implementation": "sdpa"}
    assert on_cpu.load_kwargs == {"device_map": "cpu", "dtype": "FP32"}


def test_fast_code_predictor_installed_unless_disabled(monkeypatch):
    installed = []
    monkeypatch.setattr(rt.qwen3_tts_fast, "install",
                        lambda tts, *, torch, use_cuda_graphs, int8: installed.append(
                            (tts, use_cuda_graphs)) or True)
    _design()
    _clone(fast_code_predictor=False)
    assert installed == [(FakeQwenTTS.instances[0], False)]


def test_fast_talker_installed_after_the_predictor_unless_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(rt.qwen3_tts_fast, "install",
                        lambda tts, *, torch, use_cuda_graphs, int8: calls.append("predictor") or True)
    monkeypatch.setattr(rt.qwen3_tts_fast, "install_talker",
                        lambda tts, *, torch, use_cuda_graphs, int8: calls.append("talker") or True)
    _design()
    assert calls == ["predictor", "talker"]
    calls.clear()
    _clone(fast_talker=False)
    assert calls == ["predictor"]
    calls.clear()
    _clone(fast_code_predictor=False)  # the talker needs the fast predictor
    assert calls == []


def test_fast_decoder_installed_unless_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(rt.qwen3_tts_fast, "install_decoder",
                        lambda tts, *, torch: calls.append("decoder") or True)
    _design()
    assert calls == ["decoder"]
    calls.clear()
    _clone(fast_code_predictor=False)  # independent of the talker patches
    assert calls == ["decoder"]
    calls.clear()
    _clone(fast_decoder=False)
    assert calls == []


def test_int8_weights_is_opt_in_and_reaches_both_installs(monkeypatch):
    seen = []
    for name in ("install", "install_talker"):
        monkeypatch.setattr(rt.qwen3_tts_fast, name,
                            lambda tts, *, torch, use_cuda_graphs, int8, name=name:
                            seen.append((name, int8)) or True)
    _design()
    _clone(int8_weights=True)
    assert seen == [("install", False), ("install_talker", False),
                    ("install", True), ("install_talker", True)]


def test_fast_code_predictor_uses_cuda_graphs_on_cuda(monkeypatch):
    from types import SimpleNamespace

    installed = []
    monkeypatch.setattr(rt, "torch", SimpleNamespace(
        bfloat16="BF16", float16="FP16", float32="FP32"))
    for name in ("install", "install_talker"):
        monkeypatch.setattr(rt.qwen3_tts_fast, name,
                            lambda tts, *, torch, use_cuda_graphs, int8: installed.append(
                                use_cuda_graphs) or True)
    rt.Qwen3TTSRuntime(hf_repo="r", device="cuda", model_id=DESIGN_ID, mode="design")
    rt.Qwen3TTSRuntime(hf_repo="r", device="cuda", model_id=DESIGN_ID, mode="design",
                       cuda_graphs=False)
    assert installed == [True, True, False, False]


def test_design_uses_instructions_and_auto_language():
    m = _design()
    r = m.synthesize("Hello there.", voice="alloy", speed=1.0, align=False,
                     instructions="gruff dwarf")
    (kw,) = FakeQwenTTS.instances[0].named("design")
    assert kw["instruct"] == "gruff dwarf"
    assert kw["text"] == "Hello there."
    assert kw["language"] is None
    assert r.metadata["voice"] is None
    assert m.voices == []
    assert m.default_voice is None


def test_design_default_instructions_when_absent():
    _design().synthesize("Hi.")
    _design(default_instructions="a cheerful bard").synthesize("Hi.")
    first, second = (f.named("design")[0] for f in FakeQwenTTS.instances)
    assert first["instruct"] == rt.DEFAULT_INSTRUCTIONS
    assert rt.DEFAULT_INSTRUCTIONS == (
        "A clear, neutral adult narrator with an even, friendly tone, "
        "speaking at a moderate pace.")
    assert second["instruct"] == "a cheerful bard"


def test_clone_defaults_to_narrator_and_passes_tuple():
    m = _clone()
    r = m.synthesize("The inn is warm.", voice=None, speed=1.0, align=False)
    fake = FakeQwenTTS.instances[0]
    (prompt_kw,) = fake.named("prompt")
    ref = prompt_kw["ref_audio"]
    assert isinstance(ref, tuple)
    assert isinstance(ref[0], np.ndarray) and ref[0].dtype == np.float32
    assert ref[0].ndim == 1 and ref[0].size > 24000
    assert ref[1] == 24000
    assert prompt_kw["ref_text"] == _narrator_transcript()
    (clone_kw,) = fake.named("clone")
    assert clone_kw["language"] is None
    assert clone_kw["text"] == "The inn is warm."
    assert r.metadata["voice"] == "narrator"
    assert m.default_voice == "narrator"


def test_every_line_has_a_frame_limit_from_its_text(caplog):
    """A sampled line that never emits EOS would run to qwen-tts's 4096
    frames; the runtime caps each line at FRAMES_PER_CHAR * chars + slack
    (measured: at most 1.06 frames a character of English prose over 96
    takes; digits, prices and dates up to 5.5)."""
    _design().synthesize("Hello there.")
    _clone().synthesize("The inn is warm.")
    (d,) = FakeQwenTTS.instances[0].named("design")
    (c,) = FakeQwenTTS.instances[1].named("clone")
    assert d["max_new_tokens"] == rt.frame_limit("Hello there.")
    assert c["max_new_tokens"] == rt.frame_limit("The inn is warm.")
    assert rt.frame_limit("x" * 10) == int(rt.FRAMES_PER_CHAR * 10) + rt.FRAME_SLACK
    assert rt.frame_limit("x" * 5000) == 4096  # qwen-tts's own ceiling
    # text spoken longer than it is written (measured frames on the 0.6B)
    assert rt.frame_limit("Dr. Smith paid £3,499.99 on 03/04/2025.") > 2 * 174
    assert rt.frame_limit("1234567890") > 2 * 55
    FakeQwenTTS.output = np.zeros(1920 * rt.frame_limit("Hi.") - 1920, "float32")
    with caplog.at_level("WARNING"):
        _design().synthesize("Hi.")
    assert "frame limit" in caplog.text


def test_clone_unknown_voice_raises():
    m = _clone()
    with pytest.raises(UnknownVoiceError) as exc:
        m.synthesize("Hi.", voice="alloy")
    assert "narrator" in str(exc.value)
    with pytest.raises(UnknownVoiceError):
        m.check_voice("alloy")
    m.check_voice(None)
    m.check_voice("narrator")
    with pytest.raises(UnknownVoiceError):
        next(iter(m.synthesize_stream("Hi.", voice="alloy")))


def test_clone_prompt_cached_and_evicted_on_overwrite():
    m = _clone()
    fake = FakeQwenTTS.instances[0]
    m.save_voice("grimbold", _tone(), 24000, "Old words.")
    m.synthesize("One.", voice="grimbold")
    m.synthesize("Two.", voice="grimbold")
    assert len(fake.named("prompt")) == 1
    m.save_voice("grimbold", _tone(3.0), 24000, "New words.", overwrite=True)
    m.synthesize("Three.", voice="grimbold")
    prompts = fake.named("prompt")
    assert len(prompts) == 2
    assert prompts[1]["ref_text"] == "New words."
    assert prompts[1]["ref_audio"][0].size == 3 * 24000
    m.delete_voice("grimbold")
    with pytest.raises(UnknownVoiceError):
        m.synthesize("Four.", voice="grimbold")


def test_clone_prompt_cache_holds_every_saved_voice(monkeypatch):
    monkeypatch.setenv("MUSE_MAX_SAVED_VOICES", "40")
    m = _clone()
    fake = FakeQwenTTS.instances[0]
    names = [f"v{i}" for i in range(40)]
    for name in names:
        m.save_voice(name, _tone(1.0), 24000, f"Voice {name}.")
    for name in ["narrator", *names, "narrator", *names]:
        m.synthesize("Hi.", voice=name)
    assert len(fake.named("prompt")) == 41  # each voice prepared once


def test_clone_prompt_cache_is_lru_bounded():
    m = _clone()
    m._prompt_cache_size = 2
    fake = FakeQwenTTS.instances[0]
    for name in ("a", "b", "c"):
        m.save_voice(name, _tone(), 24000, f"Voice {name}.")
        m.synthesize("Hi.", voice=name)
    assert len(m._prompts) == 2
    m.synthesize("Hi.", voice="c")  # still cached
    assert len(fake.named("prompt")) == 3
    m.synthesize("Hi.", voice="a")  # evicted as least recently used
    assert len(fake.named("prompt")) == 4


def test_save_and_delete_go_through_store(tmp_path):
    m = _clone()
    stored = m.save_voice("grimbold", _tone(), 24000, "Hello, traveler.",
                          source_filename="g.wav")
    assert stored.name == "grimbold"
    assert m.voices == ["narrator", "grimbold"]
    voice_dir = tmp_path / "home" / "voices" / BASE_ID / "grimbold"
    assert (voice_dir / "reference.wav").is_file()
    # The store root is private even under a permissive umask.
    assert voice_dir.parent.stat().st_mode & 0o077 == 0
    with pytest.raises(VoiceError) as exc:
        m.save_voice("grimbold", _tone(), 24000, "Again.")
    assert exc.value.code == "voice_exists"
    with pytest.raises(VoiceError) as exc:
        m.delete_voice("narrator")
    assert exc.value.code == "voice_read_only"
    m.delete_voice("grimbold")
    assert m.voices == ["narrator"]
    assert not voice_dir.exists()


def test_save_with_metadata_reads_back_through_voice_entry():
    m = _clone()
    m.save_voice("grimbold", _tone(), 24000, "Hello, traveler.",
                 metadata={"category": "npc/dwarf", "tags": ["male"]})
    e = m.voice_entry("grimbold")
    assert (e.name, e.category, e.tags) == ("grimbold", "npc/dwarf", ("male",))
    assert m.voice_entry("narrator").category == "narrator/neutral"


def test_saved_voice_cap_comes_from_config(monkeypatch):
    monkeypatch.setenv("MUSE_MAX_SAVED_VOICES", "1")
    m = _clone()
    m.save_voice("one", _tone(), 24000, "First.")
    with pytest.raises(VoiceError) as exc:
        m.save_voice("two", _tone(), 24000, "Second.")
    assert exc.value.code == "voice_limit_reached"


def test_design_mode_refuses_saved_voices():
    m = _design()
    with pytest.raises(VoiceError):
        m.save_voice("grimbold", _tone(), 24000, "Hi.")
    with pytest.raises(VoiceError):
        m.delete_voice("grimbold")


def test_output_is_clipped_float32():
    FakeQwenTTS.output = np.array([2.0, -3.0, 0.5], dtype="float64")
    r = _design().synthesize("Hi.", speed=1.5)
    assert isinstance(r, AudioResult)
    assert r.audio.dtype == np.float32
    assert r.audio.tolist() == [1.0, -1.0, 0.5]
    assert r.sample_rate == 24000
    chunks = list(_design().synthesize_stream("Hi."))
    assert len(chunks) == 1 and isinstance(chunks[0], AudioChunk)
    assert chunks[0].audio.dtype == np.float32


def test_manifests_are_well_formed():
    from muse.core.capability_keys import classify
    from muse.models import qwen3_tts_0_6b_base as base
    from muse.models import qwen3_tts_1_7b_voicedesign as design

    assert design.Model is rt.Qwen3TTSRuntime
    assert base.Model is rt.Qwen3TTSRuntime
    d, b = design.MANIFEST, base.MANIFEST
    assert d["model_id"] == DESIGN_ID
    assert b["model_id"] == BASE_ID
    assert d["hf_repo"] == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert b["hf_repo"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    for m in (d, b):
        assert m["modality"] == "audio/speech"
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"])
        assert m["license"] == "Apache-2.0"
        assert "qwen-tts==0.1.1" in m["pip_extras"]
        assert any(e.startswith("torch") for e in m["pip_extras"])
        assert m["quality_tier"] == "good"
        for key in ("description", "summary", "strengths", "limits"):
            assert m.get(key), key
        caps = m["capabilities"]
        assert caps["sample_rate"] == 24000
        assert caps["device"] == "auto"
        for key in caps:
            assert classify(key) is not None, key
        assert classify("mode") == "hidden"
        assert classify("default_instructions") == "hidden"
        # the streaming chunk schedule, in 12 Hz frames
        assert classify("stream_first_frames") == "hidden"
        assert classify("stream_chunk_frames") == "hidden"
        assert caps["stream_first_frames"] == rt.STREAM_FIRST_FRAMES
        assert caps["stream_chunk_frames"] == rt.STREAM_CHUNK_FRAMES
        assert "no streaming" not in " ".join(m["limits"])
    assert d["revision"].startswith("5ecdb673")
    dc, bc = d["capabilities"], b["capabilities"]
    assert dc["mode"] == "design" and bc["mode"] == "clone"
    assert dc["memory_gb"] == 4.5 and bc["memory_gb"] == 2.8
    assert dc["int8_weights"] is True and bc["int8_weights"] is True
    assert dc["supports_instructions"] is True
    assert dc["voices"] == []
    assert not dc.get("supports_saved_voices")
    assert bc["supports_saved_voices"] is True
    assert bc["voice_cloning"] is True
    assert not bc.get("supports_instructions")


def test_saved_voice_models_install_the_reference_decoder():
    """POST /v1/audio/speech/voices decodes the upload with TorchCodec in the
    model's own venv, so every bundled script that accepts saved voices must
    install it (and its ffmpeg libraries); the 0.6B Base model once returned
    500 "audio decoder unavailable" on every save."""
    from pathlib import Path

    import muse.models
    from muse.core.discovery import discover_models

    found = discover_models([Path(muse.models.__file__).parent])
    savers = {mid: d.manifest for mid, d in found.items()
              if d.manifest.get("capabilities", {}).get("supports_saved_voices")}
    assert BASE_ID in savers
    for mid, manifest in savers.items():
        assert any(e.startswith("torchcodec") for e in manifest["pip_extras"]), mid
        assert "ffmpeg" in manifest["system_packages"], mid


def test_design_model_needs_no_decoder():
    from muse.models import qwen3_tts_1_7b_voicedesign as design

    assert design.MANIFEST["system_packages"] == ()
    assert not any(e.startswith("torchcodec") for e in design.MANIFEST["pip_extras"])


def test_bundled_narrator_is_packaged():
    data = importlib.resources.files("muse.modalities.audio_speech") / "data"
    wav_path, json_path = data / "narrator.wav", data / "narrator.json"
    assert wav_path.is_file() and json_path.is_file()
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["name"] == "narrator"
    assert meta["sample_rate"] == 24000
    assert meta["description"] == rt.DEFAULT_INSTRUCTIONS
    assert meta["transcript"] == (
        "Welcome, traveler. The road ahead is long, but the inn is warm.")
    assert meta["generated_by"] == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb673"
    with wave.open(str(wav_path), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        seconds = w.getnframes() / w.getframerate()
    assert 2.0 <= seconds <= 8.0
    assert abs(meta["duration_seconds"] - seconds) < 0.01


def test_module_import_is_light(tmp_path):
    code = (
        "import sys\n"
        "import muse.models.qwen3_tts_1_7b_voicedesign\n"
        "import muse.models.qwen3_tts_0_6b_base\n"
        "from muse.core.catalog import known_models\n"
        "known = known_models()\n"
        f"assert {DESIGN_ID!r} in known and {BASE_ID!r} in known, sorted(known)\n"
        "for mod in ('torch', 'qwen_tts', 'transformers'):\n"
        "    assert mod not in sys.modules, mod + ' was imported'\n"
        "print('ok')\n"
    )
    env = {**os.environ, "MUSE_CATALOG_DIR": str(tmp_path / "catalog")}
    env.pop("MUSE_MODELS_DIR", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("ok")
