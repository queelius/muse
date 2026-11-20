"""Qwen3-TTS runtime: designed voices (VoiceDesign) and saved voices (Base).

One class serves both bundled scripts, selected by ``mode``:

- ``"design"`` (Qwen3-TTS-12Hz-1.7B-VoiceDesign): the voice comes from a
  natural-language description, the request's ``instructions`` (else the
  manifest's ``default_instructions``). There are no named voices.
- ``"clone"`` (Qwen3-TTS-12Hz-0.6B-Base): the voice is a saved reference clip
  plus its transcript, from a :class:`VoiceStore` under
  ``<paths.home>/voices/<model_id>/`` with the bundled read-only ``narrator``.

qwen-tts has no speed control (``speed`` is ignored) and no streaming
(``synthesize_stream`` yields one chunk). ``language=None`` is qwen-tts's
"Auto" detection.

Heavy deps (torch, qwen_tts) load in ``_ensure_deps``; tests replace the
``torch`` / ``QwenTTS`` sentinels.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from muse.core import config
from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    select_device,
    set_inference_mode,
)
from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, VoiceError
from muse.modalities.audio_speech.voices import (
    StoredVoice,
    VoiceStore,
    load_bundled_voice,
    voice_store_root,
)

logger = logging.getLogger(__name__)

torch: Any = None
QwenTTS: Any = None  # qwen_tts.Qwen3TTSModel

SAMPLE_RATE = 24000
NARRATOR = "narrator"
DEFAULT_INSTRUCTIONS = (
    "A clear, neutral adult narrator with an even, friendly tone, "
    "speaking at a moderate pace."
)
PROMPT_CACHE_SIZE = 32
_DATA = Path(__file__).resolve().parents[1] / "data"


def _ensure_deps() -> None:
    global torch, QwenTTS
    if QwenTTS is not None:
        return
    try:
        import torch as _torch
        from qwen_tts import Qwen3TTSModel as _QwenTTS
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qwen3-TTS requires torch and qwen-tts; run `muse pull` to "
            "install the model dependencies"
        ) from exc
    torch = _torch
    QwenTTS = _QwenTTS


def bundled_voices() -> dict[str, StoredVoice]:
    """The read-only voices shipped with muse (currently just ``narrator``)."""
    return {NARRATOR: load_bundled_voice(
        NARRATOR, _DATA / "narrator.wav", _DATA / "narrator.json")}


class Qwen3TTSRuntime:
    """audio/speech backend for Qwen3-TTS VoiceDesign and Base checkpoints."""

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        model_id: str,
        mode: Literal["design", "clone"],
        dtype: str = "bf16",
        default_instructions: str | None = None,
        **_: Any,
    ) -> None:
        if mode not in ("design", "clone"):
            raise ValueError(f"mode must be 'design' or 'clone', got {mode!r}")
        _ensure_deps()
        self._model_id = model_id
        self._default_instructions = default_instructions or DEFAULT_INSTRUCTIONS
        self._inference_lock = threading.Lock()
        self._prompts: OrderedDict[tuple, Any] = OrderedDict()
        self._prompts_lock = threading.Lock()
        self._store: VoiceStore | None = None
        if mode == "clone":
            self._store = VoiceStore(
                voice_store_root(model_id),
                bundled=bundled_voices(),
                max_voices=int(config.get_positive("limits.max_saved_voices")),
            )

        resolved = select_device(device, torch_module=torch)
        device_map = "cuda:0" if resolved == "cuda" else resolved
        load_kwargs: dict[str, Any] = {"device_map": device_map}
        torch_dtype = dtype_for_name(
            dtype if resolved.startswith("cuda") else "fp32", torch)
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        src = local_dir or hf_repo
        with LoadTimer(model_id, logger):
            self._tts = QwenTTS.from_pretrained(src, **load_kwargs)
        set_inference_mode(getattr(self._tts, "model", None))
        self._device = device_map

    # -- protocol ------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def voices(self) -> list[str]:
        return self._store.names() if self._store is not None else []

    @property
    def default_voice(self) -> str | None:
        return NARRATOR if self._store is not None else None

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        instructions: str | None = None,
        **_: Any,
    ) -> AudioResult:
        if self._store is None:
            instruct = (instructions or "").strip() or self._default_instructions
            wavs, sr = self._tts.generate_voice_design(
                text=text, instruct=instruct, language=None)
            used = None
        else:
            used = voice or NARRATOR
            prompt = self._clone_prompt(used)
            wavs, sr = self._tts.generate_voice_clone(
                text=text, language=None, voice_clone_prompt=prompt)
        audio = np.clip(np.asarray(wavs[0], dtype=np.float32).reshape(-1), -1.0, 1.0)
        return AudioResult(audio=audio, sample_rate=int(sr), metadata={"voice": used})

    def synthesize_stream(self, text: str, **kwargs: Any) -> Iterator[AudioChunk]:
        result = self.synthesize(text, **kwargs)
        yield AudioChunk(audio=result.audio, sample_rate=result.sample_rate)

    # -- saved voices (clone mode) -------------------------------------------

    def check_voice(self, voice: str | None) -> None:
        """Raise UnknownVoiceError before a stream starts; no-op in design mode."""
        if self._store is not None:
            self._store.get(voice or NARRATOR)

    def save_voice(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
        *,
        overwrite: bool = False,
        source_filename: str | None = None,
    ) -> StoredVoice:
        store = self._require_store()
        try:
            return store.save(name, audio, sample_rate, transcript,
                              overwrite=overwrite, source_filename=source_filename)
        finally:
            self._evict(name)

    def delete_voice(self, name: str) -> None:
        store = self._require_store()
        try:
            store.delete(name)
        finally:
            self._evict(name)

    # -- internals -----------------------------------------------------------

    def _require_store(self) -> VoiceStore:
        if self._store is None:
            raise VoiceError(
                "voice_saving_not_supported",
                "This model does not support saved voices")
        return self._store

    def _clone_prompt(self, name: str) -> Any:
        store = self._store
        stored = store.get(name)
        key = (name, stored.identity)
        with self._prompts_lock:
            if key in self._prompts:
                self._prompts.move_to_end(key)
                return self._prompts[key]
        stored, audio = store.get_with_audio(name)
        # Always the (array, sr) tuple: qwen-tts reads a str as a path, URL
        # or base64 payload. Saved voices are VOICE_SAMPLE_RATE (24 kHz);
        # qwen-tts resamples anything else itself.
        prompt = self._tts.create_voice_clone_prompt(
            ref_audio=(audio, stored.sample_rate), ref_text=stored.transcript)
        with self._prompts_lock:
            self._prompts[(name, stored.identity)] = prompt
            self._prompts.move_to_end((name, stored.identity))
            while len(self._prompts) > PROMPT_CACHE_SIZE:
                self._prompts.popitem(last=False)
        return prompt

    def _evict(self, name: str) -> None:
        with self._prompts_lock:
            for key in [k for k in self._prompts if k[0] == name]:
                del self._prompts[key]
