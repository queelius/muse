"""XTTS v2 model backend.

Wraps Coqui's XTTS v2 to implement the :class:`~narro.protocol.TTSModel`
protocol.  Key feature: zero-shot voice cloning from a 6-30 second
reference audio clip.  Supports 16 languages and streaming.

Requires: ``pip install TTS`` and system ``espeak-ng``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from narro.catalog import voices_dir
from narro.protocol import AudioChunk, AudioResult

logger = logging.getLogger(__name__)

XTTS_SAMPLE_RATE = 24000

XTTS_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko",
]


class XttsModel:
    """XTTS v2 TTS backend.

    Args:
        model_path: Local model directory (None = download from HF/Coqui).
        device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``.
        compile: Unused.
        quantize: Unused.
    """

    MODEL_ID = "xtts-v2"
    VOICES: list[str] = []

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        import torch
        from TTS.api import TTS

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading XTTS v2 (device=%s)", device)
        self._tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(device != "cpu"),
        )
        self._device = device

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def sample_rate(self) -> int:
        return XTTS_SAMPLE_RATE

    def list_voices(self) -> list[str]:
        """Return custom cloned voice names."""
        return sorted(self._custom_voice_names())

    def create_voice(self, name: str, wav_path: str) -> None:
        """Import a reference WAV as a named voice for cloning."""
        src = Path(wav_path)
        if not src.exists():
            raise FileNotFoundError(f"Reference audio not found: {wav_path}")
        dest = voices_dir(self.MODEL_ID) / f"{name}.wav"
        import shutil
        shutil.copy2(str(src), str(dest))
        logger.info("Saved voice %r to %s", name, dest)

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        """Synthesize speech from *text*.

        XTTS-specific kwargs:
            voice (str): custom voice name (must be created first via
                ``create_voice``).
            language (str): language code (default ``'en'``).
        """
        voice = kwargs.get("voice")
        language = kwargs.get("language", "en")

        speaker_wav = self._resolve_voice(voice)

        if speaker_wav:
            audio_list = self._tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
            )
        else:
            audio_list = self._tts.tts(text=text, language=language)

        audio = np.array(audio_list, dtype=np.float32)

        return AudioResult(
            audio=audio,
            sample_rate=XTTS_SAMPLE_RATE,
            metadata={"voice": voice, "language": language} if voice else {"language": language},
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        """XTTS supports streaming via the lower-level model API."""
        result = self.synthesize(text, **kwargs)
        yield AudioChunk(audio=result.audio, sample_rate=result.sample_rate)

    def _resolve_voice(self, voice: str | None) -> str | None:
        """Resolve a voice name to a WAV file path."""
        if voice is None:
            return None
        wav = voices_dir(self.MODEL_ID) / f"{voice}.wav"
        if wav.exists():
            return str(wav)
        available = ", ".join(self._custom_voice_names()[:5])
        raise ValueError(
            f"Unknown voice: {voice!r}. Create one with create_voice(). "
            f"Available: {available or '(none)'}"
        )

    def _custom_voice_names(self) -> list[str]:
        vdir = voices_dir(self.MODEL_ID)
        return [f.stem for f in vdir.iterdir() if f.suffix == ".wav"]
