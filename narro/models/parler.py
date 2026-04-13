"""Parler-TTS model backend.

Wraps parler-tts/parler-tts-mini-v1 to implement the
:class:`~narro.protocol.TTSModel` protocol.  Unique feature: the
``voice`` parameter is a free-text description of the desired voice
(e.g. "A warm female voice with a slow pace").

Requires: ``pip install parler-tts``
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from narro.protocol import AudioChunk, AudioResult

logger = logging.getLogger(__name__)

PARLER_VOICES = [
    "Jon", "Laura", "Gary", "Lea", "Karen", "Rick", "Brenda", "David",
    "Eileen", "Jordan", "Mike", "Yolanda", "Patrick", "Rose", "Jerry",
    "Jenna", "Bill", "Tom", "Carol", "Barbara", "Rebecca", "Anna",
    "Bruce", "Emily",
]


class ParlerModel:
    """Parler-TTS backend.

    Args:
        model_path: Local model directory (None = download from HF).
        device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``.
        compile: Unused.
        quantize: Unused.
    """

    MODEL_ID = "parler-tts"
    VOICES = PARLER_VOICES

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        repo = model_path or "parler-tts/parler-tts-mini-v1"
        dtype = torch.float16 if device != "cpu" else torch.float32

        logger.info("Loading Parler-TTS from %s (device=%s)", repo, device)
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo, torch_dtype=dtype,
        ).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(repo)
        self._device = device
        self._sample_rate = self._model.config.sampling_rate

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        """Synthesize speech from *text*.

        Parler-specific kwargs:
            voice (str): either a speaker name (e.g. ``"Laura"``) or a
                free-text voice description (e.g. ``"A warm female voice
                with a moderate pace and clear enunciation"``).
        """
        voice = kwargs.get("voice")
        description = self._build_description(voice)

        input_ids = self._tokenizer(
            description, return_tensors="pt",
        ).input_ids.to(self._device)
        prompt_input_ids = self._tokenizer(
            text, return_tensors="pt",
        ).input_ids.to(self._device)

        import torch
        with torch.inference_mode():
            generation = self._model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )

        audio = generation.cpu().numpy().squeeze().astype(np.float32)
        if audio.ndim == 0:
            audio = np.zeros(0, dtype=np.float32)

        return AudioResult(
            audio=audio,
            sample_rate=self._sample_rate,
            metadata={"voice": voice} if voice else {},
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        """Parler supports streaming via ParlerTTSStreamer."""
        voice = kwargs.get("voice")
        description = self._build_description(voice)

        import torch
        from threading import Thread
        from parler_tts import ParlerTTSStreamer

        input_ids = self._tokenizer(
            description, return_tensors="pt",
        ).input_ids.to(self._device)
        prompt_input_ids = self._tokenizer(
            text, return_tensors="pt",
        ).input_ids.to(self._device)

        streamer = ParlerTTSStreamer(
            self._model,
            device=self._device,
            play_steps=int(self._sample_rate * 0.5),
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            streamer=streamer,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            audio = chunk.cpu().numpy().squeeze().astype(np.float32)
            if audio.ndim > 0 and len(audio) > 0:
                yield AudioChunk(audio=audio, sample_rate=self._sample_rate)

        thread.join()

    def _build_description(self, voice: str | None) -> str:
        """Turn a voice name or description into a Parler description prompt."""
        if voice is None:
            return "A clear, neutral voice with moderate pace."
        if any(voice.lower() == v.lower() for v in PARLER_VOICES):
            return f"{voice}'s voice is clear and expressive with a moderate pace."
        return voice
