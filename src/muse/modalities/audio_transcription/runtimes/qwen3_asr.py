"""Transcript-only Qwen3-ASR runtime, using native Transformers >=5.13.0.

The pinned model card documents AutoProcessor + AutoModelForMultimodalLM
(or Qwen3ASRForConditionalGeneration), apply_transcription_request(audio=...,
 language=..., prompt=...), and the equivalent explicit chat-template path.
Audio can be a path or waveform; language accepts names or ISO codes. Context
is a system message, included only for a nonempty caller prompt. We use the
explicit template because the 5.13.0 convenience helper does not yet consume
prompt. A language hint prefills the assistant with language NAME<asr_text>.
No Hub code or external model package is executed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from muse.core import config
from muse.core.runtime_helpers import LoadTimer, select_device, set_inference_mode
from muse.modalities.audio_alignment.decoding import decode_audio
from muse.modalities.audio_transcription.protocol import InvalidLanguageError, Segment, TranscriptionResult


logger = logging.getLogger(__name__)
torch: Any = None
AutoProcessor: Any = None
AutoModelForMultimodalLM: Any = None

# The checkpoint's 30 transcription languages (card names / ISO codes).
_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}


def _ensure_deps() -> None:
    global torch, AutoProcessor, AutoModelForMultimodalLM
    if torch is None:
        import torch as _torch
        torch = _torch
    if AutoProcessor is None or AutoModelForMultimodalLM is None:
        from transformers import AutoModelForMultimodalLM as _AutoModel, AutoProcessor as _AutoProcessor
        AutoProcessor = _AutoProcessor
        AutoModelForMultimodalLM = _AutoModel


class Qwen3ASRRuntime:
    """Transcribe audio without claiming measured segment or word times."""

    def __init__(
        self, *, model_id: str, hf_repo: str, local_dir: str | None = None,
        device: str = 'auto', **_: Any,
    ) -> None:
        _ensure_deps()
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = (
            torch.float32 if self._device == 'cpu'
            else torch.bfloat16 if self._device.startswith('cuda') else torch.float16
        )
        source = local_dir or hf_repo
        with LoadTimer(f'Qwen3 ASR from {source}', logger):
            self._processor = AutoProcessor.from_pretrained(source)
            self._model = AutoModelForMultimodalLM.from_pretrained(source, dtype=self._dtype)
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def transcribe(
        self, audio_path: str, *, task: Literal['transcribe', 'translate'] = 'transcribe',
        language: str | None = None, prompt: str | None = None,
        temperature: float = 0.0, word_timestamps: bool = False,
        vad_filter: bool = False, **_: Any,
    ) -> TranscriptionResult:
        if task != 'transcribe':
            raise ValueError('Qwen3-ASR does not support translation')
        if word_timestamps:
            raise ValueError('Qwen3-ASR has no word timestamps; use /v1/audio/alignments')
        max_bytes = config.get('limits.asr_max_mb') * 1024 * 1024
        if Path(audio_path).stat().st_size > max_bytes:
            raise ValueError('audio exceeds MUSE_ASR_MAX_MB')
        decoded = decode_audio(audio_path)
        language_code = None
        canonical_language = None
        if language and language.lower() != 'auto':
            for code, name in _LANGUAGES.items():
                if language.lower() in (code, name.lower()):
                    language_code, canonical_language = code, name
                    break
            if canonical_language is None:
                raise InvalidLanguageError(f'unsupported Qwen3-ASR language: {language!r}')
        messages = []
        if prompt:
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': prompt}]})
        messages.append({'role': 'user', 'content': [{
            'type': 'audio', 'audio': decoded.waveform.squeeze(0).numpy(),
        }]})
        prefill = f'language {canonical_language}<asr_text>' if canonical_language else ''
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': prefill}]})
        inputs = self._processor.apply_chat_template(
            [messages], tokenize=True, return_dict=True, continue_final_message=True,
        ).to(self._device, self._dtype)
        generation = {'max_new_tokens': 8192, 'do_sample': temperature > 0}
        if temperature > 0:
            generation['temperature'] = temperature
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        parsed = self._processor.decode(generated_ids, return_format='parsed')[0]
        text = parsed['transcription']
        detected = parsed.get('language')
        detected_code = next((code for code, name in _LANGUAGES.items()
                              if detected and name.lower() == detected.lower()), '')
        return TranscriptionResult(
            text=text, language=language_code or detected_code,
            duration=decoded.duration_seconds, task='transcribe',
            segments=[Segment(0, 0.0, decoded.duration_seconds, text, None)],
        )
