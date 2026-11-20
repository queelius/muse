"""Audio speech modality: text-to-speech.

Wire contract: POST /v1/audio/speech with {input, model, voice?, speed?,
response_format? ('wav' | 'opus'), stream?} returns WAV/Opus audio bytes
(or SSE-streamed base64 PCM chunks when stream=True).

Models declaring `modality = "audio/speech"` in their MANIFEST and
satisfying the TTSModel protocol plug into this modality.
"""
from muse.modalities.audio_speech.client import SpeechClient
from muse.modalities.audio_speech.protocol import (
    AudioChunk,
    AudioResult,
    TTSModel,
    TimestampsUnavailableError,
    UnknownVoiceError,
    VoiceError,
    VoiceUnreadableError,
)
from muse.modalities.audio_speech.routes import build_router

MODALITY = "audio/speech"

DISCOVERY = {
    "summary": (
        "Text-to-speech. Turns a script into a spoken narration track in a "
        "chosen voice, as WAV or Opus audio (or streamed PCM chunks). Set "
        "timestamps: true to return JSON with base64 audio and one timing per "
        "written word (Kokoro only). Models without native timings refuse; "
        "use audio/alignment for those models. Send pronunciations to keep "
        "input as the written text while applying respellings or supported IPA. "
        "Set align=true for Soprano's paragraph timings in the X-Alignment header. "
        "Designed voices: send instructions (a voice description) to a model with "
        "supports_instructions (qwen3-tts-1.7b-voicedesign). Saved voices: upload a "
        "reference clip plus its transcript under a name with POST "
        "/v1/audio/speech/voices on a model with supports_saved_voices "
        "(qwen3-tts-0.6b-base), then speak with voice=<name>; the same name keeps "
        "the same voice on every line. Delete with DELETE "
        "/v1/audio/speech/voices/{name}?model=."
    ),
    "use_for": [
        "narration and voiceover for a video from a written script",
        "spoken audio for a scene, character line or announcement",
        "listing the voices a speech model offers before choosing one",
        "inventing a character voice from a text description",
        "giving each game character one consistent voice across many lines: "
        "design a take, save it as a named voice, then clone it for every line",
    ],
    "not_for": [
        "music or sound effects (use audio/generation)",
        "timing a transcript you already have against existing audio (use audio/alignment)",
        "speech recognition (use audio/transcription)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/audio/speech",
            "request_content_type": "application/json",
            "response_content_type": "audio/wav",
            "compat": "openai:audio.speech",
            "mcp_tool": "muse_speak",
            "file_fields": [],
            "media_fields": ["audio.b64_json"],
            "example": {"model": "<model>", "input": "Hello from muse.", "voice": "<voice>"},
        },
        {
            "method": "GET",
            "path": "/v1/audio/speech/voices",
            "request_content_type": None,
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_list_voices",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>"},
        },
        {
            "method": "POST",
            "path": "/v1/audio/speech/voices",
            "request_content_type": "multipart/form-data",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_save_voice",
            "file_fields": ["audio"],
            "media_fields": [],
            "example": {
                "model": "<model>", "name": "grimbold",
                "transcript": "Welcome, traveler.",
            },
        },
        {
            "method": "DELETE",
            "path": "/v1/audio/speech/voices/{name}",
            "request_content_type": None,
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": None,
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>"},
        },
    ],
}

# Per-modality probe defaults read by `muse models probe`. Declares a
# representative inference shape and a callable that exercises the model
# at that shape so peak VRAM/RAM is measurable. See cli_impl/probe_worker.
PROBE_DEFAULTS = {
    "shape": "5s synthesis",
    "call": lambda m: m.synthesize("Hello, this is a probe."),
}

__all__ = [
    "MODALITY",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "SpeechClient",
    "AudioChunk",
    "AudioResult",
    "TTSModel",
    "TimestampsUnavailableError",
]
