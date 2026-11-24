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
from muse.modalities.audio_speech.light import build_light_router, model_options
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
        "the same voice on every line. Saved voices form a catalog: each has a "
        "category path (narrator/epic, npc/human/guard), tags, a description, "
        "notes and the design instructions it came from. Browse with GET "
        "/v1/audio/speech/voices?model=&category=&tag=&q=, inspect one with GET "
        "/v1/audio/speech/voices/{name}, audition its reference clip with GET "
        "/v1/audio/speech/voices/{name}/audio, edit metadata with PATCH "
        "/v1/audio/speech/voices/{name}, and delete with DELETE "
        "/v1/audio/speech/voices/{name}?model=. None of these load a model."
    ),
    "use_for": [
        "narration and voiceover for a video from a written script",
        "spoken audio for a scene, character line or announcement",
        "listing the voices a speech model offers before choosing one",
        "inventing a character voice from a text description",
        "giving each game character one consistent voice across many lines: "
        "design a take on qwen3-tts-1.7b-voicedesign with instructions, listen to "
        "it, save it with POST /v1/audio/speech/voices including category, "
        "description and design_instructions, then speak with voice=<name> on "
        "qwen3-tts-0.6b-base",
        "finding an existing voice for a role by category, tag or text search, "
        "and auditioning it before using it",
        "organizing voices into categories and annotating them for other agents",
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
            "example": {"model": "<model>", "category": "narrator"},
            "light": True,
        },
        {
            "method": "GET",
            "path": "/v1/audio/speech/voices/{name}",
            "request_content_type": None,
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_get_voice",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>", "name": "narrator"},
            "light": True,
        },
        {
            "method": "GET",
            "path": "/v1/audio/speech/voices/{name}/audio",
            "request_content_type": None,
            "response_content_type": "audio/wav",
            "compat": None,
            "mcp_tool": "muse_get_voice_audio",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>", "name": "narrator"},
            "light": True,
        },
        {
            "method": "PATCH",
            "path": "/v1/audio/speech/voices/{name}",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_update_voice",
            "file_fields": [],
            "media_fields": [],
            "example": {
                "model": "<model>", "name": "grimbold",
                "category": "npc/dwarf/blacksmith",
                "tags": ["male", "gravelly"],
                "notes": "Ironhold's blacksmith in the main quest; keep lines short.",
            },
            "light": True,
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
                "category": "npc/dwarf/blacksmith",
                "tags": "male,elderly,gravelly",
                "description": "A gruff, elderly dwarf blacksmith with a gravelly voice.",
                "design_model": "qwen3-tts-1.7b-voicedesign",
                "design_instructions": (
                    "A gruff, elderly dwarf blacksmith, gravelly and slow, "
                    "with a slight Scottish lilt."),
            },
        },
        {
            "method": "DELETE",
            "path": "/v1/audio/speech/voices/{name}",
            "request_content_type": None,
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_delete_voice",
            "file_fields": [],
            "media_fields": [],
            "example": {"model": "<model>", "name": "grimbold"},
            "light": True,
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
    "build_light_router",
    "build_router",
    "model_options",
    "SpeechClient",
    "AudioChunk",
    "AudioResult",
    "TTSModel",
    "TimestampsUnavailableError",
]
