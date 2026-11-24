"""audio-tools: deterministic audio mastering tools for ``audio/process`` (no weights).

Serves ``POST /v1/audio/process`` (silence trim + micro-fades, DC removal,
equal-power crossfade looping, peak normalization with PCM16-safe ceiling,
mono downmix, and resampling). Pure numpy + scipy on CPU: ``muse pull
audio-tools`` installs the venv only and records a code receipt instead of
downloading model weights.

The Model class is the generic ``AudioTools`` runtime.
"""
from muse.modalities.audio_process.runtime import AudioTools as Model


MANIFEST = {
    "model_id": "audio-tools",
    "modality": "audio/process",
    "weights": "none",
    "description": (
        "Deterministic audio mastering tools: silence trimming, micro-fades, "
        "DC removal, equal-power crossfade looping, peak normalization, mono "
        "downmix, and resampling. CPU, no weights, MIT."
    ),
    "summary": "Trims, normalizes, resamples, and creates seamless equal-power loops for game-ready audio",
    "strengths": [
        "Silence trimming with configurable lead/tail padding and click-free micro-fades",
        "Seamless equal-power sin/cos crossfade looping for ambient beds and music loops",
        "Peak normalization with hard PCM16-safe ceiling, DC removal, mono downmix, and polyphase resampling; CPU, no weights",
    ],
    "limits": [
        "Processes existing audio clips; it does not synthesize new audio (pair with audio/generation or audio/speech)",
        "Crossfade looping folds tail into head, reducing output duration by loop_fade_s",
        "WAV works with stdlib only; FLAC requires soundfile and Opus requires ffmpeg",
    ],
    "quality_tier": "good",
    "license": "MIT",
    "pip_extras": (
        "numpy",
        "scipy",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cpu",
        "memory_gb": 0.5,
        "supports_trim": True,
        "supports_loop": True,
        "supports_normalize": True,
        "supports_resample": True,
    },
}
