"""audio/generation modality (music + SFX text-to-audio).

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounted by the worker)
  - AudioGenerationModel Protocol
  - AudioGenerationResult dataclass
  - MusicClient + SFXClient (HTTP, added in Task D)
  - PROBE_DEFAULTS

Wire contract:
  - POST /v1/audio/music   -> WAV/MP3/Opus/FLAC bytes (capability-gated)
  - POST /v1/audio/sfx     -> same body shape, sfx-capability gate

This is muse's first modality with two routes mounted on one MIME
tag. The two URLs disambiguate user intent for the model (music vs.
sound effects); per-model capability flags `supports_music` and
`supports_sfx` decide which routes a given model serves.
"""
from muse.modalities.audio_generation.client import MusicClient, SFXClient
from muse.modalities.audio_generation.protocol import (
    AudioGenerationModel,
    AudioGenerationResult,
)
from muse.modalities.audio_generation.routes import build_router


MODALITY = "audio/generation"


# Per-modality probe defaults read by `muse models probe`. Five seconds
# keeps probe wall time low (under ~30s on a 12GB GPU for Stable Audio
# Open) while still measuring real peak memory.
PROBE_DEFAULTS = {
    "shape": "5s music",
    "call": lambda m: m.generate("ambient piano", duration=5.0),
}


__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "AudioGenerationModel",
    "AudioGenerationResult",
    "MusicClient",
    "SFXClient",
]
