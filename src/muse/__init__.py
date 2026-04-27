"""Muse: model-agnostic multi-modality generation server.

The authoritative list of supported modalities lives in
`muse.core.discovery.discover_modalities()`, which scans
`src/muse/modalities/` plus any user-configured dirs. As of v0.16.1
the bundled modalities are:

  - audio/speech: /v1/audio/speech (TTS: Soprano, Kokoro, Bark)
  - audio/transcription: /v1/audio/transcriptions, /v1/audio/translations (faster-whisper)
  - chat/completion: /v1/chat/completions (llama-cpp-python over GGUF)
  - embedding/text: /v1/embeddings (sentence-transformers)
  - image/generation: /v1/images/generations (diffusers)
  - text/classification: /v1/moderations (HF text-classification)

Heavy backends (transformers, diffusers, faster-whisper, llama-cpp,
sentence-transformers) are imported lazily inside per-modality runtime
modules to keep `muse --help` and `muse pull` instant. Each pulled
model lives in its own venv at `~/.muse/venvs/<model-id>/`.

`__version__` is read from pyproject.toml at install time; this
fallback covers in-tree imports without an installed muse.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("muse")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
