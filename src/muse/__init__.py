"""Muse: model-agnostic multi-modality generation server.

The authoritative list of supported modalities lives in
`muse.core.discovery.discover_modalities()`, which scans
`src/muse/modalities/` plus any user-configured dirs. As of v0.24.0
the bundled modalities are:

  - audio/embedding: /v1/audio/embeddings (transformers AutoModel + librosa; MERT, CLAP, wav2vec; multipart upload, OpenAI-shape envelope)
  - audio/generation: /v1/audio/music, /v1/audio/sfx (Stable Audio Open 1.0; capability-gated)
  - audio/speech: /v1/audio/speech (TTS: Soprano, Kokoro, Bark)
  - audio/transcription: /v1/audio/transcriptions, /v1/audio/translations (faster-whisper)
  - chat/completion: /v1/chat/completions (llama-cpp-python over GGUF)
  - embedding/text: /v1/embeddings (sentence-transformers)
  - image/animation: /v1/images/animations (AnimateDiff)
  - image/embedding: /v1/images/embeddings (transformers AutoModel; CLIP, SigLIP, DINOv2)
  - image/generation: /v1/images/generations, /v1/images/edits (inpaint), /v1/images/variations (diffusers)
  - text/classification: /v1/moderations (HF text-classification)
  - text/rerank: /v1/rerank (sentence-transformers CrossEncoder; Cohere-compat)
  - text/summarization: /v1/summarize (transformers AutoModelForSeq2SeqLM; Cohere-compat)

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
