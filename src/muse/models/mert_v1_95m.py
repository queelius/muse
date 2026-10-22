"""m-a-p/MERT-v1-95M (MERT music understanding audio embedder).

Curated default for `audio/embedding`. ~95M parameters; CPU-friendly.
768-dim embeddings via mean-pool over the time dimension of the
last hidden state. Trained on music with masked acoustic modeling
(MERT v1).

License: CC-BY-NC-4.0 (non-commercial).

MERT ships custom model code in the repo, so loading requires
`trust_remote_code=True`; Muse pins and downloads that code with the weights.

The Model class is the generic `AudioEmbeddingRuntime` (transformers
AutoModel + feature extractor, librosa decoding). `load_backend` splats
the capabilities below (sample_rate, dimensions, trust_remote_code)
plus the pinned `revision` into its constructor; MERT's output carries
only `last_hidden_state`, so the runtime's pooling dispatch mean-pools
it over time.
"""
from muse.modalities.audio_embedding.runtimes.transformers_audio import (
    AudioEmbeddingRuntime as Model,
)


MANIFEST = {
    "model_id": "mert-v1-95m",
    "modality": "audio/embedding",
    "hf_repo": "m-a-p/MERT-v1-95M",
    # Reviewed 2026-08-02. Source: official Hugging Face model API.
    "revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
    "description": (
        "MERT v1 95M: music understanding, 768-dim audio embeddings, "
        "CC-BY-NC-4.0 (non-commercial)"
    ),
    "summary": "Music-understanding embeddings for similarity search, tagging and clustering of music clips",
    "strengths": [
        "768-dim music representations, mean-pooled over time",
        "Trained on music: reflects timbre, pitch and rhythm",
        "Small, ~0.5 GB, runs on CPU",
    ],
    "limits": [
        "CC-BY-NC-4.0: non-commercial use only",
        "No text tower, so no text-to-audio search (use clap-htsat-fused)",
        "Clips longer than 60 s are truncated",
    ],
    "quality_tier": "good",
    "license": "CC-BY-NC-4.0",
    "pip_extras": (
        # The pinned repo has only a pickle pytorch_model.bin checkpoint.
        # Current Transformers rejects torch.load below 2.6 because of
        # CVE-2025-32434, even when weights_only=True.
        "torch>=2.6.0",
        "transformers>=4.36.0",
        "librosa>=0.10.0",
        # numpy is pulled by librosa but the runtime imports it directly;
        # listing it keeps the manifest self-describing (#110).
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU-friendly default; "auto" lets the runtime pick GPU when available.
        "device": "auto",
        "dimensions": 768,
        # MERT v1 was trained at 24kHz; resample inputs to that rate.
        "sample_rate": 24000,
        "max_duration_seconds": 60.0,
        "supports_text_embeddings_too": False,
        # MERT ships custom model architecture/config code in the repo.
        "trust_remote_code": True,
        # Measured peak inference, MERT v1 95M at fp32, single 1s 24kHz clip.
        "memory_gb": 0.5,
    },
    # This immutable MERT revision ships pytorch_model.bin plus custom model
    # architecture/config .py files required by the trust_remote_code path.
    "allow_patterns": [
        "pytorch_model.bin", "*.json", "*.txt", "*.md",
        "*.py",
        "preprocessor_config.json",
    ],
}
