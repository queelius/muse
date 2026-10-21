"""facebook/bart-large-cnn (BART large fine-tuned on CNN/DailyMail).

Curated default for `text/summarization`. ~400MB on disk; CPU-friendly.
1024-token max_input_tokens (BART's hard limit). News summarization.

License: Apache 2.0.

The Model class is the generic `BartSeq2SeqRuntime` (the same runtime
resolver-pulled summarizers use); `load_backend` splats the
capabilities below (default_length, default_format, max_input_tokens)
into its constructor. BART-CNN is not instruction-tuned, so `format` is
recorded in the result metadata but does not affect generation.
"""
from muse.modalities.text_summarization.runtimes.bart_seq2seq import (
    BartSeq2SeqRuntime as Model,
)


MANIFEST = {
    "model_id": "bart-large-cnn",
    "modality": "text/summarization",
    "hf_repo": "facebook/bart-large-cnn",
    "revision": "37f520fa929c961707657b28798b30c003dd100b",
    "description": (
        "BART large CNN: news summarization, ~400MB, CPU-friendly, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU-friendly is the muse story for this model. Honest annotation.
        "device": "auto",
        "default_length": "medium",
        "default_format": "paragraph",
        "supports_dialog_summarization": False,
        # Measured peak inference, BART-large at fp32 + 1024 ctx.
        "memory_gb": 1.5,
        "max_input_tokens": 1024,
    },
    # BART tokenizer requires merges.txt + vocab.json (BPE pair); without
    # both, the tokenizer load fails with a non-obvious error. Listed
    # explicitly so the snapshot_download allow_patterns can't drop them.
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "merges.txt", "vocab.json", "tokenizer*",
    ],
}
