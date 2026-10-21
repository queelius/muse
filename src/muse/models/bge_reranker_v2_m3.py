"""BAAI bge-reranker-v2-m3 (multilingual cross-encoder reranker).

Curated default for `text/rerank`. ~568MB on disk; works on CPU.
8192-token context window (the m3 lineage). Multilingual.

License: Apache 2.0.

The Model class is the generic `CrossEncoderRuntime` (over
`sentence_transformers.CrossEncoder`); `load_backend` splats the
capabilities below (including max_length=8192) into its constructor.
"""
from muse.modalities.text_rerank.runtimes.cross_encoder import (
    CrossEncoderRuntime as Model,
)


MANIFEST = {
    "model_id": "bge-reranker-v2-m3",
    "modality": "text/rerank",
    "hf_repo": "BAAI/bge-reranker-v2-m3",
    "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "description": (
        "BAAI bge-reranker-v2-m3: multilingual cross-encoder reranker, "
        "8192-token context, ~568MB"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "sentence-transformers>=2.2.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "max_length": 8192,
        # Measured peak fp16 inference, query + 32 documents @ 8192 ctx.
        "memory_gb": 1.2,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "tokenizer*", "spiece.model",
    ],
}
