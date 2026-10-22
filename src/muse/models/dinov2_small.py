"""facebook/dinov2-small (DINOv2 small self-supervised image embedder).

Curated default for `image/embedding`. ~88MB on disk; CPU-friendly.
384-dim embeddings via the CLS token of the last hidden state. No
text alignment (DINOv2 is purely self-supervised on image data).

License: Apache 2.0.

The Model class is the generic `ImageEmbeddingRuntime` (transformers
AutoModel + AutoProcessor). Its pooling dispatch takes DINOv2's
`pooler_output`, which transformers' Dinov2Model defines as the
layer-normed CLS token, i.e. `last_hidden_state[:, 0]`.
"""
from muse.modalities.image_embedding.runtimes.transformers_image import (
    ImageEmbeddingRuntime as Model,
)


MANIFEST = {
    "model_id": "dinov2-small",
    "modality": "image/embedding",
    "hf_repo": "facebook/dinov2-small",
    "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "description": (
        "DINOv2 small: 88MB, 384-dim self-supervised image features, Apache 2.0"
    ),
    "summary": "Image-to-vector features for near-duplicate detection, clustering and visual similarity",
    "strengths": [
        "384 dims, CPU-friendly, 88 MB",
        "Self-supervised features capture visual structure without captions",
        "Apache 2.0",
    ],
    "limits": [
        "Image only: no text tower, so no text-to-image search (use siglip2-base)",
        "Smallest of the family; dinov2-base and dinov2-large give finer features",
    ],
    "quality_tier": "draft",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        # Transformers 5 image processors use torchvision-backed operations.
        "torchvision>=0.16.0",
        "transformers>=4.36.0",
        "Pillow>=9.1.0",
        # numpy is pulled by transformers but the runtime imports it
        # directly for embedding postprocessing (#110).
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU-friendly default; "auto" lets the runtime pick GPU when available.
        "device": "auto",
        "dimensions": 384,
        "image_size": 224,
        "supports_text_embeddings_too": False,
        # Measured peak inference, DINOv2-small at fp32, single 224x224 image.
        "memory_gb": 0.4,
    },
    # DINOv2 ships a preprocessor_config.json + safetensors. The wide
    # *.json + *.txt patterns cover the full tokenizer/preprocessor set
    # without our having to enumerate every file.
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "preprocessor_config.json",
    ],
}
