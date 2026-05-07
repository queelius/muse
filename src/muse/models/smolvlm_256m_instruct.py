"""SmolVLM-256M-Instruct: tiny CPU-runnable VLM, ~500MB.

Default vision-language model bundled with muse. Image captioning,
basic VQA, multi-image inference. Apache 2.0 license. Runs on CPU
in ~5-15 seconds per inference.

Architecture: Idefics3 (HuggingFaceTB), the same family as the larger
SmolVLM-Instruct (2.2B) and SmolVLM2 video model.
"""
from muse.modalities.chat_completion.runtimes.transformers_vlm import (
    HFVisionLanguageModel as Model,
)


MANIFEST = {
    "model_id": "smolvlm-256m-instruct",
    "modality": "chat/completion",
    "hf_repo": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "description": (
        "SmolVLM-256M-Instruct: tiny CPU-runnable VLM, ~500MB. "
        "Image captioning, basic VQA, multi-image."
    ),
    "license": "apache-2.0",
    "pip_extras": [
        "torch>=2.1.0",
        "transformers>=4.46.0",
        "accelerate",
        "Pillow",
    ],
    "capabilities": {
        "memory_gb": 1.0,
        "device": "cpu",
        "supports_vision": True,
        "supports_multi_image": True,
        "supports_tools": False,
        "chat_format": None,
    },
}
