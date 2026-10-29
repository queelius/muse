"""Public versus internal capability keys.

A manifest's ``capabilities`` dict mixes feature flags and limits an agent can
act on with runtime tuning knobs. The capabilities document shows only the
public ones. Hidden wins over public (``max_concurrency`` is a knob despite
the ``max_`` prefix); a key in neither list is unclassified and dropped, and a
meta-test fails until it is classified.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

logger = logging.getLogger(__name__)

PUBLIC_PREFIXES: tuple[str, ...] = ("supports_", "max_", "min_", "default_")

PUBLIC_KEYS: frozenset[str] = frozenset({
    "size_multiple", "quality_presets",
    "voices", "sample_rate", "dimensions", "supported_languages",
    "source_language", "target_language", "context_length", "device",
    "memory_gb", "quality_axes", "primary_score", "question_types",
    "input_modalities", "image_size", "metric_depth", "word_timestamps",
    "calibrated", "lora_adapter", "base_model", "supported_scales", "supported_tile_sizes",
    "safe_labels", "languages", "multi_label", "num_labels", "output_format",
    "output_mime_type", "static_svg_only", "voice_cloning", "pronunciation_kinds",
    "prompt_prefix",
})

HIDDEN_KEYS: frozenset[str] = frozenset({
    "pipeline_class",
    "gguf_file", "chat_format", "n_gpu_layers", "n_threads",
    "n_threads_batch", "n_batch", "flash_attn", "use_mlock", "type_k",
    "type_v", "allow_patterns", "torch_compile", "cpu_offload", "vae_tiling",
    "overlapped_decode", "trust_remote_code", "precision",
    "option_cache_size", "idle_timeout_seconds", "max_concurrency",
    "lora_scale", "image_processor_overrides", "window_seconds", "num_beams",
    "temperature", "probabilities", "calibration", "confidence",
    "revision", "adapter_model_subdir", "base_model_revision", "base_model_subdir", "dtype",
})


def classify(key: str) -> Literal["public", "hidden"] | None:
    if key in HIDDEN_KEYS:
        return "hidden"
    if key in PUBLIC_KEYS or key.startswith(PUBLIC_PREFIXES):
        return "public"
    return None


def public_capabilities(caps: Mapping[str, Any]) -> dict:
    out: dict = {}
    for key, value in caps.items():
        kind = classify(key)
        if kind == "public":
            out[key] = value
        elif kind is None:
            logger.debug("unclassified capability key dropped: %s", key)
    return out
