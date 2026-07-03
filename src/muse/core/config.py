"""Central settings registry for muse.

This module is the single source of truth for every environment-variable
knob muse reads: what its dotted config key is, what env var backs it,
what type it coerces to, what its default is, which group it belongs to,
and a human-readable help string. Later tasks build a layered Config
object (env > config.yaml > defaults) on top of this registry; this task
only establishes the registry, the coercion function, and the two
bootstrap path helpers that the config file itself needs before it can
be loaded.

Import-light by design: stdlib + pathlib only. No yaml, no torch, no
fastapi, no transformers. `muse --help` and `muse pull` must work
without any ML deps installed, and this module is imported early enough
that it must not pull in anything heavy.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("muse.config")

_MB = 1024 * 1024


class ConfigError(ValueError):
    """A config value could not be coerced to its declared type."""


@dataclass(frozen=True)
class Setting:
    key: str          # dotted "group.leaf"
    env: str          # "MUSE_*"
    type: str         # int|float|str|bool|opt_int|opt_float|opt_str
    default: Any
    group: str
    help: str


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def coerce(setting: Setting, raw: str) -> Any:
    """Parse a raw string per setting.type. Raises ConfigError on failure.
    Callers choose lenient (Config.get) vs strict (config set)."""
    t = setting.type
    if t.startswith("opt_"):
        if raw is None or raw.strip() == "":
            return None
        t = t[len("opt_"):]
    try:
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            low = raw.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ValueError(f"not a boolean: {raw!r}")
        if t == "str":
            return raw
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"{setting.env} / {setting.key} must be {setting.type}, got {raw!r}: {e}"
        ) from e
    raise ConfigError(f"unknown type {setting.type!r} for {setting.key}")


SETTINGS: list[Setting] = [
    # --- server ---
    Setting("server.idle_sweep_interval_seconds", "MUSE_IDLE_SWEEP_INTERVAL_SECONDS",
            "float", 30.0, "server", "Seconds between idle-eviction sweeps."),
    Setting("server.idle_timeout_seconds", "MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS",
            "opt_float", 600.0, "server",
            "Global default idle timeout (s) before an untouched model is evicted; 0/negative disables."),
    Setting("server.shutdown_grace_seconds", "MUSE_SHUTDOWN_GRACE_SECONDS",
            "opt_float", None, "server",
            "Grace period (s) for workers to exit on shutdown; None uses the built-in default."),
    Setting("server.gpu_budget_gb", "MUSE_GPU_BUDGET_GB",
            "opt_float", None, "server",
            "Declared GPU memory cap (GB); muse uses min(declared, live)."),
    Setting("server.cpu_budget_gb", "MUSE_CPU_BUDGET_GB",
            "opt_float", None, "server", "Declared host-RAM cap (GB)."),
    Setting("server.gpu_headroom_gb", "MUSE_GPU_HEADROOM_GB",
            "float", 1.0, "server",
            "GB subtracted from live free VRAM before deciding fit."),
    Setting("server.cpu_headroom_gb", "MUSE_CPU_HEADROOM_GB",
            "float", 2.0, "server",
            "GB subtracted from live free RAM before deciding fit."),
    Setting("server.device", "MUSE_DEVICE",
            "str", "auto", "server",
            "Default device for models (auto|cpu|cuda|mps); `muse serve --device` overrides."),
    # --- admin ---
    Setting("admin.token", "MUSE_ADMIN_TOKEN",
            "opt_str", None, "admin",
            "Bearer token that unlocks /v1/admin/*; unset keeps admin closed."),
    # --- client ---
    Setting("client.server_url", "MUSE_SERVER",
            "str", "http://localhost:8000", "client",
            "Base URL muse clients + CLI target."),
    # --- paths (bootstrap: catalog_dir/config_file resolve env+default only) ---
    Setting("paths.catalog_dir", "MUSE_CATALOG_DIR",
            "str", "~/.muse", "paths", "Directory for catalog.json, venvs, config.yaml."),
    Setting("paths.home", "MUSE_HOME",
            "str", "~/.muse", "paths", "Base dir for bundled voices/assets."),
    Setting("paths.models_dir", "MUSE_MODELS_DIR",
            "opt_str", None, "paths", "Extra directory scanned for model scripts."),
    Setting("paths.modalities_dir", "MUSE_MODALITIES_DIR",
            "opt_str", None, "paths", "Extra directory scanned for modality packages."),
    Setting("paths.config_file", "MUSE_CONFIG",
            "opt_str", None, "paths",
            "Explicit config.yaml path; overrides <catalog_dir>/config.yaml."),
    # --- fetch ---
    Setting("fetch.allow_private", "MUSE_ALLOW_PRIVATE_FETCH",
            "bool", False, "fetch",
            "Allow image/URL fetches to non-public IPs (SSRF guard off)."),
    Setting("fetch.mcp_allowed_path_prefixes", "MUSE_MCP_ALLOWED_PATH_PREFIXES",
            "str", "", "fetch",
            "Colon-separated dir prefixes MCP *_path inputs may read from."),
    # --- limits (per-modality request caps) ---
    Setting("limits.image_input_max_bytes", "MUSE_IMAGE_INPUT_MAX_BYTES",
            "opt_int", 10 * _MB, "limits", "Max bytes per image upload / data URL."),
    Setting("limits.audio_cls_max_bytes", "MUSE_AUDIO_CLS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-classification upload."),
    Setting("limits.audio_embeddings_max_bytes", "MUSE_AUDIO_EMBEDDINGS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-embedding upload."),
    Setting("limits.asr_max_mb", "MUSE_ASR_MAX_MB",
            "int", 100, "limits", "Max MB per transcription/translation upload."),
    Setting("limits.embeddings_max_batch", "MUSE_EMBEDDINGS_MAX_BATCH",
            "int", 2048, "limits", "Max inputs per /v1/embeddings request."),
    Setting("limits.embeddings_max_chars_per_item", "MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per embedding input."),
    Setting("limits.image_embeddings_max_batch", "MUSE_IMAGE_EMBEDDINGS_MAX_BATCH",
            "int", 64, "limits", "Max inputs per /v1/images/embeddings request."),
    Setting("limits.segmentation_max_input_side", "MUSE_SEGMENTATION_MAX_INPUT_SIDE",
            "int", 2048, "limits", "Max px on the long side of a segmentation input."),
    Setting("limits.upscale_max_input_side", "MUSE_UPSCALE_MAX_INPUT_SIDE",
            "int", 1024, "limits", "Max px on the long side of an upscale input."),
    Setting("limits.model_3d_input_max_bytes", "MUSE_3D_INPUT_MAX_BYTES",
            "opt_int", 20 * _MB, "limits", "Max bytes per image-to-3D upload."),
    Setting("limits.moderations_max_batch", "MUSE_MODERATIONS_MAX_BATCH",
            "int", 1024, "limits", "Max inputs per /v1/moderations request."),
    Setting("limits.moderations_max_chars_per_item", "MUSE_MODERATIONS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per moderation input."),
    Setting("limits.classifications_max_labels", "MUSE_CLASSIFICATIONS_MAX_LABELS",
            "int", 200, "limits", "Max candidate labels per zero-shot classification."),
    Setting("limits.rerank_max_documents", "MUSE_RERANK_MAX_DOCUMENTS",
            "int", 1000, "limits", "Max documents per /v1/rerank request."),
    Setting("limits.rerank_max_query_chars", "MUSE_RERANK_MAX_QUERY_CHARS",
            "int", 4000, "limits", "Max chars in a rerank query."),
    Setting("limits.rerank_max_doc_chars", "MUSE_RERANK_MAX_DOC_CHARS",
            "int", 100000, "limits", "Max chars per rerank document."),
    Setting("limits.summarize_max_text_chars", "MUSE_SUMMARIZE_MAX_TEXT_CHARS",
            "int", 100000, "limits", "Max chars per /v1/summarize request."),
    Setting("limits.video_max_frames_b64", "MUSE_VIDEO_MAX_FRAMES_B64",
            "int", 240, "limits", "Max frames returned as base64 from /v1/video/generations."),
]

SETTINGS_BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}
SETTINGS_BY_ENV: dict[str, Setting] = {s.env: s for s in SETTINGS}


def _catalog_dir() -> pathlib.Path:
    """Resolve the catalog/config directory from env+default only.
    Standalone by design: must NOT import catalog.py (import cycle)."""
    raw = os.environ.get("MUSE_CATALOG_DIR")
    base = raw if raw else "~/.muse"
    return pathlib.Path(base).expanduser()


def config_path() -> pathlib.Path:
    """Resolve the config.yaml path from env+default only (bootstrap)."""
    raw = os.environ.get("MUSE_CONFIG")
    if raw:
        return pathlib.Path(raw).expanduser()
    return _catalog_dir() / "config.yaml"
