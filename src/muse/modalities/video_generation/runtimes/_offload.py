"""Shared CPU-offload dispatch for diffusers video pipelines.

Both WanRuntime and CogVideoXRuntime (and the bundled wan2_1_t2v_1_3b
Model script, which duplicates WanRuntime's construction logic for the
one-off-script loading path) place their pipeline the same way: either
the whole-pipeline `.to(device)` move, or one of diffusers' two CPU
offload modes. This module is the single place that dispatch lives, so
all three call sites stay in lockstep.

Import-light by design: only `muse.core.config` (no torch, no
diffusers) at module top, matching the config module's own
import-light discipline. `place_pipeline` receives an already-loaded
pipeline object; it never imports torch/diffusers itself.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core import config

logger = logging.getLogger(__name__)

_OFF = {"", "off", "false", "none", "no", "0"}
_MODES = {"model", "sequential"}


def resolve_offload_mode(capability_mode: Any) -> str | None:
    """Effective offload mode: global config override > per-model capability > off.

    Returns "model", "sequential", or None (no offload). An unset
    global override (`server.video_cpu_offload`, env
    MUSE_VIDEO_CPU_OFFLOAD) falls through to `capability_mode`; a set
    override always wins, including forcing "off" over a model that
    declares sequential offload.
    """
    override = config.get("server.video_cpu_offload")
    raw = override if override is not None else capability_mode
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _OFF:
        return None
    if s in _MODES:
        return s
    logger.warning("unknown cpu_offload mode %r; ignoring (no offload)", raw)
    return None


def place_pipeline(
    pipe: Any,
    device: str,
    *,
    cpu_offload: Any = None,
    vae_tiling: bool = False,
) -> Any:
    """Place a diffusers pipeline on its device.

    Dispatches to CPU offload (model or sequential granularity) when
    requested and the device is not cpu, OR falls back to the plain
    `.to(device)` move -- the two are mutually exclusive, offload
    manages its own placement. On a cpu device the pipeline is left
    alone (matches the pre-offload `device != "cpu"` guard). VAE tiling
    / slicing is applied best-effort afterward when requested.
    """
    if device == "cpu":
        return pipe
    mode = resolve_offload_mode(cpu_offload)
    if mode == "model":
        logger.info("enabling model cpu offload (device=%s)", device)
        pipe.enable_model_cpu_offload(device=device)
    elif mode == "sequential":
        logger.info("enabling sequential cpu offload (device=%s)", device)
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)
    if vae_tiling:
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    return pipe
