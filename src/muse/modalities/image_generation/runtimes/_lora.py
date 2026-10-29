"""LoRA helpers shared by the image_generation runtimes.

Base-pipeline resolution (`resolve_lora_base`) and the adapter-weight scale
factor (`lora_scale_factor`), derived from the adapter's safetensors header
when the manifest does not set one.

Import-light on purpose: no torch, diffusers, safetensors or catalog import
at module load (the catalog is reached lazily through
`resolve_model_source`); the header is parsed with the stdlib.
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


def resolve_lora_base(
    model_id: str,
    base_model: str | None,
    *,
    resolve: Callable[[str], str] | None = None,
) -> str:
    """Return the from_pretrained source for a LoRA entry's base pipeline.

    A muse catalog id maps to that entry's snapshot dir; an HF repo id
    (contains "/") passes through for from_pretrained to fetch. `resolve`
    defaults to `muse.core.runtime_helpers.resolve_model_source`; runtimes
    pass their own module-level name so tests that patch it keep working.

    Raises RuntimeError with the fix when base_model is missing, or when a
    muse id did not resolve to a local dir (removed after pull, or never
    pulled), instead of letting from_pretrained fail with a cryptic HF
    RepositoryNotFoundError for a bare id.
    """
    if not base_model:
        raise RuntimeError(
            f"{model_id}: lora_adapter is set but base_model is "
            f"missing; re-pull with `muse pull <id> --base "
            f"<muse-id-or-hf-repo>`"
        )
    if resolve is None:
        from muse.core.runtime_helpers import resolve_model_source as resolve
    load_src = resolve(base_model)
    if "/" not in base_model and load_src == base_model:
        raise RuntimeError(
            f"{model_id}: LoRA base {base_model!r} is not pulled "
            f"(it may have been removed); run `muse pull "
            f"{base_model}` and retry"
        )
    return load_src


# ---------------------------------------------------------------------------
# Adapter scale factor
#
# The runtimes hand diffusers `lora_scale * factor` as the adapter weight.
# diffusers reads `__metadata__["lora_adapter_metadata"]` (a JSON string whose
# keys are prefixed by the component, e.g. "transformer.r") and builds the
# peft LoraConfig from the keys under each component's prefix
# (loaders/peft.py load_lora_adapter, utils/peft_utils.py _create_lora_config;
# diffusers 0.38 and 0.41). peft then scales each layer by lora_alpha/sqrt(r)
# when use_rslora is set, else lora_alpha/r (tuners/lora/layer.py). ComfyUI
# uses a scale of 1 for a diffusers-format file without `.alpha` tensors, and
# LoRA cards quote ComfyUI strengths, so for an rsLoRA adapter without
# `.alpha` tensors the factor sqrt(r)/alpha makes lora_scale 1.0 mean what the
# card means by 1.0. Anything else keeps factor 1.0.
# ---------------------------------------------------------------------------

LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"
# diffusers' own adapter file name, which load_lora_weights prefers.
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"
# The safetensors format caps the JSON header at 100 MB.
_MAX_HEADER_BYTES = 100 * 1024 * 1024
# Files diffusers' weight-name guess skips (lora_base._best_guess_weight_name).
_UNALLOWED_SUBSTRINGS = ("scheduler", "optimizer", "checkpoint")
_CONFIG_FIELDS = ("r", "lora_alpha", "use_rslora", "rank_pattern", "alpha_pattern")


def positive_factor(value: Any) -> float:
    """lora_scale_factor as a finite float > 0, else ValueError."""
    try:
        factor = math.nan if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        factor = math.nan
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(
            f"lora_scale_factor must be a finite number > 0, got {value!r}"
        )
    return factor


def read_safetensors_header(path: str | os.PathLike) -> dict:
    """The JSON header of a safetensors file, without reading tensor bytes.

    Layout: an 8-byte little-endian header length, then that many bytes of
    UTF-8 JSON (tensor entries plus an optional `__metadata__` str->str map).
    Raises ValueError when the file does not have that shape.
    """
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: shorter than the safetensors header length")
        (n,) = struct.unpack("<Q", prefix)
        size = os.fstat(f.fileno()).st_size
        if n > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: safetensors header of {n} bytes exceeds 100 MB")
        if n > size - 8:
            raise ValueError(f"{path}: safetensors header length {n} runs past the file end")
        raw = f.read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"{path}: safetensors header is not UTF-8 JSON: {e}") from e
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not a JSON object")
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{path}: safetensors __metadata__ is not an object")
    return header


def adapter_weight_file(src: str | os.PathLike | None) -> Path | None:
    """The local adapter file `load_lora_weights(src)` would load, or None.

    Mirrors diffusers' guess for a directory: top-level `.safetensors`
    files, minus scheduler/optimizer/checkpoint ones, preferring
    `pytorch_lora_weights.safetensors`. None when src is not a local path,
    holds no candidate, or holds several (diffusers would pick one by
    directory order; nothing is derived from a guess).
    """
    if not src:
        return None
    p = Path(src)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    names = sorted(
        n for n in os.listdir(p)
        if n.endswith(".safetensors")
        and not any(s in n for s in _UNALLOWED_SUBSTRINGS)
        and (p / n).is_file()
    )
    preferred = [n for n in names if n.endswith(LORA_WEIGHT_NAME_SAFE)]
    if preferred:
        names = preferred
    return p / names[0] if len(names) == 1 else None


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) and v > 0 else None


def rslora_scale_factor(header: dict) -> float:
    """The scale factor a safetensors header implies (see the block comment).

    sqrt(r)/lora_alpha when every component in `lora_adapter_metadata`
    declares use_rslora with the same r and lora_alpha, no per-module
    rank/alpha patterns, and the file has no `.alpha` tensors; 1.0 when
    there is no such metadata, rsLoRA is off, or `.alpha` tensors exist.
    Raises ValueError when the metadata is present but unusable or
    ambiguous (the caller logs it and uses 1.0).
    """
    if any(k.endswith(".alpha") for k in header if k != "__metadata__"):
        return 1.0
    raw = (header.get("__metadata__") or {}).get(LORA_ADAPTER_METADATA_KEY)
    if not raw:
        return 1.0
    try:
        meta = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{LORA_ADAPTER_METADATA_KEY} is not JSON: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"{LORA_ADAPTER_METADATA_KEY} is not a JSON object")
    # Only "<component>.<field>" keys reach peft; diffusers drops bare ones.
    components: dict[str, dict] = {}
    for key, value in meta.items():
        prefix, _, field = str(key).rpartition(".")
        if prefix and field in _CONFIG_FIELDS:
            components.setdefault(prefix, {})[field] = value
    rs = {c: cfg for c, cfg in components.items() if cfg.get("use_rslora") is True}
    if not rs:
        return 1.0
    factors = set()
    for comp, cfg in components.items():
        if cfg.get("use_rslora") is not True:
            raise ValueError(f"components disagree on use_rslora ({sorted(components)})")
        if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            raise ValueError(f"{comp} has per-module rank/alpha patterns")
        r = _finite_positive(cfg.get("r"))
        alpha = _finite_positive(cfg.get("lora_alpha"))
        if r is None or alpha is None:
            raise ValueError(
                f"{comp} r={cfg.get('r')!r} lora_alpha={cfg.get('lora_alpha')!r} "
                "are not positive numbers"
            )
        factors.add(math.sqrt(r) / alpha)
    if len(factors) != 1:
        raise ValueError(f"components imply different factors {sorted(factors)}")
    return factors.pop()


def derive_lora_scale_factor(src: str | os.PathLike | None, *, model_id: str) -> float:
    """The scale factor for the adapter at `src`, derived from its header.

    Never raises: a missing, ambiguous or corrupt adapter file gives 1.0
    (with a warning when a file was found but could not be used). Logs the
    factor derived from a file once at info.
    """
    try:
        path = adapter_weight_file(src)
    except OSError as e:
        logger.warning("%s: cannot list LoRA adapter %s (%s); lora_scale_factor 1.0",
                       model_id, src, e)
        return 1.0
    if path is None:
        logger.debug("%s: no single local adapter file at %s; lora_scale_factor 1.0",
                     model_id, src)
        return 1.0
    try:
        factor = rslora_scale_factor(read_safetensors_header(path))
    except (OSError, ValueError) as e:
        logger.warning(
            "%s: cannot derive lora_scale_factor from %s (%s); using 1.0. "
            "Set capabilities.lora_scale_factor to override",
            model_id, path.name, e,
        )
        return 1.0
    logger.info(
        "%s: derived lora_scale_factor %g from %s (%s)",
        model_id, factor, path.name,
        "rsLoRA metadata; lora_scale 1.0 = ComfyUI strength 1.0"
        if factor != 1.0 else "no rsLoRA correction",
    )
    return factor


def resolve_lora_scale_factor(
    model_id: str, src: str | os.PathLike | None, *, explicit: Any = None,
) -> float:
    """The adapter-weight multiplier: an explicit capability wins (validated,
    ValueError when not finite and > 0), else derived from the adapter."""
    if explicit is not None:
        return positive_factor(explicit)
    return derive_lora_scale_factor(src, model_id=model_id)
