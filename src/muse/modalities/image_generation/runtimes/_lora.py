"""LoRA base-pipeline resolution shared by the image_generation runtimes.

Import-light on purpose: no torch, diffusers or catalog import at module
load (the catalog is reached lazily through `resolve_model_source`).
"""
from __future__ import annotations

from typing import Callable


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
