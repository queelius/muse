"""decision/evaluation modality (in progress; see docs/superpowers/plans/2026-10-18-decision-api.md).

Exports so far:
  - PROBE_DEFAULTS: the three-type probe run by `muse models probe` and
    after `muse pull`.

MODALITY and build_router (POST /v1/systemone) land with the router.
"""
from muse.modalities.decision_evaluation.probe import PROBE_SHAPE, probe_backend


# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": PROBE_SHAPE,
    "call": probe_backend,
}


__all__ = ["PROBE_DEFAULTS"]
