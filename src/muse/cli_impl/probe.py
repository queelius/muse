"""muse models probe: measure model VRAM/RAM by loading + (default) inference.

Spawns a subprocess in the model's per-model venv so the measurement is
clean (no test-fixture interference, no other-model confounding). The
subprocess runs an internal entry point (`muse _probe_worker`) that
imports the model, captures pre/post memory, optionally runs a
representative inference, captures peak, prints a JSON line on stdout
that this command parses.

Per-modality default inference shape is read from the modality's
PROBE_DEFAULTS dict (declared in each `muse.modalities.<name>.__init__`).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from muse.core.catalog import (
    _read_catalog,
    _reset_known_models_cache,
    _write_catalog,
    is_pulled,
    known_models,
)
from muse.core.venv import venv_python


log = logging.getLogger("muse.probe")


def run_probe(
    *,
    model_id: str,
    no_inference: bool,
    device: str | None,
    as_json: bool,
) -> int:
    """Spawn an in-venv probe; persist the resulting measurement.

    Returns 0 on success, non-zero on the various failure modes
    (unknown model / not pulled / subprocess crash / non-JSON output).
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        print(f"error: unknown model {model_id!r}", file=sys.stderr)
        return 2
    if not is_pulled(model_id):
        print(
            f"error: model {model_id!r} is not pulled; "
            f"run `muse pull {model_id}` first",
            file=sys.stderr,
        )
        return 2

    catalog = _read_catalog()
    venv_path = Path(catalog[model_id]["venv_path"])
    py = venv_python(venv_path)

    # Resolve the device the worker will pass to load_backend. Capability
    # device pin still takes precedence inside load_backend; this just
    # picks the request that gets forwarded.
    entry = catalog_known[model_id]
    cap_device = entry.extra.get("device", "auto")
    effective_device = device or cap_device or "auto"

    cmd = [
        str(py), "-m", "muse.cli", "_probe_worker",
        "--model", model_id,
        "--device", effective_device,
    ]
    if no_inference:
        cmd.append("--no-inference")

    if not as_json:
        print(f"Probing {model_id} on {effective_device}...")
        if no_inference:
            print("  (load only; --no-inference)")
        print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("error: probe timed out (>10 min)", file=sys.stderr)
        return 3

    if result.returncode != 0:
        print(f"error: probe subprocess exited {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode

    # The worker writes progress logs on stderr; the final stdout line
    # is a JSON object the parent persists. Parse the last non-empty line.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        print("error: probe produced no output", file=sys.stderr)
        return 4
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError:
        print("error: probe output was not JSON; full stdout:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        return 4

    # Persist under measurements.<device>. Per-device keying so the same
    # model can carry both a CPU and a GPU probe; `muse models list`
    # picks based on capabilities.device.
    catalog = _read_catalog()
    if model_id in catalog:
        catalog[model_id].setdefault("measurements", {})
        device_key = record.get("device", effective_device)
        catalog[model_id]["measurements"][device_key] = record
        _write_catalog(catalog)
        _reset_known_models_cache()

    if as_json:
        print(json.dumps(record, indent=2))
    else:
        _print_human(record)
    return 0


def _print_human(r: dict) -> None:
    """Format a measurement record for terminal display."""
    if "error" in r and not r.get("ran_inference"):
        # Load-failure record (worker exited 2 with an error JSON line).
        print(f"  device:   {r.get('device', '?')}")
        print(f"  error:    {r['error']}")
        print()
        print("Persisted to catalog.json (with error field)")
        return
    device = r.get("device", "?")
    weights_gb = r.get("weights_bytes", 0) / (1024**3)
    peak_gb = r.get("peak_bytes", 0) / (1024**3)
    load_seconds = r.get("load_seconds")
    print(f"  device:           {device}")
    print(f"  weights:          {weights_gb:.2f} GB")
    if load_seconds is not None:
        print(f"  load time:        {load_seconds:.1f}s")
    if r.get("ran_inference"):
        shape = r.get("shape", "?")
        print(f"  inference shape:  {shape}")
        delta_gb = max(0.0, peak_gb - weights_gb)
        print(f"  activation delta: {delta_gb:.2f} GB")
    if peak_gb > 0:
        print(f"  peak observed:    {peak_gb:.2f} GB")
    if r.get("inference_error"):
        print(f"  inference error:  {r['inference_error']}")
    print()
    print(f"Persisted to catalog.json under measurements.{device}")
