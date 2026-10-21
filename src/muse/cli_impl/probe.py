"""muse models probe: measure model VRAM/RAM by loading + (default) inference.

Spawns an owned subprocess group in the model's per-model venv so the measurement is
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
    ModelInUseError,
    _CATALOG_WRITE_LOCK,
    _model_pull_lock,
    _model_resource_lease,
    _read_catalog,
    _reset_known_models_cache,
    _write_catalog,
    is_pulled,
    known_models,
)
from muse.core.owned_command import _OwnedProcessCleanupError, run_owned_command
from muse.core.venv import venv_python


log = logging.getLogger("muse.probe")


DEFAULT_PROBE_TIMEOUT_SECONDS = 600.0


def _probe_timeout(modality: str) -> float:
    """The modality's declared PROBE_TIMEOUT_SECONDS, never below the default."""
    from muse.core.discovery import probe_timeouts

    return max(DEFAULT_PROBE_TIMEOUT_SECONDS, probe_timeouts().get(modality, 0.0))


def run_probe(
    *,
    model_id: str,
    no_inference: bool,
    device: str | None,
    as_json: bool,
) -> int:
    """Spawn an in-venv probe; persist the resulting measurement.

    Returns 0 on success, non-zero on the various failure modes
    (unknown model / not pulled / subprocess crash / non-JSON output /
    a load that succeeded but whose inference probe raised).
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

    with _model_pull_lock(model_id):
        try:
            with _model_resource_lease(model_id):
                return _run_probe_locked(
                    model_id=model_id,
                    no_inference=no_inference,
                    device=device,
                    as_json=as_json,
                )
        except ModelInUseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3


def _run_probe_locked(
    *,
    model_id: str,
    no_inference: bool,
    device: str | None,
    as_json: bool,
) -> int:
    """Run one probe while holding the model's pull/refresh/remove lock."""
    if not is_pulled(model_id):
        print(
            f"error: model {model_id!r} is no longer pulled; "
            f"run `muse pull {model_id}` first",
            file=sys.stderr,
        )
        return 2

    catalog = _read_catalog()
    venv_path_raw = catalog.get(model_id, {}).get("venv_path")
    if not venv_path_raw:
        print(
            f"error: catalog entry for {model_id!r} is missing venv_path; "
            f"re-pull",
            file=sys.stderr,
        )
        return 2
    venv_path = Path(venv_path_raw)
    py = venv_python(venv_path)

    # Resolve the device the worker will pass to load_backend. Capability
    # device pin still takes precedence inside load_backend; this just
    # picks the request that gets forwarded.
    entry = known_models().get(model_id)
    if entry is None:
        print(f"error: model {model_id!r} disappeared during probe", file=sys.stderr)
        return 2
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

    timeout = _probe_timeout(entry.modality)
    try:
        result = run_owned_command(
            cmd, capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"error: probe timed out (>{timeout / 60:g} min)", file=sys.stderr)
        return 3
    except _OwnedProcessCleanupError as exc:
        print(f"error: probe subprocess cleanup failed: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"error: could not start probe subprocess: {exc}", file=sys.stderr)
        return 3

    if result.returncode != 0:
        print(f"error: probe subprocess exited {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode

    # The worker writes progress logs on stderr; the final stdout line
    # is a JSON object the parent persists. Parse the last non-empty line.
    stdout = result.stdout or ""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        print("error: probe produced no output", file=sys.stderr)
        return 4
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError:
        print("error: probe output was not JSON; full stdout:", file=sys.stderr)
        print(stdout, file=sys.stderr)
        return 4
    if not isinstance(record, dict):
        print("error: probe output JSON was not an object", file=sys.stderr)
        return 4
    record_model_id = record.get("model_id")
    if record_model_id is not None and record_model_id != model_id:
        print(
            f"error: probe returned measurement for {record_model_id!r}, "
            f"expected {model_id!r}",
            file=sys.stderr,
        )
        return 4

    # Persist under measurements.<device>. Per-device keying so the same
    # model can carry both a CPU and a GPU probe; `muse models list`
    # picks based on capabilities.device.
    # M1: hold _CATALOG_WRITE_LOCK for the full read->mutate->write to
    # prevent a concurrent probe or observed-peak writeback from losing
    # this measurement via an interleaved read-modify-write.
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id in catalog:
            catalog[model_id].setdefault("measurements", {})
            device_key = record.get("device", effective_device)
            if not isinstance(device_key, str) or not device_key:
                print("error: probe returned an invalid device", file=sys.stderr)
                return 4
            catalog[model_id]["measurements"][device_key] = record
            _write_catalog(catalog)
    _reset_known_models_cache()

    if as_json:
        print(json.dumps(record, indent=2))
    else:
        _print_human(record)
    # The worker exits 0 when the model loads but its representative
    # inference raises (the record carries the load measurement plus
    # `inference_error`). That is still a failed probe.
    if record.get("inference_error"):
        print(
            f"error: {model_id} loaded, but its inference probe failed",
            file=sys.stderr,
        )
        return 6
    return 0


def run_probe_all(
    *,
    no_inference: bool,
    device: str | None,
    as_json: bool,
) -> int:
    """Probe every enabled+pulled model. Continue past failures.

    Returns 0 if every probe succeeded; non-zero if any failed (but the
    loop runs to completion regardless). With no enabled+pulled models
    available, prints a helpful note to stderr and returns 1.
    """
    catalog_known = known_models()
    catalog = _read_catalog()
    targets = [
        mid for mid in sorted(catalog_known)
        if is_pulled(mid)
        and catalog.get(mid, {}).get("enabled", True)
    ]

    if not targets:
        print(
            "no enabled+pulled models to probe; check `muse models list`",
            file=sys.stderr,
        )
        return 1

    if not as_json:
        n = len(targets)
        plural = "s" if n != 1 else ""
        print(f"Probing {n} enabled model{plural}...")
        print()

    results: list[tuple[str, int]] = []
    for mid in targets:
        rc = run_probe(
            model_id=mid,
            no_inference=no_inference,
            device=device,
            as_json=as_json,
        )
        results.append((mid, rc))

    successes = [m for m, rc in results if rc == 0]
    failures = [m for m, rc in results if rc != 0]

    if not as_json:
        print()
        print("=" * 60)
        print(
            f"Probed {len(targets)} models: "
            f"{len(successes)} succeeded, {len(failures)} failed"
        )
        if failures:
            print(f"Failed: {', '.join(failures)}")
        print()
        print("Run `muse models list` to see updated memory column.")

    return 0 if not failures else 1


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
