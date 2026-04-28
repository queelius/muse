"""muse _probe_worker: in-venv probe execution.

Runs in the per-model venv (where the model's heavy deps are installed).
Loads the model, captures pre/post memory, runs representative inference
(unless --no-inference), captures peak. Prints one final line of JSON
on stdout for the parent process to parse and persist.

Progress logs go to stderr so the final stdout line is unambiguous.

psutil is imported defensively because older venvs (created before
psutil was added to muse[server] in v0.18.2) won't have it. We fall
back to /proc/self/statm on Linux when needed.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone


log = logging.getLogger("muse.probe_worker")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_rss_bytes() -> int:
    """Process resident set size in bytes. psutil if present; /proc fallback."""
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        pass
    # Linux fallback: /proc/self/statm reports pages; convert via PAGESIZE.
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
        rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        return rss_pages * page_size
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return 0


def run_probe_worker(*, model_id: str, device: str, run_inference: bool) -> int:
    """In-venv probe entry. Prints JSON record on stdout's last line."""
    from muse.core.catalog import known_models, load_backend  # noqa: PLC0415

    entry = known_models()[model_id]
    actual_device = _resolve_device(device, entry)

    # Pre-load baseline. CUDA peak counter must be reset BEFORE loading
    # so reset_peak_memory_stats and the load both see the same epoch.
    ram_baseline = _process_rss_bytes()
    vram_baseline = 0
    cuda = None
    if actual_device.startswith("cuda"):
        try:
            import torch  # noqa: PLC0415
            cuda = torch
            cuda.cuda.reset_peak_memory_stats()
            cuda.cuda.empty_cache()
            vram_baseline = cuda.cuda.memory_allocated()
        except ImportError:
            cuda = None

    print(
        f"baseline: RAM={ram_baseline/1024**3:.2f} GB, "
        f"VRAM={vram_baseline/1024**3:.2f} GB",
        file=sys.stderr,
    )

    # Load
    t0 = time.time()
    try:
        backend = load_backend(model_id, device=device)
    except Exception as e:  # noqa: BLE001
        print(f"load failed: {e}", file=sys.stderr)
        record = {
            "model_id": model_id,
            "device": actual_device,
            "error": f"load failed: {e}",
            "probed_at": _utcnow_iso(),
            "ran_inference": False,
        }
        print(json.dumps(record))
        return 2
    load_seconds = time.time() - t0

    ram_post_load = _process_rss_bytes()
    vram_post_load = 0
    if cuda is not None:
        vram_post_load = cuda.cuda.memory_allocated()

    if actual_device.startswith("cuda") and cuda is not None:
        weights_bytes = max(0, vram_post_load - vram_baseline)
    else:
        weights_bytes = max(0, ram_post_load - ram_baseline)

    print(
        f"load: {load_seconds:.1f}s, weights={weights_bytes/1024**3:.2f} GB",
        file=sys.stderr,
    )

    record: dict = {
        "model_id": model_id,
        "modality": entry.modality,
        "device": actual_device,
        "weights_bytes": weights_bytes,
        "peak_bytes": weights_bytes,
        "load_seconds": load_seconds,
        "ran_inference": False,
        "probed_at": _utcnow_iso(),
    }

    if run_inference:
        try:
            shape, peak_bytes = _run_inference(backend, entry, actual_device, cuda)
            record["ran_inference"] = True
            record["shape"] = shape
            record["peak_bytes"] = max(peak_bytes, weights_bytes)
        except Exception as e:  # noqa: BLE001
            print(f"inference probe failed: {e}", file=sys.stderr)
            record["inference_error"] = str(e)

    # Drop references so any GC happens before exit.
    del backend
    gc.collect()

    print(json.dumps(record))
    return 0


def _resolve_device(requested: str, entry) -> str:
    """Mirror catalog.load_backend's device resolution.

    Capability pin > requested > torch auto-detect > "cpu".
    """
    cap = entry.extra.get("device", "auto")
    if cap and cap != "auto":
        return cap
    if requested and requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _run_inference(backend, entry, device: str, cuda) -> tuple[str, int]:
    """Run representative inference, return (shape_label, peak_bytes).

    Resets the CUDA peak counter immediately before the call so the
    measurement isolates the inference activations from the load step.
    """
    if device.startswith("cuda") and cuda is not None:
        cuda.cuda.reset_peak_memory_stats()

    defaults = _read_probe_defaults(entry.modality)
    shape = defaults["shape"]
    fn = defaults["call"]
    fn(backend)

    if device.startswith("cuda") and cuda is not None:
        peak_bytes = cuda.cuda.max_memory_allocated()
    else:
        # CPU: process RSS now is a poor proxy for activation peak (it
        # does not capture transient peaks that get freed). Documented
        # in CLAUDE.md memory accounting section.
        peak_bytes = _process_rss_bytes()

    return shape, peak_bytes


def _read_probe_defaults(modality: str) -> dict:
    """Look up PROBE_DEFAULTS on the modality package; fallback if missing."""
    try:
        import importlib  # noqa: PLC0415
        module_name = "muse.modalities." + modality.replace("/", "_")
        mod = importlib.import_module(module_name)
        defaults = getattr(mod, "PROBE_DEFAULTS", None)
        if defaults is not None:
            return defaults
    except Exception:  # noqa: BLE001
        pass
    return _hardcoded_defaults_for(modality)


def _hardcoded_defaults_for(modality: str) -> dict:
    """Fallback when a modality doesn't declare PROBE_DEFAULTS yet.

    Keeps the probe useful even for external modality plugins dropped
    into $MUSE_MODALITIES_DIR that haven't been updated for v0.18.2.
    """
    if modality == "audio/speech":
        return {
            "shape": "5s synthesis",
            "call": lambda m: m.synthesize("Hello, this is a probe."),
        }
    if modality == "audio/transcription":
        return {
            "shape": "5s 16k mono",
            "call": _transcribe_short_silence,
        }
    if modality == "embedding/text":
        return {
            "shape": "1 short string",
            "call": lambda m: m.embed(["probe text"]),
        }
    if modality == "image/generation":
        return {
            "shape": "default size, 1 image",
            "call": lambda m: m.generate("probe scene"),
        }
    if modality == "image/animation":
        return {
            "shape": "default frames @ default size",
            "call": lambda m: m.generate("probe motion"),
        }
    if modality == "text/classification":
        return {
            "shape": "1 short string",
            "call": lambda m: m.classify(["probe text"]),
        }
    if modality == "chat/completion":
        return {
            "shape": "8-token completion",
            "call": lambda m: m.chat(
                messages=[{"role": "user", "content": "probe"}],
                max_tokens=8,
            ),
        }
    raise NotImplementedError(f"no probe default for modality {modality!r}")


def _transcribe_short_silence(model):
    """Generate 5s of silent wav and pass to backend.transcribe."""
    import io  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import wave  # noqa: PLC0415

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 5)
    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(buf.read())
        path = f.name
    try:
        return model.transcribe(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
