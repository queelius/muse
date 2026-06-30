"""AceStepRuntime: text-to-music over ACE-Step/ACE-Step-v1-3.5B.

ACE-Step generates full songs (genre/style prompt + optional structured
lyrics) or instrumentals, 48kHz stereo, via the `acestep` package's
`ACEStepPipeline`. Unlike StableAudio (which returns in-memory numpy
arrays), ACE-Step WRITES audio file(s) to disk and returns the path(s).
This runtime therefore:

  1. hands the pipeline a temp `save_path`,
  2. resolves the real on-disk output defensively (returned path if it
     exists, else the save_path we provided),
  3. reads the WAV back into a float32 array via soundfile, and
  4. deletes the temp dir in a `finally` (leak-safe, per the #200
     temp-WAV lesson).

`trust_remote_code` is NOT needed: `ACEStepPipeline` is a direct package
import, not an `AutoModel.from_pretrained(..., trust_remote_code=True)`
boundary.

Deferred imports follow the muse pattern: torch + ACEStepPipeline +
soundfile stay as module-top sentinels (None) until `_ensure_deps()`
lazy-imports them. Tests patch the sentinels directly; `_ensure_deps`
short-circuits on non-None.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any

import numpy as np

from muse.core.runtime_helpers import select_device
from muse.modalities.audio_generation.protocol import AudioGenerationResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
ACEStepPipeline: Any = None
sf: Any = None


# ACE-Step's constructor wants a dtype STRING ("bfloat16"/"float16"/
# "float32"), not a torch.dtype object. This normalizer maps muse's
# dtype aliases onto ACE-Step's expected strings. It is string->string
# (never string->torch.dtype), so it is distinct from the canonical
# dtype_for_name helper and does not trip the runtime-helpers meta-test.
_ACESTEP_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def _ensure_deps() -> None:
    """Lazy-import torch + acestep.ACEStepPipeline + soundfile.

    Each symbol is imported independently so tests can patch one without
    forcing the others to load.
    """
    global torch, ACEStepPipeline, sf
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("AceStepRuntime: torch unavailable: %s", e)
    if ACEStepPipeline is None:
        try:
            from acestep.pipeline_ace_step import ACEStepPipeline as _p
            ACEStepPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("AceStepRuntime: acestep.ACEStepPipeline unavailable: %s", e)
    if sf is None:
        try:
            import soundfile as _sf
            sf = _sf
        except Exception as e:  # noqa: BLE001
            logger.debug("AceStepRuntime: soundfile unavailable: %s", e)


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


def _acestep_dtype(name: str) -> str:
    """Map a muse dtype alias to ACE-Step's expected dtype string."""
    return _ACESTEP_DTYPES.get(str(name).lower(), "bfloat16")


def _resolve_output_path(outputs: Any, fallback: str) -> str:
    """Resolve the on-disk audio file ACE-Step produced.

    The pipeline returns path(s) and writes file(s) to disk. Prefer a
    returned path that exists; fall back to the save_path we provided
    (ACE-Step may add a suffix or return a relative path). Raise if
    nothing readable is on disk.
    """
    candidates: list[str] = []
    if isinstance(outputs, str):
        candidates.append(outputs)
    elif isinstance(outputs, (list, tuple)):
        for o in outputs:
            if isinstance(o, str):
                candidates.append(o)
            elif isinstance(o, (list, tuple)):
                candidates.extend(x for x in o if isinstance(x, str))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError(
        "ACE-Step produced no readable audio output "
        f"(returned {outputs!r}, save_path {fallback!r} not on disk)"
    )


class AceStepRuntime:
    """ACE-Step text-to-music runtime.

    Construction kwargs (set by catalog at load_backend time, sourced
    from manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response metadata echoes this)
      - default_duration, default_steps, default_guidance,
        default_sample_rate: defaults from manifest capabilities
      - min_duration, max_duration: clamp bounds
      - cpu_offload, overlapped_decode, torch_compile: optional low-VRAM
        / speed knobs forwarded to the pipeline constructor (default off)
    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "bf16",
        default_duration: float = 60.0,
        default_steps: int = 60,
        default_guidance: float = 15.0,
        default_sample_rate: int = 48000,
        min_duration: float = 1.0,
        max_duration: float = 240.0,
        cpu_offload: bool = False,
        overlapped_decode: bool = False,
        torch_compile: bool = False,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if ACEStepPipeline is None:
            raise RuntimeError(
                "acestep.ACEStepPipeline is not installed; run "
                "`muse pull ace-step-v1-3.5b` to install deps into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._default_duration = default_duration
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._default_sample_rate = default_sample_rate
        self._min_duration = min_duration
        self._max_duration = max_duration

        acestep_dt = _acestep_dtype(dtype)
        self._dtype = acestep_dt
        # ACE-Step indexes the GPU by integer id; it is GPU-oriented.
        device_id = 0

        src = local_dir or hf_repo
        logger.info(
            "loading ACEStepPipeline from %s (model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, acestep_dt,
        )
        self._pipe = ACEStepPipeline(
            checkpoint_dir=src,
            device_id=device_id,
            dtype=acestep_dt,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode,
            torch_compile=torch_compile,
        )

    def generate(
        self,
        prompt: str,
        *,
        duration: float | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        lyrics: str | None = None,
        **_: Any,
    ) -> AudioGenerationResult:
        eff_duration = duration if duration is not None else self._default_duration
        eff_duration = max(self._min_duration, min(self._max_duration, eff_duration))
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        # Empty/blank lyrics => instrumental. ACE-Step's instrumental
        # convention is the literal "[instrumental]" tag.
        instrumental = not (lyrics and lyrics.strip())
        lyrics_eff = "[instrumental]" if instrumental else lyrics
        manual_seeds = str(seed) if seed is not None else None

        tmpdir = tempfile.mkdtemp(prefix="muse-acestep-")
        try:
            save_path = os.path.join(tmpdir, "out.wav")
            outputs = self._pipe(
                format="wav",
                audio_duration=float(eff_duration),
                prompt=prompt,
                lyrics=lyrics_eff,
                infer_step=int(n_steps),
                guidance_scale=float(cfg),
                manual_seeds=manual_seeds,
                save_path=save_path,
                batch_size=1,
            )
            path = _resolve_output_path(outputs, save_path)
            data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        audio = np.asarray(data, dtype=np.float32)
        channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        n_samples = int(audio.shape[0])
        duration_seconds = n_samples / float(sample_rate)

        return AudioGenerationResult(
            audio=audio,
            sample_rate=int(sample_rate),
            channels=channels,
            duration_seconds=duration_seconds,
            metadata={
                "prompt": prompt,
                "lyrics": lyrics_eff,
                "instrumental": instrumental,
                "steps": int(n_steps),
                "guidance": float(cfg),
                "seed": seed,
                "model": self.model_id,
                "duration_requested": float(eff_duration),
            },
        )
