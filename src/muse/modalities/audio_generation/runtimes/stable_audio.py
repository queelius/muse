"""StableAudioRuntime: generic runtime over diffusers.StableAudioPipeline.

One class wraps any Stable Audio Open-shaped repo on HuggingFace.
Pulled via the HF resolver (`muse pull
hf://stabilityai/stable-audio-open-1.0`) -> manifest with
backend_path pointing at this class.

Deferred imports follow the muse pattern: torch + StableAudioPipeline
stay as module-top sentinels (None) until _ensure_deps() lazy-imports
them. Tests patch the sentinels directly; _ensure_deps short-circuits
on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.audio_generation.protocol import AudioGenerationResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
StableAudioPipeline: Any = None


def _ensure_deps() -> None:
    global torch, StableAudioPipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("StableAudioRuntime: torch unavailable: %s", e)
    if StableAudioPipeline is None:
        try:
            from diffusers import StableAudioPipeline as _p
            StableAudioPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "StableAudioRuntime: diffusers.StableAudioPipeline unavailable: %s",
                e,
            )


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


def _normalize_pipeline_output(audio: Any) -> tuple[np.ndarray, int]:
    """Normalize StableAudioPipeline output to (samples,) or (samples, channels).

    StableAudioPipeline emits a numpy array. Mono outputs come back as
    `(samples,)`. Multi-channel outputs come back as `(channels, samples)`
    (channel-first), but soundfile + the codec want
    `(samples, channels)`. This helper transposes so downstream code
    only deals with the second shape.

    Returns (audio, channels).
    """
    if isinstance(audio, list):
        # Some test mocks may use a list; tolerate.
        audio = np.asarray(audio)
    if not isinstance(audio, np.ndarray):
        # Tensor: convert via .cpu().numpy() if torch is available.
        if torch is not None and isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        else:
            audio = np.asarray(audio)
    audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 1:
        return audio, 1
    if audio.ndim == 2:
        # diffusers StableAudioPipeline emits (channels, samples) when
        # multi-channel (matches transformer convention). Detect by
        # the smaller dim being channels (1 or 2).
        if audio.shape[0] <= 2 and audio.shape[1] > 2:
            return audio.T.copy(), audio.shape[0]
        # Already (samples, channels).
        return audio, audio.shape[1]
    raise ValueError(
        f"unsupported StableAudio output shape: {audio.shape}"
    )


class StableAudioRuntime:
    """Generic Stable Audio runtime.

    Construction kwargs (set by catalog at load_backend time, sourced
    from manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response metadata echoes this)
      - default_duration, default_steps, default_guidance,
        default_sample_rate: defaults from manifest capabilities
      - min_duration, max_duration: clamp bounds
    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        default_duration: float = 10.0,
        default_steps: int = 50,
        default_guidance: float = 7.0,
        default_sample_rate: int = 44100,
        min_duration: float = 1.0,
        max_duration: float = 47.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if StableAudioPipeline is None:
            raise RuntimeError(
                "diffusers.StableAudioPipeline is not installed; ensure "
                "diffusers>=0.27.0 is in this venv (e.g. via "
                "`muse pull stable-audio-open-1.0`)"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._default_duration = default_duration
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._default_sample_rate = default_sample_rate
        self._min_duration = min_duration
        self._max_duration = max_duration

        # Access torch through this module so tests' patches survive.
        import muse.modalities.audio_generation.runtimes.stable_audio as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        self._dtype = dtype
        self._torch_dtype = torch_dtype

        src = local_dir or hf_repo
        logger.info(
            "loading StableAudioPipeline from %s (model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, dtype,
        )
        self._pipe = StableAudioPipeline.from_pretrained(
            src,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        duration: float | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        **_: Any,
    ) -> AudioGenerationResult:
        eff_duration = duration if duration is not None else self._default_duration
        # Clamp to capability bounds.
        eff_duration = max(self._min_duration, min(self._max_duration, eff_duration))
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.modalities.audio_generation.runtimes.stable_audio as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "audio_end_in_s": float(eff_duration),
            "num_inference_steps": int(n_steps),
            "guidance_scale": float(cfg),
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        audios = getattr(out, "audios", None)
        if audios is None:
            # Some pipeline variants return a tuple.
            audios = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(audios, (list, tuple)):
            audio = audios[0]
        else:
            audio = audios

        normalized, channels = _normalize_pipeline_output(audio)
        sample_rate = getattr(
            self._pipe, "sample_rate", self._default_sample_rate,
        )
        if sample_rate is None:
            sample_rate = self._default_sample_rate
        n_samples = normalized.shape[0]
        duration_seconds = n_samples / float(sample_rate)

        return AudioGenerationResult(
            audio=normalized,
            sample_rate=int(sample_rate),
            channels=int(channels),
            duration_seconds=duration_seconds,
            metadata={
                "prompt": prompt,
                "steps": int(n_steps),
                "guidance": float(cfg),
                "seed": seed,
                "model": self.model_id,
                "duration_requested": float(eff_duration),
            },
        )
