"""``AudioTools``: the deterministic, weightless audio-processing runtime.

Wraps :func:`muse.modalities.audio_process.dsp.process_audio` behind the
:class:`AudioProcessModel` protocol. The bundled ``audio-tools`` model script
aliases it as ``Model``. CPU only; ``device`` is accepted and ignored.
"""
from __future__ import annotations

from typing import Any

from muse.modalities.audio_process.protocol import (
    AudioProcessOptions,
    AudioProcessResult,
)

# numpy and scipy are imported inside dsp.py when process() is invoked, never
# at module load so script discovery stays light on bare hosts.


class AudioTools:
    """Deterministic audio mastering and looping backend (CPU, no weights)."""

    def __init__(
        self,
        *,
        model_id: str = "audio-tools",
        hf_repo: str = "",
        local_dir: str | None = None,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.local_dir = local_dir
        self.device = "cpu"

    def process(
        self,
        audio: Any,
        sample_rate: int,
        options: AudioProcessOptions | None = None,
    ) -> AudioProcessResult:
        from muse.modalities.audio_process.dsp import process_audio

        opts = options if options is not None else AudioProcessOptions()
        processed = process_audio(
            audio,
            sample_rate,
            mono=opts.mono,
            target_sample_rate=opts.sample_rate,
            remove_dc=opts.remove_dc,
            trim_silence=opts.trim_silence,
            trim_threshold_db=opts.trim_threshold_db,
            pad_lead_ms=opts.pad_lead_ms,
            pad_tail_ms=opts.pad_tail_ms,
            fade_in_ms=opts.fade_in_ms,
            fade_out_ms=opts.fade_out_ms,
            loop_fade_s=opts.loop_fade_s,
            target_peak_dbfs=opts.target_peak_dbfs,
        )
        return AudioProcessResult(
            audio=processed.audio,
            sample_rate=processed.sample_rate,
            channels=processed.channels,
            metrics=processed.metrics,
        )
