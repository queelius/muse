"""``AbcMidiLMRuntime``: generic HuggingFace causal-LM runtime for ABC symbolic music.

Wraps ``transformers.AutoModelForCausalLM`` + ``AutoTokenizer`` for
symbolic text-to-ABC checkpoints (e.g. ``sander-wood/text-to-music``,
NotaGen, ABC-GPT), parses generated ABC notation into ``MidiScore`` via
``muse.modalities.music_midi.abc.parse_abc``, and encodes Standard MIDI
File Type 1 bytes.

Follows Muse's deferred-import pattern: ``torch``,
``AutoModelForCausalLM``, and ``AutoTokenizer`` stay ``None`` at module
import until ``_ensure_deps()`` runs, and device / dtype / inference-mode /
timing / source resolution delegate to ``muse.core.runtime_helpers``.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    resolve_model_source,
    select_device,
    set_inference_mode,
)
from muse.modalities.music_midi.abc import extract_abc_block, parse_abc
from muse.modalities.music_midi.protocol import MidiGenerationResult
from muse.modalities.music_midi.smf import encode_smf, resolve_gm_instrument


logger = logging.getLogger(__name__)


# Module-level sentinels patched in unit tests; populated lazily by _ensure_deps().
torch: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("AbcMidiLMRuntime: torch unavailable: %s", e)
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        try:
            from transformers import (
                AutoModelForCausalLM as _am,
                AutoTokenizer as _at,
            )
            if AutoModelForCausalLM is None:
                AutoModelForCausalLM = _am
            if AutoTokenizer is None:
                AutoTokenizer = _at
        except Exception as e:  # noqa: BLE001
            logger.debug("AbcMidiLMRuntime: transformers unavailable: %s", e)


def _format_abc_conditioning_prompt(
    prompt: str,
    *,
    key: str | None,
    mode: str | None,
    bpm: int | None,
    time_signature: str,
    bars: int,
    instruments: list[str] | None,
) -> str:
    """Build a text + ABC header prompt to steer symbolic causal LMs."""
    parts = [prompt.strip()]
    hints: list[str] = []
    if key:
        hints.append(f"Key: {key} {mode or 'major'}")
    elif mode:
        hints.append(f"Mode: {mode}")
    if bpm is not None:
        hints.append(f"Tempo: {bpm} BPM")
    if time_signature:
        hints.append(f"Meter: {time_signature}")
    if bars:
        hints.append(f"Length: {bars} bars")
    if instruments:
        hints.append(f"Instruments: {', '.join(instruments)}")
    if hints:
        parts.append(" | ".join(hints))

    header_lines = ["X:1", f"M:{time_signature}", "L:1/8"]
    if bpm is not None:
        header_lines.append(f"Q:1/4={int(bpm)}")
    if key:
        mode_tag = (mode or "major").lower()
        key_spec = f"{key}m" if mode_tag in ("minor", "aeolian") else (
            key if mode_tag == "major" else f"{key} {mode_tag[:3]}"
        )
        header_lines.append(f"K:{key_spec}")
    return "\n".join(parts) + "\n" + "\n".join(header_lines) + "\n"


class AbcMidiLMRuntime:
    """Generic causal-LM runtime generating ABC notation -> SMF Type 1 MIDI."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str = "",
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 512,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers (AutoModelForCausalLM, AutoTokenizer) is not "
                "installed in this venv; install transformers>=4.36.0 and torch"
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._max_new_tokens = max(64, int(max_new_tokens))

        # Access torch through module namespace so test patches survive.
        import muse.modalities.music_midi.runtimes.abc_lm as _mod
        _torch = _mod.torch
        eff_dtype_name = (
            "float32"
            if self._device == "cpu" and dtype in ("float16", "fp16")
            else dtype
        )
        torch_dtype = dtype_for_name(eff_dtype_name, _torch)
        self._dtype = eff_dtype_name
        self._torch_dtype = torch_dtype

        src = local_dir or resolve_model_source(hf_repo)
        logger.info(
            "loading AbcMidiLMRuntime from %s (model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, eff_dtype_name,
        )
        with LoadTimer(model_id, logger):
            self._tokenizer = AutoTokenizer.from_pretrained(src)
            load_kwargs: dict[str, Any] = {}
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            self._model = AutoModelForCausalLM.from_pretrained(src, **load_kwargs)
            if self._device != "cpu" and hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
            set_inference_mode(self._model)

    def generate(
        self,
        prompt: str,
        *,
        key: str | None = None,
        mode: str | None = None,
        bpm: int | None = None,
        time_signature: str = "4/4",
        bars: int = 16,
        instruments: list[str] | None = None,
        layers: list[str] | None = None,
        seed: int | None = None,
        temperature: float = 0.9,
        max_new_tokens: int | None = None,
        **_: Any,
    ) -> MidiGenerationResult:
        import muse.modalities.music_midi.runtimes.abc_lm as _mod
        _torch = _mod.torch

        if seed is not None and _torch is not None and hasattr(_torch, "manual_seed"):
            _torch.manual_seed(int(seed))

        cond_prompt = _format_abc_conditioning_prompt(
            prompt,
            key=key,
            mode=mode,
            bpm=bpm,
            time_signature=time_signature,
            bars=bars,
            instruments=instruments,
        )
        encoded = self._tokenizer(cond_prompt, return_tensors="pt")
        if self._device != "cpu" and hasattr(encoded, "to"):
            encoded = encoded.to(self._device)

        eff_temp = max(0.0, min(2.0, float(temperature)))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self._max_new_tokens),
            "do_sample": eff_temp > 0.0,
        }
        if eff_temp > 0.0:
            gen_kwargs["temperature"] = eff_temp
            gen_kwargs["top_p"] = 0.95

        no_grad_ctx = (
            _torch.no_grad()
            if (_torch is not None and hasattr(_torch, "no_grad"))
            else nullcontext()
        )
        with no_grad_ctx:
            if isinstance(encoded, dict):
                output_ids = self._model.generate(**encoded, **gen_kwargs)
            else:
                output_ids = self._model.generate(**dict(encoded), **gen_kwargs)

        first_seq = output_ids[0] if isinstance(output_ids, (list, tuple)) or hasattr(output_ids, "__getitem__") else output_ids
        raw_text = self._tokenizer.decode(first_seq, skip_special_tokens=True)
        abc_block = extract_abc_block(raw_text)

        ts_parts = time_signature.split("/")
        default_ts = (int(ts_parts[0]), int(ts_parts[1])) if len(ts_parts) == 2 else (4, 4)

        score = parse_abc(
            abc_block,
            default_bpm=int(bpm) if bpm is not None else 120,
            default_time_signature=default_ts,
            default_key=key or "C",
            default_mode=mode or "major",
        )

        # Apply explicit caller overrides cleanly on top of parsed ABC score.
        if bpm is not None:
            score.bpm = int(bpm)
        if instruments:
            for idx, track in enumerate(score.tracks):
                inst_name = instruments[idx % len(instruments)]
                prog, is_perc = resolve_gm_instrument(inst_name)
                track.instrument = inst_name
                track.gm_program = prog
                if is_perc:
                    track.channel = 9
        if layers:
            primary_layer = layers[0]
            for track in score.tracks:
                track.layer = primary_layer

        # Trim notes beyond requested bar ceiling if the LM ran long
        max_beat = bars * score.beats_per_bar
        for track in score.tracks:
            trimmed = []
            for note in track.notes:
                if note.start_beat >= max_beat:
                    continue
                if note.start_beat + note.duration_beats > max_beat:
                    note.duration_beats = max(0.0625, max_beat - note.start_beat)
                trimmed.append(note)
            track.notes = trimmed
        score.bars = min(score.bars, int(bars)) if score.bars > bars else score.bars

        midi_bytes = encode_smf(score)
        return MidiGenerationResult(
            score=score,
            midi_bytes=midi_bytes,
            seed=seed,
            model_id=self.model_id,
            metadata={
                "prompt": prompt,
                "seed": seed,
                "temperature": eff_temp,
                "runtime": "abc_lm",
                "abc": abc_block,
            },
        )
