"""Qwen3-TTS runtime: designed voices (VoiceDesign) and saved voices (Base).

One class serves both bundled scripts, selected by ``mode``:

- ``"design"`` (Qwen3-TTS-12Hz-1.7B-VoiceDesign): the voice comes from a
  natural-language description, the request's ``instructions`` (else the
  manifest's ``default_instructions``). There are no named voices.
- ``"clone"`` (Qwen3-TTS-12Hz-0.6B-Base): the voice is a saved reference clip
  plus its transcript, from a :class:`VoiceStore` under
  ``<paths.home>/voices/<model_id>/`` with the bundled read-only ``narrator``.

On CUDA the checkpoint loads with SDPA attention, and qwen-tts's two
``generate()`` loops (the talker's, and its per-frame code predictor's) are
replaced by static-KV decoders that run each 12 Hz frame as one CUDA graph
(:mod:`.qwen3_tts_fast`; ``fast_code_predictor`` / ``fast_talker`` /
``cuda_graphs`` kwargs). The speech decoder's SnakeBeta activations run as
one Triton kernel each (``fast_decoder``), bit-identical to upstream.
``int8_weights`` (off by default; CUDA with Triton) stores the talker and
code-predictor layer weights as int8 only: about 40% less frame time and less
VRAM, for a small, measured change in the token distributions.

``synthesize_stream`` streams real chunks where the fast talker runs (CUDA):
qwen-tts builds the prompt as for :meth:`synthesize` and is unwound at the
talker call (:meth:`FastTalker.capture`); the talker's frames then come one
at a time (``start`` / ``step``) and the speech decoder, split into its
causal front and its conv stack (:mod:`.qwen3_tts_stream`), turns each
chunk into audio (on CUDA as graphs per padded shape). The first chunk is
``stream_first_frames`` 12 Hz frames, each later one twice the last up to
``stream_chunk_frames``. The codes equal the one-call path's and
the audio its one-pass decode within the decoder's bf16 floor. Elsewhere (CPU,
another qwen-tts, a fast path disabled after a failure) a stream is one chunk.

qwen-tts has no speed control (``speed`` is ignored). ``language=None`` is
qwen-tts's "Auto" detection.

Heavy deps (torch, qwen_tts) load in ``_ensure_deps``; tests replace the
``torch`` / ``QwenTTS`` sentinels.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Iterator, Literal, Mapping

import numpy as np

from muse.core import config
from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    select_device,
    set_inference_mode,
)
from muse.modalities.audio_speech.runtimes import (
    qwen3_tts_batch,
    qwen3_tts_fast,
    qwen3_tts_stream,
)
from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, VoiceError
from muse.modalities.audio_speech.voice_catalog import BUNDLED_VOICES, VoiceEntry
from muse.modalities.audio_speech.voices import (
    StoredVoice,
    VoiceStore,
    load_bundled_voice,
    voice_store_root,
)

logger = logging.getLogger(__name__)

torch: Any = None
QwenTTS: Any = None  # qwen_tts.Qwen3TTSModel

SAMPLE_RATE = 24000
NARRATOR = "narrator"
DEFAULT_INSTRUCTIONS = (
    "A clear, neutral adult narrator with an even, friendly tone, "
    "speaking at a moderate pace."
)
# Streaming chunk schedule, in 12 Hz frames (1920 samples, 80 ms each): a
# small first chunk for a fast start, then each chunk twice the last, up to
# the steady size. Doubling keeps every chunk inside the playback of the audio
# already sent (a 2-frame first chunk straight to 16 frames underran on the
# 1.7B: 16 frames take ~170 ms there, the first chunk plays for 160 ms).
STREAM_FIRST_FRAMES = 2
STREAM_CHUNK_FRAMES = 16
# Batched engine (max_batch_size > 1): streams are paced to stay this far
# ahead of their playback, so the GPU idles instead of racing ahead.
STREAM_LEAD_SECONDS = 1.5
# Each line's frame budget (qwen-tts's max_new_tokens), from its text: a
# sampled line that never emits EOS would otherwise run to 4096 frames
# (5.5 min) and, batched, grow every row's KV cache. Measured over 96 takes
# (24 lines, 2 seeds, both models) and a 750-character paragraph: at most
# 1.06 frames a character (short lines; 0.62-0.68 on the paragraph).
FRAMES_PER_CHAR = 3.0
FRAME_SLACK = 40
MAX_FRAMES = 4096


def frame_limit(text: str) -> int:
    """max_new_tokens for ``text``: FRAMES_PER_CHAR per character plus slack,
    at most qwen-tts's own 4096. A line that reaches it ends there."""
    return min(MAX_FRAMES, int(FRAMES_PER_CHAR * len(text)) + FRAME_SLACK)


# Padded prefill lengths captured by a warm-up (tokens, the batched prefill's step).
WARM_PREFILL_LENGTHS = tuple(range(32, 257, 32))
WARM_TEXT = "The lanterns stay lit until dawn."


def _ensure_deps() -> None:
    global torch, QwenTTS
    if QwenTTS is not None:
        return
    try:
        import torch as _torch
        from qwen_tts import Qwen3TTSModel as _QwenTTS
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qwen3-TTS requires torch and qwen-tts; run `muse pull` to "
            "install the model dependencies"
        ) from exc
    torch = _torch
    QwenTTS = _QwenTTS


def bundled_voices() -> dict[str, StoredVoice]:
    """The read-only voices shipped with muse (``voice_catalog.BUNDLED_VOICES``)."""
    return {name: load_bundled_voice(name, wav, meta)
            for name, (wav, meta) in BUNDLED_VOICES.items()}


class Qwen3TTSRuntime:
    """audio/speech backend for Qwen3-TTS VoiceDesign and Base checkpoints."""

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        model_id: str,
        mode: Literal["design", "clone"],
        dtype: str = "bf16",
        default_instructions: str | None = None,
        fast_code_predictor: bool = True,
        fast_talker: bool = True,
        fast_decoder: bool = True,
        int8_weights: bool = False,
        cuda_graphs: bool = True,
        stream_first_frames: int = STREAM_FIRST_FRAMES,
        stream_chunk_frames: int = STREAM_CHUNK_FRAMES,
        max_batch_size: int = 1,
        stream_lead_seconds: float = STREAM_LEAD_SECONDS,
        stream_buffer_seconds: float = 0.0,
        warm_rows: int = 0,
        **_: Any,
    ) -> None:
        if mode not in ("design", "clone"):
            raise ValueError(f"mode must be 'design' or 'clone', got {mode!r}")
        if int(stream_first_frames) < 1 or int(stream_chunk_frames) < 1:
            raise ValueError("stream_first_frames and stream_chunk_frames must be >= 1")
        self._schedule = (int(stream_first_frames), int(stream_chunk_frames))
        max_batch_size = int(max_batch_size)
        if not 1 <= max_batch_size <= qwen3_tts_batch.BUCKETS[-1]:
            raise ValueError(f"max_batch_size must be 1..{qwen3_tts_batch.BUCKETS[-1]}")
        self._engine: Any = None
        _ensure_deps()
        self._model_id = model_id
        self._default_instructions = default_instructions or DEFAULT_INSTRUCTIONS
        self._inference_lock = threading.Lock()
        self._prompts: OrderedDict[tuple, Any] = OrderedDict()
        self._prompts_lock = threading.Lock()
        self._store: VoiceStore | None = None
        # Every saved voice stays cached once touched (a prompt is a speaker
        # embedding plus the reference codes: tiny next to the model).
        self._prompt_cache_size = 1
        if mode == "clone":
            bundled = bundled_voices()
            max_voices = int(config.get_positive("limits.max_saved_voices"))
            self._store = VoiceStore(
                voice_store_root(model_id), bundled=bundled, max_voices=max_voices)
            self._prompt_cache_size = max_voices + len(bundled)

        resolved = select_device(device, torch_module=torch)
        device_map = "cuda:0" if resolved == "cuda" else resolved
        load_kwargs: dict[str, Any] = {"device_map": device_map}
        torch_dtype = dtype_for_name(
            dtype if resolved.startswith("cuda") else "fp32", torch)
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        on_cuda = resolved.startswith("cuda")
        if on_cuda:
            # Without flash-attn, transformers would otherwise pick eager.
            load_kwargs["attn_implementation"] = "sdpa"
        src = local_dir or hf_repo
        with LoadTimer(model_id, logger):
            self._tts = QwenTTS.from_pretrained(src, **load_kwargs)
        set_inference_mode(getattr(self._tts, "model", None))
        graphs = bool(cuda_graphs) and on_cuda
        wide = {}  # float weights beside the int8, for frames past INT8_MAX_ROWS rows
        if on_cuda and max_batch_size > qwen3_tts_batch.INT8_MAX_ROWS:
            wide["keep_float"] = True
        if fast_code_predictor:
            qwen3_tts_fast.install(self._tts, torch=torch, use_cuda_graphs=graphs,
                                   int8=bool(int8_weights), **wide)
            if fast_talker:
                qwen3_tts_fast.install_talker(self._tts, torch=torch, use_cuda_graphs=graphs,
                                              int8=bool(int8_weights), **wide)
        if fast_decoder:
            qwen3_tts_fast.install_decoder(self._tts, torch=torch)
        self._device = device_map
        # Streaming needs the fast talker (checked per request) and the split
        # speech decoder; CPU keeps the one-chunk stream.
        self._split = (qwen3_tts_stream.split_decoder(self._tts, torch=torch,
                                                      use_cuda_graphs=graphs)
                       if on_cuda else None)
        talker = self._stream_talker()
        if max_batch_size > 1 and talker is not None:
            # One engine thread owns the GPU; up to max_batch_size requests
            # share its frames, so the route's lock admits that many.
            self._gpu_lock = threading.Lock()  # one-at-a-time after an engine failure
            self._engine = qwen3_tts_batch.Engine(
                talker, decoder=self._split, decode_whole=self._decode_whole, torch=torch,
                max_rows=max_batch_size, first_frames=self._schedule[0],
                chunk_frames=self._schedule[1], lead_seconds=float(stream_lead_seconds),
                buffer_seconds=float(stream_buffer_seconds), use_cuda_graphs=graphs)
            self._inference_lock = threading.Semaphore(max_batch_size)
            if int(warm_rows) > 0:
                # Allocate the rows and capture the graphs now: on the first
                # joins at each new bucket they would stall live lines.
                self._engine.warm(int(warm_rows),
                                  prepare=lambda: self._prepare(WARM_TEXT, None, None),
                                  prefill_lengths=WARM_PREFILL_LENGTHS if graphs else ())

    # -- protocol ------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def voices(self) -> list[str]:
        return self._store.names() if self._store is not None else []

    @property
    def default_voice(self) -> str | None:
        return NARRATOR if self._store is not None else None

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        instructions: str | None = None,
        **_: Any,
    ) -> AudioResult:
        engine = self._engine
        if engine is not None:
            try:
                handle = engine.submit(lambda: self._prepare(text, voice, instructions),
                                       stream=False)
                audio = handle.result()
                used = None if self._store is None else voice or NARRATOR
                return AudioResult(audio=audio, sample_rate=SAMPLE_RATE, metadata={"voice": used})
            except qwen3_tts_batch.EngineUnavailable:
                return self._one_at_a_time(
                    lambda: self._synthesize_one(text, voice=voice, instructions=instructions))
        return self._synthesize_one(text, voice=voice, instructions=instructions)

    def _synthesize_one(
        self,
        text: str,
        *,
        voice: str | None = None,
        instructions: str | None = None,
    ) -> AudioResult:
        limit = frame_limit(text)
        if self._store is None:
            instruct = (instructions or "").strip() or self._default_instructions
            wavs, sr = self._tts.generate_voice_design(
                text=text, instruct=instruct, language=None, max_new_tokens=limit)
            used = None
        else:
            used = voice or NARRATOR
            prompt = self._clone_prompt(used)
            wavs, sr = self._tts.generate_voice_clone(
                text=text, language=None, voice_clone_prompt=prompt, max_new_tokens=limit)
        audio = np.clip(np.asarray(wavs[0], dtype=np.float32).reshape(-1), -1.0, 1.0)
        if len(audio) // self._samples_per_frame() >= limit - 1:
            self._warn_limit(limit)
        return AudioResult(audio=audio, sample_rate=int(sr), metadata={"voice": used})

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Iterator[AudioChunk]:
        """Audio as it is generated: upstream's prompt building (captured at
        the talker call), then the fast talker's frames from a loop here,
        decoded in chunks by the split speech decoder. The codes are the
        one-call path's; the audio is its one-pass decode to within the
        decoder's own bf16 floor. Without the fast talker or the split
        decoder (CPU, another qwen-tts, either disabled after a failure), and
        after a failure before the first chunk, the line streams as one chunk."""
        kwargs.update(voice=voice, instructions=instructions)
        if self._engine is not None:
            yield from self._engine_stream(text, voice, instructions)
            return
        talker = self._stream_talker()
        if talker is None:
            yield self._one_chunk(text, **kwargs)
            return
        entry, ref_codes = self._entry(text, voice, instructions)
        req = talker.request(talker.capture(entry), ref_codes=ref_codes)
        if req is None:
            yield self._one_chunk(text, **kwargs)
            return
        first, steady = self._schedule
        sent, part = False, "talker"
        try:
            run = talker.start(req)
            decoder = self._split.stream(req.ref_codes)
            target = first
            while True:
                part = "talker"
                codes = talker.step(run)
                if codes is not None:
                    decoder.push(codes)
                if codes is None or decoder.pending >= target:
                    part = "decoder"
                    audio = decoder.decode()
                    if len(audio) or (codes is None and not sent):
                        part = None  # anything thrown in at the yield is not ours
                        yield AudioChunk(audio=audio, sample_rate=SAMPLE_RATE)
                        sent, target = True, max(1, min(steady, 2 * target))
                if codes is None:
                    if (run.produced >= run.request.max_new_tokens
                            and bool(run.state.unfinished)):
                        self._warn_limit(run.request.max_new_tokens)
                    return
        except Exception as exc:
            if part is None or type(exc).__name__ == "OutOfMemoryError":
                raise
            self._stream_failed(part, talker)
            if sent:
                raise
        yield self._one_chunk(text, **kwargs)

    def _engine_stream(self, text: str, voice: str | None,
                       instructions: str | None) -> Iterator[AudioChunk]:
        try:
            handle = self._engine.submit(lambda: self._prepare(text, voice, instructions),
                                         stream=True)
        except qwen3_tts_batch.EngineUnavailable:
            handle = None
        sent = False
        if handle is not None:
            try:
                for audio in handle:
                    sent = True
                    yield AudioChunk(audio=audio, sample_rate=SAMPLE_RATE)
                return
            except qwen3_tts_batch.EngineUnavailable:
                if sent:
                    raise
            finally:
                handle.cancel()  # a client that left frees its row at the next frame
        result = self._one_at_a_time(
            lambda: self._synthesize_one(text, voice=voice, instructions=instructions))
        yield AudioChunk(audio=result.audio, sample_rate=result.sample_rate)

    def _one_at_a_time(self, call: Any) -> Any:
        """The one-at-a-time path beside a batched engine: on the engine's
        thread while it runs (its GPU work must not overlap), else serialized."""
        engine = self._engine
        if engine is not None and not engine.failed:
            try:
                return engine.run_on_engine(call)
            except qwen3_tts_batch.EngineUnavailable:
                pass
        with self._gpu_lock:
            return call()

    def _prepare(self, text: str, voice: str | None, instructions: str | None) -> Any:
        """This line as a TalkerRequest (on the engine thread), or None when
        the fast talker does not handle it."""
        talker = self._stream_talker()
        if talker is None:
            return None
        entry, ref_codes = self._entry(text, voice, instructions)
        return talker.request(talker.capture(entry), ref_codes=ref_codes)

    def _decode_whole(self, codes: Any, ref_codes: Any) -> np.ndarray:
        """A finished line's audio as qwen-tts's one-call path decodes it: the
        reference's codes in front, one decode, the reference's share cut."""
        tok = self._tts.model.speech_tokenizer
        full = codes if ref_codes is None else torch.cat((ref_codes.to(codes.device), codes))
        wav = tok.decode([{"audio_codes": full}])[0][0]
        if ref_codes is not None:
            ref, total = int(ref_codes.shape[0]), int(full.shape[0])
            wav = wav[int(ref / max(total, 1) * wav.shape[0]):]
        return np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)

    def _samples_per_frame(self) -> int:
        split = getattr(self, "_split", None)
        return int(split.upsample) if split is not None else 1920

    def _warn_limit(self, limit: int) -> None:
        logger.warning("Qwen3-TTS line reached its frame limit (%d) without ending; "
                       "stopped there", limit - 1)

    def _one_chunk(self, text: str, **kwargs: Any) -> AudioChunk:
        result = self.synthesize(text, **kwargs)
        return AudioChunk(audio=result.audio, sample_rate=result.sample_rate)

    # -- saved voices (clone mode) -------------------------------------------

    def voice_entry(self, name: str) -> VoiceEntry:
        """One voice's catalog entry (metadata, no audio); shared-lock read."""
        return self._require_store().catalog.entry(name)

    def check_voice(self, voice: str | None) -> None:
        """Raise UnknownVoiceError before a stream starts; no-op in design mode."""
        if self._store is not None:
            self._store.get(voice or NARRATOR)

    def save_voice(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
        *,
        overwrite: bool = False,
        source_filename: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredVoice:
        store = self._require_store()
        try:
            return store.save(name, audio, sample_rate, transcript,
                              overwrite=overwrite, source_filename=source_filename,
                              metadata=metadata)
        finally:
            self._evict(name)

    def delete_voice(self, name: str) -> None:
        store = self._require_store()
        try:
            store.delete(name)
        finally:
            self._evict(name)

    # -- internals -----------------------------------------------------------

    def _require_store(self) -> VoiceStore:
        if self._store is None:
            raise VoiceError(
                "voice_saving_not_supported",
                "This model does not support saved voices")
        return self._store

    def _clone_prompt(self, name: str) -> Any:
        store = self._store
        stored = store.get(name)
        key = (name, stored.identity)
        with self._prompts_lock:
            if key in self._prompts:
                self._prompts.move_to_end(key)
                return self._prompts[key]
        stored, audio = store.get_with_audio(name)
        # Always the (array, sr) tuple: qwen-tts reads a str as a path, URL
        # or base64 payload. Saved voices are VOICE_SAMPLE_RATE (24 kHz);
        # qwen-tts resamples anything else itself.
        prompt = self._tts.create_voice_clone_prompt(
            ref_audio=(audio, stored.sample_rate), ref_text=stored.transcript)
        with self._prompts_lock:
            self._prompts[(name, stored.identity)] = prompt
            self._prompts.move_to_end((name, stored.identity))
            while len(self._prompts) > self._prompt_cache_size:
                self._prompts.popitem(last=False)
        return prompt

    def _stream_talker(self) -> Any:
        """The active fast talker when streaming is possible, else None."""
        if self._split is None:
            return None
        talker = getattr(getattr(self._tts, "model", None), "talker", None)
        fast = getattr(getattr(talker, "generate", None), "__self__", None)
        if isinstance(fast, qwen3_tts_fast.FastTalker) and not fast.disabled:
            return fast
        return None

    def _entry(self, text: str, voice: str | None,
               instructions: str | None) -> tuple[Any, Any]:
        """The qwen-tts call :meth:`synthesize` makes, unrun, and the clone
        prompt's reference codes (None in design mode)."""
        limit = frame_limit(text)
        if self._store is None:
            instruct = (instructions or "").strip() or self._default_instructions
            return (lambda: self._tts.generate_voice_design(
                text=text, instruct=instruct, language=None, max_new_tokens=limit)), None
        prompt = self._clone_prompt(voice or NARRATOR)
        ref = getattr(prompt[0], "ref_code", None) if isinstance(prompt, list) and prompt else None
        return (lambda: self._tts.generate_voice_clone(
            text=text, language=None, voice_clone_prompt=prompt, max_new_tokens=limit)), ref

    def _stream_failed(self, part: str, talker: Any) -> None:
        if part == "talker":
            talker.disable("fast Qwen3-TTS talker failed while streaming; using qwen-tts "
                           "generate() for the rest of this process")
        else:
            logger.warning("Qwen3-TTS streaming decode failed; streams are one chunk "
                           "for the rest of this process", exc_info=True)
            self._split = None

    def _evict(self, name: str) -> None:
        with self._prompts_lock:
            for key in [k for k in self._prompts if k[0] == name]:
                del self._prompts[key]
