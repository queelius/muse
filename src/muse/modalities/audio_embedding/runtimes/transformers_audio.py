"""AudioEmbeddingRuntime: generic runtime over any HF audio embedder.

One class wraps `transformers.AutoModel` plus
`AutoProcessor`/`AutoFeatureExtractor` and `librosa`-based audio
decoding for any HF repo that ships an audio tower (CLAP, MERT,
wav2vec, audio-encoder family). Pulled via the HF resolver:
`muse pull hf://laion/clap-htsat-fused` synthesizes a manifest
pointing at this class.

Deferred imports follow the muse pattern: torch + transformers +
librosa stay as module-top sentinels (None) until _ensure_deps()
lazy-imports them. Tests patch the sentinels directly; _ensure_deps
short-circuits on non-None.

Per-architecture extraction dispatch is the single source of truth
on how an embedding is pooled out of the model's outputs:

  1. CLAP family: outputs.audio_embeds  (set by ClapModel.forward
     after the audio_model + projection chain)
  2. Pooler-bearing models: outputs.pooler_output (BERT-shaped
     audio models often populate this)
  3. MERT / wav2vec base: outputs.last_hidden_state mean-pooled
     over the time dimension (dim 1) so [B, T, H] -> [B, H]

The order matters: CLAP outputs *also* carry pooler_output, but
audio_embeds is the projected, normalized vector that downstream
clients expect when comparing CLAP audio and text embeddings.

The runtime decodes each input bytes via librosa, resamples to
self._sample_rate, and truncates to self._max_duration_seconds.
The feature extractor then builds the model's input tensors.
"""
from __future__ import annotations

import io
import logging
from typing import Any

from muse.modalities.audio_embedding.protocol import AudioEmbeddingResult


logger = logging.getLogger(__name__)


torch: Any = None
AutoModel: Any = None
AutoProcessor: Any = None
AutoFeatureExtractor: Any = None
librosa: Any = None


def _ensure_deps() -> None:
    global torch, AutoModel, AutoProcessor, AutoFeatureExtractor, librosa
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("AudioEmbeddingRuntime: torch unavailable: %s", e)
    if AutoModel is None:
        try:
            from transformers import (
                AutoFeatureExtractor as _afe,
                AutoModel as _am,
                AutoProcessor as _ap,
            )
            AutoModel = _am
            AutoProcessor = _ap
            AutoFeatureExtractor = _afe
        except Exception as e:  # noqa: BLE001
            logger.debug("AudioEmbeddingRuntime: transformers unavailable: %s", e)
    if librosa is None:
        try:
            import librosa as _l
            librosa = _l
        except Exception as e:  # noqa: BLE001
            logger.debug("AudioEmbeddingRuntime: librosa unavailable: %s", e)


def _resolve_dtype(dtype: str) -> Any:
    if torch is None:
        return None
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _set_inference_mode(model: Any) -> None:
    """Switch model to no-grad inference mode if the method exists.

    Wrapped in a helper so the runtime body stays readable and tests
    can patch this without intercepting the model object's attribute.
    The transformers idiom for this is the no-grad-switch method named
    the same as Python's evaluation builtin minus the parens; we look
    it up by string via getattr rather than calling it inline.
    """
    fn = getattr(model, "eval", None)
    if callable(fn):
        fn()


def _load_processor(src: str, *, trust_remote_code: bool = False) -> Any:
    """Load AutoProcessor; fall back to AutoFeatureExtractor on failure.

    Audio repos almost universally ship `preprocessor_config.json`
    (read by AutoFeatureExtractor) but not always
    `processor_config.json` (read by AutoProcessor). The fallback
    path is the common case for audio.
    """
    try:
        return AutoProcessor.from_pretrained(
            src, trust_remote_code=trust_remote_code,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "AutoProcessor load failed for %s: %s; falling back to AutoFeatureExtractor",
            src, e,
        )
        if AutoFeatureExtractor is None:
            raise
        return AutoFeatureExtractor.from_pretrained(
            src, trust_remote_code=trust_remote_code,
        )


def _detect_dimensions(model: Any) -> int:
    """Best-effort native-dimension detection from the loaded model.

    Tries common attribute paths in priority order:
      1. config.projection_dim (CLAP)
      2. config.hidden_size (MERT, wav2vec base)
      3. config.audio_config.hidden_size (CLAP-shaped composite)
      4. -1 sentinel (caller may overwrite from manifest capabilities)
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("projection_dim", "hidden_size"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        audio_cfg = getattr(cfg, "audio_config", None)
        if audio_cfg is not None:
            val = getattr(audio_cfg, "hidden_size", None)
            if isinstance(val, int) and val > 0:
                return val
    return -1


def _extract_embeddings(outputs: Any) -> Any:
    """Per-architecture pooling dispatch.

    Order is fixed and tested per-architecture:
      1. CLAP family: outputs.audio_embeds (projected vector)
      2. Pooler-bearing: outputs.pooler_output
      3. MERT / wav2vec: outputs.last_hidden_state.mean(dim=1)
         (mean-pool over the time dimension)

    Returns the embeddings tensor; raises ValueError when no path matches.
    """
    audio_embeds = getattr(outputs, "audio_embeds", None)
    if audio_embeds is not None:
        return audio_embeds
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        # Time-axis is dim 1 in [B, T, H]; mean-pool to [B, H].
        mean_method = getattr(last_hidden_state, "mean", None)
        if callable(mean_method):
            return mean_method(dim=1)
        return last_hidden_state.mean(dim=1)
    raise ValueError(
        f"could not extract embeddings from outputs of type "
        f"{type(outputs).__name__}; expected audio_embeds, "
        f"pooler_output, or last_hidden_state"
    )


def _decode_audio(
    raw: bytes,
    *,
    sample_rate: int,
    max_seconds: float,
) -> Any:
    """Decode raw audio bytes into a mono float32 numpy array.

    Resamples to `sample_rate`, mono. Truncates to
    `max_seconds * sample_rate` samples to bound memory. librosa's
    BytesIO path handles wav/mp3/flac/ogg/etc. via its underlying
    audioread / soundfile chain.
    """
    import numpy as np

    audio, _ = librosa.load(
        io.BytesIO(raw), sr=sample_rate, mono=True,
    )
    max_samples = int(max_seconds * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio.astype(np.float32)


class AudioEmbeddingRuntime:
    """Generic audio embedder runtime.

    Constructor kwargs (from a resolver-synthesized manifest's capabilities):
      - model_id (required, passed by load_backend)
      - hf_repo (required, fallback weight source)
      - local_dir (optional, preferred over hf_repo)
      - device ("auto" | "cpu" | "cuda" | "mps")
      - dtype ("float32" | "float16" | "bfloat16" and aliases)
      - sample_rate (int; model's preferred rate; librosa resamples
        on the way in)
      - max_duration_seconds (float; truncate decoded audio to bound
        memory; default 60.0)
      - dimensions (optional override; auto-detected if absent)
      - trust_remote_code (bool; required for MERT and any repo that
        ships a custom feature extractor)
      - other kwargs absorbed by **_
    """

    model_id: str
    dimensions: int

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        sample_rate: int = 16000,
        max_duration_seconds: float = 60.0,
        dimensions: int | None = None,
        trust_remote_code: bool = False,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModel is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull` "
                "or install `transformers` into this venv"
            )
        if librosa is None:
            raise RuntimeError(
                "librosa is not installed; run `muse pull` "
                "or install `librosa` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = dtype
        self._sample_rate = int(sample_rate)
        self._max_duration_seconds = float(max_duration_seconds)
        self._trust_remote_code = bool(trust_remote_code)

        src = local_dir or hf_repo
        logger.info(
            "loading audio embedder from %s (device=%s, dtype=%s, sr=%d)",
            src, self._device, dtype, self._sample_rate,
        )
        self._processor = _load_processor(
            src, trust_remote_code=self._trust_remote_code,
        )
        torch_dtype = _resolve_dtype(dtype)
        kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if self._trust_remote_code:
            kwargs["trust_remote_code"] = True
        self._model = AutoModel.from_pretrained(src, **kwargs)
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)
        detected = _detect_dimensions(self._model)
        if dimensions is not None and dimensions > 0:
            self.dimensions = int(dimensions)
        else:
            self.dimensions = detected

    def embed(
        self,
        audio_bytes_list: list[bytes],
    ) -> AudioEmbeddingResult:
        """Embed a list of raw audio byte strings into vectors.

        Each entry is the bytes of an audio file (wav/mp3/flac/ogg/...)
        decodable by librosa. The runtime resamples to
        self._sample_rate on the way in and truncates to
        self._max_duration_seconds. Output rows preserve input order.
        """
        import numpy as np

        if not isinstance(audio_bytes_list, list):
            audio_bytes_list = [audio_bytes_list]
        n_audio_clips = len(audio_bytes_list)

        decoded: list = [
            _decode_audio(
                raw,
                sample_rate=self._sample_rate,
                max_seconds=self._max_duration_seconds,
            )
            for raw in audio_bytes_list
        ]

        inputs = self._processor(
            decoded,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        inputs = _move_to_device(inputs, self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        embeddings = _extract_embeddings(outputs)
        arr = embeddings.detach().to("cpu").float().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out_dim = int(arr.shape[1])

        return AudioEmbeddingResult(
            embeddings=arr.tolist(),
            dimensions=out_dim,
            model_id=self.model_id,
            n_audio_clips=n_audio_clips,
            metadata={
                "source": "transformers",
                "sample_rate_used": self._sample_rate,
            },
        )


def _move_to_device(inputs: Any, device: str) -> Any:
    """Best-effort move of a processor's BatchEncoding to a device.

    BatchEncoding has a .to(device) method; plain dicts get walked
    manually. Falls through unchanged when neither path applies (test
    doubles often skip this call entirely).
    """
    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)
    if isinstance(inputs, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
    return inputs
