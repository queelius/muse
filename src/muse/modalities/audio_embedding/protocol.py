"""Protocol + dataclasses for audio/embedding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class AudioEmbeddingResult:
    """N audio clips in, N embedding vectors out, plus provenance.

    embeddings: list[list[float]] (float32, native dim per row).
                Pure-Python at the protocol boundary; backends may use
                numpy internally and convert via `.tolist()` before
                returning.
    dimensions: vector length (model's native dim).
    model_id: catalog id of the producing model.
    n_audio_clips: count of inputs the runtime processed (for usage
                   roll-up).
    metadata: optional per-call extras (sample_rate_used, source
              backend tag, etc.).
    """

    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    n_audio_clips: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AudioEmbeddingModel(Protocol):
    """Structural protocol any audio embedder backend satisfies.

    AudioEmbeddingRuntime (the generic runtime) and the bundled
    mert-v1-95m Model satisfy this without inheritance.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(
        self,
        audio_bytes_list: list[bytes],
    ) -> AudioEmbeddingResult:
        """Embed a list of raw audio byte strings into vectors.

        Each entry is the bytes of an audio file (wav/mp3/flac/ogg/...)
        decodable by librosa. The runtime resamples to its preferred
        rate on the way in. Output rows preserve input order.
        """
        ...
