"""Protocol + dataclasses for image/embedding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ImageEmbeddingResult:
    """N images in, N embedding vectors out, plus provenance.

    embeddings: list[list[float]] (float32, native dim per row).
                Pure-Python at the protocol boundary; backends may use
                numpy internally and convert via `.tolist()` before
                returning.
    dimensions: vector length after any matryoshka truncation; equals
                the model's native dimensionality when no truncation
                was applied.
    model_id: catalog id of the producing model.
    n_images: count of inputs the runtime processed (for usage roll-up).
    metadata: optional per-call extras (truncation_warning, source
              backend tag, etc.).
    """

    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    n_images: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageEmbeddingModel(Protocol):
    """Structural protocol any image embedder backend satisfies.

    ImageEmbeddingRuntime (the generic runtime) and the bundled
    dinov2-small Model satisfy this without inheritance.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(
        self,
        images: list,
        *,
        dimensions: int | None = None,
    ) -> ImageEmbeddingResult:
        """Encode a list of PIL images into vectors.

        `images` is always a list (the route layer wraps single-image
        inputs into a one-element list before calling). `dimensions`
        triggers matryoshka-style truncation when smaller than the
        model's native dim; backends re-normalize after slicing.
        """
        ...
