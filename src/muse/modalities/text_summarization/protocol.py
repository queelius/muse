"""Protocol + dataclasses for text/summarization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SummarizationResult:
    """One summary produced by a `text/summarization` model.

    summary: the produced summary text (paragraph or bullet shape;
             see format).
    length: the length budget used ("short"|"medium"|"long"). Echoed
            so a client can confirm the runtime honored its request.
    format: the format requested ("paragraph"|"bullets"). For BART-CNN
            and similar non-instruction models this is metadata only;
            the produced summary is whatever the model gave.
    model_id: catalog id of the model that produced this summary.
    prompt_tokens: input token count (post-truncation when applicable).
    completion_tokens: output token count.
    metadata: optional per-call extras the runtime wants surfaced
              (e.g. truncation_warning, language detected).
    """

    summary: str
    length: str
    format: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class SummarizationModel(Protocol):
    """Structural protocol any summarizer backend satisfies.

    BartSeq2SeqRuntime (the generic runtime) and the bundled
    bart-large-cnn Model satisfy this without inheritance.
    """

    def summarize(
        self,
        text: str,
        length: str = "medium",
        format: str = "paragraph",
    ) -> SummarizationResult:
        """Produce a summary for `text`.

        length controls max_new_tokens; format is metadata for non-
        instruction models, instructional for instruction-tuned ones.
        """
        ...
