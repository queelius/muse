"""Validate pronunciation overrides and map spoken characters to written input."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Literal

MAX_PRONUNCIATIONS = 2000
MAX_SPANS = 2000
_IPA_FORBIDDEN = frozenset("()[]/\n\r")
MAX_TERM_CHARS = 200
MAX_REPLACEMENT_CHARS = 500


class PronunciationError(ValueError):
    def __init__(self, message: str, index: int | None = None) -> None:
        self.index = index
        super().__init__(message if index is None else f"Pronunciation entry {index}: {message}")


@dataclass(frozen=True)
class Pronunciation:
    index: int
    term: str | None
    spans: tuple[tuple[int, int], ...] | None
    ipa: str | None
    respell: str | None

    @property
    def kind(self) -> Literal["ipa", "respell"]:
        return "ipa" if self.ipa is not None else "respell"


def parse_pronunciations(raw: object, text: str) -> list[Pronunciation]:
    """Validate the request list, retaining entry order and original values."""
    if not isinstance(raw, list):
        raise PronunciationError("pronunciations must be a list")
    if len(raw) > MAX_PRONUNCIATIONS:
        raise PronunciationError(f"at most {MAX_PRONUNCIATIONS} pronunciations are allowed")

    entries = []
    occupied: list[tuple[int, int]] = []  # sorted by start, never overlapping
    span_total = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PronunciationError("must be an object", index)
        if item.keys() - {"term", "spans", "ipa", "respell"}:
            raise PronunciationError("unknown fields", index)
        if ("term" in item) == ("spans" in item):
            raise PronunciationError("specify exactly one of term or spans", index)
        if ("ipa" in item) == ("respell" in item):
            raise PronunciationError("specify exactly one of ipa or respell", index)

        for key, limit in (("term", MAX_TERM_CHARS), ("ipa", MAX_REPLACEMENT_CHARS),
                           ("respell", MAX_REPLACEMENT_CHARS)):
            if key in item:
                value = item[key]
                if not isinstance(value, str) or not value.strip() or len(value) > limit:
                    raise PronunciationError(f"{key} must be a nonblank string of at most {limit} characters", index)

        if "ipa" in item and _IPA_FORBIDDEN & set(item["ipa"]):
            raise PronunciationError(
                "ipa must not contain parentheses, brackets, slashes or newlines", index)

        spans = None
        if "spans" in item:
            ranges = item["spans"]
            if not isinstance(ranges, list) or not ranges:
                raise PronunciationError("spans must be a nonempty list of ranges", index)
            span_total += len(ranges)
            if span_total > MAX_SPANS:
                raise PronunciationError(f"at most {MAX_SPANS} spans are allowed in total", index)
            parsed = []
            for span in ranges:
                if (not isinstance(span, list) or len(span) != 2
                        or any(type(offset) is not int for offset in span)):
                    raise PronunciationError("each span must contain two integer offsets", index)
                start, end = span
                if not 0 <= start < end <= len(text):
                    raise PronunciationError("span must satisfy 0 <= start < end <= len(input)", index)
                at = bisect_right(occupied, (start, end))
                if (at and occupied[at - 1][1] > start) or (
                        at < len(occupied) and occupied[at][0] < end):
                    raise PronunciationError("spans must not overlap", index)
                occupied.insert(at, (start, end))
                parsed.append((start, end))
            spans = tuple(parsed)
        entries.append(Pronunciation(index, item.get("term"), spans, item.get("ipa"), item.get("respell")))
    return entries


def pronunciations_digest(raw: list) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Replacement:
    char_start: int
    char_end: int
    entry: Pronunciation


def match_pronunciations(text: str, entries: list[Pronunciation]) -> tuple[list[Replacement], list[dict]]:
    """Match once from left to right; explicit spans suppress overlapping terms."""
    replacements = [Replacement(start, end, entry) for entry in entries
                    for start, end in (entry.spans or ())]
    spans = sorted((rep.char_start, rep.char_end) for rep in replacements)
    span_ends = [end for _, end in spans]
    counts = {entry.index: len(entry.spans or ()) for entry in entries}
    terms = [entry for entry in entries if entry.term is not None]
    if terms:
        # Keep the pipeline's alternation order and normalized lookup semantics.
        ordered = sorted(terms, key=lambda entry: len(entry.term), reverse=True)
        alternation = "|".join(
            "(?:" + r"\s+".join(re.escape(word) for word in entry.term.split()) + ")"
            for entry in ordered
        )
        pattern = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
        lookup = {re.sub(r"\s+", " ", entry.term).lower(): entry for entry in terms}
        for match in pattern.finditer(text):
            at = bisect_right(span_ends, match.start())
            if at < len(spans) and spans[at][0] < match.end():
                continue
            entry = lookup.get(re.sub(r"\s+", " ", match.group()).lower())
            if entry is not None:
                replacements.append(Replacement(match.start(), match.end(), entry))
                counts[entry.index] += 1

    applied = []
    for entry in entries:
        target = ({"term": entry.term} if entry.term is not None else
                  {"spans": [list(span) for span in entry.spans]})
        applied.append({"index": entry.index, **target, "kind": entry.kind, "count": counts[entry.index]})
    return sorted(replacements, key=lambda rep: rep.char_start), applied


@dataclass
class SpokenText:
    text: str
    _input_spans: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def to_input_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a nonempty spoken range to the full input range it covers."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError("spoken span must satisfy 0 <= start < end <= len(text)")
        if not self._input_spans:
            return start, end
        return self._input_spans[start][0], self._input_spans[end - 1][1]


def render_spoken(
    text: str,
    replacements: list[Replacement],
    render_ipa: Callable[[str, str], str] | None,
) -> SpokenText:
    """Render sorted matches, mapping each inserted character to its whole target."""
    parts = []
    input_spans = []
    cursor = 0
    for replacement in replacements:
        start, end, entry = replacement.char_start, replacement.char_end, replacement.entry
        parts.append(text[cursor:start])
        input_spans.extend((i, i + 1) for i in range(cursor, start))
        if entry.ipa is not None:
            if render_ipa is None:
                raise PronunciationError("IPA pronunciation requires a renderer", entry.index)
            original = text[start:end]
            if "[" in original or "]" in original:
                raise PronunciationError(
                    "IPA cannot be applied to text containing square brackets", entry.index)
            # Newlines would split the link before the pipeline parses it.
            spoken = render_ipa(re.sub(r"\s+", " ", original), entry.ipa)
        else:
            spoken = entry.respell
        parts.append(spoken)
        input_spans.extend([(start, end)] * len(spoken))
        cursor = end
    parts.append(text[cursor:])
    input_spans.extend((i, i + 1) for i in range(cursor, len(text)))
    return SpokenText("".join(parts), input_spans)
