"""Assemble written-word timings from measured character spans."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TimedSpan:
    char_start: int
    char_end: int
    start: float
    end: float


def assemble_words(text: str, spans: list[TimedSpan], duration: float) -> list[dict]:
    """Prefer single-word measurements, then divide coarse spans and gaps."""
    words = [
        {"word": match.group(), "char_start": match.start(), "char_end": match.end(),
         "measured": False}
        for match in re.finditer(r"\S+", text)
    ]
    ends = [word["char_end"] for word in words]
    times: list[tuple[float, float] | None] = [None] * len(words)
    coarse = []

    def include(index: int, start: float, end: float) -> None:
        previous = times[index]
        times[index] = (start, end) if previous is None else (
            min(previous[0], start), max(previous[1], end)
        )

    def divide(indices: list[int], start: float, end: float) -> None:
        length = sum(len(words[index]["word"]) for index in indices)
        used = 0
        for index in indices:
            left = start + (end - start) * used / length
            used += len(words[index]["word"])
            right = start + (end - start) * used / length
            include(index, left, right)

    for span in spans:
        # Words are sorted and disjoint, so overlap is a contiguous run.
        first = bisect_right(ends, span.char_start)
        last = first
        while last < len(words) and words[last]["char_start"] < span.char_end:
            last += 1
        overlapping = list(range(first, last)) if span.char_start < span.char_end else []
        if len(overlapping) == 1:
            index = overlapping[0]
            include(index, span.start, span.end)
            words[index]["measured"] = True
        elif overlapping:
            coarse.append((span, overlapping))

    # Collect all finer measurements before considering any phrase span.
    for span, overlapping in coarse:
        indices = [index for index in overlapping if not words[index]["measured"]]
        divide(indices, span.start, span.end)

    index = 0
    while index < len(words):
        if times[index] is not None:
            index += 1
            continue
        first = index
        while index < len(words) and times[index] is None:
            index += 1
        start = times[first - 1][1] if first else 0.0
        end = times[index][0] if index < len(words) else duration
        divide(list(range(first, index)), start, max(start, end))

    # Use the last representable millisecond if rounding would exceed the audio.
    rounded_duration = round(duration, 3)
    if rounded_duration > duration:
        rounded_duration = round(rounded_duration - 0.001, 3)

    def normalize(value: float) -> float:
        return min(rounded_duration, round(max(0.0, min(duration, value)), 3))

    previous_start = 0.0
    for word, (start, end) in zip(words, times):
        start = max(previous_start, normalize(start))
        end = max(start, normalize(end))
        word.update(start=start, end=end)
        previous_start = start
    return words
