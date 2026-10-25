from dataclasses import FrozenInstanceError

import pytest

from muse.modalities.audio_speech.word_timing import TimedSpan, assemble_words


def test_one_entry_per_split_word_measured():
    text = "Coulomb's law"
    words = assemble_words(text, [TimedSpan(0, 9, 0.1, 0.6), TimedSpan(10, 13, 0.62, 0.9)], 1.0)
    assert [w["word"] for w in words] == text.split()
    assert words[0] == {"word": "Coulomb's", "char_start": 0, "char_end": 9, "start": 0.1, "end": 0.6, "measured": True}


def test_respelled_word_spans_all_tokens():
    words = assemble_words("a monoid", [TimedSpan(0, 1, 0.0, 0.1), TimedSpan(2, 8, 0.2, 0.4), TimedSpan(2, 8, 0.4, 0.7)], 1.0)
    assert (words[1]["start"], words[1]["end"], words[1]["measured"]) == (0.2, 0.7, True)


def test_phrase_span_divided_unmeasured():
    words = assemble_words("Runge Kutta", [TimedSpan(0, 11, 1.0, 2.0)], 3.0)
    assert [w["measured"] for w in words] == [False, False]
    assert words[0]["start"] == 1.0 and words[1]["end"] == 2.0 and words[0]["end"] == words[1]["start"]


def test_unmeasured_word_interpolated_between_neighbours():
    words = assemble_words("one two three", [TimedSpan(0, 3, 0.0, 0.5), TimedSpan(8, 13, 1.5, 2.0)], 2.0)
    assert words[1]["measured"] is False and 0.5 <= words[1]["start"] <= words[1]["end"] <= 1.5


def test_words_match_split_with_odd_whitespace():
    text = "  Hello,\n\nworld — ok...  "
    words = assemble_words(text, [], 2.0)
    assert [w["word"] for w in words] == text.split()
    assert all(text[w["char_start"]:w["char_end"]] == w["word"] for w in words)
    assert words[-1]["end"] <= 2.0


def test_times_clamped_to_duration():
    words = assemble_words("hi", [TimedSpan(0, 2, 0.9, 1.4)], 1.0)
    assert words[0]["end"] == 1.0


def timings(words):
    return [(w["start"], w["end"], w["measured"]) for w in words]


@pytest.mark.parametrize("text", ["", " \t\n\u2003"])
def test_empty_input_has_no_words(text):
    assert assemble_words(text, [], 2.0) == []


def test_timed_span_is_frozen():
    span = TimedSpan(0, 1, 0.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        span.start = 2.0


def test_partial_overlap_and_half_open_boundaries():
    words = assemble_words("one two", [
        TimedSpan(3, 4, 0.0, 9.0),  # Whitespace touches both words, overlaps neither.
        TimedSpan(1, 3, 0.2, 0.4),
        TimedSpan(4, 5, 0.6, 0.8),
    ], 1.0)
    assert timings(words) == [(0.2, 0.4, True), (0.6, 0.8, True)]


def test_phrase_division_weights_character_lengths():
    words = assemble_words("a bbb", [TimedSpan(0, 5, 1.0, 3.0)], 4.0)
    assert timings(words) == [(1.0, 1.5, False), (1.5, 3.0, False)]


def test_finer_measurements_win_regardless_of_span_order():
    spans = [TimedSpan(0, 8, 0.0, 2.0), TimedSpan(0, 1, 0.1, 0.2)]
    assert timings(assemble_words("a b cccc", spans, 2.0)) == [
        (0.1, 0.2, True), (0.1, 0.4, False), (0.4, 2.0, False),
    ]
    assert assemble_words("a b cccc", spans, 2.0) == assemble_words("a b cccc", spans[::-1], 2.0)


def test_missing_runs_use_character_lengths_and_audio_edges():
    words = assemble_words("a bbb mid c ddd", [TimedSpan(6, 9, 1.0, 2.0)], 4.0)
    assert timings(words) == [
        (0.0, 0.25, False), (0.25, 1.0, False), (1.0, 2.0, True),
        (2.0, 2.5, False), (2.5, 4.0, False),
    ]


def test_missing_run_uses_next_divided_word_as_boundary():
    words = assemble_words("a bb ccc", [TimedSpan(2, 8, 1.0, 3.0)], 4.0)
    assert timings(words) == [(0.0, 1.0, False), (1.0, 1.8, False), (1.8, 3.0, False)]


def test_no_spans_divides_full_duration_by_character_lengths():
    assert timings(assemble_words("a bbb", [], 2.0)) == [(0.0, 0.5, False), (0.5, 2.0, False)]


def test_times_rounded_clamped_and_starts_non_decreasing():
    words = assemble_words("a b c d", [
        TimedSpan(0, 1, -0.2, 0.1236),
        TimedSpan(2, 3, 0.8004, 0.9),
        TimedSpan(4, 5, 0.3, 0.4),
        TimedSpan(6, 7, 1.2, 1.4),
    ], 1.0)
    assert timings(words) == [(0.0, 0.124, True), (0.8, 0.9, True), (0.8, 0.8, True), (1.0, 1.0, True)]


def test_missing_word_in_reversed_gap_has_valid_times():
    words = assemble_words("a bb c", [TimedSpan(0, 1, 0.7, 0.9), TimedSpan(5, 6, 0.2, 0.3)], 1.0)
    assert timings(words) == [(0.7, 0.9, True), (0.9, 0.9, False), (0.9, 0.9, True)]


def test_zero_duration_has_zero_times():
    assert timings(assemble_words("a bb", [TimedSpan(0, 1, -1.0, 2.0)], 0.0)) == [
        (0.0, 0.0, True), (0.0, 0.0, False),
    ]


def test_rounding_at_duration_stays_inside_audio():
    words = assemble_words("a bb", [TimedSpan(0, 1, 1.2346, 2.0)], 1.2346)
    assert timings(words) == [(1.234, 1.234, True), (1.234, 1.234, False)]


def test_assembly_is_near_linear():
    import time
    text = " ".join(["word"] * 8000)
    spans = [TimedSpan(5 * i, 5 * i + 4, i * 0.01, i * 0.01 + 0.005) for i in range(8000)]
    started = time.perf_counter()
    words = assemble_words(text, spans, 100.0)
    assert time.perf_counter() - started < 2
    assert len(words) == 8000 and all(w["measured"] for w in words)
