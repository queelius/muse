import pytest

from muse.modalities.audio_speech.pronunciations import (
    MAX_PRONUNCIATIONS,
    MAX_REPLACEMENT_CHARS,
    MAX_TERM_CHARS,
    PronunciationError,
    match_pronunciations,
    parse_pronunciations,
    pronunciations_digest,
    render_spoken,
)


def test_term_whole_word_and_counts():
    entries = parse_pronunciations([{"term": "json", "respell": "JAY-sawn"}], "json and JSON, not jsonl")
    reps, applied = match_pronunciations("json and JSON, not jsonl", entries)
    assert [(r.char_start, r.char_end) for r in reps] == [(0, 4), (9, 13)]
    assert applied == [{"index": 0, "term": "json", "kind": "respell", "count": 2}]

def test_longest_term_first_and_flexible_whitespace():
    raw = [{"term": "kutta", "respell": "KOO"}, {"term": "runge kutta", "respell": "Roong-uh Koo-tuh"}]
    text = "the Runge  Kutta method, then kutta"
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert [(text[r.char_start:r.char_end], r.entry.index) for r in reps] == [("Runge  Kutta", 1), ("kutta", 0)]
    assert [a["count"] for a in applied] == [1, 1]

def test_span_wins_over_overlapping_term():
    text = "a cat, not a dog"
    raw = [{"term": "a", "respell": "uh"}, {"spans": [[11, 12]], "respell": "eigh"}]
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert [(r.char_start, r.entry.index) for r in reps] == [(0, 0), (11, 1)]
    assert applied[0]["count"] == 1 and applied[1] == {"index": 1, "spans": [[11, 12]], "kind": "respell", "count": 1}

def test_unmatched_entry_reported_with_zero():
    text = "hello"
    _, applied = match_pronunciations(text, parse_pronunciations([{"term": "tanh", "respell": "hyperbolic tangent"}], text))
    assert applied[0]["count"] == 0

@pytest.mark.parametrize("raw", [
    [{"respell": "x"}], [{"term": "a", "spans": [[0, 1]], "respell": "x"}],
    [{"term": "a"}], [{"term": "a", "ipa": "x", "respell": "y"}],
    [{"spans": [[0, 3], [2, 4]], "respell": "x"}], [{"spans": [[0, 99]], "respell": "x"}],
    [{"spans": [[0, 1]], "respell": "x"}, {"spans": [[0, 2]], "respell": "y"}],
    [{"term": "a", "respell": "x", "extra": 1}],
])
def test_invalid_entries_name_index(raw):
    with pytest.raises(PronunciationError) as e:
        parse_pronunciations(raw, "abcdef")
    assert e.value.index == len(raw) - 1

def test_rejects_hostile_lists():
    for raw in ({"term": "a"}, [{"term": "a", "respell": "b"}] * (MAX_PRONUNCIATIONS + 1),
                [{"term": "x" * (MAX_TERM_CHARS + 1), "respell": "b"}],
                [{"spans": [[0.5, 1]], "respell": "b"}], [{"spans": [[-1, 1]], "respell": "b"}],
                [{"term": "   ", "respell": "b"}], [{"spans": [[1, 1]], "respell": "b"}]):
        with pytest.raises(PronunciationError):
            parse_pronunciations(raw, "abcdef")

def test_digest_is_canonical():
    a = [{"term": "x", "respell": "y"}]
    b = [{"respell": "y", "term": "x"}]
    assert pronunciations_digest(a) == pronunciations_digest(b)
    assert len(pronunciations_digest(a)) == 64

def test_render_spoken_maps_back():
    text = "Coulomb's law and monoid"
    raw = [{"term": "monoid", "respell": "mah noyd"}, {"spans": [[0, 7]], "ipa": "kuːˈlɒm"}]
    reps, _ = match_pronunciations(text, parse_pronunciations(raw, text))
    spoken = render_spoken(text, reps, lambda orig, ipa: f"[{orig}](/{ipa}/)")
    assert spoken.text == "[Coulomb](/kuːˈlɒm/)'s law and mah noyd"
    assert spoken.to_input_span(1, 8) == (0, 7)
    i = spoken.text.index("noyd")
    assert spoken.to_input_span(i, i + 4) == (18, 24)
    j = spoken.text.index("law")
    assert spoken.to_input_span(j, j + 3) == (10, 13)

def test_render_spoken_ipa_without_renderer_raises():
    text = "tanh"
    reps, _ = match_pronunciations(text, parse_pronunciations([{"term": "tanh", "ipa": "tæn"}], text))
    with pytest.raises(PronunciationError):
        render_spoken(text, reps, None)


@pytest.mark.parametrize("raw, index", [
    (None, None), ((), None),
    ([{}] * (MAX_PRONUNCIATIONS + 1), None),
    ([None], 0), ([42], 0),
    ([{"term": None, "respell": "x"}], 0),
    ([{"term": 1, "respell": "x"}], 0),
    ([{"term": "a", "respell": None}], 0),
    ([{"term": "a", "respell": ""}], 0),
    ([{"term": "a", "ipa": "  "}], 0),
    ([{"term": "a", "ipa": 12}], 0),
    ([{"term": "a", "respell": "x" * (MAX_REPLACEMENT_CHARS + 1)}], 0),
    ([{"term": "a", "ipa": "x" * (MAX_REPLACEMENT_CHARS + 1)}], 0),
    ([{"spans": None, "respell": "x"}], 0),
    ([{"spans": [], "respell": "x"}], 0),
    ([{"spans": [None], "respell": "x"}], 0),
    ([{"spans": [[0]], "respell": "x"}], 0),
    ([{"spans": [[0, 1, 2]], "respell": "x"}], 0),
    ([{"spans": [[False, 1]], "respell": "x"}], 0),
    ([{"spans": [[0, True]], "respell": "x"}], 0),
    ([{"spans": [[2, 1]], "respell": "x"}], 0),
])
def test_validation_error_index(raw, index):
    with pytest.raises(PronunciationError) as exc:
        parse_pronunciations(raw, "abc")
    assert exc.value.index == index


def test_valid_entries_preserve_values_and_limits():
    raw = [
        {"term": "x" * MAX_TERM_CHARS, "ipa": "ə" * MAX_REPLACEMENT_CHARS},
        {"spans": [[2, 3], [0, 2]], "respell": " spoken "},
    ]
    entries = parse_pronunciations(raw, "abc")
    assert entries[0].term == raw[0]["term"]
    assert entries[0].ipa == raw[0]["ipa"]
    assert entries[0].kind == "ipa"
    assert entries[1].spans == ((2, 3), (0, 2))
    assert entries[1].respell == " spoken "
    assert entries[1].kind == "respell"
    assert len(parse_pronunciations([{"term": "a", "respell": "b"}] * MAX_PRONUNCIATIONS, "a")) == MAX_PRONUNCIATIONS


def test_single_pass_does_not_chain_and_preserves_word_boundaries():
    text = "json JSON jsonl _json json_ éjson jsoné json's"
    raw = [{"term": "json", "respell": "code"}, {"term": "code", "respell": "other"}]
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert render_spoken(text, reps, None).text == "code code jsonl _json json_ éjson jsoné code's"
    assert [entry["count"] for entry in applied] == [3, 0]


def test_phrase_whitespace_and_literal_regex_characters():
    text = "Runge\n\tKutta and a.b axb"
    raw = [{"term": "runge kutta", "respell": "method"}, {"term": "a.b", "respell": "literal"}]
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert render_spoken(text, reps, None).text == "method and literal axb"
    assert [entry["count"] for entry in applied] == [1, 1]


def test_span_suppresses_entire_phrase_without_shorter_term_fallback():
    text = "runge kutta"
    raw = [
        {"term": "runge kutta", "respell": "method"},
        {"term": "kutta", "respell": "short"},
        {"spans": [[0, 1]], "respell": "R"},
    ]
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert [(r.char_start, r.char_end) for r in reps] == [(0, 1)]
    assert [entry["count"] for entry in applied] == [0, 0, 1]


def test_adjacent_spans_all_apply_in_input_order():
    text = "abcdef"
    raw = [{"spans": [[4, 6], [0, 2]], "respell": "X"}, {"spans": [[2, 4]], "respell": "Y"}]
    reps, applied = match_pronunciations(text, parse_pronunciations(raw, text))
    assert [(r.char_start, r.char_end) for r in reps] == [(0, 2), (2, 4), (4, 6)]
    assert applied == [
        {"index": 0, "spans": [[4, 6], [0, 2]], "kind": "respell", "count": 2},
        {"index": 1, "spans": [[2, 4]], "kind": "respell", "count": 1},
    ]
    spoken = render_spoken(text, reps, None)
    assert spoken.text == "XYX"
    assert spoken.to_input_span(0, 3) == (0, 6)
    assert spoken.to_input_span(1, 2) == (2, 4)


def test_empty_list_and_identity_mapping():
    assert parse_pronunciations([], "") == []
    assert match_pronunciations(" hello ", []) == ([], [])
    spoken = render_spoken(" hello ", [], None)
    assert spoken.text == " hello "
    assert spoken.to_input_span(1, 6) == (1, 6)
    assert render_spoken("", [], None).text == ""


def test_map_expansion_contraction_and_unicode():
    text = "α cat dog!"
    raw = [{"term": "cat", "respell": "kit ten"}, {"term": "dog", "respell": "D"}]
    reps, _ = match_pronunciations(text, parse_pronunciations(raw, text))
    spoken = render_spoken(text, reps, None)
    assert spoken.text == "α kit ten D!"
    assert spoken.to_input_span(3, 4) == (2, 5)
    assert spoken.to_input_span(10, 11) == (6, 9)
    assert spoken.to_input_span(0, 1) == (0, 1)
    assert spoken.to_input_span(4, 12) == (2, 10)


def test_ipa_renderer_receives_exact_original_and_phonemes():
    text = "JSON"
    reps, _ = match_pronunciations(text, parse_pronunciations([{"term": "json", "ipa": "dʒ"}], text))
    calls = []

    def renderer(original, ipa):
        calls.append((original, ipa))
        return "spoken"

    spoken = render_spoken(text, reps, renderer)
    assert calls == [("JSON", "dʒ")]
    assert spoken.text == "spoken"
    assert spoken.to_input_span(0, 1) == (0, 4)


def test_digest_unicode_and_list_order():
    assert pronunciations_digest([{"term": "é", "ipa": "ə"}]) == "41e7c65b67bed61fb5113b5b888b1f3705186467a0164764d9f89799a60ba049"
    a = {"term": "a", "respell": "b"}
    b = {"term": "c", "respell": "d"}
    assert pronunciations_digest([a, b]) != pronunciations_digest([b, a])
