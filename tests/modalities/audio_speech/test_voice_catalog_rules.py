"""Voice metadata rules: normalization, limits, entries, filters, categories."""
import pytest

from muse.modalities.audio_speech.voice_catalog import (
    VoiceEntry,
    category_counts,
    entry_from_meta,
    filter_entries,
    normalize_metadata,
)
from muse.modalities.audio_speech.voice_errors import (
    UnknownVoiceError,
    VoiceError,
    VoiceUnreadableError,
)


def _e(name, *, category=None, tags=(), description=None, notes=None):
    return VoiceEntry(
        name=name, transcript="t", sample_rate=24000, duration_seconds=1.0,
        sha256=None, read_only=False, category=category, tags=tuple(tags),
        description=description, notes=notes, design=None,
        created_at=None, updated_at=None,
    )


def test_protocol_reexports_the_same_exception_classes():
    from muse.modalities.audio_speech import protocol

    assert protocol.VoiceError is VoiceError
    assert protocol.UnknownVoiceError is UnknownVoiceError
    assert protocol.VoiceUnreadableError is VoiceUnreadableError


def test_normalize_lowercases_trims_and_strips_slashes():
    out = normalize_metadata(
        {"category": " Narrator/Epic/ ", "tags": " Male , GRAVELLY ,male"},
        partial=True,
    )
    assert out == {"category": "narrator/epic", "tags": ["gravelly", "male"]}


def test_tags_accept_list_or_comma_string():
    assert normalize_metadata({"tags": ["b", "a"]}, partial=True)["tags"] == ["a", "b"]


def test_empty_string_and_none_clear():
    assert normalize_metadata({"category": "", "notes": None}, partial=True) == {
        "category": None, "notes": None}


def test_full_fills_absent_keys():
    assert normalize_metadata({}, partial=False) == {
        "category": None, "tags": [], "description": None, "notes": None,
        "design": None}


def test_unknown_keys_are_ignored():
    assert normalize_metadata({"colour": "red", "notes": "n"}, partial=True) == {
        "notes": "n"}


def test_design_allows_one_half_missing():
    out = normalize_metadata({"design": {"model": None, "instructions": "gruff"}},
                             partial=True)
    assert out == {"design": {"model": None, "instructions": "gruff"}}


@pytest.mark.parametrize("fields,needle", [
    ({"category": "a/b/c/d/e"}, "category"),
    ({"category": "bad seg"}, "category"),
    ({"category": "a//b"}, "category"),
    ({"tags": [f"t{i}" for i in range(17)]}, "tags"),
    ({"tags": ["no spaces allowed"]}, "tags"),
    ({"tags": [1]}, "tags"),
    ({"description": "x" * 501}, "description"),
    ({"notes": "x" * 2001}, "notes"),
    ({"notes": 5}, "notes"),
    ({"design": {"model": "m", "instructions": "x" * 4097}}, "design"),
    ({"design": {"model": "m" * 257, "instructions": "i"}}, "design"),
    ({"design": {"model": "m", "extra": 1}}, "design"),
    ({"design": {"model": None, "instructions": "  "}}, "design"),
    ({"design": "text"}, "design"),
])
def test_rule_violations_raise_invalid_voice_metadata(fields, needle):
    with pytest.raises(VoiceError) as e:
        normalize_metadata(fields, partial=True)
    assert e.value.code == "invalid_voice_metadata"
    assert e.value.status == 400
    assert needle in str(e.value)


def test_entry_from_old_meta_has_null_metadata():
    e = entry_from_meta("old", {
        "name": "old", "transcript": "t", "sample_rate": 24000,
        "duration_seconds": 2.0, "sha256": "ab"}, read_only=False)
    assert (e.category, e.tags, e.description, e.design, e.updated_at) == (
        None, (), None, None, None)
    assert e.sha256 == "ab"


def test_entry_from_meta_tolerates_malformed_optional_fields():
    e = entry_from_meta("x", {
        "transcript": "t", "sample_rate": 24000, "duration_seconds": 1,
        "category": 7, "tags": "notalist", "design": [1]}, read_only=True)
    assert (e.category, e.tags, e.design, e.read_only) == (None, (), None, True)


def test_entry_from_meta_missing_required_field_is_unreadable():
    with pytest.raises(VoiceUnreadableError):
        entry_from_meta("x", {"sample_rate": 24000}, read_only=False)


def test_entry_to_dict_wire_shape():
    d = _e("grim", category="npc/dwarf", tags=("male",)).to_dict("qwen3-tts-0.6b-base")
    assert d["object"] == "audio.voice"
    assert d["model"] == "qwen3-tts-0.6b-base"
    assert d["name"] == "grim" and d["tags"] == ["male"]
    assert d["audio_url"] == (
        "/v1/audio/speech/voices/grim/audio?model=qwen3-tts-0.6b-base")
    for key in ("transcript", "duration_seconds", "sample_rate", "sha256",
                "read_only", "category", "description", "notes", "design",
                "created_at", "updated_at"):
        assert key in d


def test_filter_category_matches_subtree_tags_and_q():
    es = [_e("a", category="narrator/epic", tags=("deep",)),
          _e("b", category="narrator"),
          _e("c", category="npc/guard", description="Gruff city guard"),
          _e("d", category="narratorish")]
    assert [e.name for e in filter_entries(es, category="narrator", tags=[], q=None)] == ["a", "b"]
    assert [e.name for e in filter_entries(es, category="Narrator/", tags=[], q=None)] == ["a", "b"]
    assert [e.name for e in filter_entries(es, category=None, tags=["deep"], q=None)] == ["a"]
    assert [e.name for e in filter_entries(es, category=None, tags=[], q="GRUFF")] == ["c"]
    assert [e.name for e in filter_entries(es, category=None, tags=["deep", "x"], q=None)] == []


def test_category_counts_include_ancestors():
    es = [_e("a", category="narrator/epic"), _e("b", category="narrator/epic"),
          _e("c", category="narrator"), _e("d")]
    assert category_counts(es) == [
        {"path": "narrator", "count": 3}, {"path": "narrator/epic", "count": 2}]
