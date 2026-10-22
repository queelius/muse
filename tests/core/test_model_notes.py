"""Tests for muse.core.model_notes: agent-facing summary/strengths/limits/tier."""
from importlib.resources import files
from unittest.mock import MagicMock, patch

import pytest

from muse.core.catalog_models import get_manifest, known_models
from muse.core.curated import (
    _entry_from_dict,
    _reset_curated_cache_for_tests,
    load_curated,
)
from muse.core.model_notes import QUALITY_TIERS, ModelNotes, model_notes


@pytest.fixture(autouse=True)
def _reset_cache():
    _reset_curated_cache_for_tests()
    yield
    _reset_curated_cache_for_tests()


def _patch_yaml(yaml_text: str):
    fake_resource = MagicMock()
    fake_resource.read_text.return_value = yaml_text
    return patch("muse.core.curated._curated_yaml_path", return_value=fake_resource)


_CURATED_WITH_NOTES = """
- id: uri-model
  uri: hf://org/uri-model
  modality: image/generation
  description: curated description
  summary: Fast stills for drafts
  strengths: [fast, small]
  limits: [square only]
  quality_tier: draft
"""


def test_quality_tiers_are_the_three_documented_values():
    assert QUALITY_TIERS == frozenset({"best", "good", "draft"})


def test_manifest_notes_win():
    manifest = {
        "description": "manifest description",
        "summary": "Narration voice",
        "strengths": ["24 kHz"],
        "limits": ["no word timings"],
        "quality_tier": "good",
    }
    with _patch_yaml(_CURATED_WITH_NOTES):
        n = model_notes("uri-model", manifest)
    assert n == ModelNotes("Narration voice", ("24 kHz",), ("no word timings",), "good")


def test_curated_notes_used_for_uri_models():
    # A resolver-synthesized manifest carries no note fields.
    manifest = {"description": "synthesized description", "capabilities": {}}
    with _patch_yaml(_CURATED_WITH_NOTES):
        n = model_notes("uri-model", manifest)
    assert n == ModelNotes("Fast stills for drafts", ("fast", "small"), ("square only",), "draft")


def test_description_fallback():
    n = model_notes("x", {"description": "d"})
    assert n == ModelNotes("d", (), (), None)


def test_description_fallback_without_description():
    assert model_notes("x", {}) == ModelNotes(None, (), (), None)


def test_curated_entry_without_notes_falls_back_to_description():
    yaml_text = """
- id: plain
  uri: hf://org/plain
  description: curated only
"""
    with _patch_yaml(yaml_text):
        n = model_notes("plain", {"description": "manifest description"})
    assert n == ModelNotes("manifest description", (), (), None)


def test_manifest_invalid_tier_degrades_to_none():
    n = model_notes("x", {"summary": "s", "quality_tier": "great"})
    assert n.summary == "s"
    assert n.quality_tier is None


def test_curated_entry_parses_note_fields():
    with _patch_yaml(_CURATED_WITH_NOTES):
        (entry,) = load_curated()
    assert entry.summary == "Fast stills for drafts"
    assert entry.strengths == ("fast", "small")
    assert entry.limits == ("square only",)
    assert entry.quality_tier == "draft"


def test_curated_note_fields_default_empty():
    with _patch_yaml("- id: bare\n  uri: hf://a/b\n"):
        (entry,) = load_curated()
    assert entry.summary is None
    assert entry.strengths == ()
    assert entry.limits == ()
    assert entry.quality_tier is None


def test_bad_tier_rejected_at_load():
    with pytest.raises(ValueError, match="bad-tier"):
        _entry_from_dict({"id": "bad-tier", "uri": "hf://a/b", "quality_tier": "great"})
    # load_curated keeps its skip-and-warn contract: the entry is dropped.
    yaml_text = """
- id: bad-tier
  uri: hf://a/b
  quality_tier: great
- id: ok
  uri: hf://c/d
  quality_tier: good
"""
    with _patch_yaml(yaml_text):
        assert [e.id for e in load_curated()] == ["ok"]


@pytest.mark.parametrize("field,value", [
    ("summary", ["not", "a", "string"]),
    ("summary", ""),
    ("strengths", "not a list"),
    ("limits", [1, 2]),
    ("limits", [""]),
])
def test_bad_note_shapes_rejected_at_load(field, value):
    with pytest.raises(ValueError, match="bad-shape"):
        _entry_from_dict({"id": "bad-shape", "uri": "hf://a/b", field: value})


def test_bundled_curated_entry_cannot_carry_notes():
    # Bundled aliases take their notes from the script MANIFEST.
    with pytest.raises(ValueError, match="alias"):
        _entry_from_dict({"id": "alias", "bundled": True, "summary": "s"})


def test_every_shipped_curated_entry_loads():
    """A note typo must not silently drop a shipped entry."""
    text = files("muse").joinpath("curated.yaml").read_text()
    expected = sum(1 for line in text.splitlines() if line.startswith("- id:"))
    assert len(load_curated()) == expected


def test_every_bundled_and_curated_model_has_notes():
    bundled = [e for e in known_models().values() if e.bundled]
    scripts = [
        p for p in files("muse").joinpath("models").iterdir()
        if p.name.endswith(".py") and p.name != "__init__.py"
    ]
    assert len(bundled) == len(scripts)
    for entry in bundled:
        n = model_notes(entry.model_id, get_manifest(entry.model_id))
        assert n.summary and n.quality_tier in QUALITY_TIERS, entry.model_id
        assert 1 <= len(n.strengths) <= 3, entry.model_id
        assert 1 <= len(n.limits) <= 3, entry.model_id
    for c in load_curated():
        if not c.bundled:
            assert c.summary and c.quality_tier in QUALITY_TIERS, c.id
            assert 1 <= len(c.strengths) <= 3, c.id
            assert 1 <= len(c.limits) <= 3, c.id


def test_non_commercial_licenses_are_named_in_limits():
    """CC-BY-NC models must say so in limits, not only in description."""
    rows = []
    for e in known_models().values():
        if e.bundled:
            m = get_manifest(e.model_id)
            rows.append((e.model_id, str(m.get("license", "")), model_notes(e.model_id, m).limits))
    for c in load_curated():
        if not c.bundled:
            rows.append((c.id, c.description or "", c.limits))
    for model_id, license_text, limits in rows:
        if "NC" in license_text.upper().replace("-", " ").split():
            assert any("non-commercial" in lim.lower() for lim in limits), model_id


def test_kokoro_limits_mention_word_timings_and_cloning():
    n = model_notes("kokoro-82m", get_manifest("kokoro-82m"))
    assert n.quality_tier == "good"
    text = " ".join(n.limits).lower()
    assert "word timing" in text
    assert "cloning" in text
