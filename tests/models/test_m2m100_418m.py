"""Tests for the bundled m2m100_418m script (fully mocked) + the three
new curated.yaml translation entries (nllb-200-distilled-600m,
opus-mt-en-es, opus-mt-en-de).

Module-level imports of `muse.models.m2m100_418m` are DELIBERATELY
avoided (mirrors tests/models/test_bart_large_cnn.py): another test in
the suite (test_discovery_robust_to_broken_deps) pops `muse.models.*`
from sys.modules and re-imports them, which would leave a top-level
import holding a stale module reference.
"""
from __future__ import annotations

import importlib

from muse.core.curated import load_curated, _reset_curated_cache_for_tests
from muse.modalities.text_translation.runtimes.hf_translation import (
    TranslationRuntime,
)


def _m2m100_script():
    return importlib.import_module("muse.models.m2m100_418m")


def _manifest():
    return _m2m100_script().MANIFEST


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "m2m100-418m"
    assert m["modality"] == "text/translation"
    assert m["hf_repo"] == "facebook/m2m100_418M"
    assert m["license"] == "MIT"


def test_manifest_description_mentions_languages_and_route():
    desc = _manifest()["description"]
    assert "100 languages" in desc
    assert "/v1/translate" in desc or "LibreTranslate" in desc


def test_manifest_pip_extras():
    extras = _manifest()["pip_extras"]
    dists = {e.split(">")[0].split("=")[0].split("<")[0].strip() for e in extras}
    assert "transformers" in dists
    assert "torch" in dists
    assert "sentencepiece" in dists


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "auto"
    assert caps["memory_gb"] == 2.5
    assert caps["num_beams"] == 4


def test_model_is_translation_runtime_alias():
    """The VLM-bundled pattern: Model IS the shared runtime class, not
    a wrapper that duplicates construction logic."""
    m2m100 = _m2m100_script()
    assert m2m100.Model is TranslationRuntime


# ---------- curated.yaml: nllb-200-distilled-600m ----------


def _reset_curated():
    _reset_curated_cache_for_tests()


def _find(entries, id_):
    for e in entries:
        if e.id == id_:
            return e
    return None


def test_curated_nllb_entry_present():
    _reset_curated()
    entries = load_curated()
    e = _find(entries, "nllb-200-distilled-600m")
    assert e is not None
    assert e.uri == "hf://facebook/nllb-200-distilled-600M"
    assert e.modality == "text/translation"


def test_curated_nllb_description_carries_nc_warning_and_lang_count():
    _reset_curated()
    entries = load_curated()
    e = _find(entries, "nllb-200-distilled-600m")
    assert e is not None
    assert "CC-BY-NC-4.0 (non-commercial)" in e.description
    assert "200 languages" in e.description


# ---------- curated.yaml: opus-mt-en-es / opus-mt-en-de ----------


def test_curated_opus_en_es_entry_present():
    _reset_curated()
    entries = load_curated()
    e = _find(entries, "opus-mt-en-es")
    assert e is not None
    assert e.uri == "hf://Helsinki-NLP/opus-mt-en-es"
    assert e.modality == "text/translation"


def test_curated_opus_en_de_entry_present():
    _reset_curated()
    entries = load_curated()
    e = _find(entries, "opus-mt-en-de")
    assert e is not None
    assert e.uri == "hf://Helsinki-NLP/opus-mt-en-de"
    assert e.modality == "text/translation"


def test_curated_opus_entries_declare_source_and_target_language():
    """T2 review finding: an opus-mt model without a declared
    source_language/target_language pair is silently non-functional --
    TranslationRuntime derives the fixed pair ONLY from the manifest.
    Both curated opus entries MUST carry both capability keys."""
    _reset_curated()
    entries = load_curated()
    for id_, src, tgt in (
        ("opus-mt-en-es", "en", "es"),
        ("opus-mt-en-de", "en", "de"),
    ):
        e = _find(entries, id_)
        assert e is not None, f"missing curated entry {id_}"
        assert e.capabilities.get("source_language") == src, id_
        assert e.capabilities.get("target_language") == tgt, id_
