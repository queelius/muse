"""Tests for the text/translation HF resolver plugin.

Mirrors tests/modalities/text_summarization/test_hf_plugin.py in shape.
Also pins the cross-plugin disambiguation behavior described in
docs/superpowers/specs/2026-07-09-text-translation-design.md: a repo
tagged BOTH `summarization` and `translation` resolves to translation
only when its name matches a known translation family (m2m100, nllb,
opus-mt, madlad); otherwise it is left to text/summarization's plugin.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.text_translation.hf import HF_PLUGIN


def _fake_info(repo_id, tags=(), siblings=(), card=None):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "text/translation"
    assert HF_PLUGIN["priority"] == 110


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.text_translation.runtimes.hf_translation"
        ":TranslationRuntime"
    )


def test_plugin_pip_extras_includes_torch_transformers_sentencepiece():
    extras = HF_PLUGIN["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any("transformers" in e for e in extras)
    assert any("sentencepiece" in e for e in extras)


# ---------- sniff: translation tag ----------

def test_sniff_true_for_translation_tag():
    info = _fake_info("facebook/m2m100_418M", tags=["translation"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_translation_tag_with_others():
    info = _fake_info(
        "facebook/nllb-200-distilled-600M",
        tags=["translation", "text2text-generation", "transformers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


# ---------- sniff: name-pattern fallback (each family) ----------

def test_sniff_true_for_m2m100_name_no_tags():
    info = _fake_info("facebook/m2m100_418M", tags=["text2text-generation"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_nllb_name_no_tags():
    info = _fake_info("facebook/nllb-200-distilled-600M", tags=[])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_opus_mt_name_no_tags():
    info = _fake_info("Helsinki-NLP/opus-mt-en-es", tags=[])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_madlad_name_no_tags():
    info = _fake_info("google/madlad400-3b-mt", tags=["text2text-generation"])
    assert HF_PLUGIN["sniff"](info) is True


# ---------- sniff: rejections ----------

def test_sniff_false_for_bare_summarization_repo():
    """A generic summarization repo (no translation tag, no family-name
    match) must not be claimed here -- it belongs to text/summarization.
    """
    info = _fake_info("facebook/bart-large-cnn", tags=["summarization"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_text_classification_only():
    info = _fake_info("KoalaAI/Text-Moderation", tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_repo_without_tags_or_family_name():
    info = _fake_info("acme/empty", tags=[])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_no_tags_attribute():
    info = SimpleNamespace(id="x/y", siblings=[], card_data=None)
    assert HF_PLUGIN["sniff"](info) is False


# ---------- sniff: both-tagged disambiguation (spec-normative) ----------

def test_sniff_both_tagged_true_when_name_matches_translation_family():
    """Tagged both summarization and translation, name matches a
    translation family (opus-mt) -> claimed here.
    """
    info = _fake_info(
        "acme/opus-mt-en-es-hybrid",
        tags=["summarization", "translation"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_both_tagged_false_when_name_does_not_match_translation_family():
    """Tagged both summarization and translation, name does NOT match
    any translation family -> left to text/summarization's plugin.
    """
    info = _fake_info(
        "acme/generic-seq2seq-model",
        tags=["summarization", "translation"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_summarization_plugin_wins_priority_tie_for_bare_summarization_tag():
    """Documents the actual cross-plugin dispatch outcome (see the
    comment above HF_PLUGIN['priority'] in hf.py): text/summarization
    and text/translation share priority 110, and HFResolver's dispatch
    sorts ties alphabetically by modality ('text/summarization' <
    'text/translation'), so summarization's plugin is always consulted
    FIRST. Its own sniff (`"summarization" in tags`) claims any
    summarization-tagged repo unconditionally, independent of this
    plugin's disambiguation logic. This test pins that ordering fact
    directly (no live HFResolver instantiation needed).
    """
    from muse.modalities.text_summarization.hf import HF_PLUGIN as SUMM_PLUGIN

    assert SUMM_PLUGIN["priority"] == HF_PLUGIN["priority"] == 110
    plugins = sorted(
        [HF_PLUGIN, SUMM_PLUGIN], key=lambda p: (p["priority"], p["modality"]),
    )
    assert plugins[0] is SUMM_PLUGIN
    assert plugins[1] is HF_PLUGIN


# ---------- resolve: manifest shape ----------

def test_resolve_synthesizes_manifest_for_m2m100():
    info = _fake_info(
        "facebook/m2m100_418M",
        tags=["translation"],
        card=SimpleNamespace(license="mit"),
    )
    resolved = HF_PLUGIN["resolve"]("facebook/m2m100_418M", None, info)
    m = resolved.manifest
    assert m["model_id"] == "m2m100_418m"
    assert m["modality"] == "text/translation"
    assert m["hf_repo"] == "facebook/m2m100_418M"
    assert m["license"] == "mit"
    assert m["pip_extras"] == list(HF_PLUGIN["pip_extras"])
    assert m["capabilities"]["device"] == "auto"
    # No opus pair for a non-opus-mt repo.
    assert "source_language" not in m["capabilities"]
    assert "target_language" not in m["capabilities"]


def test_resolve_backend_path_points_to_translation_runtime():
    info = _fake_info("acme/x-nllb-mini", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("acme/x-nllb-mini", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_resolve_when_card_data_missing_license_is_none():
    info = _fake_info("acme/x-madlad-mini", tags=["translation"], card=None)
    resolved = HF_PLUGIN["resolve"]("acme/x-madlad-mini", None, info)
    assert resolved.manifest["license"] is None


# ---------- resolve: opus-mt pair parsing ----------

def test_resolve_parses_opus_pair_two_letter_codes():
    info = _fake_info("Helsinki-NLP/opus-mt-en-es", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("Helsinki-NLP/opus-mt-en-es", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["source_language"] == "en"
    assert caps["target_language"] == "es"


def test_resolve_parses_opus_pair_en_de():
    info = _fake_info("Helsinki-NLP/opus-mt-en-de", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("Helsinki-NLP/opus-mt-en-de", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["source_language"] == "en"
    assert caps["target_language"] == "de"


def test_resolve_skips_opus_pair_for_non_two_letter_target():
    """opus-mt-en-ROMANCE targets a language *group*, not an ISO-639-1
    code. Per design decision: skip pair extraction and synthesize
    WITHOUT the pair rather than guessing.
    """
    info = _fake_info("Helsinki-NLP/opus-mt-en-ROMANCE", tags=["translation"])
    resolved = HF_PLUGIN["resolve"](
        "Helsinki-NLP/opus-mt-en-ROMANCE", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert "source_language" not in caps
    assert "target_language" not in caps


def test_resolve_skips_opus_pair_for_multi_part_variant_name():
    """opus-mt-tc-big-en-pt has more than two dash-separated segments
    after the opus-mt- prefix; do not guess a pair from it.
    """
    info = _fake_info("Helsinki-NLP/opus-mt-tc-big-en-pt", tags=["translation"])
    resolved = HF_PLUGIN["resolve"](
        "Helsinki-NLP/opus-mt-tc-big-en-pt", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert "source_language" not in caps
    assert "target_language" not in caps


def test_resolve_skips_opus_pair_for_three_letter_code():
    """A 3-letter macro-language code (not plain ISO-639-1) should also
    be skipped rather than guessed.
    """
    info = _fake_info("Helsinki-NLP/opus-mt-en-itc", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("Helsinki-NLP/opus-mt-en-itc", None, info)
    caps = resolved.manifest["capabilities"]
    assert "source_language" not in caps
    assert "target_language" not in caps


# ---------- search ----------

def test_search_yields_results():
    api = MagicMock()
    repo1 = SimpleNamespace(id="facebook/m2m100_418M", downloads=2_000_000)
    repo2 = SimpleNamespace(id="Helsinki-NLP/opus-mt-en-es", downloads=800_000)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](
        api, "translat", sort="downloads", limit=10,
    ))
    assert len(out) == 2
    assert out[0].uri == "hf://facebook/m2m100_418M"
    assert out[0].modality == "text/translation"
    assert out[0].downloads == 2_000_000
    assert out[1].model_id == "opus-mt-en-es"


def test_search_calls_list_models_with_translation_filter():
    api = MagicMock()
    api.list_models.return_value = []
    list(HF_PLUGIN["search"](api, "x", sort="downloads", limit=5))
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == "translation"
    assert kwargs["sort"] == "downloads"
    assert kwargs["limit"] == 5
    assert kwargs["search"] == "x"
